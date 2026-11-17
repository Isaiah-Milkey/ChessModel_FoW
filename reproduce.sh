#!/usr/bin/env bash
# Reproduce submission/ (agents/turn_transformer_rl) end to end, from a fresh clone. See REPRODUCE.md.
#
#   bash reproduce.sh          # uses $STOCKFISH, $DEVICE (cpu or cuda, default: cuda if available)
#
# Stages (each skipped if its output already exists, so a job that hits a time limit can rerun it):
#   1. 75,000 bot games labelled by Stockfish -> train agents/move_history
#   2. 54,000 branching games played by move_history, labelled by Stockfish
#   3. train agents/move_history_branch on them (it plays the value warm-up games in stage 5)
#   4. encode the same games as turn sequences -> train agents/turn_transformer
#   5. 60,000 games of move_history_branch against the bots (value warm-up for the loop)
#   6. learning loop from turn_transformer until the rating plateaus -> runs/rl_tf/best
#   7. copy the best round's weights into submission/ and check it
# It retrains the committed agents in place (agents/move_history, agents/move_history_branch,
# agents/turn_transformer). About 6.5 hours on a 16-thread PC with one RTX 3060 Ti.
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
: "${STOCKFISH:?set STOCKFISH to your Stockfish binary (bash scripts/get_stockfish.sh)}"
python -c "import psutil" 2>/dev/null || echo "note: psutil is not installed (pip install psutil); the loop then skips its memory checks"
t0=$(date +%s)
stamp() { echo "== $1 ($(( ($(date +%s) - t0) / 60 )) min so far)"; }

stamp "1a. data for move_history: 75,000 bot-vs-bot games, Stockfish labels (5,000 nodes)"
if [ ! -d data/rec75k_s3_hist ]; then
  python baselines/imitation/generate_data.py --games 75000 --workers 12 --seed 3 --out data/rec75k_s3 \
      --agent-dir agents/move_history --record
  mkdir -p data/rec75k_s3_hist && mv data/rec75k_s3/shard_*.npz data/rec75k_s3_hist/
fi
stamp "1b. train agents/move_history"
[ -f data/.done_move_history ] || { python baselines/imitation/train.py --data data/rec75k_s3_hist \
    --agent-dir agents/move_history --channels 64 --blocks 4 --epochs 16 --seed 0 --device "$DEVICE" --save-best \
    && touch data/.done_move_history; }

stamp "2. 54,000 branching games played by move_history, Stockfish labels"
if [ ! -d data/branch54k_s4_hist ]; then
  python baselines/imitation/generate_data.py --games 54000 --workers 10 --seed 4 --out data/branch54k_s4 \
      --agent-dir agents/move_history --record --random-plies 10 --keep-schedule "0:0.1,10:0.5,20:1" \
      --matchups long --play-agent agents/move_history
  mkdir -p data/branch54k_s4_hist && mv data/branch54k_s4/shard_*.npz data/branch54k_s4_hist/
fi

stamp "3. train agents/move_history_branch"
[ -f data/.done_move_history_branch ] || { python baselines/imitation/train.py --data data/branch54k_s4_hist \
    --agent-dir agents/move_history_branch --channels 64 --blocks 4 --epochs 16 --seed 0 --device "$DEVICE" \
    --save-best --split view && touch data/.done_move_history_branch; }

stamp "4. turn sequences of the same games -> train agents/turn_transformer"
[ -d data/branch54k_s4_seq ] || python baselines/sequence/encode_seq.py --games data/branch54k_s4 --out data/branch54k_s4_seq
[ -f data/.done_turn_transformer ] || { python baselines/sequence/train_seq.py --data data/branch54k_s4_seq \
    --agent-dir agents/turn_transformer --device "$DEVICE" && touch data/.done_turn_transformer; }

stamp "5. 60,000 games of move_history_branch against the bots (value warm-up)"
[ -f data/fow_games/meta.json ] || python baselines/fow_games/play_games.py --agent-dir agents/move_history_branch \
    --games 60000 --workers 12 --seed 100 --out data/fow_games

stamp "6. learning loop from turn_transformer until the rating plateaus"
if [ -f runs/rl_tf/config.json ]; then
  python baselines/fow_games/rl_loop.py --out runs/rl_tf --resume --rounds 40 --patience 3 --workers 12
else
  python baselines/fow_games/rl_loop.py --init agents/turn_transformer --out runs/rl_tf --rounds 40 --patience 3 \
      --workers 12 --value-games data/fow_games --device "$DEVICE"
fi

stamp "7. assemble and check submission/ (its code is committed; only the weights are copied)"
cp runs/rl_tf/best/weights/model.safetensors submission/weights/model.safetensors
python -m fow.params submission
python -m fow.validate submission --device cpu
sha256sum submission/weights/*.safetensors
stamp "done"
