#!/usr/bin/env bash
# Reproduce submission/ (agents/turn_transformer_late) end to end, from a fresh clone. See REPRODUCE.md.
#
#   bash reproduce.sh            # uses $STOCKFISH, $DEVICE (cpu or cuda, default: cuda if available)
#   QUICK=1 bash reproduce.sh    # test run: every stage with tiny sizes (300 games, 1 epoch, 2 rounds),
#                                # all output under quick/ (data, agents, runs, submission); about 30 min
#
# Stages (each skipped if its output already exists, so a job that hits a time limit can rerun it):
#   1. 75,000 bot games labelled by Stockfish -> train agents/move_history
#   2. 54,000 branching games played by move_history, labelled by Stockfish
#   3. train agents/move_history_branch on them (it plays the value warm-up games in stage 5)
#   4. encode the same games as turn sequences -> train agents/turn_transformer
#   5. 60,000 games of move_history_branch against the bots (value warm-up for the loop)
#   6. learning loop from turn_transformer until the rating plateaus -> runs/rl_tf/best (= agents/turn_transformer_rl)
#   7. learning loop from that model, half its games started shortly before losses from stage 6 -> runs/rl_late/best
#   8. copy the best round's weights into submission/ and check it
# It retrains the committed agents in place (agents/move_history, agents/move_history_branch,
# agents/turn_transformer). About 8 hours on a 16-thread PC with one RTX 3060 Ti.
set -euo pipefail
cd "$(dirname "$0")"
DEVICE=${DEVICE:-$(python -c "import torch; print('cuda' if torch.cuda.is_available() else 'cpu')")}
: "${STOCKFISH:?set STOCKFISH to your Stockfish binary (bash scripts/get_stockfish.sh)}"
python -c "import psutil" 2>/dev/null || echo "note: psutil is not installed (pip install psutil); the loop then skips its memory checks"

if [ "${QUICK:-0}" = 1 ]; then
  # Test run: same commands, tiny sizes, separate folders. Checks that every stage runs here, not the result.
  D=quick/data A=quick/agents R=quick/runs S=quick/submission
  G1=300 G2=300 G5=300 EP=1 W=4
  LOOP="--games 300 --eval-games 10 --holdout-games 10"; R6=2 R7=2 MIDPLIES=10,60
  for a in move_history move_history_branch turn_transformer; do   # code only: the weights are trained below
    mkdir -p "$A/$a" && cp agents/$a/*.py agents/$a/MODEL.md "$A/$a/"
  done
  [ -d "$S" ] || cp -r submission "$S"
  echo "QUICK test run: output in quick/ (delete it afterwards)"
else
  D=data A=agents R=runs S=submission
  G1=75000 G2=54000 G5=60000 EP=16 W=12
  LOOP=""; R6=40 R7=20 MIDPLIES=20,60
fi
t0=$(date +%s)
stamp() { echo "== $1 ($(( ($(date +%s) - t0) / 60 )) min so far)"; }

stamp "1a. data for move_history: $G1 bot games, Stockfish labels (5,000 nodes)"
if [ ! -d $D/rec75k_s3_hist ]; then
  python baselines/imitation/generate_data.py --games $G1 --workers $W --seed 3 --out $D/rec75k_s3 \
      --agent-dir $A/move_history --record
  mkdir -p $D/rec75k_s3_hist && mv $D/rec75k_s3/shard_*.npz $D/rec75k_s3_hist/
fi
stamp "1b. train $A/move_history"
[ -f $D/.done_move_history ] || { python baselines/imitation/train.py --data $D/rec75k_s3_hist \
    --agent-dir $A/move_history --channels 64 --blocks 4 --epochs $EP --seed 0 --device "$DEVICE" --save-best \
    && touch $D/.done_move_history; }

stamp "2. $G2 branching games played by move_history, Stockfish labels"
if [ ! -d $D/branch54k_s4_hist ]; then
  python baselines/imitation/generate_data.py --games $G2 --workers $(( W * 10 / 12 )) --seed 4 --out $D/branch54k_s4 \
      --agent-dir $A/move_history --record --random-plies 10 --keep-schedule "0:0.1,10:0.5,20:1" \
      --matchups long --play-agent $A/move_history
  mkdir -p $D/branch54k_s4_hist && mv $D/branch54k_s4/shard_*.npz $D/branch54k_s4_hist/
fi

stamp "3. train $A/move_history_branch"
[ -f $D/.done_move_history_branch ] || { python baselines/imitation/train.py --data $D/branch54k_s4_hist \
    --agent-dir $A/move_history_branch --channels 64 --blocks 4 --epochs $EP --seed 0 --device "$DEVICE" \
    --save-best --split view && touch $D/.done_move_history_branch; }

stamp "4. turn sequences of the same games -> train $A/turn_transformer"
[ -d $D/branch54k_s4_seq ] || python baselines/sequence/encode_seq.py --games $D/branch54k_s4 --out $D/branch54k_s4_seq
[ -f $D/.done_turn_transformer ] || { python baselines/sequence/train_seq.py --data $D/branch54k_s4_seq \
    --agent-dir $A/turn_transformer --epochs $EP --device "$DEVICE" && touch $D/.done_turn_transformer; }

stamp "5. $G5 games of move_history_branch against the bots (value warm-up)"
[ -f $D/fow_games/meta.json ] || python baselines/fow_games/play_games.py --agent-dir $A/move_history_branch \
    --games $G5 --workers $W --seed 100 --out $D/fow_games

stamp "6. learning loop from turn_transformer until the rating plateaus"
if [ -f $R/rl_tf/config.json ]; then
  python baselines/fow_games/rl_loop.py --out $R/rl_tf --resume --rounds $R6 --patience 3 --workers $W
else
  python baselines/fow_games/rl_loop.py --init $A/turn_transformer --out $R/rl_tf --rounds $R6 --patience 3 \
      --workers $W --value-games $D/fow_games --device "$DEVICE" $LOOP
fi

stamp "7. learning loop from $R/rl_tf/best, half its games started 10-30 plies before a loss from stage 6"
if [ -f $R/rl_late/config.json ]; then
  python baselines/fow_games/rl_loop.py --out $R/rl_late --resume --rounds $R7 --patience 3 --workers $W \
      --holdout $A/move_history_branch
else
  python baselines/fow_games/rl_loop.py --init $R/rl_tf/best --out $R/rl_late --rounds $R7 --patience 3 --workers $W \
      --seed 8 --midgame-share 0.5 --midgame-lost --midgame-plies $MIDPLIES --midgame-games "$R/rl_tf/games/games_*.jsonl.gz" \
      --bot-weights 1,1,2 --value-init $R/rl_tf/value.safetensors --holdout $A/move_history_branch --device "$DEVICE" $LOOP
fi

stamp "8. assemble and check $S (its code is committed; only the weights are copied)"
cp $R/rl_late/best/weights/model.safetensors $S/weights/model.safetensors
python -m fow.params $S
python -m fow.validate $S --device cpu
sha256sum $S/weights/*.safetensors
stamp "done"
