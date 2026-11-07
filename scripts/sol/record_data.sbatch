#!/bin/bash
# CPU job: play and label games once, keeping the games so later input changes need no Stockfish.
#
#     export STOCKFISH=<path from setup_env.sh>
#     sbatch scripts/sol/record_data.sbatch                      # 150,000 games, seed 3
#     GAMES=50000 SEED=4 sbatch scripts/sol/record_data.sbatch   # other sizes or seeds
#
# Writes (NAME defaults to rec<GAMES/1000>k_s<SEED>):
#   data/NAME/       games_<seed>_<worker>.jsonl.gz: moves and teacher labels (keep these)
#   data/NAME_hist/  shards encoded for agents/move_history (29 planes)
#   data/NAME_base/  the same positions with the baseline encoding (22 planes), as a control
# Any other agent's shards: python baselines/imitation/reencode.py --games data/NAME --agent-dir ... --out ...
#SBATCH -J fow-record
#SBATCH -q class
#SBATCH -p general            # partition and QoS names: see ASU Research Computing docs
#SBATCH -c 32
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -o runs/%x-%j.out
module load mamba/latest
source activate fow
set -eo pipefail
export STOCKFISH=${STOCKFISH:?set STOCKFISH to your Stockfish binary}
GAMES=${GAMES:-150000}
SEED=${SEED:-3}
NAME=${NAME:-rec$((GAMES / 1000))k_s$SEED}
OUT=data/$NAME
echo "recording $GAMES games, seed $SEED, into $OUT ($(date))"
SECONDS=0
python baselines/imitation/generate_data.py --games "$GAMES" --workers 32 --seed "$SEED" \
    --out "$OUT" --agent-dir agents/move_history --record
mkdir -p "${OUT}_hist"
mv "$OUT"/shard_*.npz "${OUT}_hist"/
echo "generation took $((SECONDS / 60)) min"
SECONDS=0
python baselines/imitation/reencode.py --games "$OUT" --out "${OUT}_base" --workers 32
echo "baseline re-encoding took $((SECONDS / 60)) min"
ls -la "$OUT" "${OUT}_hist" "${OUT}_base"
