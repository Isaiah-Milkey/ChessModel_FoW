#!/usr/bin/env bash
# One command on a Sol login node, from the repository root:
#     bash scripts/sol/submit_move_history.sh
#
# Sets up the environment if needed, then submits three jobs:
#   1. record_data.sbatch: 150,000 games (seed 3), recorded, encoded for move_history and baseline
#   2. train_agent.sbatch: agents/move_history on data/rec150k_s3_hist       (after job 1)
#   3. train_agent.sbatch: runs/base150k (baseline_75k code) on data/rec150k_s3_base, the control
# It stops without submitting anything if a step fails or the output folders already exist.
set -euo pipefail
[ -f scripts/sol/record_data.sbatch ] || { echo "run this from the repository root"; exit 1; }
if [ -e data/rec150k_s3 ] || [ -e data/rec150k_s3_hist ] || [ -e data/rec150k_s3_base ] || [ -e runs/base150k ]; then
    echo "data/rec150k_s3* or runs/base150k already exists; not submitting again."
    echo "Check 'squeue -u ${USER:-$(whoami)}' and runs/*.out, or move those folders away to start over."
    exit 1
fi

set +u   # module and conda scripts use unset variables
module load mamba/latest
if ! source activate fow 2>/dev/null; then
    echo "== first run: creating the fow environment (about 5 minutes)"
    bash scripts/sol/setup_env.sh
    source activate fow
fi
set -u
python -c "import fow, torch, chess, safetensors" || { echo "the fow environment is missing packages"; exit 1; }

BIN=$(find "$PWD/stockfish" -type f -name 'stockfish-ubuntu-x86-64-avx2' 2>/dev/null | head -n 1 || true)
if [ -z "$BIN" ]; then
    echo "== downloading Stockfish"
    bash scripts/get_stockfish.sh
    BIN=$(find "$PWD/stockfish" -type f -name 'stockfish-ubuntu-x86-64-avx2' | head -n 1)
fi
export STOCKFISH=$BIN
SF_OUT=$(echo uci | timeout 20 "$STOCKFISH" 2>&1 || true)
grep -q uciok <<< "$SF_OUT" || { echo "Stockfish at $STOCKFISH does not start"; exit 1; }
echo "Stockfish: $STOCKFISH"

mkdir -p runs data
REC=$(sbatch --parsable scripts/sol/record_data.sbatch)
HIST=$(AGENT=agents/move_history DATA=data/rec150k_s3_hist \
       sbatch --parsable --dependency=afterok:"$REC" scripts/sol/train_agent.sbatch)
BASE=$(AGENT=runs/base150k FROM=agents/baseline_75k DATA=data/rec150k_s3_base \
       sbatch --parsable --dependency=afterok:"$REC" scripts/sol/train_agent.sbatch)
echo
echo "submitted: record $REC, train move_history $HIST, train control $BASE (both wait for $REC)"
echo "watch:     squeue -u ${USER:-$(whoami)}        tail -f runs/fow-record-$REC.out"
echo "cancel:    scancel $REC $HIST $BASE"
