#!/bin/bash
# GPU job: train an agent's network on a data directory, keep the best validation epoch, validate.
#
#     AGENT=agents/move_history DATA=data/rec150k_s3_hist sbatch scripts/sol/train_agent.sbatch
#     AGENT=runs/base150k FROM=agents/baseline_75k DATA=data/rec150k_s3_base sbatch scripts/sol/train_agent.sbatch
#
# AGENT is the directory the weights and config.json are written into; it must hold the agent's
# code (model.py, features.py, ...). FROM copies the code of another agent into AGENT first.
# Optional: EPOCHS (16), CHANNELS (64), BLOCKS (4), TRAIN_SEED (0).
#SBATCH -J fow-train
#SBATCH -q class
#SBATCH -p general            # partition and QoS names: see ASU Research Computing docs
#SBATCH -G 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o runs/%x-%j.out
module load mamba/latest
source activate fow
set -eo pipefail
AGENT=${AGENT:?set AGENT to the agent directory to train into}
DATA=${DATA:?set DATA to a directory of shard_*.npz files}
if [ -n "$FROM" ]; then
    mkdir -p "$AGENT"
    cp "$FROM"/*.py "$AGENT"/
fi
[ -f "$AGENT/MODEL.md" ] || echo "# Model card: placeholder written by train_agent.sbatch" > "$AGENT/MODEL.md"
echo "training $AGENT on $DATA ($(date)); GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
SECONDS=0
python baselines/imitation/train.py --data "$DATA" --agent-dir "$AGENT" --channels "${CHANNELS:-64}" \
    --blocks "${BLOCKS:-4}" --epochs "${EPOCHS:-16}" --seed "${TRAIN_SEED:-0}" --device cuda --save-best
echo "training took $((SECONDS / 60)) min of GPU time"
python -m fow.params "$AGENT"
sha256sum "$AGENT/weights/model.safetensors"
python -m fow.validate "$AGENT" --device cuda
