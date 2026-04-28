#!/bin/bash
#SBATCH --job-name=poi-serve
#SBATCH --output=logs/slurm-serve-%j.out
#SBATCH --error=logs/slurm-serve-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=shard:1
# Serving needs more RAM because Qwen2.5-VL-7B sits resident.
# Still fits a single shard for VRAM.

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
echo "Compute node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'none')"

source "${POI_VENV:-$HOME/poi/.venv}/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs

CONFIG="${CONFIG:-configs/default.yaml}"
INDEX="${INDEX:-artifacts/celeba_default.index}"
PORT="${POI_PORT:-8080}"

echo "Config: ${CONFIG}"
echo "Index:  ${INDEX}"
echo ""
echo "After the server starts, tunnel from your laptop with:"
echo "  ssh -L ${PORT}:$(hostname):${PORT} <login-node>"
echo "Then open http://localhost:${PORT}"

python -m poi.ui.app \
    --config "${CONFIG}" \
    --index "${INDEX}" \
    --host 0.0.0.0 \
    --port "${PORT}"
