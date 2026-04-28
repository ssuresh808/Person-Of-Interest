#!/bin/bash
#SBATCH --job-name=poi-build-index
#SBATCH --output=logs/slurm-build-%j.out
#SBATCH --error=logs/slurm-build-%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:1
# A shared GPU shard (12-16 GB VRAM) is plenty for index build.
# Use --gres=gpu:1 only if you're fine-tuning.

set -euo pipefail

# --- Environment ---
echo "Job started on $(hostname) at $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader || echo 'none')"

# Activate venv. Adjust path to wherever yours lives on the cluster.
source "${POI_VENV:-$HOME/poi/.venv}/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs artifacts

# --- Config ---
CONFIG="${CONFIG:-configs/default.yaml}"
IMAGES="${IMAGES:-data/celeba/img_align_celeba}"
OUT="${OUT:-artifacts/celeba_$(basename "${CONFIG}" .yaml).index}"

echo "Config: ${CONFIG}"
echo "Images: ${IMAGES}"
echo "Output: ${OUT}"

# --- Run ---
python scripts/build_index.py \
    --config "${CONFIG}" \
    --images "${IMAGES}" \
    --out "${OUT}"

echo "Job finished at $(date)"
