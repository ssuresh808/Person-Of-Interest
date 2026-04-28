#!/bin/bash
#SBATCH --job-name=poi-evaluate
#SBATCH --output=logs/slurm-eval-%j.out
#SBATCH --error=logs/slurm-eval-%j.err
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=shard:1

set -euo pipefail

echo "Job started on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

source "${POI_VENV:-$HOME/poi/.venv}/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs evals

CONFIG="${CONFIG:-configs/default.yaml}"
INDEX="${INDEX:-artifacts/celeba_default.index}"
N_QUERIES="${N_QUERIES:-500}"
OUT="${OUT:-evals/results_$(basename "${CONFIG}" .yaml).json}"

echo "Config: ${CONFIG}"
echo "Index:  ${INDEX}"
echo "N:      ${N_QUERIES}"
echo "Out:    ${OUT}"

python scripts/evaluate.py \
    --config "${CONFIG}" \
    --index "${INDEX}" \
    --n-queries "${N_QUERIES}" \
    --no-vlm \
    --out "${OUT}"

echo "Job finished at $(date)"
