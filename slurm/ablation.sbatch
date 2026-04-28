#!/bin/bash
#SBATCH --job-name=poi-ablation
#SBATCH --output=logs/slurm-ablation-%j.out
#SBATCH --error=logs/slurm-ablation-%j.err
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=shard:1

# Encoder ablation: build an index and evaluate retrieval quality for
# CLIP, SigLIP-2 base, and SigLIP-2 large in one job. Each combination
# emits a JSON file under evals/. The aggregate table in evals/results.md
# is regenerated from these files by notebooks/03_evaluation.ipynb.

set -euo pipefail

echo "Job started on $(hostname) at $(date)"

source "${POI_VENV:-$HOME/poi/.venv}/bin/activate"
cd "${SLURM_SUBMIT_DIR}"

mkdir -p logs artifacts evals

CONFIGS=(
    "configs/clip_baseline.yaml"
    "configs/default.yaml"
    "configs/siglip2_large.yaml"
)
IMAGES="${IMAGES:-data/celeba/img_align_celeba}"
N_QUERIES="${N_QUERIES:-500}"

for cfg in "${CONFIGS[@]}"; do
    name="$(basename "${cfg}" .yaml)"
    index_path="artifacts/celeba_${name}.index"
    eval_out="evals/results_${name}.json"

    echo ""
    echo "=========================================="
    echo "  ${name}"
    echo "=========================================="

    # Skip rebuild if the index already exists — cheaper to just re-eval.
    if [ ! -f "${index_path}" ]; then
        echo "Building index: ${index_path}"
        python scripts/build_index.py \
            --config "${cfg}" \
            --images "${IMAGES}" \
            --out "${index_path}"
    else
        echo "Reusing existing index: ${index_path}"
    fi

    echo "Evaluating: ${eval_out}"
    python scripts/evaluate.py \
        --config "${cfg}" \
        --index "${index_path}" \
        --n-queries "${N_QUERIES}" \
        --no-vlm \
        --out "${eval_out}"
done

echo ""
echo "Ablation complete at $(date)"
echo "Results: evals/results_*.json"
