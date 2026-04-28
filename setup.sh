#!/usr/bin/env bash
# Initializes a fresh git repository with the project as an initial commit.
# Run once after extracting the archive.

set -euo pipefail

if [ -d .git ]; then
    echo "git repo already initialized; aborting"
    exit 1
fi

git init -q
git add .
git commit -q -m "Initial commit: Person of Interest

Multimodal face retrieval with vision-language reasoning.
SigLIP-2 + FAISS + Qwen2.5-VL pipeline over CelebA.
"

echo "Repository initialized."
echo ""
echo "Next steps:"
echo "  1. Create a new repository on GitHub: https://github.com/new"
echo "  2. Add the remote and push:"
echo "       git remote add origin git@github.com:<you>/person-of-interest.git"
echo "       git branch -M main"
echo "       git push -u origin main"
echo ""
echo "  3. On the cluster, install dependencies:"
echo "       python -m venv .venv && source .venv/bin/activate"
echo "       pip install -e \".[dev,eval]\""
echo ""
echo "  4. Build an index and run the eval:"
echo "       sbatch slurm/build_index.sbatch"
echo "       sbatch slurm/evaluate.sbatch"
