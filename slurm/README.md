# SLURM submission scripts

These scripts target the SupportVectors cluster, which has a mix of RTX 3090, 4090, 5090, and RTX PRO 6000 nodes accessed via SLURM.

## When to use shared shards vs full GPUs

Per the cluster's resource policy:

- **Shared shard (`--gres=shard:1`)** is the default. Each shard provides 12–16 GB VRAM, sufficient for everything in this repo: index build, evaluation, and serving.
- **Full GPU (`--gres=gpu:1`)** is only needed for fine-tuning. None of the scripts here ask for one.

If you're running on a different cluster, edit the `--gres` lines at the top of each `.sbatch` file.

## Setup (once)

```bash
# 1. Clone and create a venv on the cluster
git clone https://github.com/<you>/person-of-interest.git
cd person-of-interest
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,eval]"

# 2. Set the venv path so the SLURM scripts can find it
export POI_VENV="$PWD/.venv"
echo "export POI_VENV=$PWD/.venv" >> ~/.bashrc

# 3. Download data
python scripts/download_data.py --dataset celeba --out data/
```

## Building an index

```bash
sbatch slurm/build_index.sbatch
```

Defaults: `configs/default.yaml`, `data/celeba/img_align_celeba`, output at `artifacts/celeba_default.index`. Override with environment variables:

```bash
CONFIG=configs/siglip2_large.yaml \
    OUT=artifacts/celeba_large.index \
    sbatch slurm/build_index.sbatch
```

Expected runtime on a 4090 shard: ~25 min for SigLIP-2 base on full CelebA, ~75 min for SigLIP-2 large.

## Running the evaluation

```bash
INDEX=artifacts/celeba_default.index sbatch slurm/evaluate.sbatch
```

Writes results to `evals/results_default.json`. Combine with the ablation script below to get the full encoder comparison table.

## Encoder ablation (CLIP vs SigLIP-2 base vs SigLIP-2 large)

```bash
sbatch slurm/ablation.sbatch
```

Builds three indexes (skipping any that already exist) and evaluates each. Total runtime ~3 hours on a 4090 shard. The output JSON files are aggregated by `notebooks/03_evaluation.ipynb` into the table in [evals/results.md](../evals/results.md).

## Serving the UI

```bash
INDEX=artifacts/celeba_default.index sbatch slurm/serve.sbatch
```

The job logs will show which compute node it landed on. Tunnel from your local machine:

```bash
ssh -L 8080:<compute-node>:8080 <cluster-login>
```

Then open `http://localhost:8080`.

## Picking a node

You can target a specific node type by extending the `--gres` line. From the cluster's `nodes.md`:

| Node | GPU | Use case |
|---|---|---|
| `archimedes`, `deepseek` | RTX 5090 (32 GB) | Best per-node throughput; SigLIP-2 large + Qwen-VL-7B comfortably |
| `sapphire` | RTX PRO 6000 (96 GB) | When you want headroom, e.g. trying Qwen-VL-32B someday |
| `hinton`, `gradientdescent`, `kolmogorov`, `inference`, `bourbaki`, `ansatz` | RTX 4090 (24 GB) ×2 | Default workhorses |
| `feynman`, `saras` | RTX 3090 (24 GB) ×2 | Fine for everything except the largest VLMs |

Example, request a 5090:

```bash
sbatch --gres=shard:1 --partition=archimedes,deepseek slurm/build_index.sbatch
```
