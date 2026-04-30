# Person of Interest

> **Multimodal face retrieval with vision-language reasoning.**
> Find a person from a natural-language description. Built on SigLIP-2, FAISS, and Qwen2.5-VL. Pipeline validated end-to-end on synthetic data.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

> **Status.** The retrieval pipeline (encoder → FAISS index → eval harness → NiceGUI frontend) is complete and validated end-to-end on a 500-image synthetic corpus. The real-model ablation comparing CLIP-ViT-L/14, SigLIP-2-base, and SigLIP-2-large on full CelebA is scripted in [`slurm/ablation.sbatch`](slurm/ablation.sbatch) and is the next step. README will be updated with measured numbers when the run completes.

## What this is

You describe someone in natural language — "a woman with curly dark hair, glasses, and a thoughtful expression" — and the system retrieves the closest matches from a face corpus. A vision-language model (Qwen2.5-VL) then captions each match in context of your query, so you can see *why* the system thinks each face fits.

The corpus this is designed for is CelebA (~200k images). The system has been demonstrated end-to-end on a 500-image synthetic corpus that mirrors CelebA's attribute structure — enough to validate the architecture, the index, the retrieval logic, and the eval harness. The next step is the real-model run on full CelebA.

This is the geometry of meaning made tangible: text and faces inhabit a shared embedding space, and similarity in that space is similarity of meaning.

![Demo](docs/screenshots/demo.gif)

## Why this project

Most face-recognition demos do identity matching: "is this the same person?" That's a closed-world problem with a known gallery and known identities.

This project does something different and harder: **describe-then-retrieve over an open corpus**. The query has no fixed schema. The matches are not guaranteed to exist. The system has to operate on narrative, not on labels — the way a human witness describes someone they met at a conference.

That mirrors how multimodal retrieval actually behaves in production: ambiguous queries, noisy corpora, no ground truth, and a generative model that has to make sense of what came back.

## Architecture

```
                    ┌──────────────────────────┐
   Natural language │  "a woman with curly     │
   query ──────────▶│   dark hair, glasses..." │
                    └────────────┬─────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │     SigLIP-2 text        │   shared
                    │       encoder            │   hypersphere
                    └────────────┬─────────────┘   (1024-D)
                                 │
                                 ▼
                    ┌──────────────────────────┐
                    │   FAISS index (IVF-Flat) │
                    │   ~200k face embeddings  │◀── built once,
                    │   normalized, cosine sim │    cached on disk
                    └────────────┬─────────────┘
                                 │ top-k
                                 ▼
                    ┌──────────────────────────┐
                    │   Qwen2.5-VL-7B          │
                    │   per-result captioning  │   "Why does this
                    │   conditioned on query   │    face match?"
                    └────────────┬─────────────┘
                                 │
                                 ▼
                          NiceGUI frontend
```

Three components, three different roles:

1. **SigLIP-2** does the alignment. Text and image embeddings live on the same unit hypersphere because contrastive training put them there. Retrieval is angular proximity.
2. **FAISS** does the search. Brute-force cosine similarity is fast on small corpora; IVF-Flat scales to the full 200k.
3. **Qwen2.5-VL** does the reasoning. It looks at the retrieved face, reads the original query, and writes a sentence about whether they match — and where the system might have gotten it wrong.

The full design rationale, including why SigLIP-2 over CLIP and why a separate VLM stage, is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Results

### Pipeline validation (synthetic corpus)

A 500-image synthetic corpus generated to mirror CelebA's attribute structure. The encoder is a deterministic hash over attribute combinations — *not* a real visual encoder. The point of this run is to confirm the wiring: query → encode → search → top-K → metrics. Real measured numbers from a verified run:

| Metric | hash-encoder, 500 synthetic images |
|---|---|
| Recall@1 | 0.474 |
| Recall@5 | 0.877 |
| Recall@10 | 0.965 |
| Recall@50 | 1.000 |
| Mean reciprocal rank | 0.650 |
| p95 query latency | 0.12 ms |

The numbers are high because the encoder has direct access to the same attributes used to generate the queries — this is closer to a closed-loop test of the retrieval logic than a model-quality result. They prove the pipeline produces coherent output. They do not prove anything about SigLIP-2 vs CLIP — that's the next experiment.

Recall curves and latency distribution are in [`evals/figures/`](evals/figures/). Full methodology in [`evals/results.md`](evals/results.md).

### Real-model ablation (next step)

The next experiment is a head-to-head comparison of CLIP-ViT-L/14, SigLIP-2-base, and SigLIP-2-large on full CelebA. The SLURM script in [`slurm/ablation.sbatch`](slurm/ablation.sbatch) builds three indexes and evaluates each:

```bash
sbatch slurm/ablation.sbatch
python scripts/generate_figures.py
```

The hypothesis, based on published results for these encoders on description-to-image retrieval, is that SigLIP-2 should outperform CLIP at every K, with SigLIP-2-large adding meaningful Recall@1 over SigLIP-2-base at higher inference cost. The geometric explanation in [ARCHITECTURE.md](ARCHITECTURE.md) is that SigLIP's sigmoid loss produces a more uniform embedding distribution on the hypersphere than CLIP's batch-softmax, and that uniformity should translate into better recall.

This README will be updated with measured numbers once the run completes.

## Hardware

Designed for the SupportVectors GPU cluster: SLURM-scheduled, mixed RTX 3090 / 4090 / 5090 / RTX PRO 6000 nodes, shared GPU shards (12–16 GB VRAM per shard) sufficient for inference. SLURM submission scripts are in [`slurm/`](slurm/).

The synthetic-data validation runs in under a second on a single VM CPU — no GPU required for that path.

## Quick start

### Reproduce the synthetic-data validation (no GPU, no Kaggle, no HF Hub)

This is the path that's been verified end-to-end. Total runtime: under a minute.

```bash
git clone https://github.com/ssuresh808/Person-Of-Interest.git
cd Person-Of-Interest

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"

# Synthetic CelebA-format corpus
python scripts/generate_synthetic_data.py

# Build the index
python scripts/build_index.py \
    --config configs/offline_demo.yaml \
    --images data/celeba_synthetic/img_align_celeba \
    --out artifacts/offline_demo.index

# Run eval (no VLM, no GPU)
python scripts/evaluate.py \
    --config configs/offline_demo.yaml \
    --index artifacts/offline_demo.index \
    --out evals/results_offline_demo.json \
    --no-vlm

# Optional: launch the UI
python -m poi.ui.app \
    --config configs/offline_demo.yaml \
    --index artifacts/offline_demo.index
```

### Full pipeline (GPU, real CelebA, real models)

```bash
# 1. Set up Kaggle credentials (~/.kaggle/kaggle.json) for CelebA download
python scripts/download_data.py --dataset celeba --out data/

# 2. Build the FAISS index on a GPU node
python scripts/build_index.py \
    --config configs/default.yaml \
    --images data/celeba/img_align_celeba \
    --out artifacts/celeba_siglip2.index

# 3. Launch the UI
python -m poi.ui.app \
    --config configs/default.yaml \
    --index artifacts/celeba_siglip2.index
```

Or, on the cluster:

```bash
sbatch slurm/build_index.sbatch
sbatch slurm/serve.sbatch
```

## Repo layout

```
person-of-interest/
├── src/poi/
│   ├── embeddings/      # SigLIP-2, CLIP, DeepFace, hash-encoder (unified API)
│   ├── index/           # FAISS index build, save, load, query
│   ├── retrieval/       # End-to-end search pipeline
│   ├── vlm/             # Qwen2.5-VL captioning with prompt templates
│   ├── data/            # CelebA loader, attribute → query synthesis
│   ├── ui/              # NiceGUI frontend
│   └── utils/           # Logging, config, device management
├── scripts/             # CLI entry points (download, build, eval)
├── slurm/               # Cluster submission scripts
├── notebooks/           # Exploratory analysis + figures for the writeup
├── tests/               # pytest suite — fast tests run in CI
├── evals/               # Eval results, plots, qualitative analysis
└── docs/                # Screenshots, architecture diagrams
```

## What's next

In priority order:

1. **Run the real-model ablation.** `sbatch slurm/ablation.sbatch` on the cluster. Replaces the "next step" section above with measured CLIP vs SigLIP-2 numbers on full CelebA.
2. **Hard-negative mining** for a fine-tune of SigLIP-2 on CelebA's attribute pairs. The gap between Recall@1 and Recall@10 will mostly be visually-similar distractors.
3. **Re-ranking** with a cross-encoder. Bi-encoder retrieval is fast but loses fine-grained interaction; a small cross-encoder over the top-100 should add several points of Recall@1.
4. **Query rewriting** with the VLM. Real users don't write structured descriptions — they write fragments. An LLM rewrite step before encoding consistently helps in production.
5. **Late interaction** (ColBERT-style). A single vector throws away fine-grained attribute information that a multi-vector representation could preserve.
6. **VLM-based image regeneration** of the matched person, conditioned on the original query, as a sanity check on what the system "thinks" the person looks like.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

Built as a final project for the SupportVectors LLM Bootcamp, Spring 2026. The geometric framing (concentration of measure, the hypersphere, contrastive alignment) follows the course's treatment of embeddings — see [ARCHITECTURE.md](ARCHITECTURE.md) for how the theory shows up in the code.

Models used:
- [SigLIP-2](https://huggingface.co/google/siglip2-base-patch16-256) (Google)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Alibaba)
- [DeepFace](https://github.com/serengil/deepface) for baseline face embeddings

Dataset: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (Liu et al., 2015) — used here for academic, non-commercial research as per the dataset license.
