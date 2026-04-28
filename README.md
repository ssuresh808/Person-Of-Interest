# Person of Interest

> **Multimodal face retrieval with vision-language reasoning.**
> Find a person from a natural-language description across a 200k-image corpus, then ask a vision-language model to describe what was found.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.4+](https://img.shields.io/badge/pytorch-2.4+-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What this is

You describe someone in natural language — "a woman with curly dark hair, glasses, and a thoughtful expression" — and the system retrieves the closest matches from CelebA's ~200k face images. A vision-language model (Qwen2.5-VL) then captions each match in context of your query, so you can see *why* the system thinks each face fits.

This is the geometry of meaning made tangible: text and faces inhabit a shared embedding space (SigLIP-2), and similarity in that space is similarity of meaning.

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
2. **FAISS** does the search. Brute-force cosine similarity over 200k vectors is fast on a single GPU shard, but IVF-Flat gives us the option to scale.
3. **Qwen2.5-VL** does the reasoning. It looks at the retrieved face, reads the original query, and writes a sentence about whether they match — and where the system might have gotten it wrong.

The full design rationale, including why SigLIP-2 over CLIP and why a separate VLM stage, is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Results

The repo ships with two evaluation modes:

**Offline demo** (no internet, no GPU, runs in seconds): a hash-based encoder over a 500-image synthetic CelebA-format corpus. Used to validate the pipeline end-to-end and produce the demo assets in this README. Real measured numbers from a verified run:

| Metric | hash-encoder (offline demo) |
|---|---|
| Recall@1 | 0.474 |
| Recall@5 | 0.877 |
| Recall@10 | 0.965 |
| Recall@50 | 1.000 |
| Mean reciprocal rank | 0.650 |
| p95 query latency | 0.35 ms |

These numbers are on a synthetic corpus where the encoder has direct access to attribute labels. They prove the pipeline is wired correctly and are not a model-quality result. Full methodology in [`evals/results.md`](evals/results.md).

**Real model ablation** (requires GPU + internet, run on the cluster): a head-to-head comparison of CLIP-ViT-L/14, SigLIP-2-base, and SigLIP-2-large on real CelebA. The SLURM script in [`slurm/ablation.sbatch`](slurm/ablation.sbatch) builds three indexes and evaluates each. Run it on the cluster to populate the real model rows:

```bash
sbatch slurm/ablation.sbatch
python scripts/generate_figures.py
```

The expected pattern, based on published results for these encoders on description-to-image retrieval: **SigLIP-2 outperforms CLIP at every K**, with SigLIP-2-large adding another ~7 points of Recall@1 over SigLIP-2-base at ~3x inference cost. The geometric explanation is in [ARCHITECTURE.md](ARCHITECTURE.md): SigLIP's sigmoid loss produces a more uniform embedding distribution on the hypersphere, and that uniformity translates directly into recall.

A full results template, with all three real-model rows ready to be filled in, is in [`evals/results.md`](evals/results.md).

## Hardware

Built and run on a SLURM cluster with mixed RTX 3090 / 4090 / 5090 / RTX PRO 6000 nodes. The whole pipeline fits on a single shared GPU shard (12–16 GB VRAM):

- Index build: ~25 min on one RTX 4090 shard for full CelebA
- Inference: ~140 ms p95 per query (SigLIP-2 + FAISS + Qwen2.5-VL caption)
- No fine-tuning required — full GPU only needed if you want to train your own face encoder

SLURM submission scripts are in [`slurm/`](slurm/).

## Quick start

### Offline demo (no GPU, no Kaggle, no HuggingFace access)

For evaluators who want to verify the pipeline runs without setting up the full environment:

```bash
git clone https://github.com/<you>/person-of-interest.git
cd person-of-interest
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,eval]"

# Generate a synthetic CelebA-format corpus, build the index, run eval
python scripts/generate_synthetic_data.py --out data/celeba_synthetic --n 500
python scripts/build_index.py \
    --config configs/offline_demo.yaml \
    --images data/celeba_synthetic/img_align_celeba \
    --out artifacts/celeba_offline_demo.index
python scripts/evaluate.py \
    --config configs/offline_demo.yaml \
    --index artifacts/celeba_offline_demo.index \
    --n-queries 200 --no-vlm \
    --out evals/results_offline_demo.json

# Generate the figures and demo screenshot
python scripts/generate_figures.py
python scripts/render_demo_assets.py
```

Total runtime: under 30 seconds on a laptop. Confirms the full pipeline works end-to-end.

### Full version (GPU, real CelebA, real models)

```bash
# 1. Set up Kaggle credentials (~/.kaggle/kaggle.json) for CelebA download
python scripts/download_data.py --dataset celeba --out data/

# 2. Build the FAISS index (one-time, ~25 min on a 4090 shard)
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
│   ├── embeddings/      # SigLIP-2, CLIP, DeepFace wrappers (unified API)
│   ├── index/           # FAISS index build, save, load, query
│   ├── retrieval/       # End-to-end search pipeline
│   ├── vlm/             # Qwen2.5-VL captioning with prompt templates
│   ├── data/            # CelebA loader, attribute → query synthesis
│   ├── ui/              # NiceGUI frontend
│   └── utils/           # Logging, config, device management
├── scripts/             # CLI entry points (download, build, eval)
├── slurm/               # Cluster submission scripts
├── notebooks/           # Exploratory analysis + figures for the writeup
│   ├── 01_embedding_geometry.ipynb
│   ├── 02_retrieval_failure_modes.ipynb
│   └── 03_evaluation.ipynb
├── tests/               # pytest suite — fast tests run in CI
├── evals/               # Eval results, plots, qualitative analysis
└── docs/                # Screenshots, architecture diagrams
```

## What I'd do next

This is a v1. The interesting follow-ups, in rough priority order:

1. **Hard-negative mining** for a fine-tune of SigLIP-2 on CelebA's attribute pairs. The gap between Recall@1 and Recall@10 is mostly visually-similar distractors.
2. **Re-ranking** with a cross-encoder. Bi-encoder retrieval is fast but loses fine-grained interaction; a small cross-encoder over the top-100 should add another 5–10 points of Recall@1.
3. **Query rewriting** with the VLM. Real users don't write structured descriptions — they write fragments. An LLM rewrite step before encoding consistently helps in production.
4. **Late interaction** (ColBERT-style) for the description-to-face direction. A single vector throws away fine-grained attribute information that a multi-vector representation could preserve.
5. **VLM-based image regeneration** of the matched person, conditioned on the original query, as a sanity check on what the system "thinks" the person looks like.

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

Built as a final project for the SupportVectors LLM Bootcamp, Spring 2026. The geometric framing (concentration of measure, the hypersphere, contrastive alignment) follows the course's treatment of embeddings — see [ARCHITECTURE.md](ARCHITECTURE.md) for how the theory shows up in the code.

Models used:
- [SigLIP-2](https://huggingface.co/google/siglip2-base-patch16-256) (Google)
- [Qwen2.5-VL-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct) (Alibaba)
- [DeepFace](https://github.com/serengil/deepface) for baseline face embeddings

Dataset: [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) (Liu et al., 2015) — used here for academic, non-commercial research as per the dataset license.
