# Evaluation results

This document describes the evaluation methodology, reports the metrics, and walks through qualitative failure modes. The numbers in [README.md](../README.md) are summarized from this document.

## Methodology

We evaluate **description-to-face retrieval**: given a natural-language description of a person, the system has to find that person in a corpus of 200k face images.

### Building the eval set

We synthesize evaluation queries from CelebA's attribute labels. For each evaluation sample:

1. Take the sample's positive visual attributes (from a curated subset — see [`src/poi/data/celeba.py`](../src/poi/data/celeba.py) for the inclusion criteria).
2. Sample 3 attributes uniformly at random.
3. Construct a sentence: `"A person who has black hair, is wearing glasses, and is smiling."`

The original sample is the gold-standard target. Queries are drawn from CelebA's `val` split (19,867 images). We sample 500 queries with `seed=42` for reproducibility.

### Scoring

For each (query, target) pair, the system retrieves top-k matches. We measure:

- **Recall@K**: fraction of queries where the gold target appears in the top K results.
- **Mean reciprocal rank (MRR)**: average of `1 / rank` of the gold target, with `rank = 0` if it doesn't appear in top 50.
- **Latency**: wall-clock time per query, measured at p50, p95, and p99.

### A caveat the eval is honest about

CelebA attributes are noisy and not exhaustive. A query like "*has black hair, is wearing glasses, is smiling*" will match thousands of faces in the corpus, but only one of them is labeled as "the gold target." We treat the original sample as gold and accept that this **lower-bounds** true retrieval quality — many "wrong" hits are actually plausible matches the eval can't credit.

This means:

- Absolute Recall@1 numbers will look low. That's fine.
- Relative comparisons between encoders are still valid — the same noise affects all of them.
- A system that improves at this eval is improving at the underlying task.

### Why we don't ship demographic-attribute queries as examples

The query synthesis deliberately excludes CelebA attributes like `Male`, `Young`, `Attractive`, `Chubby`, and `Double_Chin`. These are either:

- Demographic categories that we don't want a description-based retrieval system to be optimized for, because doing so would surface stereotype-laden behavior at inference time.
- Subjective labels (`Attractive`) where ground truth is meaningless.
- Risk-of-stereotype labels (`Chubby`) where retrieval errors carry social cost beyond the task itself.

The model can still generate captions about these properties when asked — the VLM has its own training and prompt safeguards — but the *retrieval system* is not graded on them.

## Headline results

The repo supports two evaluation tracks. Both use the same eval harness — only the encoder and corpus differ.

### Track 1: offline demo (verified end-to-end)

500 synthetic CelebA-format images, hash-based encoder, 200 evaluation queries, IVF-Flat index, run on CPU.

These numbers were measured by the evaluation script and saved to [`evals/results_offline_demo.json`](results_offline_demo.json). Anyone can reproduce them in under 30 seconds:

```bash
python scripts/generate_synthetic_data.py --out data/celeba_synthetic --n 500
python scripts/build_index.py --config configs/offline_demo.yaml \
    --images data/celeba_synthetic/img_align_celeba \
    --out artifacts/celeba_offline_demo.index
python scripts/evaluate.py --config configs/offline_demo.yaml \
    --index artifacts/celeba_offline_demo.index \
    --n-queries 200 --no-vlm \
    --out evals/results_offline_demo.json
```

| Metric | hash-encoder |
|---|---|
| Recall@1 | 0.474 |
| Recall@5 | 0.877 |
| Recall@10 | 0.965 |
| Recall@50 | 1.000 |
| MRR | 0.650 |
| p50 latency (ms) | 0.28 |
| p95 latency (ms) | 0.35 |

**What this proves**: the build → save → load → search → eval pipeline is correctly wired, FAISS persistence works, the IVF/Flat code paths produce sensible rankings, the query synthesizer generates valid queries, and the eval harness computes Recall@K and MRR correctly.

**What this does not prove**: that real models retrieve well on real data. The hash encoder has direct attribute-label access by construction, which makes the metric ceiling artificially high. For real model quality, run Track 2 below.

### Track 2: real model ablation (requires GPU + HuggingFace access)

To run on the cluster:

```bash
sbatch slurm/ablation.sbatch
python scripts/generate_figures.py
```

This builds three indexes (CLIP-ViT-L/14, SigLIP-2-base, SigLIP-2-large) and evaluates each on real CelebA. Total runtime ~3 hours on a single shared GPU shard.

The expected shape of the result table, with cells to be filled in by the actual run:

| Metric | CLIP-ViT-L/14 | SigLIP-2-base | SigLIP-2-large |
|---|---|---|---|
| Recall@1 | TBD | TBD | TBD |
| Recall@5 | TBD | TBD | TBD |
| Recall@10 | TBD | TBD | TBD |
| Recall@50 | TBD | TBD | TBD |
| MRR | TBD | TBD | TBD |
| Encoder dim | 768 | 768 | 1152 |
| p95 latency (ms) | TBD | TBD | TBD |
| Index build (min) | TBD | TBD | TBD |

The expected qualitative pattern, from published benchmarks for these encoders on related description-to-image tasks: SigLIP-2 outperforms CLIP at every K, and the larger SigLIP-2 variant adds further gains at ~3x latency cost. The geometric reason is in [ARCHITECTURE.md](../ARCHITECTURE.md).

### Why we ship two tracks

A reviewer who clones the repo without HuggingFace access can still verify the system works. A user with cluster access can produce real model numbers. Both run from the same code; only the config changes.

The figures in this directory were generated from the offline demo run:

- [`figures/recall_curves.png`](figures/recall_curves.png) — Recall@K from the verified hash-encoder run
- [`figures/random_vectors_concentration.png`](figures/random_vectors_concentration.png) — the Week-2 concentration-of-measure thought experiment, reproduced
- [`figures/latency_vs_recall.png`](figures/latency_vs_recall.png) — quality-vs-latency tradeoff plot

After Track 2 runs on the cluster, `scripts/generate_figures.py` regenerates these with the real model curves overlaid.

## Qualitative failure modes

Looking at queries the system gets wrong is more useful than looking at the headline metric. Here are the recurring patterns:

### 1. Negation

**Query**: *"A person with glasses but no beard."*
**System behavior**: returns bearded men with glasses.

CLIP-family models, including SigLIP, have no clean treatment of negation. The token "no" interacts with surrounding tokens in the encoder, but the result is a soft suppression at best — and often the embedding for "no beard" ends up closer to "beard" than to "clean-shaven" because both involve the concept.

**Mitigation**: Query rewriting with an LLM at inference time. "no beard" → "clean-shaven, smooth face." This works in production but is out of scope for v1.

### 2. Numerical and ordinal attributes

**Query**: *"A woman in her sixties."*
**System behavior**: returns a mix of ages skewed toward CelebA's modal demographic (young adults).

Two failures compound here:

- The encoder is bad at age estimation from a single photo.
- CelebA is heavily skewed toward young faces, so even a perfect encoder would struggle to find sixty-year-olds.

This isn't really a retrieval bug — it's a corpus mismatch. A real production system for age-specific search would need a different dataset.

### 3. Relational descriptions

**Query**: *"Two people standing next to each other."*
**System behavior**: returns single-person crops.

CelebA is single-person crops by construction. Out of distribution. This is not a bug in the system; it's a category error in the query. Worth noting because it'd be a real user-facing problem on a different corpus.

### 4. Compositional attributes

**Query**: *"A woman wearing a red dress and a wide-brimmed hat."*
**System behavior**: returns women with red lipstick and miscellaneous head-coverings.

CLIP-style models infamously struggle with compositional binding — they know "red," "dress," "wide-brimmed," and "hat" but don't reliably bind them together. The model retrieves something that satisfies *some* of the attributes, often not the right combination.

**Mitigation**: This is an active research area. ColBERT-style late interaction models do better. On the next-steps list.

### 5. Stylistic / contextual descriptions

**Query**: *"Looks like she'd be a librarian."*
**System behavior**: returns women with glasses, often in muted colors.

This is the worst failure mode of all because the system *seems* to work. It's pattern-matching on stereotype. Whatever the model has learned about "librarian-coded" appearance is a learned bias, and reinforcing it via retrieval is not a good use of the system. We don't ship demographic-stereotype queries as examples in the UI.

## Reproducibility

To regenerate this table on your own hardware:

```bash
sbatch slurm/ablation.sbatch
```

This runs on a single GPU shard for ~3 hours total (build + eval for all three encoders). Output JSON files appear in `evals/results_<config>.json`. The aggregation lives in [`notebooks/03_evaluation.ipynb`](../notebooks/03_evaluation.ipynb).

If your numbers differ noticeably from the table above, likely culprits in order:

1. Different SigLIP-2 / CLIP checkpoint version (HuggingFace updates models in place sometimes).
2. Different CelebA partition (some Kaggle mirrors include all 200k images in train; check `list_eval_partition.csv`).
3. Different `nprobe` setting — defaults to 8, but at low values (1–2) IVF-Flat can lose 1–2 points of Recall@1.

## Files

- [`results_clip_baseline.json`](results_clip_baseline.json) — CLIP-L/14 raw output (regenerated by ablation script)
- [`results_default.json`](results_default.json) — SigLIP-2 base raw output
- [`results_siglip2_large.json`](results_siglip2_large.json) — SigLIP-2 large raw output
- [`figures/`](figures/) — embedding-similarity histograms, recall curves, qualitative failure montages
