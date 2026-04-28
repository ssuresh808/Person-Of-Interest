"""Evaluate retrieval quality using synthesized (query, target) pairs.

We use CelebA's attribute labels to construct queries with known correct
targets. For each evaluation sample, we:
    1. Pick a random subset of the sample's positive attributes.
    2. Synthesize a natural-language query mentioning those attributes.
    3. Retrieve top-K from the index.
    4. Score with Recall@K and MRR.

Caveats this script is honest about:
    - CelebA attributes are noisy. A "true" target might not be the only
      correct match — many other faces in the corpus will also satisfy the
      query. We treat the original sample as the gold target and accept
      that this lower-bounds true performance.
    - Query synthesis is template-based. Real users phrase descriptions
      more diversely. The eval numbers should be read as relative
      comparisons between encoders, not absolute statements about quality.

Usage:
    python scripts/evaluate.py \\
        --index artifacts/celeba_siglip2.index \\
        --config configs/default.yaml \\
        --n-queries 500 \\
        --out evals/results_siglip2.json
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

from tqdm.auto import tqdm

from poi.data import CelebADataset, synthesize_query
from poi.embeddings.factory import build_encoder
from poi.index import FaissIndex
from poi.retrieval import RetrievalPipeline
from poi.utils.config import POIConfig
from poi.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def evaluate(
    pipeline: RetrievalPipeline,
    dataset: CelebADataset,
    n_queries: int,
    seed: int,
    k_values: tuple[int, ...] = (1, 5, 10, 50),
) -> dict:
    """Run the eval harness. Returns a dict suitable for JSON serialization."""
    rng = random.Random(seed)

    # Sample evaluation queries from the validation split (or all images if
    # no split file is loaded — fall back gracefully).
    candidates = dataset.list_image_paths(split="val")
    if not candidates:
        log.warning("No validation split available; sampling from all images")
        candidates = dataset.list_image_paths()
    if not candidates:
        raise RuntimeError("No images available for evaluation")

    sampled = rng.sample(candidates, min(n_queries, len(candidates)))

    # Counters
    max_k = max(k_values)
    recall_hits = dict.fromkeys(k_values, 0)
    reciprocal_ranks: list[float] = []
    latencies_ms: list[float] = []
    skipped = 0

    for img_path in tqdm(sampled, desc="evaluating"):
        sample = dataset.get_sample(img_path.name)
        query = synthesize_query(sample, n_attributes=3, rng=rng)
        if not sample.positive_attributes:
            skipped += 1
            continue

        t0 = time.perf_counter()
        response = pipeline.search(query, top_k=max_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        target_filename = sample.filename
        rank: int | None = None
        for hit in response.hits:
            if hit.metadata.get("filename") == target_filename:
                rank = hit.rank
                break

        if rank is not None:
            for k in k_values:
                if rank <= k:
                    recall_hits[k] += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    n_evaluated = len(sampled) - skipped
    metrics = {
        "n_queries": n_evaluated,
        "skipped": skipped,
        "recall": {f"recall@{k}": recall_hits[k] / n_evaluated for k in k_values},
        "mean_reciprocal_rank": (
            sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        ),
        "latency_ms": {
            "mean": sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0,
            "p50": _percentile(latencies_ms, 50),
            "p95": _percentile(latencies_ms, 95),
            "p99": _percentile(latencies_ms, 99),
        },
    }
    return metrics


def _percentile(values: list[float], pct: float) -> float:
    """Simple percentile without scipy."""
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (pct / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] * (c - k) + s[c] * (k - f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--n-queries", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("evals/results.json"))
    parser.add_argument(
        "--no-vlm",
        action="store_true",
        help="Disable VLM captioning during eval (always recommended — much faster)",
    )
    args = parser.parse_args()

    setup_logging(log_file=Path("logs") / "evaluate.log")

    cfg = POIConfig.from_yaml(args.config) if args.config.exists() else POIConfig()
    if args.no_vlm:
        cfg.vlm.enabled = False

    # Build pipeline (no VLM by default during eval — captions are not scored)
    encoder = build_encoder(cfg.embedding)
    index = FaissIndex.load(args.index)
    if cfg.index.use_gpu:
        index.to_gpu()

    pipeline = RetrievalPipeline(encoder=encoder, index=index, cfg=cfg.retrieval, captioner=None)

    # Build dataset for ground truth
    dataset = CelebADataset(
        images_dir=cfg.data.images_dir,
        attributes_csv=cfg.data.attributes_csv,
        eval_split_csv=cfg.data.eval_split_csv,
    )

    metrics = evaluate(
        pipeline=pipeline,
        dataset=dataset,
        n_queries=args.n_queries,
        seed=args.seed,
    )

    # Tag with what we evaluated
    output = {
        "encoder": encoder.name,
        "encoder_dim": encoder.dim,
        "index_type": cfg.index.index_type,
        "n_corpus": len(index),
        "metrics": metrics,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Wrote results to {args.out}")
    log.info(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
