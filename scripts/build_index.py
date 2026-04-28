"""Build a FAISS index over a face image corpus.

Usage:
    python scripts/build_index.py \\
        --config configs/default.yaml \\
        --images data/celeba/img_align_celeba \\
        --out artifacts/celeba_siglip2.index

The index is built once per (encoder, corpus) combination. Subsequent
searches reuse the cached file. Building all of CelebA with SigLIP-2-base
takes ~25 minutes on a single RTX 4090 shard.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from poi.embeddings.factory import build_encoder
from poi.index import FaissIndex
from poi.utils.config import POIConfig
from poi.utils.device import cuda_clear
from poi.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


def encode_corpus_in_chunks(
    encoder,
    image_paths: list[Path],
    chunk_size: int = 512,
) -> tuple[np.ndarray, list[dict]]:
    """Encode the corpus in chunks to keep peak memory bounded.

    Returns (vectors, metadata) where metadata[i] = {"image_path": str(path)}.
    """
    all_vectors: list[np.ndarray] = []
    metadata: list[dict] = []

    n = len(image_paths)
    log.info(f"Encoding {n} images with {encoder.name} (chunk_size={chunk_size})")

    for chunk_start in tqdm(range(0, n, chunk_size), desc="encoding corpus"):
        chunk = image_paths[chunk_start : chunk_start + chunk_size]
        vecs = encoder.encode_images(chunk)
        all_vectors.append(vecs)
        for p in chunk:
            metadata.append({"image_path": str(p), "filename": p.name})
        # Aggressively reclaim VRAM between chunks
        cuda_clear()

    vectors = np.concatenate(all_vectors, axis=0)
    return vectors, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a FAISS index over a face corpus")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--images",
        type=Path,
        required=True,
        help="Directory of face images (e.g. data/celeba/img_align_celeba)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output .index file path")
    parser.add_argument("--limit", type=int, default=None, help="Cap N images (for smoke tests)")
    parser.add_argument("--chunk-size", type=int, default=512)
    args = parser.parse_args()

    setup_logging(log_file=Path("logs") / "build_index.log")

    # Load config — fall back to defaults if missing
    if args.config.exists():
        cfg = POIConfig.from_yaml(args.config)
    else:
        log.warning(f"Config not found at {args.config}; using defaults")
        cfg = POIConfig()

    # Discover images
    image_paths = sorted(args.images.glob("*.jpg"))
    if not image_paths:
        log.error(f"No .jpg files found in {args.images}")
        return
    if args.limit:
        image_paths = image_paths[: args.limit]
    log.info(f"Found {len(image_paths)} images")

    # Build encoder + encode corpus
    encoder = build_encoder(cfg.embedding)
    t0 = time.perf_counter()
    vectors, metadata = encode_corpus_in_chunks(encoder, image_paths, chunk_size=args.chunk_size)
    encode_seconds = time.perf_counter() - t0
    log.info(f"Encoded {len(vectors)} vectors in {encode_seconds:.1f}s")

    # Build index
    index = FaissIndex(dim=encoder.dim, cfg=cfg.index)
    index.train(vectors)
    index.add(vectors, metadata=metadata)

    # Save
    args.out.parent.mkdir(parents=True, exist_ok=True)
    index.save(args.out)

    # Also save the config used so the index is self-describing
    config_out = args.out.with_suffix(args.out.suffix + ".config.yaml")
    cfg.to_yaml(config_out)
    log.info(f"Saved config alongside index at {config_out}")
    log.info("Done.")


if __name__ == "__main__":
    main()
