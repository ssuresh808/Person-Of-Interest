"""FAISS index management.

This wraps FAISS in a small object so callers don't have to think about
GPU↔CPU handoffs, normalization, or which factory string maps to which
index type. All FAISS-specific code lives here; the rest of the project
sees a simple build/save/load/query API.

The index stores normalized vectors and uses inner product as the metric,
so similarity scores are cosines in [-1, 1]. With SigLIP-2's tight
distribution this means above ~0.1 is a meaningful match.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from poi.utils.config import IndexConfig
from poi.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class SearchResult:
    """A single retrieval hit."""

    index: int  # Position in the corpus
    score: float  # Cosine similarity (or whatever the metric is)
    metadata: dict | None = None  # Arbitrary, populated downstream


class FaissIndex:
    """Cosine-similarity FAISS index with metadata co-storage.

    The index itself only stores vectors. Per-vector metadata (image filename,
    CelebA attributes if loaded) is kept in a parallel list. We persist both
    to disk side-by-side so a saved index is fully self-describing.

    Args:
        dim: Embedding dimensionality. Must match what the encoder produces.
        cfg: Index parameters (type, nlist, nprobe, metric).
    """

    SIDECAR_SUFFIX = ".meta.json"

    def __init__(self, dim: int, cfg: IndexConfig) -> None:
        self.dim = dim
        self.cfg = cfg
        self._cpu_index = self._build_empty_index(dim, cfg)
        self._gpu_resources: faiss.GpuResources | None = None
        self._index: faiss.Index = self._cpu_index  # active handle (CPU or GPU)
        self._metadata: list[dict] = []
        self._is_trained = False

    # ----- Construction -----

    @staticmethod
    def _build_empty_index(dim: int, cfg: IndexConfig) -> faiss.Index:
        """Construct a fresh CPU index of the requested type."""
        metric = faiss.METRIC_INNER_PRODUCT if cfg.metric == "ip" else faiss.METRIC_L2

        if cfg.index_type == "flat":
            # IndexFlatIP / IndexFlatL2 — exact, brute force. Sufficient up to
            # ~1M vectors on a single GPU.
            return (
                faiss.IndexFlatIP(dim)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dim)
            )

        if cfg.index_type == "ivf_flat":
            # IVF-Flat: partition by k-means into nlist Voronoi cells.
            # Search nprobe cells per query.
            quantizer = (
                faiss.IndexFlatIP(dim)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dim)
            )
            return faiss.IndexIVFFlat(quantizer, dim, cfg.nlist, metric)

        if cfg.index_type == "ivf_pq":
            # IVF + Product Quantization. Compresses vectors to ~bytes each
            # at modest recall cost. Use for >10M vectors.
            quantizer = (
                faiss.IndexFlatIP(dim)
                if metric == faiss.METRIC_INNER_PRODUCT
                else faiss.IndexFlatL2(dim)
            )
            m = 8  # subquantizers
            nbits = 8  # bits per subquantizer
            return faiss.IndexIVFPQ(quantizer, dim, cfg.nlist, m, nbits, metric)

        raise ValueError(f"Unknown index_type: {cfg.index_type}")

    # ----- Build -----

    def train(self, vectors: np.ndarray) -> None:
        """Train the quantizer. No-op for flat indexes.

        For IVF, FAISS recommends >= 30 * nlist training vectors. We sample
        if more are provided.
        """
        if self.cfg.index_type == "flat":
            self._is_trained = True
            return

        if self._is_trained:
            log.warning("Index already trained; skipping")
            return

        target = max(30 * self.cfg.nlist, 10_000)
        if len(vectors) > target:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(vectors), size=target, replace=False)
            sample = np.ascontiguousarray(vectors[idx])
        else:
            sample = np.ascontiguousarray(vectors)

        log.info(
            f"Training {self.cfg.index_type} on {len(sample)} vectors (nlist={self.cfg.nlist})"
        )
        self._index.train(sample)
        self._is_trained = True

    def add(self, vectors: np.ndarray, metadata: list[dict] | None = None) -> None:
        """Add vectors and their per-vector metadata to the index."""
        vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or vectors.shape[1] != self.dim:
            raise ValueError(f"Expected vectors of shape (N, {self.dim}); got {vectors.shape}")
        if metadata is not None and len(metadata) != len(vectors):
            raise ValueError(
                f"metadata length ({len(metadata)}) != vectors length ({len(vectors)})"
            )

        if not self._is_trained:
            # Auto-train on first add for convenience. Larger pipelines should
            # call train() explicitly with a representative sample.
            self.train(vectors)

        self._index.add(vectors)
        if metadata is not None:
            self._metadata.extend(metadata)
        else:
            self._metadata.extend({} for _ in range(len(vectors)))

    # ----- GPU lifecycle -----

    def to_gpu(self) -> FaissIndex:
        """Move the index to GPU. Faster search; required for large nprobe."""
        if not self.cfg.use_gpu:
            log.info("use_gpu=False; staying on CPU")
            return self
        try:
            self._gpu_resources = faiss.StandardGpuResources()
            self._index = faiss.index_cpu_to_gpu(self._gpu_resources, 0, self._cpu_index)
            log.info("Index moved to GPU")
        except (AttributeError, RuntimeError) as e:
            # faiss-cpu doesn't expose StandardGpuResources; not all wheels do.
            log.warning(f"Could not move index to GPU ({e}); staying on CPU")
            self._index = self._cpu_index
        return self

    def to_cpu(self) -> FaissIndex:
        """Move the index back to CPU before saving."""
        if self._index is not self._cpu_index:
            self._cpu_index = faiss.index_gpu_to_cpu(self._index)
            self._index = self._cpu_index
        return self

    # ----- Query -----

    def search(self, queries: np.ndarray, k: int) -> list[list[SearchResult]]:
        """Search for top-k matches. Returns one list of results per query."""
        if self.cfg.index_type.startswith("ivf"):
            # nprobe is settable on the active index handle (GPU or CPU).
            try:
                self._index.nprobe = self.cfg.nprobe
            except AttributeError:
                pass  # GPU IVF sets nprobe via a different API on some versions

        queries = np.ascontiguousarray(queries, dtype=np.float32)
        if queries.ndim == 1:
            queries = queries[None, :]

        scores, indices = self._index.search(queries, k)

        results: list[list[SearchResult]] = []
        for q_scores, q_indices in zip(scores, indices, strict=True):
            row = []
            for s, i in zip(q_scores, q_indices, strict=True):
                if i == -1:
                    continue  # FAISS pads underfilled rows with -1
                row.append(
                    SearchResult(
                        index=int(i),
                        score=float(s),
                        metadata=self._metadata[i] if i < len(self._metadata) else None,
                    )
                )
            results.append(row)
        return results

    # ----- Persistence -----

    def save(self, path: Path | str) -> None:
        """Write index + metadata sidecar to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Always save from CPU
        self.to_cpu()
        faiss.write_index(self._cpu_index, str(path))

        sidecar = path.with_suffix(path.suffix + self.SIDECAR_SUFFIX)
        with sidecar.open("w") as f:
            json.dump(
                {
                    "dim": self.dim,
                    "config": self.cfg.model_dump(mode="json"),
                    "metadata": self._metadata,
                },
                f,
            )
        log.info(f"Saved index to {path} ({len(self._metadata)} vectors)")

    @classmethod
    def load(cls, path: Path | str) -> FaissIndex:
        """Load an index + sidecar. Reconstruct the same config used at build time."""
        path = Path(path)
        sidecar = path.with_suffix(path.suffix + cls.SIDECAR_SUFFIX)
        if not sidecar.exists():
            raise FileNotFoundError(f"Metadata sidecar not found: {sidecar}")

        with sidecar.open() as f:
            data = json.load(f)

        cfg = IndexConfig(**data["config"])
        instance = cls(dim=data["dim"], cfg=cfg)
        instance._cpu_index = faiss.read_index(str(path))
        instance._index = instance._cpu_index
        instance._metadata = data["metadata"]
        instance._is_trained = True
        log.info(f"Loaded index from {path} ({len(instance._metadata)} vectors)")
        return instance

    # ----- Inspection -----

    @property
    def n_vectors(self) -> int:
        return self._index.ntotal

    def __len__(self) -> int:
        return self.n_vectors

    def get_metadata(self, idx: int) -> dict:
        """Fetch metadata for a single index position."""
        return self._metadata[idx]
