"""Retrieval pipeline.

Glues encoders, the FAISS index, and (optionally) the VLM caption stage into
a single search() call. This is the object the UI talks to. Nothing else
should call FAISS directly.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image

from poi.embeddings.base import Encoder
from poi.index import FaissIndex, SearchResult
from poi.utils.config import RetrievalConfig
from poi.utils.logging import get_logger

if TYPE_CHECKING:
    # Imported only for type checking. At runtime, the captioner is passed in
    # as an instance, so the import would just add a torch dependency for no
    # reason. The string annotation works fine because of __future__ annotations.
    from poi.vlm import Captioner

log = get_logger(__name__)


@dataclass
class RetrievalHit:
    """One match with everything the UI needs to render it."""

    rank: int
    score: float
    image_path: Path
    caption: str | None = None  # Filled by VLM stage if enabled
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievalResponse:
    """A complete response to one query."""

    query: str
    hits: list[RetrievalHit]
    timings_ms: dict[str, float]


class RetrievalPipeline:
    """Search a face corpus by natural-language description.

    Args:
        encoder: Multimodal encoder (must support encode_texts).
        index: Pre-built FAISS index over the corpus.
        cfg: Retrieval parameters (top_k, min_score, etc.).
        captioner: Optional VLM captioner. If provided, every hit is annotated
            with a context-aware caption. Adds noticeable latency.
    """

    def __init__(
        self,
        encoder: Encoder,
        index: FaissIndex,
        cfg: RetrievalConfig,
        captioner: Captioner | None = None,
    ) -> None:
        self.encoder = encoder
        self.index = index
        self.cfg = cfg
        self.captioner = captioner

        if encoder.dim != index.dim:
            raise ValueError(
                f"Encoder dim ({encoder.dim}) does not match index dim ({index.dim}). "
                f"You probably built the index with a different encoder."
            )

    def search(self, query: str, top_k: int | None = None) -> RetrievalResponse:
        """Run the full pipeline on a single query."""
        k = top_k or self.cfg.top_k
        timings: dict[str, float] = {}

        # 1. Encode the text query.
        t0 = time.perf_counter()
        query_vec = self.encoder.encode_texts([query])  # shape (1, D)
        timings["encode_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Search FAISS. Over-fetch if we're going to filter.
        t0 = time.perf_counter()
        fetch_k = self.cfg.rerank_top_k or k
        search_results = self.index.search(query_vec, k=fetch_k)[0]  # only one query
        timings["search_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Filter by min_score and trim to top_k.
        filtered = [r for r in search_results if r.score >= self.cfg.min_score][:k]

        # 4. Materialize hits with image paths from metadata.
        hits = self._build_hits(filtered)

        # 5. Optionally caption each hit with the VLM.
        if self.captioner is not None and hits:
            t0 = time.perf_counter()
            self._caption_hits(hits, query=query)
            timings["caption_ms"] = (time.perf_counter() - t0) * 1000

        return RetrievalResponse(query=query, hits=hits, timings_ms=timings)

    # ----- Internal -----

    def _build_hits(self, results: list[SearchResult]) -> list[RetrievalHit]:
        hits: list[RetrievalHit] = []
        for rank, r in enumerate(results, start=1):
            meta = r.metadata or {}
            path_str = meta.get("image_path")
            if path_str is None:
                log.warning(
                    f"Hit at rank {rank} (idx={r.index}) has no image_path metadata; skipping"
                )
                continue
            hits.append(
                RetrievalHit(
                    rank=rank,
                    score=r.score,
                    image_path=Path(path_str),
                    metadata=meta,
                )
            )
        return hits

    def _caption_hits(self, hits: list[RetrievalHit], query: str) -> None:
        """Annotate each hit with a VLM-generated caption.

        Loads images one at a time. For real workloads this should batch,
        but Qwen2.5-VL's batch handling is finicky and the latency win is
        small for top_k=12. We can revisit if it shows up in profiling.
        """
        for hit in hits:
            try:
                image = Image.open(hit.image_path).convert("RGB")
                hit.caption = self.captioner.caption(image=image, query=query)  # type: ignore[union-attr]
            except Exception as e:
                log.warning(f"VLM captioning failed for {hit.image_path}: {e}")
                hit.caption = None
