"""Pipeline tests with the mock encoder.

Verifies that the (encoder, index, retrieval cfg) wiring is correct
without needing real models.
"""

from __future__ import annotations

import pytest

from poi.index import FaissIndex
from poi.retrieval import RetrievalPipeline
from poi.utils.config import IndexConfig, RetrievalConfig


def _build_pipeline(mock_encoder, items: list[str], retrieval_cfg=None):
    """Helper: build a small searchable pipeline from string items."""
    vectors = mock_encoder.encode_texts(items)  # treat strings as items
    metadata = [{"image_path": f"/fake/{i}.jpg", "filename": f"{i}.jpg"} for i in range(len(items))]

    cfg = IndexConfig(index_type="flat", use_gpu=False)
    index = FaissIndex(dim=mock_encoder.dim, cfg=cfg)
    index.add(vectors, metadata=metadata)

    return RetrievalPipeline(
        encoder=mock_encoder,
        index=index,
        cfg=retrieval_cfg or RetrievalConfig(top_k=5),
    )


class TestRetrievalPipeline:
    def test_dim_mismatch_raises(self, mock_encoder) -> None:
        cfg = IndexConfig(index_type="flat", use_gpu=False)
        bad_index = FaissIndex(dim=mock_encoder.dim + 1, cfg=cfg)
        with pytest.raises(ValueError, match="does not match"):
            RetrievalPipeline(encoder=mock_encoder, index=bad_index, cfg=RetrievalConfig())

    def test_search_returns_top_k(self, mock_encoder) -> None:
        items = [f"item-{i}" for i in range(50)]
        pipeline = _build_pipeline(mock_encoder, items)
        response = pipeline.search("item-0", top_k=5)
        # Same string → identical vector → top hit is the matching item with score ~1
        assert response.hits[0].score == pytest.approx(1.0, abs=1e-4)
        assert len(response.hits) == 5

    def test_search_records_timings(self, mock_encoder) -> None:
        items = [f"item-{i}" for i in range(20)]
        pipeline = _build_pipeline(mock_encoder, items)
        response = pipeline.search("item-3")
        assert "encode_ms" in response.timings_ms
        assert "search_ms" in response.timings_ms
        assert response.timings_ms["encode_ms"] >= 0
        assert response.timings_ms["search_ms"] >= 0

    def test_min_score_filters(self, mock_encoder) -> None:
        items = [f"item-{i}" for i in range(20)]
        # Set min_score above what's achievable for non-matching items
        pipeline = _build_pipeline(
            mock_encoder, items, retrieval_cfg=RetrievalConfig(top_k=10, min_score=0.99)
        )
        response = pipeline.search("item-0")
        # Only the exact match should pass; mock vectors are otherwise random.
        assert all(hit.score >= 0.99 for hit in response.hits)
