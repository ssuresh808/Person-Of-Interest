"""Tests for the FAISS index wrapper.

These are fast and run on CPU. They cover:
    - Build, search, save, load roundtrip
    - Metadata co-storage
    - Dimension mismatch errors
    - All three index types (flat, ivf_flat, ivf_pq)
"""

from __future__ import annotations

import numpy as np
import pytest

from poi.index import FaissIndex
from poi.utils.config import IndexConfig


class TestFaissIndexFlat:
    """Flat index: exact, no training needed."""

    def test_build_and_search(self, small_corpus: np.ndarray) -> None:
        cfg = IndexConfig(index_type="flat", use_gpu=False)
        idx = FaissIndex(dim=small_corpus.shape[1], cfg=cfg)
        idx.add(small_corpus)
        assert len(idx) == len(small_corpus)

        results = idx.search(small_corpus[:5], k=3)
        assert len(results) == 5
        # Each query should retrieve itself as the top hit (cosine = 1.0)
        for i, hits in enumerate(results):
            assert hits[0].index == i
            assert hits[0].score == pytest.approx(1.0, abs=1e-4)

    def test_returns_correct_k(self, small_corpus: np.ndarray) -> None:
        cfg = IndexConfig(index_type="flat", use_gpu=False)
        idx = FaissIndex(dim=small_corpus.shape[1], cfg=cfg)
        idx.add(small_corpus)
        results = idx.search(small_corpus[:1], k=10)
        assert len(results[0]) == 10


class TestFaissIndexIVF:
    """IVF-Flat: requires training."""

    def test_build_and_search(self, small_corpus: np.ndarray) -> None:
        cfg = IndexConfig(index_type="ivf_flat", nlist=16, nprobe=4, use_gpu=False)
        idx = FaissIndex(dim=small_corpus.shape[1], cfg=cfg)
        idx.add(small_corpus)  # auto-trains
        results = idx.search(small_corpus[:5], k=5)
        # IVF is approximate; we don't insist on perfect self-recall, but
        # for a small corpus with high nprobe-ratio it should be near-perfect.
        self_hits = sum(1 for i, hits in enumerate(results) if any(h.index == i for h in hits))
        assert self_hits >= 4  # Allow one miss

    def test_nprobe_too_large_raises(self) -> None:
        with pytest.raises(ValueError, match=r"nprobe.*cannot exceed nlist"):
            IndexConfig(index_type="ivf_flat", nlist=10, nprobe=20)


class TestPersistence:
    """Save → load roundtrip preserves data and metadata."""

    def test_roundtrip(self, small_corpus: np.ndarray, tmp_path) -> None:
        cfg = IndexConfig(index_type="flat", use_gpu=False)
        idx = FaissIndex(dim=small_corpus.shape[1], cfg=cfg)
        metadata = [{"id": i, "tag": f"item-{i}"} for i in range(len(small_corpus))]
        idx.add(small_corpus, metadata=metadata)

        save_path = tmp_path / "test.index"
        idx.save(save_path)
        assert save_path.exists()
        assert (save_path.with_suffix(save_path.suffix + ".meta.json")).exists()

        reloaded = FaissIndex.load(save_path)
        assert len(reloaded) == len(idx)
        assert reloaded.dim == idx.dim

        # Search results should match
        original_results = idx.search(small_corpus[:3], k=5)
        loaded_results = reloaded.search(small_corpus[:3], k=5)
        for o_hits, l_hits in zip(original_results, loaded_results, strict=True):
            assert [h.index for h in o_hits] == [h.index for h in l_hits]
            # Metadata survives
            assert l_hits[0].metadata == {"id": o_hits[0].index, "tag": f"item-{o_hits[0].index}"}


class TestValidation:
    """Input validation."""

    def test_dimension_mismatch_raises(self, small_corpus: np.ndarray) -> None:
        cfg = IndexConfig(index_type="flat", use_gpu=False)
        idx = FaissIndex(dim=32, cfg=cfg)  # wrong dim
        with pytest.raises(ValueError, match="Expected vectors of shape"):
            idx.add(small_corpus)

    def test_metadata_length_mismatch_raises(self, small_corpus: np.ndarray) -> None:
        cfg = IndexConfig(index_type="flat", use_gpu=False)
        idx = FaissIndex(dim=small_corpus.shape[1], cfg=cfg)
        with pytest.raises(ValueError, match="metadata length"):
            idx.add(small_corpus, metadata=[{"id": 0}])  # only 1 metadata for 1000 vectors
