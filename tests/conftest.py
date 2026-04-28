"""Shared pytest fixtures.

Tests that need real models are marked @pytest.mark.gpu and skipped in CI.
The fast tests use mock encoders to exercise the index, retrieval pipeline,
and config logic without touching the network or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for tests."""
    return np.random.default_rng(42)


class MockEncoder:
    """A deterministic, dependency-free encoder for fast tests.

    Maps inputs to vectors via a hash → seed → random, so the same input
    always produces the same vector. Outputs are L2-normalized so they
    behave like real embeddings under cosine similarity.
    """

    def __init__(self, dim: int = 64, name: str = "mock") -> None:
        self._dim = dim
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim

    def _vec_from_seed(self, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(self._dim).astype(np.float32)
        return v / (np.linalg.norm(v) + 1e-12)

    def encode_images(self, images, batch_size=None) -> np.ndarray:
        # Hash the path or PIL.Image id; either way, deterministic per item.
        out = np.stack([self._vec_from_seed(hash(str(x)) & 0xFFFFFFFF) for x in images])
        return out.astype(np.float32)

    def encode_texts(self, texts, batch_size=None) -> np.ndarray:
        out = np.stack([self._vec_from_seed(hash(t) & 0xFFFFFFFF) for t in texts])
        return out.astype(np.float32)


@pytest.fixture
def mock_encoder() -> MockEncoder:
    """A 64-dim mock encoder with stable outputs."""
    return MockEncoder(dim=64)


@pytest.fixture
def small_corpus(rng: np.random.Generator) -> np.ndarray:
    """1000 random unit vectors in 64 dimensions."""
    v = rng.standard_normal(size=(1000, 64)).astype(np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v
