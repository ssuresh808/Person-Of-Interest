"""Tests for the hash-based offline encoder.

The hash encoder isn't a real model, but it has a contract — same query
keyword should produce a consistent direction in embedding space, and
images with shared attributes should be closer than images with disjoint
attributes. Failing those is a regression that would silently break the
offline demo pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from poi.embeddings.hash_encoder import HashEncoder


@pytest.fixture
def attribute_csv(tmp_path):
    """Write a tiny CelebA-format attribute CSV for testing."""
    rows = [
        # Glasses + smiling
        {"image_id": "001.jpg", "Eyeglasses": 1, "Smiling": 1, "Bangs": -1, "Wearing_Hat": -1},
        # Glasses, not smiling
        {"image_id": "002.jpg", "Eyeglasses": 1, "Smiling": -1, "Bangs": -1, "Wearing_Hat": -1},
        # Hat + bangs
        {"image_id": "003.jpg", "Eyeglasses": -1, "Smiling": -1, "Bangs": 1, "Wearing_Hat": 1},
        # No matching attributes
        {"image_id": "004.jpg", "Eyeglasses": -1, "Smiling": -1, "Bangs": -1, "Wearing_Hat": -1},
    ]
    path = tmp_path / "attrs.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


class TestHashEncoder:
    def test_dim_and_name(self, attribute_csv) -> None:
        enc = HashEncoder(attributes_csv=attribute_csv, dim=128)
        assert enc.dim == 128
        assert enc.name == "hash-encoder"

    def test_encode_images_normalized(self, attribute_csv) -> None:
        enc = HashEncoder(attributes_csv=attribute_csv, dim=128)
        vecs = enc.encode_images(["001.jpg", "002.jpg"])
        assert vecs.shape == (2, 128)
        # L2-norm should be 1 for any image with at least one positive attribute
        for v in vecs:
            assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)

    def test_encode_texts_normalized(self, attribute_csv) -> None:
        enc = HashEncoder(attributes_csv=attribute_csv, dim=128)
        vecs = enc.encode_texts(["a person wearing glasses", "someone with bangs"])
        for v in vecs:
            assert np.linalg.norm(v) == pytest.approx(1.0, abs=1e-4)

    def test_query_matches_relevant_images(self, attribute_csv) -> None:
        """A query mentioning 'glasses' should rank glasses-wearing images higher."""
        enc = HashEncoder(attributes_csv=attribute_csv, dim=512)
        img_vecs = enc.encode_images(["001.jpg", "002.jpg", "003.jpg", "004.jpg"])
        query_vec = enc.encode_texts(["a person wearing glasses"])[0]
        sims = img_vecs @ query_vec
        # 001 (glasses+smiling) and 002 (glasses) should both rank above 003 (hat) and 004 (none)
        assert sims[0] > sims[2]
        assert sims[0] > sims[3]
        assert sims[1] > sims[2]
        assert sims[1] > sims[3]

    def test_no_matching_attributes_returns_zero(self, attribute_csv) -> None:
        """Image 004 has no positive attributes — gets a zero vector before normalization."""
        enc = HashEncoder(attributes_csv=attribute_csv, dim=128)
        vec = enc.encode_images(["004.jpg"])[0]
        # Norm guard kicks in: zero vector stays zero (we divide by 1 to avoid NaN)
        assert np.allclose(vec, 0.0)

    def test_negation_handled(self, attribute_csv, tmp_path) -> None:
        """'no beard' should not push toward the same direction as 'beard'."""
        # Build a fixture that includes No_Beard so the negation logic engages.
        rows = [
            {"image_id": "001.jpg", "Eyeglasses": 1, "No_Beard": 1, "Smiling": 1},
            {"image_id": "002.jpg", "Eyeglasses": -1, "No_Beard": -1, "Smiling": -1},
        ]
        path = tmp_path / "attrs_with_beard.csv"
        pd.DataFrame(rows).to_csv(path, index=False)

        enc = HashEncoder(attributes_csv=path, dim=512)
        with_beard = enc.encode_texts(["a man with a beard"])[0]
        without_beard = enc.encode_texts(["a man with no beard"])[0]
        # Both vectors must be non-zero (the keyword logic engaged) and different
        assert np.linalg.norm(with_beard) > 0.5
        assert np.linalg.norm(without_beard) > 0.5
        assert not np.allclose(with_beard, without_beard)
        # And the inner product should be < 1 — they're meaningfully different
        assert float(with_beard @ without_beard) < 0.99

    def test_unknown_image_returns_zero(self, attribute_csv) -> None:
        """An image not in the attribute table gets a zero vector, not a crash."""
        enc = HashEncoder(attributes_csv=attribute_csv, dim=128)
        vec = enc.encode_images(["does_not_exist.jpg"])[0]
        assert np.allclose(vec, 0.0)
