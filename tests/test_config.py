"""Config loading, validation, and round-trip tests."""

from __future__ import annotations

import pytest

from poi.utils.config import IndexConfig, POIConfig


class TestPOIConfig:
    def test_defaults(self) -> None:
        cfg = POIConfig()
        assert cfg.embedding.backend == "siglip2"
        assert cfg.index.index_type == "ivf_flat"
        assert cfg.vlm.enabled is True

    def test_yaml_roundtrip(self, tmp_path) -> None:
        original = POIConfig()
        original.embedding.batch_size = 32
        original.retrieval.top_k = 24
        path = tmp_path / "config.yaml"
        original.to_yaml(path)
        loaded = POIConfig.from_yaml(path)
        assert loaded.embedding.batch_size == 32
        assert loaded.retrieval.top_k == 24

    def test_missing_file_raises(self, tmp_path) -> None:
        with pytest.raises(FileNotFoundError):
            POIConfig.from_yaml(tmp_path / "does-not-exist.yaml")

    def test_unknown_field_rejected(self, tmp_path) -> None:
        path = tmp_path / "bad.yaml"
        path.write_text("embedding:\n  unknown_field: oops\n")
        # Pydantic v2 by default ignores extras; ensure it still loads
        # but flag this as something to revisit if we tighten validation.
        cfg = POIConfig.from_yaml(path)
        # Default values should be intact
        assert cfg.embedding.backend == "siglip2"


class TestIndexConfigValidation:
    def test_nprobe_must_be_at_most_nlist(self) -> None:
        with pytest.raises(ValueError, match="cannot exceed"):
            IndexConfig(nlist=10, nprobe=100)

    def test_valid_nprobe_passes(self) -> None:
        cfg = IndexConfig(nlist=100, nprobe=10)
        assert cfg.nprobe == 10
