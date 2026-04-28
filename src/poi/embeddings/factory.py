"""Encoder factory.

Centralizes the (config) → (encoder instance) mapping so callers don't have
to know about specific encoder classes. This is the single place to add a
new backend.
"""

from __future__ import annotations

from poi.embeddings.base import Encoder
from poi.embeddings.clip import CLIPEncoder
from poi.embeddings.siglip import SigLIPEncoder
from poi.utils.config import EmbeddingConfig
from poi.utils.logging import get_logger

log = get_logger(__name__)


def build_encoder(cfg: EmbeddingConfig) -> Encoder:
    """Instantiate an encoder from an EmbeddingConfig.

    Raises:
        ValueError: if the backend is not recognized.
    """
    backend = cfg.backend.lower()
    cache_dir = str(cfg.cache_dir) if cfg.cache_dir else None

    if backend == "siglip2":
        return SigLIPEncoder(
            model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
            cache_dir=cache_dir,
        )
    if backend == "clip":
        return CLIPEncoder(
            model_name=cfg.model_name,
            batch_size=cfg.batch_size,
            normalize=cfg.normalize,
            cache_dir=cache_dir,
        )
    if backend == "deepface":
        # Lazy import — deepface is heavy and we'd rather not pay for it
        # unless someone explicitly asks for the baseline.
        from poi.embeddings.deepface_baseline import DeepFaceEncoder

        return DeepFaceEncoder(model_name=cfg.model_name, normalize=cfg.normalize)
    if backend == "hash":
        # Offline / no-internet demo encoder. See hash_encoder.py for why
        # this exists and why it's not a substitute for SigLIP-2 in production.
        # The "model_name" field is repurposed as the path to the attribute CSV.
        from poi.embeddings.hash_encoder import HashEncoder

        return HashEncoder(
            attributes_csv=cfg.model_name,
            dim=512,
            normalize=cfg.normalize,
        )

    raise ValueError(
        f"Unknown encoder backend: {cfg.backend!r}. Expected one of: siglip2, clip, deepface, hash."
    )
