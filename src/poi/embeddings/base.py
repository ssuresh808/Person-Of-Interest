"""Encoder protocol.

Every embedding backend (SigLIP-2, CLIP, DeepFace) implements the same small
surface area. This makes it trivial to swap encoders for ablations: the
index, retrieval, and UI layers don't know which backend is in use.

Design note: we use a Protocol rather than an abstract base class because
DeepFace's encoder is a function, not a class instance, and we don't want
to force a wrapper just for type-system purity. Structural typing handles
both cases.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
from PIL import Image


@runtime_checkable
class Encoder(Protocol):
    """Common interface for any (image, text) → vector encoder.

    All embeddings returned should be:
        - float32 numpy arrays
        - shape (N, D) where D is the same for both modalities
        - L2-normalized if the config says so (the standard case)

    This last point matters: cosine similarity == inner product only when
    both sides are unit-norm. Get this wrong and the FAISS index returns
    silently meaningless results.
    """

    @property
    def name(self) -> str:
        """Human-readable identifier, used for logging and artifact naming."""
        ...

    @property
    def dim(self) -> int:
        """Output embedding dimensionality."""
        ...

    def encode_images(
        self,
        images: list[Image.Image] | list[Path] | list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode a batch of images. Accepts PIL images or file paths."""
        ...

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Encode a batch of text descriptions."""
        ...
