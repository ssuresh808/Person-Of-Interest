"""DeepFace baseline.

DeepFace is a face-recognition library — its embeddings are tuned for
identity matching, not for description-to-face retrieval. We include it as
a baseline to make a point: a face-specific encoder is *not* automatically
better at this task than a generic vision-language model. The right
embedding depends on what you're matching, not on what you're matching it
against.

DeepFace doesn't natively encode text, so this encoder raises on
encode_texts. Calling it in a multimodal pipeline is a misuse — the
factory only instantiates it for image-to-image baselines.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from poi.embeddings.base import Encoder
from poi.utils.logging import get_logger

log = get_logger(__name__)


class DeepFaceEncoder(Encoder):
    """Image-only face encoder via the DeepFace library.

    Args:
        model_name: One of DeepFace's supported models. Facenet512 is the
            default — 512-dim embeddings, strong identity discrimination.
        normalize: L2-normalize outputs.
    """

    def __init__(
        self,
        model_name: str = "Facenet512",
        normalize: bool = True,
    ) -> None:
        try:
            from deepface import DeepFace  # noqa: F401  (verifies install)
        except ImportError as e:
            raise ImportError(
                "DeepFace baseline requested but not installed. Run: pip install deepface"
            ) from e

        self.model_name = model_name
        self.normalize = normalize
        self._dim_cache: int | None = None
        log.info(f"DeepFace baseline configured with model={model_name}")

    @property
    def name(self) -> str:
        return f"deepface-{self.model_name.lower()}"

    @property
    def dim(self) -> int:
        if self._dim_cache is None:
            # Deepface only tells us the dim by encoding something.
            # Use a small synthetic image so the first real call is fast.
            dummy = Image.new("RGB", (160, 160), color=(128, 128, 128))
            self._dim_cache = self.encode_images([dummy]).shape[-1]
        return self._dim_cache

    def encode_images(
        self,
        images: list[Image.Image] | list[Path] | list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        from deepface import DeepFace

        # DeepFace processes paths or numpy arrays; not PIL images.
        # We coerce to numpy arrays in-memory rather than write temp files.
        embeddings: list[np.ndarray] = []
        for item in images:
            if isinstance(item, Image.Image):
                arr = np.array(item.convert("RGB"))
            else:
                arr = np.array(Image.open(item).convert("RGB"))

            try:
                result = DeepFace.represent(
                    img_path=arr,
                    model_name=self.model_name,
                    enforce_detection=False,  # CelebA crops are pre-aligned
                    detector_backend="skip",
                )
                vec = np.asarray(result[0]["embedding"], dtype=np.float32)
            except Exception as e:
                log.warning(f"DeepFace failed on an image: {e}; using zero vector")
                vec = np.zeros(self.dim if self._dim_cache else 512, dtype=np.float32)

            if self.normalize:
                norm = np.linalg.norm(vec)
                if norm > 0:
                    vec = vec / norm
            embeddings.append(vec)

        return np.stack(embeddings).astype(np.float32)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        raise NotImplementedError("DeepFace is image-only. Use SigLIP-2 or CLIP for text queries.")
