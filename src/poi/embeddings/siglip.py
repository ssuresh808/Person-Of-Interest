"""SigLIP-2 encoder.

SigLIP replaces CLIP's softmax contrastive loss with independent sigmoid
losses on each pair. The practical consequence for retrieval: the
embedding space is more uniformly distributed on the unit hypersphere,
so above-zero similarity scores carry more signal.

We use the HuggingFace transformers integration. SigLIP-2 was added in
transformers >= 4.45.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoModel, AutoProcessor

from poi.embeddings.base import Encoder
from poi.utils.device import get_device_info
from poi.utils.logging import get_logger

log = get_logger(__name__)


def _load_image(item: Image.Image | Path | str) -> Image.Image:
    """Coerce inputs to RGB PIL images."""
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    return Image.open(item).convert("RGB")


class SigLIPEncoder(Encoder):
    """SigLIP / SigLIP-2 unified text-image encoder.

    Args:
        model_name: HuggingFace identifier. Defaults to siglip2-base-patch16-256.
            For higher quality at ~3x cost, use google/siglip2-large-patch16-256.
        batch_size: Forward pass batch size. 64 fits on a 4090 shard for the
            base model; drop to 32 for large.
        normalize: If True (default), L2-normalize outputs. Required for
            cosine similarity via inner product.
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-base-patch16-256",
        batch_size: int = 64,
        normalize: bool = True,
        cache_dir: str | Path | None = None,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        device_info = get_device_info()
        self._device = device_info.device
        self._dtype = device_info.dtype

        log.info(f"Loading {model_name} on {self._device} ({self._dtype})")

        self.processor = AutoProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            cache_dir=cache_dir,
        )
        self.model = self.model.to(self._device).eval()

        # Probe the embedding dimension once.
        with torch.no_grad():
            dummy = self.processor(
                text=["dimension probe"],
                return_tensors="pt",
                padding="max_length",
            ).to(self._device)
            out = self.model.get_text_features(**dummy)
            self._dim = int(out.shape[-1])
        log.info(f"  dim = {self._dim}")

    # ----- Encoder protocol -----

    @property
    def name(self) -> str:
        # Strip org prefix for compactness in artifact filenames.
        short = self.model_name.split("/")[-1]
        return f"siglip-{short}"

    @property
    def dim(self) -> int:
        return self._dim

    @torch.inference_mode()
    def encode_images(
        self,
        images: list[Image.Image] | list[Path] | list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        bs = batch_size or self.batch_size
        out_chunks: list[np.ndarray] = []

        # tqdm only for batches > 1, otherwise it's noise
        iterator = range(0, len(images), bs)
        if len(images) > bs:
            iterator = tqdm(iterator, desc=f"  encoding images [{self.name}]", leave=False)

        for i in iterator:
            batch_items = [_load_image(x) for x in images[i : i + bs]]
            inputs = self.processor(images=batch_items, return_tensors="pt").to(self._device)
            features = self.model.get_image_features(**inputs)
            if self.normalize:
                features = torch.nn.functional.normalize(features, dim=-1)
            out_chunks.append(features.float().cpu().numpy())

        return np.concatenate(out_chunks, axis=0).astype(np.float32)

    @torch.inference_mode()
    def encode_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        bs = batch_size or self.batch_size
        out_chunks: list[np.ndarray] = []

        for i in range(0, len(texts), bs):
            batch = texts[i : i + bs]
            inputs = self.processor(
                text=batch,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
            ).to(self._device)
            features = self.model.get_text_features(**inputs)
            if self.normalize:
                features = torch.nn.functional.normalize(features, dim=-1)
            out_chunks.append(features.float().cpu().numpy())

        return np.concatenate(out_chunks, axis=0).astype(np.float32)
