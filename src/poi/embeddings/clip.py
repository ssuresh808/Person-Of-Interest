"""CLIP encoder.

Exists as a baseline for the SigLIP-2 vs CLIP ablation. The implementation
mirrors SigLIPEncoder so the comparison is apples-to-apples — same dtype,
same batch handling, same normalization.

The fact that we can swap encoders without touching downstream code is the
point of the Encoder protocol.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor

from poi.embeddings.base import Encoder
from poi.utils.device import get_device_info
from poi.utils.logging import get_logger

log = get_logger(__name__)


def _load_image(item: Image.Image | Path | str) -> Image.Image:
    if isinstance(item, Image.Image):
        return item.convert("RGB")
    return Image.open(item).convert("RGB")


class CLIPEncoder(Encoder):
    """OpenAI CLIP encoder, used as the baseline in the encoder ablation.

    Args:
        model_name: HuggingFace identifier. Defaults to ViT-L/14 — the standard
            "good CLIP" variant. ViT-B/32 is faster but noticeably weaker.
        batch_size: Forward pass batch size.
        normalize: L2-normalize outputs for cosine-via-inner-product.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
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

        self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=cache_dir)
        self.model = CLIPModel.from_pretrained(
            model_name,
            torch_dtype=self._dtype,
            cache_dir=cache_dir,
        )
        self.model = self.model.to(self._device).eval()
        self._dim = int(self.model.config.projection_dim)
        log.info(f"  dim = {self._dim}")

    @property
    def name(self) -> str:
        short = self.model_name.split("/")[-1]
        return f"clip-{short}"

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
                padding=True,
                truncation=True,
            ).to(self._device)
            features = self.model.get_text_features(**inputs)
            if self.normalize:
                features = torch.nn.functional.normalize(features, dim=-1)
            out_chunks.append(features.float().cpu().numpy())

        return np.concatenate(out_chunks, axis=0).astype(np.float32)
