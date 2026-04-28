"""Hash-based encoder for offline demos.

This is NOT a production encoder. It does not learn semantic similarity.
It exists for two reasons:

1. **Validation in environments without internet access.** When you can't
   download SigLIP-2 weights from HuggingFace (CI, air-gapped clusters,
   sandboxed dev environments), you still want to verify that the
   pipeline glues together correctly: build → save → load → search →
   eval, end to end, with real measured numbers.

2. **A baseline that's strictly worse than CLIP.** Including a known-bad
   encoder in the ablation table makes it impossible to mistake the
   "real" results for fabrication. If hash-encoder Recall@1 is non-zero,
   it's because of label-set overlap in the synthetic corpus, not because
   the model learned anything.

Mechanism: encode_images reads the image filename's attribute row and
projects it through a fixed hash matrix. encode_texts hashes attribute
keywords found in the query. The space is shared because both use the
same hash matrix. There is no learning.

For real retrieval quality numbers, use SigLIPEncoder or CLIPEncoder.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from poi.embeddings.base import Encoder
from poi.utils.logging import get_logger

log = get_logger(__name__)


# Same vocabulary used in CelebA attribute names + their natural-language
# forms. We hash on the lowercase normalized form.
_ATTR_KEYWORDS = {
    "smiling": "Smiling",
    "smile": "Smiling",
    "glasses": "Eyeglasses",
    "eyeglasses": "Eyeglasses",
    "hat": "Wearing_Hat",
    "earrings": "Wearing_Earrings",
    "necklace": "Wearing_Necklace",
    "lipstick": "Wearing_Lipstick",
    "makeup": "Heavy_Makeup",
    "bangs": "Bangs",
    "black hair": "Black_Hair",
    "blond": "Blond_Hair",
    "blonde": "Blond_Hair",
    "brown hair": "Brown_Hair",
    "gray hair": "Gray_Hair",
    "grey hair": "Gray_Hair",
    "curly": "Curly_Hair",
    "wavy": "Wavy_Hair",
    "straight": "Straight_Hair",
    "bald": "Bald",
    "beard": "No_Beard_INVERTED",  # special: "no beard" → No_Beard, "beard" → not(No_Beard)
    "clean-shaven": "No_Beard",
    "stubble": "5_o_Clock_Shadow",
    "mustache": "Mustache",
    "moustache": "Mustache",
    "goatee": "Goatee",
    "sideburns": "Sideburns",
    "bushy eyebrows": "Bushy_Eyebrows",
    "arched eyebrows": "Arched_Eyebrows",
    "rosy": "Rosy_Cheeks",
    "pale": "Pale_Skin",
    "high cheekbones": "High_Cheekbones",
    "pointy nose": "Pointy_Nose",
    "big nose": "Big_Nose",
    "full lips": "Big_Lips",
    "oval face": "Oval_Face",
    "receding hairline": "Receding_Hairline",
    "bags under": "Bags_Under_Eyes",
    "mouth slightly open": "Mouth_Slightly_Open",
}


def _attr_to_unit_vector(attr_name: str, dim: int) -> np.ndarray:
    """Stable random unit vector seeded by the attribute name.

    Same attribute → same vector across processes, runs, machines.
    """
    h = hashlib.sha256(attr_name.encode("utf-8")).digest()
    seed = int.from_bytes(h[:8], "big")
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v) + 1e-12
    return v


class HashEncoder(Encoder):
    """Attribute-aware deterministic encoder. CPU-only, no network.

    Args:
        attributes_csv: Path to a CelebA-format attribute CSV. Image
            filenames are looked up here to construct embeddings.
        dim: Output embedding dimension.
        normalize: L2-normalize the output (required for cosine via inner
            product, just like the real encoders).
    """

    def __init__(
        self,
        attributes_csv: str | Path,
        dim: int = 256,
        normalize: bool = True,
    ) -> None:
        self._dim = dim
        self.normalize = normalize
        self._attrs_path = Path(attributes_csv)

        log.info(f"HashEncoder loading attribute table from {self._attrs_path}")
        df = pd.read_csv(self._attrs_path)
        id_col = "image_id" if "image_id" in df.columns else df.columns[0]
        self._attrs = df.set_index(id_col)
        self._attr_columns = list(self._attrs.columns)

        # Pre-compute per-attribute basis vectors
        self._basis = {a: _attr_to_unit_vector(a, dim) for a in self._attr_columns}

    @property
    def name(self) -> str:
        return "hash-encoder"

    @property
    def dim(self) -> int:
        return self._dim

    def encode_images(
        self,
        images: list[Image.Image] | list[Path] | list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Look up the attribute row for each image and sum its positive-attribute basis vectors."""
        out = np.zeros((len(images), self._dim), dtype=np.float32)
        for i, item in enumerate(images):
            if isinstance(item, str | Path):
                filename = Path(item).name
            else:
                # PIL image — try the filename hint; otherwise return zero vector
                filename = getattr(item, "filename", None)
                if filename is None:
                    continue
                filename = Path(filename).name

            if filename not in self._attrs.index:
                continue

            row = self._attrs.loc[filename]
            for attr in self._attr_columns:
                if row[attr] > 0:
                    out[i] += self._basis[attr]

        if self.normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            out = out / norms
        return out.astype(np.float32)

    def encode_texts(
        self,
        texts: list[str],
        batch_size: int | None = None,
    ) -> np.ndarray:
        """Match keywords against the same attribute basis vectors."""
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i, text in enumerate(texts):
            t = text.lower()
            for keyword, attr in _ATTR_KEYWORDS.items():
                if keyword in t:
                    if attr.endswith("_INVERTED"):
                        # Special case for "beard": the CelebA attribute is
                        # "No_Beard" (positive when there is no beard). So:
                        #   "no beard"  → push toward  No_Beard direction
                        #   "beard"     → push away from No_Beard direction
                        clean_attr = attr.replace("_INVERTED", "")
                        if clean_attr not in self._basis:
                            continue
                        if _has_negation_for(t, keyword):
                            out[i] += self._basis[clean_attr]
                        else:
                            out[i] -= self._basis[clean_attr]
                    else:
                        if attr in self._basis:
                            out[i] += self._basis[attr]

        if self.normalize:
            norms = np.linalg.norm(out, axis=1, keepdims=True)
            norms = np.where(norms < 1e-12, 1.0, norms)
            out = out / norms
        return out.astype(np.float32)


def _has_negation_for(text: str, keyword: str) -> bool:
    """Detect a 'no <keyword>' / 'without <keyword>' pattern in `text`.

    A real model does this implicitly (badly, as Notebook 02 documents).
    Here we make the keyword logic explicit so the demo encoder behaves
    plausibly on negation-bearing queries.
    """
    idx = text.find(keyword)
    if idx == -1:
        return False
    before = text[max(0, idx - 20) : idx]
    return any(neg in before for neg in [" no ", " not ", " without ", "n't "])
