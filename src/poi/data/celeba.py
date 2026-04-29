"""CelebA dataset loader.

CelebA ships with 40 binary face attributes per image — Smiling, Eyeglasses,
Wearing_Hat, Black_Hair, etc. We use these for two things:

1. Loading images by path (the easy part).
2. Synthesizing realistic text queries from attribute combinations, so we
   have ground truth (query, target) pairs for evaluation.

The query synthesis is more interesting than it sounds. CelebA's attributes
are noisy and imbalanced — Wearing_Earrings is annotated only on faces where
they're clearly visible, and many "negative" examples actually have them.
Treating attributes as a clean signal would inflate eval numbers. The
synthesizer here uses conservative attribute combinations and simple
templates that produce queries a human might plausibly type.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from poi.utils.logging import get_logger

log = get_logger(__name__)


# A curated subset of CelebA attributes that produce visually meaningful queries.
# The full 40-attribute set includes labels like "Attractive" and "Young" that
# are subjective, demographic, or just bad ideas to surface in a portfolio
# project. We deliberately exclude those.
_VISUAL_ATTRIBUTES: dict[str, str] = {
    "Bald": "is bald",
    "Bangs": "has bangs",
    "Black_Hair": "has black hair",
    "Blond_Hair": "has blond hair",
    "Brown_Hair": "has brown hair",
    "Gray_Hair": "has gray hair",
    "Curly_Hair": "has curly hair",
    "Straight_Hair": "has straight hair",
    "Wavy_Hair": "has wavy hair",
    "Receding_Hairline": "has a receding hairline",
    "Eyeglasses": "is wearing glasses",
    "Wearing_Hat": "is wearing a hat",
    "Wearing_Earrings": "is wearing earrings",
    "Wearing_Necklace": "is wearing a necklace",
    "Wearing_Lipstick": "is wearing lipstick",
    "Heavy_Makeup": "has heavy makeup",
    "Smiling": "is smiling",
    "Mouth_Slightly_Open": "has their mouth slightly open",
    "Mustache": "has a mustache",
    "Goatee": "has a goatee",
    "No_Beard": "is clean-shaven",
    "5_o_Clock_Shadow": "has stubble",
    "Sideburns": "has sideburns",
    "Bushy_Eyebrows": "has bushy eyebrows",
    "Arched_Eyebrows": "has arched eyebrows",
    "Bags_Under_Eyes": "has bags under their eyes",
    "Big_Lips": "has full lips",
    "Big_Nose": "has a prominent nose",
    "High_Cheekbones": "has high cheekbones",
    "Pointy_Nose": "has a pointy nose",
    "Rosy_Cheeks": "has rosy cheeks",
    "Pale_Skin": "has pale skin",
    "Oval_Face": "has an oval face",
}

# Excluded attributes and why:
#   Attractive, Young — subjective / demographic, not a visual property
#   Male — gendered prediction we explicitly don't want to evaluate against
#   Chubby, Double_Chin, Narrow_Eyes — risk of stereotype reinforcement
#   Blurry — image quality, not a person attribute


@dataclass
class CelebASample:
    """One row of CelebA: an image and its (selected) attributes."""

    filename: str
    path: Path
    attributes: dict[str, bool]
    split: str  # "train" / "val" / "test"

    @property
    def positive_attributes(self) -> list[str]:
        """Attributes that are present (value True)."""
        return [k for k, v in self.attributes.items() if v]


class CelebADataset:
    """Lazy loader for CelebA.

    Args:
        images_dir: Path to img_align_celeba/.
        attributes_csv: Path to list_attr_celeba.csv (or .txt converted).
        eval_split_csv: Path to list_eval_partition.csv. If None, all samples
            are reported as 'train'.
    """

    def __init__(
        self,
        images_dir: Path | str,
        attributes_csv: Path | str | None = None,
        eval_split_csv: Path | str | None = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        if not self.images_dir.exists():
            raise FileNotFoundError(
                f"CelebA images directory not found: {self.images_dir}\n"
                f"Run scripts/download_data.py first."
            )

        self._attrs: pd.DataFrame | None = None
        self._splits: pd.Series | None = None

        if attributes_csv is not None:
            attrs_path = Path(attributes_csv)
            if attrs_path.exists():
                self._attrs = self._load_attributes(attrs_path)
                log.info(f"Loaded {len(self._attrs)} attribute rows")

        if eval_split_csv is not None:
            split_path = Path(eval_split_csv)
            if split_path.exists():
                self._splits = self._load_splits(split_path)

    # ----- Loading -----

    @staticmethod
    def _load_attributes(path: Path) -> pd.DataFrame:
        """Load attributes from CSV. Format varies between distributions —
        Kaggle ships CSV; the original release ships a custom whitespace format."""
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            # Original CelebA format: header line with attribute names, then
            # one row per image with -1/+1 values.
            df = pd.read_csv(path, sep=r"\s+", skiprows=1, header=0)
            df = df.reset_index().rename(columns={"index": "image_id"})

        # Normalize: image_id (or first column) becomes the index, values become bool
        id_col = "image_id" if "image_id" in df.columns else df.columns[0]
        df = df.set_index(id_col)
        df = (df > 0).astype(bool)
        return df

    @staticmethod
    def _load_splits(path: Path) -> pd.Series:
        """Map image_id -> split name."""
        if path.suffix.lower() == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_csv(path, sep=r"\s+", header=None, names=["image_id", "partition"])

        split_map = {0: "train", 1: "val", 2: "test"}
        id_col = "image_id" if "image_id" in df.columns else df.columns[0]
        part_col = "partition" if "partition" in df.columns else df.columns[1]
        return df.set_index(id_col)[part_col].map(split_map)

    # ----- Access -----

    def list_image_paths(self, split: str | None = None) -> list[Path]:
        """All image paths, optionally filtered to a split."""
        all_paths = sorted(self.images_dir.glob("*.jpg"))
        if split is None or self._splits is None:
            return all_paths
        keep = {fn for fn, s in self._splits.items() if s == split}
        return [p for p in all_paths if p.name in keep]

    def get_sample(self, filename: str) -> CelebASample:
        """Load attributes + split metadata for one image."""
        path = self.images_dir / filename
        attrs: dict[str, bool] = {}
        if self._attrs is not None and filename in self._attrs.index:
            row = self._attrs.loc[filename]
            attrs = {col: bool(row[col]) for col in row.index if col in _VISUAL_ATTRIBUTES}
        split = "train"
        if self._splits is not None and filename in self._splits.index:
            split = str(self._splits.loc[filename])
        return CelebASample(filename=filename, path=path, attributes=attrs, split=split)

    def __len__(self) -> int:
        return len(self.list_image_paths())


# ----- Query synthesis for evaluation -----


def synthesize_query(
    sample: CelebASample,
    n_attributes: int = 3,
    rng: random.Random | None = None,
) -> str:
    """Turn a CelebA sample's attributes into a natural-language query.

    The query mentions a small number of randomly-chosen positive attributes,
    not all of them. This mirrors how a real user describes someone — they
    pick a few salient features, not an attribute checklist.

    Args:
        sample: The CelebA sample.
        n_attributes: How many attributes to mention.
        rng: Optional Random instance for reproducibility.

    Returns:
        A sentence like "A person who has black hair and is wearing glasses."
        Returns a generic description if too few visual attributes are present.
    """
    rng = rng or random.Random()

    positive = [a for a in sample.positive_attributes if a in _VISUAL_ATTRIBUTES]
    if len(positive) < n_attributes:
        # Not enough attributes to make a discriminative query.
        # Fall back to whatever we have.
        chosen = positive
    else:
        chosen = rng.sample(positive, n_attributes)

    if not chosen:
        return "A person."

    phrases = [_VISUAL_ATTRIBUTES[a] for a in chosen]
    if len(phrases) == 1:
        body = phrases[0]
    elif len(phrases) == 2:
        body = f"{phrases[0]} and {phrases[1]}"
    else:
        body = f"{', '.join(phrases[:-1])}, and {phrases[-1]}"

    return f"A person who {body}."
