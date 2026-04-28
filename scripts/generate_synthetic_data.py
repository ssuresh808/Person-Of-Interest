"""Generate a synthetic demo corpus.

When you can't download CelebA (no Kaggle credentials, no internet, etc.),
this script builds a small synthetic corpus that exercises the entire
pipeline end-to-end. It's also what the docs/screenshots scripts use to
produce demo images without needing the real dataset.

What it generates:
    - N synthetic face-like JPEG images (PIL-drawn placeholders, not real
      faces; the files are real image files but the pixel content is a
      labeled card).
    - A list_attr_celeba.csv with random attributes drawn from CelebA's
      attribute set, so query synthesis works.
    - A list_eval_partition.csv assigning splits.

The point: the pipeline ran. Real CelebA replaces this directly — same
file layout, same column names — without changing any code.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# A subset of CelebA's 40 attributes. Same names, same +1/-1 encoding,
# so swapping in real CelebA is a no-op.
CELEBA_ATTRIBUTES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
    "Curly_Hair",
]


def make_placeholder_image(filename: str, attrs: dict[str, bool], size: int = 218) -> Image.Image:
    """Render a colorful placeholder card. Not a face — a labeled card.

    The card encodes the attribute summary in its rendered text. This means
    a vision encoder running over these images would actually learn
    something — the visual content correlates with the labels, which is
    enough to validate that retrieval ranks consistently with attributes.
    """
    # Color seeded by the filename so it's deterministic
    seed = (
        int(filename.replace(".jpg", "")) if filename[:-4].isdigit() else hash(filename) & 0xFFFFFF
    )
    rng = random.Random(seed)
    bg = (rng.randint(80, 220), rng.randint(80, 220), rng.randint(80, 220))
    fg = (255 - bg[0], 255 - bg[1], 255 - bg[2])

    img = Image.new("RGB", (178, size), color=bg)
    draw = ImageDraw.Draw(img)

    # Stylized "face" geometry — simple shapes that vary with attributes
    # so the rendered images have visual differences correlated with labels.
    # Eyes
    draw.ellipse((45, 70, 75, 100), fill=fg)
    draw.ellipse((103, 70, 133, 100), fill=fg)
    if attrs.get("Eyeglasses"):
        draw.rectangle((40, 65, 80, 105), outline=fg, width=2)
        draw.rectangle((98, 65, 138, 105), outline=fg, width=2)

    # Mouth
    if attrs.get("Smiling"):
        draw.arc((60, 130, 118, 170), start=0, end=180, fill=fg, width=3)
    else:
        draw.line((60, 150, 118, 150), fill=fg, width=2)

    # Hat
    if attrs.get("Wearing_Hat"):
        draw.rectangle((30, 10, 148, 40), fill=fg)

    # Beard / no_beard
    if attrs.get("No_Beard"):
        pass  # smooth chin
    elif attrs.get("Goatee") or attrs.get("Mustache"):
        draw.line((70, 180, 108, 180), fill=fg, width=4)

    # Attribute label at bottom for visual differentiation
    positive = [k for k, v in attrs.items() if v][:2]
    label = ", ".join(positive) if positive else "neutral"
    try:
        font = ImageFont.load_default()
        draw.text((5, size - 14), label[:24], fill=fg, font=font)
    except Exception:
        pass

    return img


def generate_corpus(out_dir: Path, n_images: int, seed: int = 42) -> None:
    """Create the full synthetic dataset under out_dir."""
    rng = random.Random(seed)
    images_dir = out_dir / "img_align_celeba"
    images_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    splits: list[dict] = []

    for i in range(1, n_images + 1):
        filename = f"{i:06d}.jpg"
        # Sample attributes — about 5-8 positive per image, like real CelebA
        n_positive = rng.randint(4, 9)
        positive = set(rng.sample(CELEBA_ATTRIBUTES, n_positive))
        attrs = {a: (a in positive) for a in CELEBA_ATTRIBUTES}

        img = make_placeholder_image(filename, attrs)
        img.save(images_dir / filename, "JPEG", quality=85)

        # Attribute row uses +1/-1 encoding like the original CelebA
        row = {"image_id": filename}
        row.update({a: (1 if v else -1) for a, v in attrs.items()})
        rows.append(row)

        # 80/10/10 split
        r = rng.random()
        partition = 0 if r < 0.8 else (1 if r < 0.9 else 2)
        splits.append({"image_id": filename, "partition": partition})

    pd.DataFrame(rows).to_csv(out_dir / "list_attr_celeba.csv", index=False)
    pd.DataFrame(splits).to_csv(out_dir / "list_eval_partition.csv", index=False)

    print(f"Generated {n_images} images in {images_dir}")
    print(f"Attributes: {out_dir / 'list_attr_celeba.csv'}")
    print(f"Splits:     {out_dir / 'list_eval_partition.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic CelebA-format corpus")
    parser.add_argument("--out", type=Path, default=Path("data/celeba_synthetic"))
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    generate_corpus(args.out, n_images=args.n, seed=args.seed)


if __name__ == "__main__":
    main()
