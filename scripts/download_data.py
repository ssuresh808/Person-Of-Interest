"""Download CelebA via Kaggle.

Usage:
    python scripts/download_data.py --dataset celeba --out data/

Requires Kaggle API credentials in ~/.kaggle/kaggle.json or in env vars
KAGGLE_USERNAME / KAGGLE_KEY. See https://github.com/Kaggle/kaggle-api.

We pull the "jessicali9530/celeba-dataset" mirror — the original
mmlab.ie.cuhk.edu.hk distribution requires manual Google Drive sign-in,
which doesn't work on a headless cluster.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

from poi.utils.logging import get_logger, setup_logging

log = get_logger(__name__)


CELEBA_KAGGLE_REF = "jessicali9530/celeba-dataset"
EXPECTED_FILES = [
    "img_align_celeba.zip",
    "list_attr_celeba.csv",
    "list_eval_partition.csv",
]


def check_kaggle_credentials() -> None:
    """Verify Kaggle credentials are available before attempting download."""
    has_env = bool(os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))
    has_file = Path("~/.kaggle/kaggle.json").expanduser().exists()
    if not (has_env or has_file):
        log.error(
            "Kaggle credentials not found. Either:\n"
            "  1. Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or\n"
            "  2. Place kaggle.json at ~/.kaggle/kaggle.json (chmod 600)\n"
            "Get a token at https://www.kaggle.com/settings/account → 'Create New Token'"
        )
        sys.exit(1)


def download_celeba(out_dir: Path) -> None:
    """Pull CelebA from Kaggle and unpack the image archive."""
    check_kaggle_credentials()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download. We shell out to the kaggle CLI rather than importing it —
    # the CLI's handling of progress bars is much nicer on a long download.
    log.info(f"Downloading {CELEBA_KAGGLE_REF} to {out_dir}")
    cmd = [
        "kaggle",
        "datasets",
        "download",
        "-d",
        CELEBA_KAGGLE_REF,
        "-p",
        str(out_dir),
        "--unzip",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        log.error("Kaggle download failed. Is the kaggle CLI installed and authenticated?")
        sys.exit(result.returncode)

    # The unzipped archive may have nested structure; flatten if needed.
    img_dir = out_dir / "img_align_celeba" / "img_align_celeba"
    if img_dir.exists():
        # Move from nested location up one level.
        target = out_dir / "img_align_celeba"
        for f in img_dir.iterdir():
            shutil.move(str(f), str(target / f.name))
        img_dir.rmdir()

    n_images = len(list((out_dir / "img_align_celeba").glob("*.jpg")))
    log.info(f"CelebA download complete: {n_images} images in {out_dir / 'img_align_celeba'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download face dataset")
    parser.add_argument("--dataset", choices=["celeba"], default="celeba")
    parser.add_argument("--out", type=Path, default=Path("data"))
    args = parser.parse_args()

    setup_logging()

    out_dir = args.out / args.dataset
    if args.dataset == "celeba":
        download_celeba(out_dir)
    else:
        log.error(f"Unsupported dataset: {args.dataset}")
        sys.exit(1)


if __name__ == "__main__":
    main()
