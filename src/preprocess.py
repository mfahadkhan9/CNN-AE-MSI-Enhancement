"""Preprocess multispectral image folders.

This script resizes RGB images, optionally creates grayscale images, and saves
histograms for inspection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def preprocess_folder(input_dir: str, output_dir: str, image_size: int = 256, save_histograms: bool = True) -> None:
    input_root = Path(input_dir)
    output_root = Path(output_dir)
    rgb_dir = output_root / "rgb"
    gray_dir = output_root / "grayscale"
    hist_dir = output_root / "histograms"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    gray_dir.mkdir(parents=True, exist_ok=True)
    if save_histograms:
        hist_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in input_root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise RuntimeError(f"No image files found in {input_root}")

    for path in tqdm(paths, desc="Preprocessing"):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"Warning: failed to read {path}")
            continue

        image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_norm = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        rel = path.relative_to(input_root)
        rgb_out = rgb_dir / rel
        gray_out = gray_dir / rel.with_suffix(".png")
        rgb_out.parent.mkdir(parents=True, exist_ok=True)
        gray_out.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(rgb_out), image)
        cv2.imwrite(str(gray_out), gray_norm)

        if save_histograms:
            hist_out = hist_dir / rel.with_suffix(".png")
            hist_out.parent.mkdir(parents=True, exist_ok=True)
            plt.figure(figsize=(6, 4))
            plt.hist(gray_norm.ravel(), bins=256, range=(0, 255))
            plt.xlabel("Pixel intensity")
            plt.ylabel("Frequency")
            plt.title(f"Histogram: {path.stem}")
            plt.tight_layout()
            plt.savefig(hist_out, dpi=300)
            plt.close()

    print(f"Preprocessed images saved to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess multispectral image folders.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--no_histograms", action="store_true")
    args = parser.parse_args()
    preprocess_folder(args.input_dir, args.output_dir, args.image_size, not args.no_histograms)
