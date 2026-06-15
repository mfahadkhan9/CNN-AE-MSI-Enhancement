"""Evaluate a trained CNN-AE checkpoint.

Example:
    python src/evaluate.py --config configs/config.yaml --checkpoint checkpoints/best_model.pth
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
from skimage import restoration
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageFolderAutoEncoderDataset
from metrics import compute_metrics
from model import CNNAutoEncoder
from utils import ensure_dir, get_device, load_config, save_json, save_tensor_image, tensor_to_numpy_image


def _safe_load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def evaluate(config_path: str, checkpoint_path: str, input_dir: str | None = None, compare_nlm: bool = False) -> None:
    cfg = load_config(config_path)
    device = get_device(cfg.get("device", "auto"))

    eval_dir = input_dir or cfg["data"].get("test_dir") or cfg["data"].get("val_dir") or cfg["data"]["train_dir"]
    image_size = int(cfg["data"].get("image_size", 256))

    dataset = ImageFolderAutoEncoderDataset(eval_dir, image_size=image_size, target_root=cfg["data"].get("target_test_dir"))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = CNNAutoEncoder(
        in_channels=int(cfg["model"].get("in_channels", 3)),
        base_channels=int(cfg["model"].get("base_channels", 64)),
    ).to(device)
    model.load_state_dict(_safe_load_checkpoint(checkpoint_path, device))
    model.eval()

    enhanced_dir = ensure_dir(cfg["paths"].get("enhanced_dir", "results/enhanced_images"))
    metrics_dir = ensure_dir(cfg["paths"].get("metrics_dir", "results/metrics"))

    rows = []
    with torch.no_grad():
        for inputs, targets, paths, wavelength in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)

            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            outputs = model(inputs)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            input_np = tensor_to_numpy_image(targets[0])
            output_np = tensor_to_numpy_image(outputs[0])
            values = compute_metrics(input_np, output_np, elapsed_seconds=elapsed)

            input_path = Path(paths[0])
            wavelength_name = wavelength[0]
            out_path = enhanced_dir / wavelength_name / f"enhanced_{input_path.name}"
            save_tensor_image(outputs[0], out_path)

            row = {"image": str(input_path), "wavelength": wavelength_name, **values, "output": str(out_path)}
            rows.append(row)

            if compare_nlm:
                start_nlm = time.perf_counter()
                nlm = restoration.denoise_nl_means(input_np, h=1.15 * np.std(input_np), fast_mode=True, channel_axis=-1)
                elapsed_nlm = time.perf_counter() - start_nlm
                nlm_values = compute_metrics(input_np, nlm, elapsed_seconds=elapsed_nlm)
                nlm_path = enhanced_dir / wavelength_name / f"nlm_{input_path.name}"
                from PIL import Image
                Image.fromarray((np.clip(nlm, 0, 1) * 255).round().astype(np.uint8)).save(nlm_path)
                rows.append({"image": str(input_path), "wavelength": wavelength_name, "method": "NLM", **nlm_values, "output": str(nlm_path)})

    csv_path = metrics_dir / "evaluation_metrics.csv"
    fieldnames = ["image", "wavelength", "CC", "MI", "RMSE", "PSNR", "RT_seconds", "output"]
    extra_keys = sorted(set().union(*(r.keys() for r in rows)) - set(fieldnames))
    fieldnames = fieldnames[:2] + extra_keys + fieldnames[2:]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # Aggregate only CNN-AE rows unless compare_nlm added a method column.
    ae_rows = [r for r in rows if r.get("method", "CNN-AE") != "NLM"]
    summary = {}
    for key in ["CC", "MI", "RMSE", "PSNR", "RT_seconds"]:
        vals = np.array([float(r[key]) for r in ae_rows if key in r and np.isfinite(float(r[key]))])
        if vals.size:
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": int(vals.size)}
    summary_path = metrics_dir / "summary_metrics.json"
    save_json(summary, summary_path)

    print(f"Saved image-level metrics to: {csv_path}")
    print(f"Saved summary metrics to: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CNN-AE checkpoint.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth", help="Path to checkpoint file.")
    parser.add_argument("--input_dir", default=None, help="Optional input directory overriding config test_dir.")
    parser.add_argument("--compare_nlm", action="store_true", help="Also evaluate Non-Local Means baseline.")
    args = parser.parse_args()
    evaluate(args.config, args.checkpoint, args.input_dir, args.compare_nlm)
