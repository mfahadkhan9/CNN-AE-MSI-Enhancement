"""Run CNN-AE inference and save enhanced images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageFolderAutoEncoderDataset
from model import CNNAutoEncoder
from utils import ensure_dir, get_device, load_config, save_tensor_image


def _safe_load_checkpoint(path: str | Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint


def run_inference(config_path: str, checkpoint_path: str, input_dir: str, output_dir: str) -> None:
    cfg = load_config(config_path)
    device = get_device(cfg.get("device", "auto"))
    image_size = int(cfg["data"].get("image_size", 256))

    dataset = ImageFolderAutoEncoderDataset(input_dir, image_size=image_size)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    model = CNNAutoEncoder(
        in_channels=int(cfg["model"].get("in_channels", 3)),
        base_channels=int(cfg["model"].get("base_channels", 64)),
    ).to(device)
    model.load_state_dict(_safe_load_checkpoint(checkpoint_path, device))
    model.eval()

    output_root = ensure_dir(output_dir)
    with torch.no_grad():
        for inputs, _, paths, wavelength in tqdm(loader, desc="Enhancing images"):
            inputs = inputs.to(device=device, dtype=torch.float32)
            outputs = model(inputs)
            input_path = Path(paths[0])
            out_path = output_root / wavelength[0] / f"enhanced_{input_path.name}"
            save_tensor_image(outputs[0], out_path)

    print(f"Enhanced images saved to: {output_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN-AE inference.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", default="results/enhanced_images")
    args = parser.parse_args()
    run_inference(args.config, args.checkpoint, args.input_dir, args.output_dir)
