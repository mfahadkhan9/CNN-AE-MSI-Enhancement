"""Train the CNN-AE model.

Example:
    python src/train.py --config configs/config.yaml
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import ImageFolderAutoEncoderDataset
from model import CNNAutoEncoder
from utils import ensure_dir, get_device, load_config, set_seed


def train(config_path: str) -> None:
    cfg = load_config(config_path)
    set_seed(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))

    train_dir = cfg["data"]["train_dir"]
    val_dir = cfg["data"].get("val_dir")
    target_train_dir = cfg["data"].get("target_train_dir")
    target_val_dir = cfg["data"].get("target_val_dir")
    image_size = int(cfg["data"].get("image_size", 256))

    train_dataset = ImageFolderAutoEncoderDataset(train_dir, image_size=image_size, target_root=target_train_dir)
    val_dataset = None
    if val_dir:
        val_dataset = ImageFolderAutoEncoderDataset(val_dir, image_size=image_size, target_root=target_val_dir)

    batch_size = int(cfg["training"].get("batch_size", 32))
    num_workers = int(cfg["training"].get("num_workers", 2))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = CNNAutoEncoder(
        in_channels=int(cfg["model"].get("in_channels", 3)),
        base_channels=int(cfg["model"].get("base_channels", 64)),
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=float(cfg["training"].get("learning_rate", 1e-3)))
    epochs = int(cfg["training"].get("epochs", 50))

    output_dir = ensure_dir(cfg["paths"].get("checkpoint_dir", "checkpoints"))
    metrics_dir = ensure_dir(cfg["paths"].get("metrics_dir", "results/metrics"))
    figures_dir = ensure_dir(cfg["paths"].get("figures_dir", "results/figures"))

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for inputs, targets, _, _ in pbar:
            inputs = inputs.to(device=device, dtype=torch.float32)
            targets = targets.to(device=device, dtype=torch.float32)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            batch_size_now = inputs.size(0)
            train_loss_sum += loss.item() * batch_size_now
            train_count += batch_size_now
            pbar.set_postfix({"loss": f"{loss.item():.6f}"})

        train_loss = train_loss_sum / max(train_count, 1)

        val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss_sum = 0.0
            val_count = 0
            with torch.no_grad():
                for inputs, targets, _, _ in val_loader:
                    inputs = inputs.to(device=device, dtype=torch.float32)
                    targets = targets.to(device=device, dtype=torch.float32)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    batch_size_now = inputs.size(0)
                    val_loss_sum += loss.item() * batch_size_now
                    val_count += batch_size_now
            val_loss = val_loss_sum / max(val_count, 1)
            is_best = val_loss < best_val_loss
        else:
            is_best = train_loss < best_val_loss
            val_loss = None

        monitor_loss = val_loss if val_loss is not None else train_loss
        if is_best:
            best_val_loss = monitor_loss
            torch.save(
                {"epoch": epoch, "model_state_dict": model.state_dict(), "config": cfg},
                output_dir / "best_model.pth",
            )

        torch.save(
            {"epoch": epoch, "model_state_dict": model.state_dict(), "config": cfg},
            output_dir / "last_model.pth",
        )

        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss if val_loss is not None else ""}
        history.append(row)
        print(f"Epoch {epoch:03d}: train_loss={train_loss:.6f}" + (f", val_loss={val_loss:.6f}" if val_loss is not None else ""))

    history_path = metrics_dir / "training_history.csv"
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(history)

    plt.figure(figsize=(7, 5))
    epochs_list = [r["epoch"] for r in history]
    plt.plot(epochs_list, [r["train_loss"] for r in history], label="Training loss")
    if any(r["val_loss"] != "" for r in history):
        plt.plot(epochs_list, [float(r["val_loss"]) for r in history], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("CNN-AE training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "training_loss.png", dpi=300)
    plt.close()

    print(f"Training complete. Best checkpoint: {output_dir / 'best_model.pth'}")
    print(f"Training history saved to: {history_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CNN-AE for multispectral image enhancement.")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file.")
    args = parser.parse_args()
    train(args.config)
