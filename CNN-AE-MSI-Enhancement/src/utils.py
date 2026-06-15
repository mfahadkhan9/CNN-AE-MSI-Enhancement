"""Common utility functions."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from PIL import Image


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(config_device: str = "auto") -> torch.device:
    if config_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config_device)


def tensor_to_numpy_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW tensor in [0, 1] to HWC numpy image in [0, 1]."""
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor.squeeze(0)
    return tensor.permute(1, 2, 0).numpy()


def save_tensor_image(tensor: torch.Tensor, path: str | Path) -> None:
    arr = tensor_to_numpy_image(tensor)
    img = Image.fromarray((arr * 255).round().astype(np.uint8))
    ensure_dir(Path(path).parent)
    img.save(path)


def save_json(data: Dict[str, Any], path: str | Path) -> None:
    ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
