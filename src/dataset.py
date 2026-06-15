"""Dataset utilities for multispectral transmission image folders."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


class ImageFolderAutoEncoderDataset(Dataset):
    """Image-folder dataset for CNN-AE training or evaluation.

    The default training mode is self-reconstruction: the input image is also
    used as the target. This matches low-SNR experimental data when clean paired
    targets are unavailable. If `target_root` is provided, the dataset attempts
    to match each input image by relative path.
    """

    def __init__(
        self,
        input_root: str | Path,
        image_size: int = 256,
        target_root: Optional[str | Path] = None,
        recursive: bool = True,
    ) -> None:
        self.input_root = Path(input_root)
        self.target_root = Path(target_root) if target_root else None
        self.image_size = image_size

        if not self.input_root.exists():
            raise FileNotFoundError(f"Input directory does not exist: {self.input_root}")

        pattern = "**/*" if recursive else "*"
        self.image_paths = sorted(
            p for p in self.input_root.glob(pattern) if p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not self.image_paths:
            raise RuntimeError(f"No image files found in {self.input_root}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),  # scales to [0, 1]
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_rgb(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return self.transform(image)

    def __getitem__(self, index: int):
        input_path = self.image_paths[index]
        x = self._load_rgb(input_path)

        if self.target_root is not None:
            rel = input_path.relative_to(self.input_root)
            target_path = self.target_root / rel
            if not target_path.exists():
                raise FileNotFoundError(f"Target image not found for {input_path}: {target_path}")
            y = self._load_rgb(target_path)
        else:
            y = x.clone()

        wavelength = input_path.parent.name
        return x, y, str(input_path), wavelength
