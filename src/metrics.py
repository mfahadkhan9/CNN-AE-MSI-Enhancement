"""Image-quality and registration-oriented evaluation metrics."""

from __future__ import annotations

import math
from typing import Dict, Optional

import numpy as np


def _as_float_image(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def mse(reference: np.ndarray, enhanced: np.ndarray) -> float:
    reference = _as_float_image(reference)
    enhanced = _as_float_image(enhanced)
    return float(np.mean((reference - enhanced) ** 2))


def rmse(reference: np.ndarray, enhanced: np.ndarray) -> float:
    return float(math.sqrt(mse(reference, enhanced)))


def psnr(reference: np.ndarray, enhanced: np.ndarray, max_value: float = 1.0) -> float:
    error = mse(reference, enhanced)
    if error == 0:
        return float("inf")
    return float(10.0 * math.log10((max_value**2) / error))


def correlation_coefficient(reference: np.ndarray, enhanced: np.ndarray) -> float:
    reference = _as_float_image(reference).ravel()
    enhanced = _as_float_image(enhanced).ravel()
    ref_std = reference.std()
    enh_std = enhanced.std()
    if ref_std == 0 or enh_std == 0:
        return float("nan")
    return float(np.corrcoef(reference, enhanced)[0, 1])


def mutual_information(reference: np.ndarray, enhanced: np.ndarray, bins: int = 256) -> float:
    """Estimate mutual information from a 2D joint histogram.

    This metric is useful for registration-oriented comparison. It is sensitive
    to intensity distribution, so it should be interpreted alongside CC, RMSE,
    PSNR, and runtime.
    """
    reference = _as_float_image(reference).ravel()
    enhanced = _as_float_image(enhanced).ravel()
    hist_2d, _, _ = np.histogram2d(reference, enhanced, bins=bins, range=[[0, 1], [0, 1]])
    pxy = hist_2d / np.sum(hist_2d)
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    px_py = px[:, None] * py[None, :]
    nz = pxy > 0
    return float(np.sum(pxy[nz] * np.log(pxy[nz] / px_py[nz])))


def compute_metrics(
    reference: np.ndarray,
    enhanced: np.ndarray,
    elapsed_seconds: Optional[float] = None,
    bins: int = 256,
) -> Dict[str, float]:
    values = {
        "CC": correlation_coefficient(reference, enhanced),
        "MI": mutual_information(reference, enhanced, bins=bins),
        "RMSE": rmse(reference, enhanced),
        "PSNR": psnr(reference, enhanced),
    }
    if elapsed_seconds is not None:
        values["RT_seconds"] = float(elapsed_seconds)
    return values
