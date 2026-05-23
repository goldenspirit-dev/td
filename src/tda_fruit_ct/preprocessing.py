"""Denoising, resampling, intensity normalization, and segmentation for CT volumes."""
from __future__ import annotations

import numpy as np


def normalize_intensity(volume: np.ndarray) -> np.ndarray:
    raise NotImplementedError


def resample(volume: np.ndarray, target_spacing: tuple[float, float, float]) -> np.ndarray:
    raise NotImplementedError


def segment_fruit(volume: np.ndarray) -> np.ndarray:
    """Return a boolean mask separating fruit tissue from background/air."""
    raise NotImplementedError
