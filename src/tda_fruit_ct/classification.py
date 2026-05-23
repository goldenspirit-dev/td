"""Downstream ML on TDA feature vectors (fruit-type classification, ripeness regression)."""
from __future__ import annotations

import numpy as np


def train_classifier(X: np.ndarray, y: np.ndarray):
    raise NotImplementedError


def evaluate(model, X: np.ndarray, y: np.ndarray) -> dict:
    raise NotImplementedError
