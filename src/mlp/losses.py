"""Stable categorical cross-entropy computed directly from logits."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def validate_targets(targets: NDArray, batch_size: int, classes: int) -> NDArray:
    """Validate integer labels or probability-distribution target rows."""
    targets = np.asarray(targets)
    if targets.shape == (batch_size,):
        if not np.issubdtype(targets.dtype, np.integer):
            raise ValueError("Class labels must be integers.")
        if np.any(targets < 0) or np.any(targets >= classes):
            raise ValueError("Class labels are outside the output class range.")
    elif targets.shape == (batch_size, classes):
        if not np.isfinite(targets).all() or np.any(targets < 0):
            raise ValueError("Target distributions must be finite and nonnegative.")
        if not np.allclose(targets.sum(axis=1), 1.0, rtol=1e-5, atol=1e-6):
            raise ValueError("Target distribution rows must sum to one.")
    else:
        raise ValueError("Targets must have shape (batch,) or (batch, classes).")
    return targets


class CrossEntropyLoss:
    """Mean softmax cross-entropy; backward returns the gradient of logits."""

    def __init__(self) -> None:
        self._gradient: NDArray[np.floating] | None = None

    def forward(self, logits: NDArray[np.floating], targets: NDArray) -> float:
        logits = np.asarray(logits)
        if logits.ndim != 2 or min(logits.shape) == 0:
            raise ValueError("Logits must be a nonempty two-dimensional array.")
        if not np.issubdtype(logits.dtype, np.floating) or not np.isfinite(logits).all():
            raise ValueError("Logits must be finite floating-point values.")
        n, classes = logits.shape
        targets = validate_targets(targets, n, classes)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exponentials = np.exp(shifted)
        normalizer = exponentials.sum(axis=1, keepdims=True)
        log_probabilities = shifted - np.log(normalizer)
        gradient = exponentials / normalizer
        if targets.ndim == 1:
            loss = -np.mean(log_probabilities[np.arange(n), targets], dtype=np.float64)
            gradient[np.arange(n), targets] -= 1
        else:
            loss = -np.sum(targets * log_probabilities, dtype=np.float64) / n
            gradient -= targets
        self._gradient = gradient / n
        return float(loss)

    def backward(self) -> NDArray[np.floating]:
        if self._gradient is None:
            raise RuntimeError("Run CrossEntropyLoss.forward() before backward().")
        return self._gradient.copy()
