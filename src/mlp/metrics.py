"""Classification metrics without external machine learning libraries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .losses import validate_targets


def accuracy(scores: NDArray[np.floating], targets: NDArray) -> float:
    """Return the fraction of correct predictions from logits or probabilities."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or min(scores.shape) == 0 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty finite two-dimensional array.")
    targets = validate_targets(targets, scores.shape[0], scores.shape[1])
    expected = targets if targets.ndim == 1 else targets.argmax(axis=1)
    return float(np.mean(scores.argmax(axis=1) == expected))
