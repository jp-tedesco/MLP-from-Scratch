"""Vectorized activation functions with explicit backward propagation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def softmax(logits: NDArray[np.floating]) -> NDArray[np.floating]:
    """Convert finite, two-dimensional logits into row-wise probabilities."""
    logits = np.asarray(logits)
    if logits.ndim != 2 or min(logits.shape) == 0:
        raise ValueError("Logits must be a nonempty two-dimensional array.")
    if not np.issubdtype(logits.dtype, np.floating) or not np.isfinite(logits).all():
        raise ValueError("Logits must contain finite floating-point values.")
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    scores = np.exp(shifted)
    return scores / np.sum(scores, axis=1, keepdims=True)


class ReLU:
    """Rectified linear unit; the derivative at zero is defined as zero."""

    def __init__(self) -> None:
        self._mask: NDArray[np.bool_] | None = None

    def forward(self, inputs: NDArray[np.floating], training: bool = False) -> NDArray[np.floating]:
        if training:
            self._mask = inputs > 0
        return np.maximum(inputs, 0)

    def backward(self, gradient: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._mask is None:
            raise RuntimeError("Run a training forward pass before ReLU.backward().")
        if gradient.shape != self._mask.shape:
            raise ValueError("ReLU gradient shape does not match its cached input.")
        return gradient * self._mask


class Softmax:
    """Softmax activation, including a general Jacobian-vector product."""

    def __init__(self) -> None:
        self._probabilities: NDArray[np.floating] | None = None

    def forward(self, inputs: NDArray[np.floating], training: bool = False) -> NDArray[np.floating]:
        probabilities = softmax(inputs)
        if training:
            self._probabilities = probabilities
        return probabilities

    def backward(self, gradient: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._probabilities is None:
            raise RuntimeError("Run a training forward pass before Softmax.backward().")
        if gradient.shape != self._probabilities.shape:
            raise ValueError("Softmax gradient shape does not match its cached output.")
        p = self._probabilities
        return p * (gradient - np.sum(gradient * p, axis=1, keepdims=True))
