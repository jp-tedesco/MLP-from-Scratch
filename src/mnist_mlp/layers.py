"""Trainable parameters and fully connected layers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
from numpy.typing import NDArray


@dataclass
class Parameter:
    """A mutable parameter array and its corresponding gradient buffer."""

    value: NDArray[np.floating]
    grad: NDArray[np.floating]


class Dense:
    """Affine transformation with He or Xavier normal initialization."""

    def __init__(
        self,
        n_in: int,
        n_out: int,
        rng: np.random.Generator | None = None,
        initialization: str = "he",
        dtype: np.typing.DTypeLike = np.float32,
    ) -> None:
        if not isinstance(n_in, (int, np.integer)) or not isinstance(n_out, (int, np.integer)):
            raise ValueError("Layer dimensions must be integers.")
        if n_in <= 0 or n_out <= 0:
            raise ValueError("Layer dimensions must be positive.")
        dtype = np.dtype(dtype)
        if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise ValueError("Use float32 or float64 parameters.")
        if initialization not in ("he", "xavier"):
            raise ValueError("Initialization must be 'he' or 'xavier'.")
        rng = rng if rng is not None else np.random.default_rng()
        scale = np.sqrt(2.0 / n_in if initialization == "he" else 2.0 / (n_in + n_out))
        weights = (rng.standard_normal((n_in, n_out)) * scale).astype(dtype)
        bias = np.zeros((1, n_out), dtype=dtype)
        self.weight = Parameter(weights, np.zeros_like(weights))
        self.bias = Parameter(bias, np.zeros_like(bias))
        self._inputs: NDArray[np.floating] | None = None

    def forward(self, inputs: NDArray[np.floating], training: bool = False) -> NDArray[np.floating]:
        inputs = np.asarray(inputs, dtype=self.weight.value.dtype)
        if inputs.ndim != 2 or inputs.shape[1] != self.weight.value.shape[0]:
            raise ValueError("Dense input shape must be (batch_size, n_in).")
        if not np.isfinite(inputs).all():
            raise ValueError("Dense inputs must be finite.")
        if training:
            self._inputs = inputs.copy()
        return inputs @ self.weight.value + self.bias.value

    def backward(self, gradient: NDArray[np.floating]) -> NDArray[np.floating]:
        if self._inputs is None:
            raise RuntimeError("Run a training forward pass before Dense.backward().")
        expected = (self._inputs.shape[0], self.weight.value.shape[1])
        if gradient.shape != expected or not np.isfinite(gradient).all():
            raise ValueError("Dense gradient must be finite and match the output shape.")
        self.weight.grad[...] = self._inputs.T @ gradient
        self.bias.grad[...] = np.sum(gradient, axis=0, keepdims=True)
        return gradient @ self.weight.value.T

    def parameters(self) -> Iterator[Parameter]:
        yield self.weight
        yield self.bias
