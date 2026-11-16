"""Optimizers that update explicit parameter and gradient arrays."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .layers import Parameter


class SGD:
    """Mini-batch stochastic gradient descent with a constant learning rate."""

    def __init__(self, learning_rate: float = 0.1) -> None:
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("Learning rate must be finite and positive.")
        self.learning_rate = float(learning_rate)

    def step(self, parameters: Iterable[Parameter]) -> None:
        """Validate every update before modifying any parameter."""
        parameters = tuple(parameters)
        updates = []
        for parameter in parameters:
            if parameter.grad.shape != parameter.value.shape:
                raise ValueError("Parameter and gradient shapes must match.")
            if not np.isfinite(parameter.grad).all():
                raise FloatingPointError("Nonfinite gradient; reduce the learning rate or debug backward.")
            updated = parameter.value - self.learning_rate * parameter.grad
            if not np.isfinite(updated).all():
                raise FloatingPointError("The parameter update produced nonfinite values.")
            updates.append(updated)
        for parameter, updated in zip(parameters, updates):
            parameter.value[...] = updated
