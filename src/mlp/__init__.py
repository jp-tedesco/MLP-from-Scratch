"""Public interfaces for the NumPy multilayer perceptron."""

from .losses import CrossEntropyLoss
from .network import MLP
from .optimizers import SGD

__all__ = ["CrossEntropyLoss", "MLP", "SGD"]
