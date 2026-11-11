"""MNIST acquisition and preprocessing."""

from .mnist import MNISTData, load_mnist, load_test_data

__all__ = ["MNISTData", "load_mnist", "load_test_data"]
