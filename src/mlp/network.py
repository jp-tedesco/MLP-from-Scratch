"""A fully connected classifier with explicit forward and backward passes."""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
from typing import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from .activations import ReLU, softmax
from .layers import Dense, Parameter


class MLP:
    """Dense/ReLU hidden layers followed by an unnormalized output layer."""

    def __init__(
        self,
        layer_sizes: Sequence[int] = (784, 128, 64, 10),
        seed: int = 42,
        dtype: np.typing.DTypeLike = np.float32,
    ) -> None:
        if len(layer_sizes) < 2 or any(
            not isinstance(size, (int, np.integer)) or isinstance(size, bool) or size <= 0
            for size in layer_sizes
        ):
            raise ValueError("Layer sizes must contain at least two positive integers.")
        self.layer_sizes = tuple(int(size) for size in layer_sizes)
        self.dtype = np.dtype(dtype)
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.layers: list[Dense | ReLU] = []
        for index, (n_in, n_out) in enumerate(zip(self.layer_sizes[:-1], self.layer_sizes[1:])):
            hidden = index < len(self.layer_sizes) - 2
            self.layers.append(Dense(n_in, n_out, rng, "he" if hidden else "xavier", dtype))
            if hidden:
                self.layers.append(ReLU())
        self._training_ready = False

    def forward(self, inputs: NDArray, training: bool = False) -> NDArray[np.floating]:
        """Return logits; inference preserves caches from an unfinished training pass."""
        output = np.asarray(inputs, dtype=self.dtype)
        if output.ndim != 2 or output.shape[0] == 0 or output.shape[1] != self.layer_sizes[0]:
            raise ValueError("Input must have shape (nonempty_batch, input_features).")
        for layer in self.layers:
            output = layer.forward(output, training=training)
        if not np.isfinite(output).all():
            raise FloatingPointError("The network produced nonfinite logits.")
        if training:
            self._training_ready = True
        return output

    def backward(self, grad_logits: NDArray[np.floating]) -> NDArray[np.floating]:
        """Overwrite all parameter gradients and return the input gradient."""
        if not self._training_ready:
            raise RuntimeError("Run forward(training=True) before each backward call.")
        gradient = grad_logits
        for layer in reversed(self.layers):
            gradient = layer.backward(gradient)
        self._training_ready = False
        return gradient

    def parameters(self) -> Iterator[Parameter]:
        for layer in self.layers:
            if isinstance(layer, Dense):
                yield from layer.parameters()

    def predict_proba(self, inputs: NDArray) -> NDArray[np.floating]:
        return softmax(self.forward(inputs))

    def predict(self, inputs: NDArray) -> NDArray[np.intp]:
        return self.forward(inputs).argmax(axis=1)

    def save(self, path: str | Path) -> None:
        """Atomically save inference parameters as a pickle-free NumPy archive."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"parameter_{i}": p.value for i, p in enumerate(self.parameters())}
        arrays["layer_sizes"] = np.asarray(self.layer_sizes, dtype=np.int64)
        arrays["format_version"] = np.asarray(1, dtype=np.int64)
        handle, temporary = tempfile.mkstemp(prefix=destination.name, suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(handle, "wb") as stream:
                np.savez_compressed(stream, **arrays)
            os.replace(temporary, destination)
        finally:
            Path(temporary).unlink(missing_ok=True)

    @classmethod
    def load(cls, path: str | Path) -> MLP:
        """Load and validate an archive; serialized Python objects are forbidden."""
        with np.load(path, allow_pickle=False) as archive:
            if "format_version" not in archive or archive["format_version"].shape != ():
                raise ValueError("Invalid checkpoint format version.")
            if int(archive["format_version"]) != 1:
                raise ValueError("Unsupported checkpoint format version.")
            sizes = archive["layer_sizes"]
            if sizes.ndim != 1 or not np.issubdtype(sizes.dtype, np.integer):
                raise ValueError("Invalid checkpoint layer sizes.")
            first = archive["parameter_0"]
            model = cls(sizes.tolist(), dtype=first.dtype)
            parameters = tuple(model.parameters())
            expected = {"layer_sizes", "format_version"} | {f"parameter_{i}" for i in range(len(parameters))}
            if set(archive.files) != expected:
                raise ValueError("Checkpoint parameter keys do not match its architecture.")
            for i, parameter in enumerate(parameters):
                saved = archive[f"parameter_{i}"]
                if saved.shape != parameter.value.shape or saved.dtype != model.dtype or not np.isfinite(saved).all():
                    raise ValueError(f"Invalid checkpoint parameter {i}.")
                parameter.value[...] = saved
        return model
