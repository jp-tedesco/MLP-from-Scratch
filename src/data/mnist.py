"""Download, validate, and prepare the original MNIST IDX files.

Only the official training partition is split into training and validation.
The original test partition is never included in that split.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import logging
import os
from pathlib import Path
import struct
import tempfile
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
from numpy.typing import NDArray


LOGGER = logging.getLogger(__name__)
_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
_FILES = {
    "train-images-idx3-ubyte.gz": ("f68b3c2dcbeaaa9fbdd348bbdeb94873", 2051, 60_000),
    "train-labels-idx1-ubyte.gz": ("d53e105ee54ea40749a09fcbcd1e9432", 2049, 60_000),
    "t10k-images-idx3-ubyte.gz": ("9fb629c4189551a2d022fa330f9573f3", 2051, 10_000),
    "t10k-labels-idx1-ubyte.gz": ("ec29112dd5afa0611ce80d1b7f02629c", 2049, 10_000),
}


@dataclass(frozen=True)
class MNISTData:
    """Normalized image matrices and integer labels for three disjoint sets."""

    x_train: NDArray[np.float32]
    y_train: NDArray[np.int64]
    x_val: NDArray[np.float32]
    y_val: NDArray[np.int64]
    x_test: NDArray[np.float32]
    y_test: NDArray[np.int64]


def _digest(path: Path) -> str:
    """Return the published MNIST MD5 checksum (integrity, not authentication)."""
    checksum = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _ensure_file(directory: Path, name: str, download: bool) -> Path:
    """Find a verified file or download it atomically from a public mirror."""
    target = directory / name
    expected = _FILES[name][0]
    if target.is_file() and _digest(target) == expected:
        return target
    if not download:
        raise FileNotFoundError(
            f"MNIST file is missing or has an invalid checksum: {target}. "
            "Enable downloading or place the original gzip file in this directory."
        )
    directory.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []
    for mirror in _MIRRORS:
        for attempt in range(2):
            temporary: Path | None = None
            try:
                LOGGER.info("Downloading %s (attempt %d)", name, attempt + 1)
                request = Request(mirror + name, headers={"User-Agent": "NumPy-MNIST-MLP/1.0"})
                with urlopen(request, timeout=30) as response:
                    with tempfile.NamedTemporaryFile(
                        dir=directory, prefix=f".{name}.", suffix=".part", delete=False
                    ) as stream:
                        temporary = Path(stream.name)
                        while chunk := response.read(1024 * 1024):
                            stream.write(chunk)
                if _digest(temporary) != expected:
                    raise ValueError(f"Checksum mismatch for {name}")
                os.replace(temporary, target)
                return target
            except (OSError, URLError, ValueError) as error:
                errors.append(f"{mirror}: {error}")
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
    raise RuntimeError(f"Unable to obtain {name}. " + " | ".join(errors))


def _read_idx(path: Path, expected_magic: int, expected_count: int) -> NDArray[np.uint8]:
    """Decode a gzip IDX file and reject malformed headers or payloads."""
    with gzip.open(path, "rb") as stream:
        header = stream.read(8)
        if len(header) != 8:
            raise ValueError(f"Truncated IDX header: {path}")
        magic, count = struct.unpack(">II", header)
        if magic != expected_magic or count != expected_count:
            raise ValueError(f"Unexpected IDX magic/count ({magic}, {count}): {path}")
        if magic == 2051:
            dimensions = stream.read(8)
            if len(dimensions) != 8 or struct.unpack(">II", dimensions) != (28, 28):
                raise ValueError(f"MNIST images must be 28 by 28 pixels: {path}")
            shape = (count, 28, 28)
        else:
            shape = (count,)
        expected_bytes = int(np.prod(shape))
        payload = stream.read(expected_bytes + 1)
        if len(payload) != expected_bytes:
            raise ValueError(f"Unexpected IDX payload size: {path}")
    array = np.frombuffer(payload, dtype=np.uint8).reshape(shape)
    if magic == 2049 and np.any(array > 9):
        raise ValueError(f"MNIST labels must be between zero and nine: {path}")
    return array


def _load_pair(directory: Path, prefix: str, download: bool) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Load a training or test image/label pair and normalize pixels once."""
    image_name = f"{prefix}-images-idx3-ubyte.gz"
    label_name = f"{prefix}-labels-idx1-ubyte.gz"
    images = _read_idx(_ensure_file(directory, image_name, download), *_FILES[image_name][1:])
    labels = _read_idx(_ensure_file(directory, label_name, download), *_FILES[label_name][1:])
    x = images.reshape(len(images), 784).astype(np.float32)
    x /= np.float32(255.0)
    return x, labels.astype(np.int64)


def _split_indices(y: NDArray[np.int64], validation_size: int, seed: int) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Allocate an exact validation size with proportional class counts."""
    if isinstance(validation_size, bool) or not isinstance(validation_size, (int, np.integer)):
        raise ValueError("validation_size must be an integer")
    if not 1 <= validation_size < len(y):
        raise ValueError(f"validation_size must be between 1 and {len(y) - 1}")
    rng = np.random.default_rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    quotas = counts.astype(np.float64) * validation_size / len(y)
    allocations = np.floor(quotas).astype(np.int64)
    remainder = validation_size - int(allocations.sum())
    order = np.argsort(-(quotas - allocations), kind="stable")
    allocations[order[:remainder]] += 1
    train_parts: list[NDArray[np.int64]] = []
    val_parts: list[NDArray[np.int64]] = []
    for label, number in zip(classes, allocations):
        indices = np.flatnonzero(y == label)
        rng.shuffle(indices)
        val_parts.append(indices[:number])
        train_parts.append(indices[number:])
    train_indices = np.concatenate(train_parts)
    val_indices = np.concatenate(val_parts)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)
    return train_indices, val_indices


def load_mnist(
    data_dir: str | Path = "data",
    validation_size: int = 10_000,
    seed: int = 42,
    download: bool = True,
) -> MNISTData:
    """Load MNIST with a reproducible stratified train/validation split.

    Defaults produce 50,000 training, 10,000 validation and 10,000 test
    examples. Images are float32 matrices shaped ``(N, 784)`` in [0, 1];
    labels are int64 vectors shaped ``(N,)``. Downloaded archives remain
    cached on disk and are checksum-verified on subsequent calls.
    """
    directory = Path(data_dir).expanduser()
    if isinstance(validation_size, bool) or not isinstance(validation_size, (int, np.integer)) or not 1 <= validation_size < 60_000:
        raise ValueError("validation_size must be an integer between 1 and 59999")
    x, y = _load_pair(directory, "train", download)
    train_indices, val_indices = _split_indices(y, validation_size, seed)
    x_test, y_test = load_test_data(directory, download=download)
    return MNISTData(x[train_indices], y[train_indices], x[val_indices], y[val_indices], x_test, y_test)


def load_test_data(
    data_dir: str | Path = "data", download: bool = True
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Load only the untouched official test set for final evaluation."""
    return _load_pair(Path(data_dir).expanduser(), "t10k", download)
