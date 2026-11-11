"""Offline checks for IDX decoding and reproducible disjoint data partitions."""

import gzip
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from src.data.mnist import _read_idx, _split_indices


class IDXTests(unittest.TestCase):
    def test_image_payload_round_trip(self) -> None:
        pixels = np.arange(2 * 28 * 28, dtype=np.uint16).astype(np.uint8).reshape(2, 28, 28)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "images.gz"
            with gzip.open(path, "wb") as stream:
                stream.write(struct.pack(">IIII", 2051, 2, 28, 28))
                stream.write(pixels.tobytes())
            np.testing.assert_array_equal(_read_idx(path, 2051, 2), pixels)

    def test_truncated_payload_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.gz"
            with gzip.open(path, "wb") as stream:
                stream.write(struct.pack(">II", 2049, 3) + bytes([1, 2]))
            with self.assertRaises(ValueError):
                _read_idx(path, 2049, 3)

    def test_out_of_range_label_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "labels.gz"
            with gzip.open(path, "wb") as stream:
                stream.write(struct.pack(">II", 2049, 2) + bytes([0, 10]))
            with self.assertRaises(ValueError):
                _read_idx(path, 2049, 2)


class SplitTests(unittest.TestCase):
    def test_stratified_split_is_exact_disjoint_and_reproducible(self) -> None:
        labels = np.repeat(np.arange(10), 20)
        train, validation = _split_indices(labels, validation_size=53, seed=42)
        second_train, second_validation = _split_indices(labels, validation_size=53, seed=42)
        self.assertEqual(len(validation), 53)
        self.assertEqual(len(train), 147)
        self.assertEqual(len(np.intersect1d(train, validation)), 0)
        np.testing.assert_array_equal(np.sort(np.concatenate((train, validation))), np.arange(200))
        np.testing.assert_array_equal(train, second_train)
        np.testing.assert_array_equal(validation, second_validation)
        counts = np.bincount(labels[validation], minlength=10)
        self.assertTrue(np.all((counts == 5) | (counts == 6)))

    def test_empty_validation_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _split_indices(np.array([0, 1]), 0, 42)


if __name__ == "__main__":
    unittest.main()
