"""Evaluate a frozen checkpoint on the official MNIST test split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.mnist import load_test_data
from src.mlp.network import MLP
from src.mlp.trainer import evaluate


def main() -> None:
    """Report test loss, accuracy, and a confusion matrix with true-class rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    if args.output is not None and args.output.exists():
        raise FileExistsError("Evaluation output already exists; choose a new path.")
    model = MLP.load(args.model)
    x, y = load_test_data(args.data_dir, download=not args.no_download)
    metrics = evaluate(model, x, y, args.batch_size)
    confusion = np.zeros((10, 10), dtype=np.int64)
    for start in range(0, len(x), args.batch_size):
        predictions = model.forward(x[start:start + args.batch_size]).argmax(axis=1)
        np.add.at(confusion, (y[start:start + args.batch_size], predictions), 1)
    report = {"checkpoint": str(args.model), "test_samples": len(x),
              "test_loss": metrics["loss"], "test_accuracy": metrics["accuracy"],
              "confusion_matrix": confusion.tolist(), "matrix_axes": "rows=true, columns=predicted"}
    content = json.dumps(report, indent=2)
    print(content)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    main()
