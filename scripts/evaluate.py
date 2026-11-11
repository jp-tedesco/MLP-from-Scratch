"""Evaluate a frozen checkpoint on the official MNIST test split."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.mnist import load_test_data
from src.mlp.network import MLP
from src.mlp.metrics import classification_report
from src.mlp.reporting import save_diagnostic_figures
from src.mlp.trainer import evaluate


def main() -> None:
    """Report test loss, accuracy, and a confusion matrix with true-class rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--figures-dir", type=Path, help="Export diagnostic figures to this directory.")
    parser.add_argument("--history", type=Path, help="Optional training history CSV to plot.")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.history is not None and args.figures_dir is None:
        parser.error("--history requires --figures-dir.")
    if args.history is not None and not args.history.is_file():
        parser.error("--history must name an existing CSV file.")
    if args.output is not None and args.output.exists():
        raise FileExistsError("Evaluation output already exists; choose a new path.")
    model = MLP.load(args.model)
    x, y = load_test_data(args.data_dir, download=not args.no_download)
    metrics = evaluate(model, x, y, args.batch_size)
    probabilities = []
    for start in range(0, len(x), args.batch_size):
        probabilities.append(model.predict_proba(x[start:start + args.batch_size]))
    probabilities = np.concatenate(probabilities)
    detailed = classification_report(probabilities, y)
    report = {"checkpoint": str(args.model), "test_samples": len(x),
              "test_loss": metrics["loss"], "test_accuracy": metrics["accuracy"],
              "checkpoint_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "parameter_count": sum(p.value.size for p in model.parameters()),
              "architecture": list(model.layer_sizes), **detailed}
    content = json.dumps(report, indent=2, allow_nan=False)
    print(content)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(content, encoding="utf-8")
    if args.figures_dir is not None:
        save_diagnostic_figures(x, y, probabilities, args.figures_dir, args.history)


if __name__ == "__main__":
    main()
