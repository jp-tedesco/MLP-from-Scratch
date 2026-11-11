"""Train an MNIST MLP and export reproducible run artifacts."""
from __future__ import annotations

import argparse
import csv
import json
import logging
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from src.data.mnist import load_mnist
from src.mlp.network import MLP
from src.mlp.optimizers import SGD
from src.mlp.trainer import Trainer


def parse_args() -> argparse.Namespace:
    """Parse training settings; paths are relative to the current directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "baseline")
    parser.add_argument("--hidden-sizes", type=int, nargs="*", default=[128, 64])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--validation-size", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Optional reproducible training subset for smoke runs.")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def plot_history(history: list[dict], destination: Path) -> None:
    """Render learning curves without requiring a graphical desktop."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    epochs = [row["epoch"] for row in history]
    for axis, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            axis.plot(epochs, [row[f"{split}_{metric}"] for row in history], label=split)
        axis.set(xlabel="Epoch", ylabel=metric.title(), title=f"MNIST {metric}")
        axis.legend()
        axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(destination, dpi=160)
    plt.close(fig)


def main() -> None:
    """Create a fresh run, train, and save metrics without accessing test metrics."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.epochs <= 0 or args.batch_size <= 0 or any(n <= 0 for n in args.hidden_sizes):
        raise ValueError("Epochs, batch size, and hidden sizes must be positive.")
    if args.train_limit is not None and args.train_limit <= 0:
        raise ValueError("train-limit must be positive.")
    model = MLP([784, *args.hidden_sizes, 10], seed=args.seed)
    optimizer = SGD(args.learning_rate)
    # Refuse accidental overwrites of previous experiment artifacts.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise FileExistsError("Output directory must be empty. Choose a new --output-dir.")
    data = load_mnist(args.data_dir, args.validation_size, args.seed, not args.no_download)
    x_train, y_train = data.x_train, data.y_train
    if args.train_limit is not None:
        count = min(args.train_limit, len(x_train))
        indices = np.random.default_rng(args.seed).permutation(len(x_train))[:count]
        x_train, y_train = x_train[indices], y_train[indices]
    trainer = Trainer(model, optimizer, args.batch_size, args.seed)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update({"python_version": platform.python_version(), "numpy_version": np.__version__,
                   "train_samples": len(x_train), "validation_samples": len(data.x_val),
                   "architecture": [784, *args.hidden_sizes, 10],
                   "selection_metric": "minimum validation loss"})
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    history = trainer.fit(x_train, y_train, data.x_val, data.y_val,
                          args.epochs, args.output_dir / "best_model.npz")
    model.save(args.output_dir / "last_model.npz")
    with (args.output_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    summary = {"best_epoch": trainer.best_epoch, "best_validation": history[trainer.best_epoch - 1]}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.no_plot:
        plot_history(history, args.output_dir / "learning_curves.png")
    logging.info("Saved run to %s (best epoch: %d)", args.output_dir, trainer.best_epoch)


if __name__ == "__main__":
    main()
