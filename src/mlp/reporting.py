"""Reproducible diagnostic figures from real predictions, not synthetic images."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .metrics import binary_roc_curve


def save_diagnostic_figures(x: np.ndarray, y: np.ndarray, probabilities: np.ndarray,
                            directory: str | Path, history: str | Path | None = None) -> None:
    """Export ROC, confusion, digit examples, and the pixel-to-prediction path.

    Examples use the first occurrence of each class and the first ten errors
    in dataset order, so the selection does not hide difficult predictions.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    predicted = probabilities.argmax(axis=1)
    classes = probabilities.shape[1]

    fig, axis = plt.subplots(figsize=(7, 6))
    for digit in range(classes):
        binary = y == digit
        if 0 < binary.sum() < len(y):
            fpr, tpr, auc = binary_roc_curve(binary, probabilities[:, digit])
            axis.plot(fpr, tpr, label=f"Digit {digit}: {auc:.5f}", linewidth=1.3)
    axis.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Chance ranking")
    axis.set(xlabel="False positive rate", ylabel="True positive rate",
             title="MNIST test ROC: each digit versus all others", xlim=(0, 1), ylim=(0, 1.01))
    axis.legend(loc="lower right", fontsize=8, title="One-vs-rest AUC")
    axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(directory / "roc.png", dpi=160)
    plt.close(fig)

    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    fig, axis = plt.subplots(figsize=(8, 7))
    heatmap = axis.imshow(confusion, cmap="Blues")
    for row in range(classes):
        for column in range(classes):
            axis.text(column, row, str(confusion[row, column]), ha="center", va="center",
                      fontsize=8, color="white" if confusion[row, column] > confusion.max() / 2 else "black")
    axis.set(xticks=range(classes), yticks=range(classes), xlabel="Predicted digit",
             ylabel="True digit", title="MNIST test confusion matrix (counts)")
    fig.colorbar(heatmap, ax=axis, shrink=0.8)
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    representatives = [int(np.flatnonzero(y == digit)[0]) for digit in range(classes) if np.any(y == digit)]
    errors = np.flatnonzero(predicted != y)[:10].tolist()
    fig, axes = plt.subplots(2, 10, figsize=(15, 4.4), squeeze=False)
    for row, indices in enumerate((representatives, errors)):
        for column, axis in enumerate(axes[row]):
            axis.axis("off")
            if column < len(indices):
                index = indices[column]
                axis.imshow(x[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
                axis.set_title(f"True {y[index]} / Pred {predicted[index]}\n"
                               f"Score {probabilities[index, predicted[index]]:.1%}", fontsize=9,
                               color="darkgreen" if predicted[index] == y[index] else "firebrick")
    fig.suptitle("Top: first example of each digit | Bottom: first ten errors in test order", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(directory / "digits.png", dpi=160)
    plt.close(fig)

    index = 0
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 1.6, 1.5]})
    axes[0].imshow(x[index].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title(f"1. Image: true digit {y[index]}")
    axes[0].set(xlabel="28 columns", ylabel="28 rows")
    axes[1].imshow(x[index][None, :], cmap="gray", vmin=0, vmax=1, aspect="auto",
                   extent=(-0.5, 783.5, -0.5, 0.5))
    axes[1].set(title="2. The same pixels, read row by row", xlabel="Position in the 784-value input", yticks=[])
    colors = ["#238b45" if digit == predicted[index] else "#8cb6d9" for digit in range(classes)]
    axes[2].bar(range(classes), probabilities[index], color=colors)
    axes[2].set(title=f"3. Network output: digit {predicted[index]}", xlabel="Digit", ylabel="Softmax score",
                xticks=range(classes), ylim=(0, 1.05))
    fig.tight_layout()
    fig.savefig(directory / "pixel_to_prediction.png", dpi=160)
    plt.close(fig)

    if history is not None:
        with Path(history).open(newline="", encoding="utf-8") as stream:
            rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(stream)]
        if not rows:
            raise ValueError("History must contain at least one epoch.")
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        for axis, metric in zip(axes, ("loss", "accuracy")):
            for split in ("train", "val"):
                axis.plot([row["epoch"] for row in rows],
                          [row[f"{split}_{metric}"] for row in rows], label=split)
            axis.set(xlabel="Epoch", ylabel=metric.title(), title=f"MNIST {metric}")
            axis.legend()
            axis.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(directory / "learning_curves.png", dpi=160)
        plt.close(fig)
