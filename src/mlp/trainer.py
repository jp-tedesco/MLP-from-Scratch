"""Mini-batch training and bounded-memory evaluation."""
from __future__ import annotations

import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from .losses import CrossEntropyLoss
from .metrics import accuracy
from .network import MLP
from .optimizers import SGD

LOGGER = logging.getLogger(__name__)


def evaluate(model: MLP, x: np.ndarray, y: np.ndarray,
             batch_size: int = 1024) -> dict[str, float]:
    """Return sample-weighted loss and accuracy without changing model caches."""
    if batch_size <= 0 or len(x) == 0 or len(x) != len(y):
        raise ValueError("Evaluation requires nonempty aligned data and a positive batch size.")
    criterion = CrossEntropyLoss()
    loss_sum = accuracy_sum = 0.0
    for start in range(0, len(x), batch_size):
        xb, yb = x[start:start + batch_size], y[start:start + batch_size]
        logits = model.forward(xb, training=False)
        loss_sum += criterion.forward(logits, yb) * len(xb)
        accuracy_sum += accuracy(logits, yb) * len(xb)
    return {"loss": loss_sum / len(x), "accuracy": accuracy_sum / len(x)}


class Trainer:
    """Coordinate forward/backward calls; save the lowest-validation-loss model.

    The in-memory model contains the final epoch parameters. Load the saved
    checkpoint explicitly to use the best validation epoch.
    """

    def __init__(self, model: MLP, optimizer: SGD, batch_size: int = 64,
                 seed: int = 42) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.history: list[dict[str, float | int]] = []
        self.best_epoch = 0

    def fit(self, x_train: np.ndarray, y_train: np.ndarray,
            x_val: np.ndarray, y_val: np.ndarray, epochs: int = 20,
            checkpoint: str | Path | None = None) -> list[dict[str, float | int]]:
        """Train with shuffled mini-batches, including the final partial batch."""
        if epochs <= 0 or len(x_train) == 0 or len(x_train) != len(y_train):
            raise ValueError("Training requires positive epochs and nonempty aligned data.")
        evaluate(self.model, x_val, y_val)
        criterion = CrossEntropyLoss()
        best_loss = float("inf")
        self.history = []
        self.best_epoch = 0
        for epoch in range(1, epochs + 1):
            started = perf_counter()
            order = self.rng.permutation(len(x_train))
            for start in range(0, len(order), self.batch_size):
                indices = order[start:start + self.batch_size]
                logits = self.model.forward(x_train[indices], training=True)
                criterion.forward(logits, y_train[indices])
                self.model.backward(criterion.backward())
                self.optimizer.step(self.model.parameters())
            train = evaluate(self.model, x_train, y_train)
            val = evaluate(self.model, x_val, y_val)
            row = {"epoch": epoch, "train_loss": train["loss"],
                   "train_accuracy": train["accuracy"], "val_loss": val["loss"],
                   "val_accuracy": val["accuracy"],
                   "seconds": perf_counter() - started}
            self.history.append(row)
            if val["loss"] < best_loss:
                best_loss = val["loss"]
                self.best_epoch = epoch
                if checkpoint is not None:
                    self.model.save(checkpoint)
            LOGGER.info("Epoch %d/%d | train loss %.4f accuracy %.2f%% | "
                        "val loss %.4f accuracy %.2f%% | %.1fs", epoch, epochs,
                        train["loss"], 100 * train["accuracy"], val["loss"],
                        100 * val["accuracy"], row["seconds"])
        return self.history
