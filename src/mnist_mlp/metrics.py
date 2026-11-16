"""Classification metrics without external machine learning libraries."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .losses import validate_targets


def accuracy(scores: NDArray[np.floating], targets: NDArray) -> float:
    """Return the fraction of correct predictions from logits or probabilities."""
    scores = np.asarray(scores)
    if scores.ndim != 2 or min(scores.shape) == 0 or not np.isfinite(scores).all():
        raise ValueError("Scores must be a nonempty finite two-dimensional array.")
    targets = validate_targets(targets, scores.shape[0], scores.shape[1])
    expected = targets if targets.ndim == 1 else targets.argmax(axis=1)
    return float(np.mean(scores.argmax(axis=1) == expected))


def binary_roc_curve(targets: NDArray, scores: NDArray) -> tuple[NDArray, NDArray, float]:
    """Return FPR, TPR and trapezoidal AUC, grouping tied thresholds.

    Both positive and negative examples are required. Grouping equal scores
    makes AUC independent of ordering within ties (a constant score has AUC .5).
    """
    targets, scores = np.asarray(targets), np.asarray(scores, dtype=np.float64)
    if targets.ndim != 1 or scores.shape != targets.shape or not len(targets):
        raise ValueError("ROC needs nonempty aligned one-dimensional arrays.")
    if not np.all((targets == 0) | (targets == 1)) or not np.isfinite(scores).all():
        raise ValueError("ROC needs binary labels and finite scores.")
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if not positives or not negatives:
        raise ValueError("ROC is undefined without both positive and negative samples.")
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores)), len(scores) - 1]
    true_positives = np.cumsum(targets[order], dtype=np.int64)[endpoints]
    false_positives = 1 + endpoints - true_positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    # Explicit trapezoids keep compatibility with NumPy 1.26 and 2.x.
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) * 0.5)
    return fpr, tpr, float(auc)


def classification_report(probabilities: NDArray, targets: NDArray) -> dict:
    """Summarize hard decisions and one-vs-rest probability ranking.

    Undefined per-class ROC AUC is represented by None, never a fabricated
    score. Macro AUC is None unless every class has positive and negative data.
    Precision/recall/F1 use zero when their denominator is zero.
    """
    probabilities = np.asarray(probabilities)
    if (probabilities.ndim != 2 or min(probabilities.shape) == 0
            or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6)):
        raise ValueError("Expected nonempty normalized class probabilities.")
    y = validate_targets(targets, *probabilities.shape)
    if y.ndim != 1:
        raise ValueError("Classification reports require integer class labels.")
    classes = probabilities.shape[1]
    predicted = probabilities.argmax(axis=1)
    confusion = np.zeros((classes, classes), dtype=np.int64)
    np.add.at(confusion, (y, predicted), 1)
    support = confusion.sum(axis=1)
    predicted_count = confusion.sum(axis=0)
    tp = confusion.diagonal().astype(np.float64)
    precision = np.divide(tp, predicted_count, out=np.zeros(classes), where=predicted_count != 0)
    recall = np.divide(tp, support, out=np.zeros(classes), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros(classes), where=(precision + recall) != 0)
    rows = []
    aucs = []
    for digit in range(classes):
        binary = y == digit
        auc = binary_roc_curve(binary, probabilities[:, digit])[2] if 0 < binary.sum() < len(y) else None
        aucs.append(auc)
        rows.append({"class": digit, "support": int(support[digit]),
                     "precision": float(precision[digit]), "recall": float(recall[digit]),
                     "f1": float(f1[digit]), "roc_auc_ovr": auc})
    correct = int(tp.sum())
    proportion = correct / len(y)
    z = 1.959963984540054
    denominator = 1 + z * z / len(y)
    center = (proportion + z * z / (2 * len(y))) / denominator
    radius = z * np.sqrt(proportion * (1 - proportion) / len(y)
                         + z * z / (4 * len(y) ** 2)) / denominator
    one_hot = np.eye(classes, dtype=np.uint8)[y]
    micro_auc = binary_roc_curve(one_hot.ravel(), probabilities.ravel())[2] if classes > 1 else None
    return {"accuracy": proportion, "correct": correct, "errors": len(y) - correct,
            "accuracy_wilson_95": [float(center - radius), float(center + radius)],
            "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
            "macro_f1": float(f1.mean()), "weighted_f1": float(np.average(f1, weights=support)),
            "roc_auc_macro_ovr": float(np.mean(aucs)) if all(a is not None for a in aucs) else None,
            "roc_auc_micro_ovr": micro_auc, "per_class": rows,
            "confusion_matrix": confusion.tolist(), "matrix_axes": "rows=true, columns=predicted"}
