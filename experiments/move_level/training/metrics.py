"""Evaluation metrics for binary cheat detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

METRIC_NAMES = [
    "accuracy",
    "micro_f1",
    "macro_f1",
    "cheat_precision",
    "cheat_recall",
    "cheat_f1",
]


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the standard metric set for binary cheat detection."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "cheat_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "cheat_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "cheat_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
    }


def apply_threshold(probs: np.ndarray, thr: float) -> np.ndarray:
    """Convert probabilities to binary predictions."""
    return (probs >= thr).astype(np.int64)
