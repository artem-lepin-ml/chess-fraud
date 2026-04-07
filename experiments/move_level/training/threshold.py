"""Threshold calibration and batched inference utilities."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .metrics import compute_metrics, apply_threshold


@torch.no_grad()
def predict_proba_loader(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run *model* over *dataloader*, return ``p(cheat=1)`` as numpy array."""
    model.eval()
    parts: list[np.ndarray] = []
    for batch_x, _batch_y in dataloader:
        batch_x = batch_x.to(device, dtype=torch.float32)
        logits = model(batch_x).view(-1)
        probs = torch.sigmoid(logits)
        parts.append(probs.cpu().numpy())
    return np.concatenate(parts, axis=0)


def best_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    thr_grid: list[float] | np.ndarray,
    metric: str = "macro_f1",
) -> tuple[float, float]:
    """Grid-search for the threshold maximising *metric* on a validation set.

    Returns ``(best_thr, best_metric_value)``.
    """
    best_thr = 0.5
    best_val = -1.0
    for thr in thr_grid:
        y_pred = apply_threshold(probs, float(thr))
        m = compute_metrics(y_true, y_pred)
        val = m[metric]
        if val > best_val:
            best_val = val
            best_thr = float(thr)
    return best_thr, best_val
