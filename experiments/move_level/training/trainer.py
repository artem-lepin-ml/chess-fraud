"""Batched training loop with early stopping and threshold calibration."""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from ..configs.model_config import ModelConfig, TrainConfig
from ..models.mlp import build_model_from_config
from .threshold import predict_proba_loader, best_threshold


@dataclass
class TrainResult:
    """Output of ``train_one``."""

    tag: str
    best_epoch: int
    best_val_loss: float
    best_thr: float
    best_val_metric: float
    model_path: Path
    meta_path: Path
    curves_path: Path
    model_config: ModelConfig
    train_config: TrainConfig
    input_dim: int


def _count_pos_neg(loader: DataLoader) -> tuple[int, int]:
    """Count positives and negatives in a DataLoader (single pass)."""
    n_pos = 0
    n_total = 0
    for _x, y in loader:
        n_pos += int((y > 0.5).sum().item())
        n_total += len(y)
    return n_pos, n_total - n_pos


def train_one(
    *,
    train_loader: DataLoader,
    val_loader: DataLoader,
    y_val: np.ndarray,
    input_dim: int,
    model_config: ModelConfig,
    train_config: TrainConfig,
    device: torch.device,
    out_dir: Path,
    tag: str,
    logger: logging.Logger | None = None,
    log_every: int = 1,
) -> TrainResult:
    """Train a single model with batched DataLoader-based training.

    Saves ``model.pt``, ``meta.json``, ``curves.csv`` to *out_dir*.
    Emits per-epoch progress to *logger* (or ``logging.getLogger`` fallback).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    log = logger or logging.getLogger("move_level.train")

    # ---- build model ----
    model = build_model_from_config(input_dim, model_config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        "[%s] build model: in_dim=%d hidden=%s act=%s norm=%s dropout=%.2f params=%d",
        tag, input_dim, model_config.hidden_dims, model_config.activation,
        model_config.normalization, model_config.dropout, n_params,
    )

    # ---- pos_weight ----
    n_pos, n_neg = _count_pos_neg(train_loader)
    pos_weight = torch.tensor(
        [n_neg / max(n_pos, 1)], device=device, dtype=torch.float32
    )
    log.info(
        "[%s] train: n_pos=%d n_neg=%d pos_weight=%.4f",
        tag, n_pos, n_neg, float(pos_weight.item()),
    )

    # ---- loss / optimiser / scheduler ----
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode="min",
        factor=train_config.scheduler_factor,
        patience=train_config.scheduler_patience,
    )
    log.info(
        "[%s] optim: AdamW lr=%.2e wd=%.2e | scheduler ReduceLROnPlateau "
        "factor=%.2f patience=%d | early_stop=%d epochs_max=%d",
        tag, train_config.lr, train_config.weight_decay,
        train_config.scheduler_factor, train_config.scheduler_patience,
        train_config.early_stop_patience, train_config.epochs,
    )

    # ---- training loop ----
    best_val_loss = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    bad_epochs = 0

    curves: list[dict[str, float]] = []
    t_start = time.time()

    for epoch in range(train_config.epochs):
        # --- train ---
        model.train()
        epoch_loss = 0.0
        epoch_n = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device, dtype=torch.float32)
            batch_y = batch_y.to(device, dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            logits = model(batch_x).view(-1)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch_x)
            epoch_n += len(batch_x)
        train_loss = epoch_loss / max(epoch_n, 1)

        # --- validation ---
        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device, dtype=torch.float32)
                batch_y = batch_y.to(device, dtype=torch.float32)
                logits = model(batch_x).view(-1)
                loss = loss_fn(logits, batch_y)
                val_loss_sum += loss.item() * len(batch_x)
                val_n += len(batch_x)
        val_loss = val_loss_sum / max(val_n, 1)

        current_lr = opt.param_groups[0]["lr"]
        scheduler.step(val_loss)

        curves.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": current_lr,
        })

        # --- early stopping ---
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        # --- per-epoch log ---
        if log_every > 0 and (epoch % log_every == 0 or bad_epochs >= train_config.early_stop_patience):
            log.info(
                "[%s] epoch=%d train_loss=%.4f val_loss=%.4f lr=%.2e "
                "patience=%d/%d best_val=%.4f (ep=%d) elapsed=%.1fs",
                tag, epoch, train_loss, val_loss, current_lr,
                bad_epochs, train_config.early_stop_patience,
                best_val_loss, best_epoch, time.time() - t_start,
            )

        if bad_epochs >= train_config.early_stop_patience:
            log.info("[%s] early stopping at epoch=%d (best epoch=%d)", tag, epoch, best_epoch)
            break

    # ---- restore best weights ----
    if best_state:
        model.load_state_dict(best_state)
    model.to(device)

    # ---- threshold calibration on validation ----
    val_probs = predict_proba_loader(model, val_loader, device)
    best_thr, best_val_metric = best_threshold(
        val_probs, y_val, train_config.threshold_grid
    )
    log.info(
        "[%s] threshold calibration: best_thr=%.3f best_val_macro_f1=%.4f",
        tag, best_thr, best_val_metric,
    )

    # ---- save artefacts ----
    model_path = out_dir / "model.pt"
    meta_path = out_dir / "meta.json"
    curves_path = out_dir / "curves.csv"

    torch.save(best_state or model.state_dict(), model_path)

    meta = {
        "tag": tag,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "best_thr": best_thr,
        "best_val_metric": best_val_metric,
        "input_dim": input_dim,
        "model_config": asdict(model_config),
        "train_config": asdict(train_config),
        "elapsed_sec": time.time() - t_start,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    pd.DataFrame(curves).to_csv(curves_path, index=False)

    log.info(
        "[%s] done. best_epoch=%d best_val_loss=%.4f best_thr=%.3f "
        "elapsed=%.1fs | artefacts: %s",
        tag, best_epoch, best_val_loss, best_thr,
        time.time() - t_start, out_dir,
    )

    return TrainResult(
        tag=tag,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
        best_thr=best_thr,
        best_val_metric=best_val_metric,
        model_path=model_path,
        meta_path=meta_path,
        curves_path=curves_path,
        model_config=model_config,
        train_config=train_config,
        input_dim=input_dim,
    )
