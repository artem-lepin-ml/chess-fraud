"""Evaluate a trained model on ChessFraud-Synth test set, per rating bin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from ..configs.model_config import ModelConfig
from ..data.synth_dataset import RowPairDataset, SynthData
from ..models.mlp import build_model_from_config
from ..training.metrics import compute_metrics, apply_threshold
from ..training.threshold import predict_proba_loader


def evaluate_on_synth_test(
    *,
    synth_data: SynthData,
    cheat_name: str,
    model_path: Path,
    model_config: ModelConfig,
    threshold: float,
    input_dim: int,
    device: torch.device,
    batch_size: int = 4096,
    out_dir: Path | None = None,
) -> dict[str, dict[str, float]]:
    """Evaluate one model on synth test per rating bin + ``"ALL"``.

    Returns ``{bin_name: {metric: value}}``.
    If *out_dir* is given, saves classification reports.
    """
    # ---- load model ----
    model = build_model_from_config(input_dim, model_config).to(device)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    emb_cheat = synth_data.emb_cheat[cheat_name]
    test_idx = synth_data.test_idx
    bin_idx = synth_data.bin_idx

    results: dict[str, dict[str, float]] = {}

    # ---- per-bin evaluation ----
    for b_i, b_name in enumerate(synth_data.bin_names):
        rows_bin = np.intersect1d(
            test_idx, np.flatnonzero(bin_idx == b_i)
        )
        if len(rows_bin) == 0:
            continue

        ds = RowPairDataset(
            synth_data.emb_human, emb_cheat, rows_bin, synth_data.extra_features
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        y_true = np.concatenate([np.zeros(len(rows_bin)), np.ones(len(rows_bin))]).astype(np.int64)

        probs = predict_proba_loader(model, loader, device)
        y_pred = apply_threshold(probs, threshold)
        results[b_name] = compute_metrics(y_true, y_pred)

        if out_dir is not None:
            rep = classification_report(
                y_true, y_pred, labels=[0, 1],
                target_names=["human", "cheat"], digits=2, zero_division=0,
            )
            rpt_dir = out_dir / "tables" / "classification_report"
            rpt_dir.mkdir(parents=True, exist_ok=True)
            (rpt_dir / f"{cheat_name}__{b_name}.txt").write_text(rep)

    # ---- ALL bins ----
    ds_all = RowPairDataset(
        synth_data.emb_human, emb_cheat, test_idx, synth_data.extra_features
    )
    loader_all = DataLoader(ds_all, batch_size=batch_size, shuffle=False)
    y_all = np.concatenate([np.zeros(len(test_idx)), np.ones(len(test_idx))]).astype(np.int64)
    probs_all = predict_proba_loader(model, loader_all, device)
    y_pred_all = apply_threshold(probs_all, threshold)
    results["ALL"] = compute_metrics(y_all, y_pred_all)

    if out_dir is not None:
        rep = classification_report(
            y_all, y_pred_all, labels=[0, 1],
            target_names=["human", "cheat"], digits=2, zero_division=0,
        )
        rpt_dir = out_dir / "tables" / "classification_report"
        (rpt_dir / f"{cheat_name}__ALL.txt").write_text(rep)

    return results
