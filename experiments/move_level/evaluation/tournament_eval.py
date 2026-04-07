"""Evaluate trained models on ChessFraud (Tournament) dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from ..configs.model_config import ModelConfig
from ..data.tournament_dataset import RowsWithLabelsDataset, TournamentData
from ..models.mlp import build_model_from_config
from ..training.metrics import METRIC_NAMES, compute_metrics, apply_threshold
from ..training.threshold import predict_proba_loader


def evaluate_on_tournament(
    *,
    tournament_data: TournamentData,
    model_path: Path,
    model_config: ModelConfig,
    threshold: float,
    input_dim: int,
    device: torch.device,
    batch_size: int = 4096,
    out_dir: Path | None = None,
    cheat_name: str = "",
) -> dict[str, dict[str, float]]:
    """Evaluate one model on tournament data per Elo bin + ``"ALL"``.

    Returns ``{bin_name: {metric: value}}``.
    If *out_dir* is given, saves classification reports.
    """
    # ---- load model ----
    model = build_model_from_config(input_dim, model_config).to(device)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    td = tournament_data
    results: dict[str, dict[str, float]] = {}

    for bin_name in td.bin_names:
        rows = td.rows_by_bin[bin_name]
        if len(rows) == 0:
            continue

        ds = RowsWithLabelsDataset(td.X_emb, rows, td.y[rows], td.extra_features)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        y_true = td.y[rows]

        probs = predict_proba_loader(model, loader, device)
        y_pred = apply_threshold(probs, threshold)
        results[bin_name] = compute_metrics(y_true, y_pred)

        if out_dir is not None and cheat_name:
            rep = classification_report(
                y_true, y_pred, labels=[0, 1],
                target_names=["human", "cheat"], digits=2, zero_division=0,
            )
            rpt_dir = out_dir / "tables" / "classification_report"
            rpt_dir.mkdir(parents=True, exist_ok=True)
            (rpt_dir / f"{cheat_name}__{bin_name}.txt").write_text(rep)

    return results


def save_eval_results(
    all_results: dict[str, dict[str, dict[str, float]]],
    out_dir: Path,
    row_names: list[str],
    metric_names: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build metric tables ``(rows=bins, cols=cheat models)``, save as CSVs.

    *all_results*: ``{cheat_name: {bin_name: {metric: value}}}``.
    Returns ``{metric_name: DataFrame}``.
    """
    if metric_names is None:
        metric_names = METRIC_NAMES

    col_names = list(all_results.keys())

    tables: dict[str, pd.DataFrame] = {}
    for m in metric_names:
        tbl = pd.DataFrame(index=row_names, columns=col_names, dtype=float)
        for cheat_name, bin_results in all_results.items():
            for bin_name, metrics in bin_results.items():
                if bin_name in row_names:
                    tbl.loc[bin_name, cheat_name] = float(metrics.get(m, float("nan")))
        tables[m] = tbl

    tbl_dir = out_dir / "tables"
    tbl_dir.mkdir(parents=True, exist_ok=True)
    for m, tbl in tables.items():
        tbl.to_csv(tbl_dir / f"metrics__{m}.csv", index=True)

    return tables
