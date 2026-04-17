"""Evaluate trained models on ChessFraud (Tournament) dataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from ..configs.model_config import ModelConfig
from ..data.tournament_dataset import RowsWithLabelsDataset, TournamentData
from ..models.mlp import build_model_from_config
from ..training.metrics import METRIC_NAMES, compute_metrics, apply_threshold


_PRED_META_COLS = [
    "game_id", "half_move", "player_color", "player_elo", "opponent_elo",
    "time_control", "move_uci", "fen_before", "move_stockfish_15",
    "elo_bin",
]


@torch.no_grad()
def _predict_logits(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """Run *model* over *loader*, return raw logits as numpy array."""
    model.eval()
    parts: list[np.ndarray] = []
    for batch_x, _batch_y in loader:
        batch_x = batch_x.to(device, dtype=torch.float32)
        logits = model(batch_x).view(-1)
        parts.append(logits.cpu().numpy())
    return np.concatenate(parts, axis=0)


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
    return_predictions: bool = False,
) -> dict[str, dict[str, float]] | tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Evaluate one model on tournament data per Elo bin + ``"ALL"``.

    Returns ``{bin_name: {metric: value}}``.
    If *return_predictions* is True, returns ``(results, predictions_df)`` where
    ``predictions_df`` has one row per tournament sample (no bin duplication).
    """
    # ---- load model ----
    model = build_model_from_config(input_dim, model_config).to(device)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    td = tournament_data
    results: dict[str, dict[str, float]] = {}

    # ---- single full-pass prediction (ALL rows) ----
    all_rows = np.arange(len(td.y), dtype=np.int64)
    ds_all = RowsWithLabelsDataset(td.X, all_rows, td.y)
    loader_all = DataLoader(ds_all, batch_size=batch_size, shuffle=False)
    logits_all = _predict_logits(model, loader_all, device)
    probs_all = 1.0 / (1.0 + np.exp(-logits_all))
    y_pred_all = apply_threshold(probs_all, threshold)

    # ---- per-bin metrics (slicing into the full-pass result) ----
    for bin_name in td.bin_names:
        rows = td.rows_by_bin[bin_name]
        if len(rows) == 0:
            continue
        y_true = td.y[rows]
        y_pred_bin = y_pred_all[rows]
        results[bin_name] = compute_metrics(y_true, y_pred_bin)

        if out_dir is not None and cheat_name:
            rep = classification_report(
                y_true, y_pred_bin, labels=[0, 1],
                target_names=["human", "cheat"], digits=2, zero_division=0,
            )
            rpt_dir = out_dir / "tables" / "classification_report"
            rpt_dir.mkdir(parents=True, exist_ok=True)
            (rpt_dir / f"{cheat_name}__{bin_name}.txt").write_text(rep)

    if not return_predictions:
        return results

    # ---- predictions DataFrame ----
    meta_cols = [c for c in _PRED_META_COLS if c in td.df_meta.columns]
    preds = td.df_meta[meta_cols].copy().reset_index(drop=True)
    preds["row_id"] = all_rows
    preds["y"] = td.y
    preds["logit"] = logits_all
    preds["prob"] = probs_all
    preds["pred"] = y_pred_all
    if "move_uci" in preds.columns and "move_stockfish_15" in preds.columns:
        preds["sf15_match"] = (preds["move_uci"] == preds["move_stockfish_15"]).astype(np.int8)

    return results, preds


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


def metrics_to_long_df(
    results: dict[str, dict[str, float]],
    *,
    index_name: str = "bin",
    metric_order: list[str] | None = None,
) -> pd.DataFrame:
    """Convert ``{bin: {metric: value}}`` into a long-format DataFrame with a
    fixed column order.

    The first columns (after the bin index) are always:
    ``macro_f1, cheat_precision, cheat_recall, cheat_f1, accuracy, micro_f1``.
    """
    if metric_order is None:
        metric_order = [
            "macro_f1", "cheat_precision", "cheat_recall", "cheat_f1",
            "accuracy", "micro_f1",
        ]
    rows: list[dict[str, float | str]] = []
    for bin_name, m in results.items():
        row: dict[str, float | str] = {index_name: bin_name}
        for metric in metric_order:
            row[metric] = float(m.get(metric, float("nan")))
        rows.append(row)
    return pd.DataFrame(rows)
