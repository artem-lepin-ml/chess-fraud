"""Evaluate a trained model on ChessFraud-Synth test set, per rating bin."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.utils.data import DataLoader

from ..configs.model_config import ModelConfig
from ..data.synth_dataset import RowPairDataset, SynthData
from ..models.mlp import build_model_from_config
from ..training.metrics import compute_metrics, apply_threshold


_PRED_META_COLS = [
    "player", "game_id", "half_move", "rating_bin",
    "player_elo", "opponent_elo", "time_control", "move_thinking_time",
    "move_uci", "fen_before", "move_stockfish_15",
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


def _build_pair_predictions_df(
    synth_data: SynthData,
    cheat_name: str,
    rows_bin: np.ndarray,
    logits: np.ndarray,
    probs: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """Assemble a per-sample predictions DataFrame for a given set of rows.

    ``rows_bin`` is an array of length N (rows used for this evaluation). The
    loader yielded ``2N`` samples in order: N human samples, then N cheat
    samples. We reconstruct that here.
    """
    n = len(rows_bin)
    df_meta = synth_data.df_meta
    meta_cols = [c for c in _PRED_META_COLS if c in df_meta.columns]

    # --- move_played and cheat_source per sample ---
    move_human = synth_data.move_human[rows_bin]

    if cheat_name == "ALL":
        if synth_data.all_source_idx is None:
            raise RuntimeError("cheat_name='ALL' but synth_data.all_source_idx is None")
        src_idx = synth_data.all_source_idx[rows_bin]
        src_names = np.array(synth_data.cheat_names, dtype=object)
        # Exclude "ALL" itself from the mapping (mixture only indexes into base names).
        base_names = [n for n in synth_data.cheat_names if n != "ALL"]
        base_names_arr = np.array(base_names, dtype=object)
        cheat_source = base_names_arr[src_idx]
    else:
        cheat_source = np.array([cheat_name] * n, dtype=object)

    move_cheat = synth_data.move_cheat[cheat_name][rows_bin]

    # --- human half ---
    human_meta = df_meta.iloc[rows_bin].reset_index(drop=True)[meta_cols].copy()
    human_meta["row_id"] = rows_bin
    human_meta["sample_type"] = "human"
    human_meta["cheat_source"] = ""
    human_meta["move_played"] = move_human
    human_meta["y"] = 0
    human_meta["logit"] = logits[:n]
    human_meta["prob"] = probs[:n]
    human_meta["pred"] = y_pred[:n]

    # --- cheat half ---
    cheat_meta = df_meta.iloc[rows_bin].reset_index(drop=True)[meta_cols].copy()
    cheat_meta["row_id"] = rows_bin
    cheat_meta["sample_type"] = "cheat"
    cheat_meta["cheat_source"] = cheat_source
    cheat_meta["move_played"] = move_cheat
    cheat_meta["y"] = 1
    cheat_meta["logit"] = logits[n:]
    cheat_meta["prob"] = probs[n:]
    cheat_meta["pred"] = y_pred[n:]

    out = pd.concat([human_meta, cheat_meta], axis=0, ignore_index=True)
    # sf15_match per sample (deterministic from move_played and move_stockfish_15)
    if "move_stockfish_15" in out.columns:
        out["sf15_match"] = (out["move_played"] == out["move_stockfish_15"]).astype(np.int8)
    return out


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
    return_predictions: bool = False,
) -> dict[str, dict[str, float]] | tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Evaluate one model on synth test per rating bin + ``"ALL"``.

    Returns ``{bin_name: {metric: value}}``.
    If *return_predictions* is True, returns ``(results, predictions_df)``.
    """
    # ---- load model ----
    model = build_model_from_config(input_dim, model_config).to(device)
    state = torch.load(model_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()

    cheat_vec = synth_data.vec_cheat[cheat_name]
    test_idx = synth_data.test_idx
    bin_idx = synth_data.bin_idx

    results: dict[str, dict[str, float]] = {}

    # ---- per-bin evaluation (metrics only; predictions gathered once below) ----
    for b_i, b_name in enumerate(synth_data.bin_names):
        rows_bin = np.intersect1d(test_idx, np.flatnonzero(bin_idx == b_i))
        if len(rows_bin) == 0:
            continue

        ds = RowPairDataset(synth_data.vec_human, cheat_vec, rows_bin)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
        y_true = np.concatenate([np.zeros(len(rows_bin)), np.ones(len(rows_bin))]).astype(np.int64)

        logits = _predict_logits(model, loader, device)
        probs = 1.0 / (1.0 + np.exp(-logits))
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
    ds_all = RowPairDataset(synth_data.vec_human, cheat_vec, test_idx)
    loader_all = DataLoader(ds_all, batch_size=batch_size, shuffle=False)
    y_all = np.concatenate([np.zeros(len(test_idx)), np.ones(len(test_idx))]).astype(np.int64)
    logits_all = _predict_logits(model, loader_all, device)
    probs_all = 1.0 / (1.0 + np.exp(-logits_all))
    y_pred_all = apply_threshold(probs_all, threshold)
    results["ALL"] = compute_metrics(y_all, y_pred_all)

    if out_dir is not None:
        rep = classification_report(
            y_all, y_pred_all, labels=[0, 1],
            target_names=["human", "cheat"], digits=2, zero_division=0,
        )
        rpt_dir = out_dir / "tables" / "classification_report"
        rpt_dir.mkdir(parents=True, exist_ok=True)
        (rpt_dir / f"{cheat_name}__ALL.txt").write_text(rep)

    if return_predictions:
        # Build a single "ALL" predictions frame (don't concatenate per-bin to avoid
        # duplicates — each test row appears exactly once per bin).
        df_all = _build_pair_predictions_df(
            synth_data, cheat_name, test_idx, logits_all, probs_all, y_pred_all,
        )
        # Attach rating bin label per row
        bin_name_arr = np.array(synth_data.bin_names, dtype=object)
        df_all["bin"] = bin_name_arr[synth_data.bin_idx[df_all["row_id"].to_numpy()]]
        return results, df_all

    return results
