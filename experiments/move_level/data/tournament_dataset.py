"""Load ChessFraud (Tournament) dataset for evaluation."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..configs.dataset_config import DatasetConfig
from ..features.feature_config import FeatureConfig
from ..features.feature_builder import build_features, feature_dim


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RowsWithLabelsDataset(Dataset):
    """Single-sample dataset: ``(emb[i], label[i])``."""

    def __init__(
        self,
        emb: np.ndarray,
        row_idx: np.ndarray,
        y: np.ndarray,
        extra_features: np.ndarray | None = None,
    ) -> None:
        self.emb = emb
        self.row_idx = row_idx
        self.y = y
        self.extra_features = extra_features

    def __len__(self) -> int:
        return len(self.row_idx)

    def __getitem__(self, idx: int):
        row = self.row_idx[idx]
        x = self.emb[row].astype(np.float32)
        if self.extra_features is not None:
            feat = self.extra_features[row].astype(np.float32)
            x = np.concatenate([x, feat])
        label = np.float32(self.y[idx])
        return x, label


# ---------------------------------------------------------------------------
# TournamentData container
# ---------------------------------------------------------------------------

@dataclass
class TournamentData:
    """Container returned by ``load_tournament_data``."""

    X_emb: np.ndarray                      # (N_filtered, D)
    y: np.ndarray                          # labels (0/1)
    extra_features: np.ndarray | None
    rows_by_bin: dict[str, np.ndarray]     # bin_name -> row indices (includes "ALL")
    bin_names: list[str]                   # ordered bin names + "ALL"
    input_dim: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_tournament_data(
    cfg: DatasetConfig,
    feature_cfg: FeatureConfig | None = None,
) -> TournamentData:
    """Load tournament CSV + NPZ, filter ``is_used``, bin by ``player_elo``."""
    csv_path = Path(cfg.tournament_csv_path)
    df = pd.read_csv(csv_path)

    # ---- NPZ alignment ----
    npz = np.load(str(cfg.tournament_emb_npz_path))
    emb_key = cfg.emb_key_human  # same key convention

    if "Unnamed: 0" in df.columns:
        npz_rows_all = df["Unnamed: 0"].to_numpy(dtype=np.int64)
    else:
        npz_rows_all = np.arange(len(df), dtype=np.int64)

    # ---- filter ----
    mask = df["is_used"] == True  # noqa: E712
    df_f = df.loc[mask].reset_index(drop=True)
    npz_rows = npz_rows_all[mask.to_numpy()]

    y = df_f["move_label"].to_numpy(dtype=np.int64)

    # ---- embeddings ----
    X_emb = np.asarray(npz[emb_key])[npz_rows].astype(np.float32)
    emb_dim = X_emb.shape[1]

    # ---- elo binning ----
    elo = df_f["player_elo"].to_numpy(dtype=np.float64)
    elo_bin_arr = np.empty(len(df_f), dtype=object)
    bin_names: list[str] = []
    for name, lo, hi in cfg.tournament_elo_bins:
        bin_mask = (elo >= lo) & (elo < hi)
        elo_bin_arr[bin_mask] = name
        bin_names.append(name)

    rows_by_bin: dict[str, np.ndarray] = {}
    for name in bin_names:
        rows_by_bin[name] = np.flatnonzero(elo_bin_arr == name)
    rows_by_bin["ALL"] = np.arange(len(df_f), dtype=np.int64)
    bin_names_with_all = bin_names + ["ALL"]

    # ---- extra features ----
    extra_features: np.ndarray | None = None
    extra_dim = 0
    if feature_cfg is not None:
        extra_features = build_features(df_f, feature_cfg)
        extra_dim = feature_dim(feature_cfg)

    input_dim = emb_dim + extra_dim

    return TournamentData(
        X_emb=X_emb,
        y=y,
        extra_features=extra_features,
        rows_by_bin=rows_by_bin,
        bin_names=bin_names_with_all,
        input_dim=input_dim,
    )
