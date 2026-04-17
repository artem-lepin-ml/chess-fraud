"""Load ChessFraud-Synth, split, build paired datasets & DataLoaders."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from ..configs.dataset_config import DatasetConfig
from ..features.feature_config import FeatureConfig
from ..features.feature_builder import build_features, compute_feature_stats, feature_dim


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RowPairDataset(Dataset):
    """For *N* rows yields *2N* samples: ``(human_vec[i], 0)`` and
    ``(cheat_vec[i], 1)``.

    ``human_vec`` / ``cheat_vec`` are pre-built per-sample vectors of shape
    ``(N_total, input_dim)`` that already include any extra features
    concatenated after the embedding.
    """

    def __init__(
        self,
        human_vec: np.ndarray,
        cheat_vec: np.ndarray,
        row_idx: np.ndarray,
    ) -> None:
        self.human_vec = human_vec
        self.cheat_vec = cheat_vec
        self.row_idx = row_idx
        self._n = len(row_idx)

    def __len__(self) -> int:
        return 2 * self._n

    def __getitem__(self, idx: int):
        if idx < self._n:
            row = self.row_idx[idx]
            x = self.human_vec[row].astype(np.float32, copy=False)
            y = np.float32(0.0)
        else:
            row = self.row_idx[idx - self._n]
            x = self.cheat_vec[row].astype(np.float32, copy=False)
            y = np.float32(1.0)
        return x, y


# ---------------------------------------------------------------------------
# SynthData container
# ---------------------------------------------------------------------------

@dataclass
class SynthData:
    """Container returned by ``load_synth_data``."""

    train_idx: np.ndarray
    val_idx: np.ndarray
    test_idx: np.ndarray
    bin_idx: np.ndarray                  # per-row (filtered) rating bin index
    split_labels: np.ndarray             # per-row "train" / "test"

    # --- embeddings (kept for backward compatibility / diagnostics) ---
    emb_human: np.ndarray                # (N_filtered, D_emb)
    emb_cheat: dict[str, np.ndarray]     # cheat_name -> (N_filtered, D_emb)
    move_human: np.ndarray
    move_cheat: dict[str, np.ndarray]

    # --- concatenated [emb | features] vectors (fed to the model) ---
    vec_human: np.ndarray                # (N_filtered, input_dim)
    vec_cheat: dict[str, np.ndarray]     # cheat_name -> (N_filtered, input_dim); includes "ALL"

    # --- per-row provenance for the "ALL" mixture ---
    all_source_idx: np.ndarray | None    # (N_filtered,) int index into ``cheat_names``, or None

    # --- feature standardization stats (fit on TRAIN human-rows) ---
    feature_stats: dict[str, dict[str, float]]

    # --- legacy alias: per-row feature slice built from human move ---
    extra_features: np.ndarray | None

    input_dim: int
    bin_names: list[str]
    cheat_names: list[str]               # includes "ALL" if built

    # --- metadata DataFrame, for predictions dump and case-study ---
    df_meta: pd.DataFrame = field(default_factory=pd.DataFrame)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _split_train_val_by_player_per_bin(
    df_f: pd.DataFrame,
    *,
    split_col: str,
    bin_idx: np.ndarray,
    player_col: str,
    n_bins: int,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hold out ``val_frac`` of unique *players* per bin for validation.

    Returns ``(train_idx, val_idx, test_idx)`` — row indices into *df_f*.
    """
    rng = np.random.RandomState(seed)
    split_arr = df_f[split_col].to_numpy()
    player_arr = df_f[player_col].to_numpy()

    train_mask = np.zeros(len(df_f), dtype=bool)
    val_mask = np.zeros(len(df_f), dtype=bool)
    test_mask = split_arr == "test"

    for b in range(n_bins):
        in_bin_train = (bin_idx == b) & (split_arr == "train")
        players = np.unique(player_arr[in_bin_train])
        n_val_players = max(1, int(len(players) * val_frac))
        val_players = set(rng.choice(players, size=n_val_players, replace=False))

        rows_in_bin = np.flatnonzero(in_bin_train)
        for r in rows_in_bin:
            if player_arr[r] in val_players:
                val_mask[r] = True
            else:
                train_mask[r] = True

    return (
        np.flatnonzero(train_mask),
        np.flatnonzero(val_mask),
        np.flatnonzero(test_mask),
    )


def _build_all_mixture(
    emb_cheat: dict[str, np.ndarray],
    vec_cheat: dict[str, np.ndarray],
    move_cheat: dict[str, np.ndarray],
    bin_idx: np.ndarray,
    cheat_model_to_bin_idxs: dict[str, list[int]],
    cheat_names: list[str],
    n_rows: int,
    emb_dim: int,
    input_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Row-wise random selection from allowed cheats.

    Returns ``(emb_all, vec_all, move_all, source_idx)`` where ``source_idx[i]`` is
    the index into ``cheat_names`` of the source chosen for row *i*.
    """
    rng = np.random.default_rng(seed)
    emb_all = np.empty((n_rows, emb_dim), dtype=np.float32)
    vec_all = np.empty((n_rows, input_dim), dtype=np.float32)
    move_all = np.empty(n_rows, dtype=object)
    source_idx = np.full(n_rows, -1, dtype=np.int32)

    name_to_pos = {name: i for i, name in enumerate(cheat_names)}

    for i in range(n_rows):
        b = int(bin_idx[i])
        allowed = [n for n in cheat_names if b in cheat_model_to_bin_idxs.get(n, [])]
        if not allowed:
            allowed = list(cheat_names)
        chosen = str(rng.choice(allowed))
        emb_all[i] = emb_cheat[chosen][i]
        vec_all[i] = vec_cheat[chosen][i]
        move_all[i] = move_cheat[chosen][i]
        source_idx[i] = name_to_pos[chosen]

    return emb_all, vec_all, move_all, source_idx


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_META_COLS = [
    "player", "game_id", "half_move", "rating_bin",
    "player_elo", "opponent_elo", "time_control", "move_thinking_time",
    "move_uci", "fen_before", "fen_after", "move_stockfish_15",
]


def load_synth_data(
    cfg: DatasetConfig,
    feature_cfg: FeatureConfig | None = None,
    feature_stats: dict[str, dict[str, float]] | None = None,
) -> SynthData:
    """Load CSV + NPZ, filter ``is_used``, split train/val/test, build
    ``"ALL"`` mixture (if enabled), compute extra features.

    If ``feature_stats`` is supplied, it is used as-is for standardization
    (skipping the TRAIN-human-rows fit). Useful for re-loading data with
    training-time stats at inference time.
    """
    # ---- load CSV ----
    csv_path = Path(cfg.synth_csv_path)
    df = pd.read_csv(csv_path, index_col=0)
    df["npz_row"] = df.index

    bin_name_to_idx = {b: i for i, b in enumerate(cfg.rating_bins)}
    df["bin_idx"] = df["rating_bin"].map(bin_name_to_idx)

    # ---- filter ----
    df_f = df.loc[df["is_used"] == True].reset_index(drop=True)  # noqa: E712
    npz_rows = df["npz_row"].loc[df["is_used"] == True].to_numpy(dtype=np.int64)  # noqa: E712

    # ---- load NPZ ----
    npz = np.load(str(cfg.synth_emb_npz_path))

    emb_human = np.asarray(npz[cfg.emb_key_human])[npz_rows].astype(np.float32)
    move_human = df_f[cfg.emb_key_human].to_numpy()
    emb_dim = emb_human.shape[1]

    cheat_names = list(cfg.cheat_models)
    emb_cheat: dict[str, np.ndarray] = {}
    move_cheat: dict[str, np.ndarray] = {}
    for name in cheat_names:
        col = f"move_{name}"
        emb_cheat[name] = np.asarray(npz[col])[npz_rows].astype(np.float32)
        move_cheat[name] = df_f[col].to_numpy()

    # ---- split ----
    bin_idx_arr = df_f["bin_idx"].to_numpy(dtype=np.int64)
    train_idx, val_idx, test_idx = _split_train_val_by_player_per_bin(
        df_f,
        split_col=cfg.split_col,
        bin_idx=bin_idx_arr,
        player_col=cfg.player_col,
        n_bins=len(cfg.rating_bins),
        val_frac=cfg.val_frac,
        seed=cfg.seed,
    )

    # ---- feature stats (TRAIN human-rows only, unless overridden) ----
    if feature_cfg is not None:
        if feature_stats is None:
            feature_stats = compute_feature_stats(df_f, feature_cfg, train_idx)
    else:
        feature_stats = {}

    # ---- per-source features (standardized with TRAIN stats) ----
    extra_dim = 0
    feats_human: np.ndarray | None = None
    feats_cheat: dict[str, np.ndarray] = {}
    if feature_cfg is not None:
        extra_dim = feature_dim(feature_cfg)
        feats_human = build_features(
            df_f, feature_cfg, move_col=cfg.emb_key_human, stats=feature_stats,
        )
        for name in cheat_names:
            move_col = f"move_{name}"
            feats_cheat[name] = build_features(
                df_f, feature_cfg, move_col=move_col, stats=feature_stats,
            )

    input_dim = emb_dim + extra_dim

    # ---- concat [emb | feats] -> vec ----
    def _concat(emb: np.ndarray, feats: np.ndarray | None) -> np.ndarray:
        if feats is None:
            return emb
        return np.concatenate([emb, feats], axis=1).astype(np.float32, copy=False)

    vec_human = _concat(emb_human, feats_human)
    vec_cheat: dict[str, np.ndarray] = {
        name: _concat(emb_cheat[name], feats_cheat.get(name))
        for name in cheat_names
    }

    # ---- "ALL" mixture ----
    all_source_idx: np.ndarray | None = None
    all_cheat_names = list(cheat_names)
    if cfg.build_all_mixture and cheat_names:
        emb_all, vec_all, move_all, src_idx = _build_all_mixture(
            emb_cheat=emb_cheat,
            vec_cheat=vec_cheat,
            move_cheat=move_cheat,
            bin_idx=bin_idx_arr,
            cheat_model_to_bin_idxs=cfg.cheat_model_to_bin_idxs,
            cheat_names=cheat_names,
            n_rows=len(df_f),
            emb_dim=emb_dim,
            input_dim=input_dim,
            seed=cfg.seed,
        )
        emb_cheat["ALL"] = emb_all
        vec_cheat["ALL"] = vec_all
        move_cheat["ALL"] = move_all
        all_source_idx = src_idx
        all_cheat_names = cheat_names + ["ALL"]

    # ---- metadata slice ----
    meta_cols_present = [c for c in _META_COLS if c in df_f.columns]
    df_meta = df_f[meta_cols_present].copy()

    return SynthData(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        bin_idx=bin_idx_arr,
        split_labels=df_f[cfg.split_col].to_numpy(),
        emb_human=emb_human,
        emb_cheat=emb_cheat,
        move_human=move_human,
        move_cheat=move_cheat,
        vec_human=vec_human,
        vec_cheat=vec_cheat,
        all_source_idx=all_source_idx,
        feature_stats=feature_stats,
        extra_features=feats_human,  # legacy alias
        input_dim=input_dim,
        bin_names=list(cfg.rating_bins),
        cheat_names=all_cheat_names,
        df_meta=df_meta,
    )


def build_dataloaders(
    synth_data: SynthData,
    cheat_name: str,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, np.ndarray, np.ndarray, np.ndarray]:
    """Build train / val / test ``DataLoader``s for one cheat model.

    Returns ``(train_loader, val_loader, test_loader, y_train, y_val, y_test)``.
    """
    cheat_vec = synth_data.vec_cheat[cheat_name]

    def _make(idx: np.ndarray, shuffle: bool) -> tuple[DataLoader, np.ndarray]:
        ds = RowPairDataset(synth_data.vec_human, cheat_vec, idx)
        y = np.concatenate([np.zeros(len(idx)), np.ones(len(idx))]).astype(np.int64)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
        )
        return loader, y

    train_loader, y_train = _make(synth_data.train_idx, shuffle=True)
    val_loader, y_val = _make(synth_data.val_idx, shuffle=False)
    test_loader, y_test = _make(synth_data.test_idx, shuffle=False)

    return train_loader, val_loader, test_loader, y_train, y_val, y_test
