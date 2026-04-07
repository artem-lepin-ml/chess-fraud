"""Load ChessFraud-Synth, split, build paired datasets & DataLoaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from ..configs.dataset_config import DatasetConfig
from ..features.feature_config import FeatureConfig
from ..features.feature_builder import build_features, feature_dim


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RowPairDataset(Dataset):
    """For *N* rows yields *2N* samples: ``(human_emb[i], 0)`` and
    ``(cheat_emb[i], 1)``.

    If *extra_features* is provided it is concatenated to the embedding.
    """

    def __init__(
        self,
        human_emb: np.ndarray,
        cheat_emb: np.ndarray,
        row_idx: np.ndarray,
        extra_features: np.ndarray | None = None,
    ) -> None:
        self.human_emb = human_emb
        self.cheat_emb = cheat_emb
        self.row_idx = row_idx
        self.extra_features = extra_features
        self._n = len(row_idx)

    def __len__(self) -> int:
        return 2 * self._n

    def __getitem__(self, idx: int):
        if idx < self._n:
            row = self.row_idx[idx]
            x = self.human_emb[row].astype(np.float32)
            y = np.float32(0.0)
        else:
            row = self.row_idx[idx - self._n]
            x = self.cheat_emb[row].astype(np.float32)
            y = np.float32(1.0)
        if self.extra_features is not None:
            feat = self.extra_features[row].astype(np.float32)
            x = np.concatenate([x, feat])
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
    bin_idx: np.ndarray          # per-row (filtered) rating bin index
    split_labels: np.ndarray     # per-row "train" / "test"
    emb_human: np.ndarray        # (N_filtered, D)
    emb_cheat: dict[str, np.ndarray]   # cheat_name -> (N_filtered, D)
    move_human: np.ndarray
    move_cheat: dict[str, np.ndarray]
    extra_features: np.ndarray | None
    input_dim: int
    bin_names: list[str]
    cheat_names: list[str]       # includes "ALL" if built


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


def _build_all_cheat_mixture(
    emb_cheat: dict[str, np.ndarray],
    move_cheat: dict[str, np.ndarray],
    bin_idx: np.ndarray,
    cheat_model_to_bin_idxs: dict[str, list[int]],
    cheat_names: list[str],
    n_rows: int,
    emb_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Row-wise random selection from allowed cheats → ``(emb_all, move_all)``."""
    rng = np.random.default_rng(seed)
    emb_all = np.empty((n_rows, emb_dim), dtype=np.float32)
    move_all = np.empty(n_rows, dtype=object)

    for i in range(n_rows):
        b = int(bin_idx[i])
        allowed = [n for n in cheat_names if b in cheat_model_to_bin_idxs[n]]
        chosen = str(rng.choice(allowed))
        emb_all[i] = emb_cheat[chosen][i]
        move_all[i] = move_cheat[chosen][i]

    return emb_all, move_all


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_synth_data(
    cfg: DatasetConfig,
    feature_cfg: FeatureConfig | None = None,
) -> SynthData:
    """Load CSV + NPZ, filter ``is_used``, split train/val/test, build
    ``"ALL"`` mixture (if enabled), compute extra features.
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

    # ---- "ALL" mixture ----
    if cfg.build_all_mixture and cheat_names:
        emb_all, move_all = _build_all_cheat_mixture(
            emb_cheat, move_cheat, bin_idx_arr,
            cfg.cheat_model_to_bin_idxs, cheat_names,
            len(df_f), emb_dim, cfg.seed,
        )
        emb_cheat["ALL"] = emb_all
        move_cheat["ALL"] = move_all
        all_cheat_names = cheat_names + ["ALL"]
    else:
        all_cheat_names = cheat_names

    # ---- extra features ----
    extra_features: np.ndarray | None = None
    extra_dim = 0
    if feature_cfg is not None:
        extra_features = build_features(df_f, feature_cfg)
        extra_dim = feature_dim(feature_cfg)

    input_dim = emb_dim + extra_dim

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
        extra_features=extra_features,
        input_dim=input_dim,
        bin_names=list(cfg.rating_bins),
        cheat_names=all_cheat_names,
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
    emb_cheat = synth_data.emb_cheat[cheat_name]

    def _make(idx: np.ndarray) -> tuple[DataLoader, np.ndarray]:
        ds = RowPairDataset(
            synth_data.emb_human, emb_cheat, idx, synth_data.extra_features
        )
        y = np.concatenate([np.zeros(len(idx)), np.ones(len(idx))]).astype(np.int64)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        return loader, y

    train_loader, y_train = _make(synth_data.train_idx)
    val_loader, y_val = _make(synth_data.val_idx)
    # test: no shuffle
    test_ds = RowPairDataset(
        synth_data.emb_human, emb_cheat, synth_data.test_idx, synth_data.extra_features
    )
    y_test = np.concatenate([
        np.zeros(len(synth_data.test_idx)),
        np.ones(len(synth_data.test_idx)),
    ]).astype(np.int64)
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader, y_train, y_val, y_test
