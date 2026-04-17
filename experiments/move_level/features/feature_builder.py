"""Build extra feature vectors from DataFrame columns.

Each per-feature function has the signature
``(df: pd.DataFrame, move_col: str) -> np.ndarray`` and returns an array of
shape ``(N, k)`` (usually ``k=1``).

Position-level features (e.g. ``player_elo``, ``eval_delta``) ignore
``move_col``; per-move features (``sf15_match``) use it to identify which
move is being scored.

``build_features`` collects all enabled features, optionally standardizes
them using precomputed statistics, concatenates, and returns ``(N, total_dim)``
or ``None``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import pandas as pd

from .feature_config import FeatureConfig


# ---------------------------------------------------------------------------
# Raw column names required for each feature (used for NaN assertion and for
# computing stats on raw values before standardization).
# ---------------------------------------------------------------------------

FEATURE_RAW_COLS: dict[str, list[str]] = {
    "eval_delta":         ["eval_after", "eval_before"],
    "maia2_win_prob":     ["maia2_win_prob_2050"],
    "maia2_move_prob":    ["maia2_move_prob_nearest"],
    "allie_win_prob":     ["allie_win_prob_2500"],
    "allie_move_prob":    ["allie_move_prob_nearest"],
    "sf15_match":         ["move_stockfish_15"],   # + move_col (provided at build time)
    "player_elo":         ["player_elo"],
    "opponent_elo":       ["opponent_elo"],
}


# Features whose raw scalar is standardized (z-score). Binary / already-normalised
# features stay as-is.
FEATURE_STATS_FIELDS = {"player_elo", "opponent_elo"}


# ---------------------------------------------------------------------------
# Per-feature functions
# ---------------------------------------------------------------------------

def feat_eval_delta(df: pd.DataFrame, move_col: str) -> np.ndarray:
    """Centipawn delta: eval_after - eval_before."""
    return (df["eval_after"] - df["eval_before"]).to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_maia2_win_prob(df: pd.DataFrame, move_col: str) -> np.ndarray:
    return df["maia2_win_prob_2050"].to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_maia2_move_prob(df: pd.DataFrame, move_col: str) -> np.ndarray:
    return df["maia2_move_prob_nearest"].to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_allie_win_prob(df: pd.DataFrame, move_col: str) -> np.ndarray:
    return df["allie_win_prob_2500"].to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_allie_move_prob(df: pd.DataFrame, move_col: str) -> np.ndarray:
    return df["allie_move_prob_nearest"].to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_sf15_match(df: pd.DataFrame, move_col: str) -> np.ndarray:
    """Per-move: 1.0 iff ``df[move_col] == df["move_stockfish_15"]``."""
    return (df[move_col].to_numpy() == df["move_stockfish_15"].to_numpy()).astype(np.float32).reshape(-1, 1)


def feat_player_elo(df: pd.DataFrame, move_col: str) -> np.ndarray:
    return df["player_elo"].to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_opponent_elo(df: pd.DataFrame, move_col: str) -> np.ndarray:
    return df["opponent_elo"].to_numpy(dtype=np.float32).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Registry: FeatureConfig field name  ->  (builder_fn, output_dim)
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: OrderedDict[str, tuple[Callable[[pd.DataFrame, str], np.ndarray], int]] = OrderedDict([
    ("eval_delta",         (feat_eval_delta, 1)),
    ("maia2_win_prob",     (feat_maia2_win_prob, 1)),
    ("maia2_move_prob",    (feat_maia2_move_prob, 1)),
    ("allie_win_prob",     (feat_allie_win_prob, 1)),
    ("allie_move_prob",    (feat_allie_move_prob, 1)),
    ("sf15_match",         (feat_sf15_match, 1)),
    ("player_elo",         (feat_player_elo, 1)),
    ("opponent_elo",       (feat_opponent_elo, 1)),
])


# ---------------------------------------------------------------------------
# NaN sanity check
# ---------------------------------------------------------------------------

def _assert_no_nan(df: pd.DataFrame, cfg: FeatureConfig, move_col: str) -> None:
    cols: set[str] = set()
    for field_name in FEATURE_REGISTRY:
        if not getattr(cfg, field_name, False):
            continue
        for c in FEATURE_RAW_COLS.get(field_name, []):
            cols.add(c)
    if getattr(cfg, "sf15_match", False):
        cols.add(move_col)
    for c in cols:
        if c not in df.columns:
            raise KeyError(f"Missing required column for features: {c!r}")
        if df[c].isna().any():
            n_nan = int(df[c].isna().sum())
            raise ValueError(
                f"Column {c!r} has {n_nan} NaN values; aborting (no imputation)."
            )


# ---------------------------------------------------------------------------
# Stats for standardization
# ---------------------------------------------------------------------------

def _raw_feature_array(field_name: str, df: pd.DataFrame) -> np.ndarray:
    """Return the raw 1-D array used for z-score fitting of a feature."""
    if field_name == "player_elo":
        return df["player_elo"].to_numpy(dtype=np.float64)
    if field_name == "opponent_elo":
        return df["opponent_elo"].to_numpy(dtype=np.float64)
    raise ValueError(f"No stats fitter for feature {field_name}")


def compute_feature_stats(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    row_idx: np.ndarray,
) -> dict[str, dict[str, float]]:
    """Compute mean/std on ``df.iloc[row_idx]`` for every enabled feature that
    lives in ``FEATURE_STATS_FIELDS``.

    Returns ``{feature_name: {"mean": float, "std": float}}``.
    """
    sub = df.iloc[row_idx]
    stats: dict[str, dict[str, float]] = {}
    for field_name in FEATURE_STATS_FIELDS:
        if not getattr(cfg, field_name, False):
            continue
        arr = _raw_feature_array(field_name, sub)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr))
        if sigma < 1e-12:
            sigma = 1.0
        stats[field_name] = {"mean": mu, "std": sigma}
    return stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(
    df: pd.DataFrame,
    cfg: FeatureConfig,
    move_col: str = "move_uci",
    stats: dict[str, dict[str, float]] | None = None,
) -> np.ndarray | None:
    """Build feature matrix from *df* based on enabled flags in *cfg*.

    Args:
        df: source DataFrame with all required raw columns.
        cfg: which features are enabled.
        move_col: which column contains the move being scored. Position-level
            features ignore this; per-move features use it.
        stats: if given, z-score features listed in ``FEATURE_STATS_FIELDS``
            using the supplied ``{feature: {mean, std}}``. If ``None``,
            standardization is skipped.

    Returns:
        ``(N, total_feature_dim)`` float32 array, or ``None`` if no
        features are enabled.
    """
    _assert_no_nan(df, cfg, move_col)

    parts: list[np.ndarray] = []
    for field_name, (func, _dim) in FEATURE_REGISTRY.items():
        if not getattr(cfg, field_name, False):
            continue
        arr = func(df, move_col).astype(np.float32, copy=False)
        if (
            stats is not None
            and getattr(cfg, "standardize", False)
            and field_name in FEATURE_STATS_FIELDS
            and field_name in stats
        ):
            mu = np.float32(stats[field_name]["mean"])
            sigma = np.float32(stats[field_name]["std"])
            arr = (arr - mu) / sigma
        parts.append(arr)
    if not parts:
        return None
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def feature_dim(cfg: FeatureConfig) -> int:
    """Total dimension of enabled features (no data needed)."""
    total = 0
    for field_name, (_func, dim) in FEATURE_REGISTRY.items():
        if getattr(cfg, field_name, False):
            total += dim
    return total


def enabled_feature_names(cfg: FeatureConfig) -> list[str]:
    """Ordered list of enabled feature field names (same order as concatenation)."""
    return [n for n in FEATURE_REGISTRY if getattr(cfg, n, False)]
