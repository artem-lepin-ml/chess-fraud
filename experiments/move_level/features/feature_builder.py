"""Build extra feature vectors from DataFrame columns.

Each per-feature function has the signature ``(pd.DataFrame) -> np.ndarray``
and returns an array of shape ``(N, k)`` (usually ``k=1``).

``build_features`` collects all enabled features (via ``FeatureConfig`` flags),
concatenates them, and returns ``(N, total_dim)`` or ``None``.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable

import numpy as np
import pandas as pd

from .feature_config import FeatureConfig


# ---------------------------------------------------------------------------
# Per-feature functions
# ---------------------------------------------------------------------------

def feat_eval_delta(df: pd.DataFrame) -> np.ndarray:
    """Centipawn delta: eval_after - eval_before."""
    return (df["eval_after"] - df["eval_before"]).to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_maia2_win_prob(df: pd.DataFrame) -> np.ndarray:
    return df["maia2_win_prob_2050"].to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_maia2_move_prob(df: pd.DataFrame) -> np.ndarray:
    return df["maia2_move_prob_nearest"].to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_allie_win_prob(df: pd.DataFrame) -> np.ndarray:
    return df["allie_win_prob_2500"].to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_allie_move_prob(df: pd.DataFrame) -> np.ndarray:
    return df["allie_move_prob_nearest"].to_numpy(dtype=np.float32).reshape(-1, 1)


def feat_move_thinking_time(df: pd.DataFrame) -> np.ndarray:
    return np.log1p(df["move_thinking_time"].to_numpy(dtype=np.float32)).reshape(-1, 1)


# ---------------------------------------------------------------------------
# Registry: FeatureConfig field name  ->  (builder_fn, output_dim)
# ---------------------------------------------------------------------------

FEATURE_REGISTRY: OrderedDict[str, tuple[Callable[[pd.DataFrame], np.ndarray], int]] = OrderedDict([
    ("eval_delta",         (feat_eval_delta, 1)),
    ("maia2_win_prob",     (feat_maia2_win_prob, 1)),
    ("maia2_move_prob",    (feat_maia2_move_prob, 1)),
    ("allie_win_prob",     (feat_allie_win_prob, 1)),
    ("allie_move_prob",    (feat_allie_move_prob, 1)),
    ("move_thinking_time", (feat_move_thinking_time, 1)),
])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, cfg: FeatureConfig) -> np.ndarray | None:
    """Build feature matrix from *df* based on enabled flags in *cfg*.

    Returns ``(N, total_feature_dim)`` float32 array, or ``None`` if no
    features are enabled.
    """
    parts: list[np.ndarray] = []
    for field_name, (func, _dim) in FEATURE_REGISTRY.items():
        if getattr(cfg, field_name, False):
            parts.append(func(df))
    if not parts:
        return None
    return np.concatenate(parts, axis=1).astype(np.float32)


def feature_dim(cfg: FeatureConfig) -> int:
    """Total dimension of enabled features (no data needed)."""
    total = 0
    for field_name, (_func, dim) in FEATURE_REGISTRY.items():
        if getattr(cfg, field_name, False):
            total += dim
    return total
