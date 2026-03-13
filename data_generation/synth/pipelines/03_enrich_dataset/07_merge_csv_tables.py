#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge a main move-level CSV with multiple side CSVs that contain extra (non-overlapping) columns.

Assumptions / guarantees:
- All tables have the same row order and the same number of rows.
- You can use row number as the implicit ID (row i in every table refers to the same position).
- Additionally, for every side table, (game_id, half_move) must match the main table row-wise.
- Columns that exist in the main table must NOT be duplicated from side tables.
  (Overlap is allowed but will be dropped from side tables; all remaining side columns must be unique.)

Output:
- A single CSV with main columns + all extra columns from all side tables.

Additional allowed behavior:
- Side tables may be "empty" (None) for rows that do NOT satisfy:
    half_move > 20 AND (
      (player_color == "white" AND half_move % 2 == 1) OR
      (player_color == "black" AND half_move % 2 == 0)
    )
  For those rows we force side extra columns to "None" (string) in the output.
- If a row DOES satisfy the condition, but some side extra value is missing (None/NaN), we emit a warning.
- If a row does NOT satisfy the condition, but some side extra value is non-missing, we emit a warning
  and still force it to "None" in the output.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any, Dict, List, Sequence

import pandas as pd
from omegaconf import OmegaConf


def log(msg: str) -> None:
    print(f"[merge_csv_tables] {msg}", flush=True)


def read_cfg(path: str) -> dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _require_cols(df: pd.DataFrame, cols: Sequence[str], *, name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")


def _drop_artifact_index_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop typical CSV artifact index columns like 'Unnamed: 0' if they look like a row index.
    """
    candidates = [c for c in df.columns if c.startswith("Unnamed:")]
    if not candidates:
        return df

    n = len(df)
    drop_cols: List[str] = []

    for c in candidates:
        s = df[c]
        vals = pd.to_numeric(s, errors="coerce")
        if vals.notna().all():
            as_int = vals.astype("int64")
            if (as_int == pd.Series(range(n))).all() or (as_int == pd.Series(range(1, n + 1))).all():
                drop_cols.append(c)

    if drop_cols:
        return df.drop(columns=drop_cols)
    return df


def _resolve_side_paths(cfg: Dict[str, Any]) -> List[Path]:
    side_paths = cfg.get("side_paths", None)
    if not isinstance(side_paths, list) or not side_paths:
        raise ValueError("Config must provide a non-empty 'side_paths' list")

    out: List[Path] = []
    for p in side_paths:
        if not isinstance(p, str) or not p.strip():
            raise ValueError("Each entry in side_paths must be a non-empty string")
        out.append(Path(p))
    return out


def _assert_rowwise_keys_match(
    main_df: pd.DataFrame,
    side_df: pd.DataFrame,
    *,
    name: str,
    game_id_col: str,
    half_move_col: str,
) -> None:
    _require_cols(main_df, [game_id_col, half_move_col], name="main")
    _require_cols(side_df, [game_id_col, half_move_col], name=name)

    a = main_df[[game_id_col, half_move_col]].reset_index(drop=True)
    b = side_df[[game_id_col, half_move_col]].reset_index(drop=True)

    if len(a) != len(b):
        raise ValueError(f"{name}: row count mismatch during key validation: main={len(a)} side={len(b)}")

    a_gid = a[game_id_col].astype(str)
    b_gid = b[game_id_col].astype(str)

    a_hm = pd.to_numeric(a[half_move_col], errors="coerce")
    b_hm = pd.to_numeric(b[half_move_col], errors="coerce")

    mismatch = (a_gid != b_gid) | (a_hm != b_hm)
    if mismatch.any():
        idx = int(mismatch.idxmax())
        raise ValueError(
            f"{name}: (game_id, half_move) mismatch at row {idx}: "
            f"main=({a.loc[idx, game_id_col]!r}, {a.loc[idx, half_move_col]!r}) "
            f"side=({b.loc[idx, game_id_col]!r}, {b.loc[idx, half_move_col]!r})"
        )


def _side_extra_columns(main_cols: Sequence[str], side_cols: Sequence[str], *, name: str) -> List[str]:
    main_set = set(main_cols)
    extra = [c for c in side_cols if c not in main_set]

    if len(extra) != len(set(extra)):
        raise ValueError(f"{name}: duplicate column names within side extras after filtering")

    return extra


def _build_allowed_mask(
    df: pd.DataFrame,
    *,
    half_move_col: str,
    player_color_col: str,
) -> pd.Series:
    _require_cols(df, [half_move_col, player_color_col], name="main")

    hm = pd.to_numeric(df[half_move_col], errors="coerce")
    if hm.isna().any():
        idx = int(hm.isna().idxmax())
        raise ValueError(f"main: {half_move_col} has non-numeric value at row {idx}: {df.loc[idx, half_move_col]!r}")

    pc = df[player_color_col].astype(str).str.lower()

    allowed = (hm > 20) & (
        ((pc == "white") & (hm % 2 == 1)) | ((pc == "black") & (hm % 2 == 0))
    )
    return allowed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = read_cfg(args.config)

    input_main_csv = cfg["input_main_csv"]
    output_csv = cfg["output_csv"]

    keys_cfg = cfg.get("keys", {}) or {}
    if not isinstance(keys_cfg, dict):
        raise ValueError("Config 'keys' must be a dict if provided")

    game_id_col = keys_cfg["game_id_col"]
    half_move_col = keys_cfg["half_move_col"]

    # Required for the partial/empty side-columns rule
    player_color_col = "player_color"

    side_paths = _resolve_side_paths(cfg)

    in_path = Path(input_main_csv)
    out_path = Path(output_csv)

    if not in_path.exists():
        raise FileNotFoundError(f"Main CSV not found: {in_path}")

    log(f"Reading main: {in_path}")
    main_df = pd.read_csv(in_path)
    main_df = _drop_artifact_index_cols(main_df)

    if len(main_df) == 0:
        raise ValueError("Main CSV is empty")

    _require_cols(main_df, [game_id_col, half_move_col], name="main")
    log(f"Main rows: {len(main_df):,} | cols: {len(main_df.columns)}")

    merged = main_df.copy()

    # Precompute which rows are expected to have non-empty side extras
    allowed_mask = _build_allowed_mask(merged, half_move_col=half_move_col, player_color_col=player_color_col)

    used_new_cols: set[str] = set()

    for p in side_paths:
        name = p.name
        if not p.exists():
            raise FileNotFoundError(f"Side CSV not found: {p}")

        log(f"Reading side: {p}")
        side_df = pd.read_csv(p)
        side_df = _drop_artifact_index_cols(side_df)

        if len(side_df) != len(merged):
            raise ValueError(f"{name}: row count mismatch: main={len(merged)} side={len(side_df)}")

        _assert_rowwise_keys_match(
            merged,
            side_df,
            name=name,
            game_id_col=game_id_col,
            half_move_col=half_move_col,
        )

        extra_cols = _side_extra_columns(merged.columns, side_df.columns, name=name)
        if not extra_cols:
            log(f"{name}: no extra columns to add (all columns overlap with main)")
            continue

        collision = [c for c in extra_cols if (c in used_new_cols) or (c in merged.columns)]
        if collision:
            raise ValueError(f"{name}: extra columns collide with existing merged columns: {collision}")

        # Enforce "empty rows => None" and warn on unexpected missing/present values
        extra_df = side_df[extra_cols].reset_index(drop=True).copy()

        # Warn if any expected (allowed) row has missing values in extra columns
        for c in extra_cols:
            miss_idx = extra_df.index[allowed_mask & extra_df[c].isna()]
            if len(miss_idx) > 0:
                warnings.warn(
                    f"{name}: column {c!r} has None/NaN in {len(miss_idx)} expected rows "
                    f"(first row={int(miss_idx[0])})",
                    RuntimeWarning,
                )

        # Rows that are not required to be computed may still be non-empty (allowed).
        non_allowed = ~allowed_mask
        if non_allowed.any():
            extra_df.loc[non_allowed, extra_cols] = pd.NA


        log(f"{name}: adding {len(extra_cols)} columns")
        merged = pd.concat([merged, extra_df], axis=1)
        used_new_cols.update(extra_cols)

        if len(merged.columns) != len(set(merged.columns)):
            dupes = merged.columns[merged.columns.duplicated()].tolist()
            raise ValueError(f"{name}: merge produced duplicate columns: {dupes}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"Writing output: {out_path}")
    merged.to_csv(out_path, index=False)
    log(f"Done. Output rows: {len(merged):,} | cols: {len(merged.columns)}")


if __name__ == "__main__":
    main()
