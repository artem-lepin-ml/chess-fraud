#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Maia2 embeddings for all columns that start with "fen_".

Optimizations (same spirit as allie script):
- We do NOT die on missing FENs in "normal" places: any null/empty fen_* cell is skipped,
  and the output embedding row for that (row, col) stays NaN.
- We deduplicate (hash) positions across ALL fen_* columns:
  if the same (fen, elo_oppo) appears multiple times (often across different model columns),
  we compute its embedding once and copy it into all corresponding output slots.

Outputs:
- output_npz: a single .npz file with arrays keyed by fen_* column names.
  Each array has shape (N, D) and dtype float16/float32 (configurable).
  Also contains:
    - __fen_cols__: list of fen_* column names in the order used.

Guaranteed behavior (kept, but relaxed where sensible):
- required columns must exist
- fen_before_col must be non-null (core input position column)
- elo_oppo_col may contain nulls -> coerced to 0 with a warning
- inference must return embeddings with row count == unique evaluated positions count
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

# pipelines/<stage>/<script>.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from external_models.maia2.maia2 import inference, model


def log(msg: str) -> None:
    print(f"[compute_maia2_embs] {msg}", flush=True)


def read_cfg(path: str) -> dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def _require_no_nulls(df: pd.DataFrame, col: str) -> None:
    if df[col].isna().any():
        idx = df.index[df[col].isna()].tolist()[:10]
        raise ValueError(f"Column '{col}' contains nulls (showing up to 10 row indices): {idx}")


def _parse_dtype(s: str) -> np.dtype:
    s_norm = str(s).strip().lower()
    if s_norm == "float16":
        return np.float16
    if s_norm == "float32":
        return np.float32
    raise ValueError(f"Unsupported embeddings_dtype='{s}'. Use float16/float32.")


def _is_valid_fen_cell(x: object) -> bool:
    if x is None:
        return False
    # pandas NA/NaN
    if pd.isna(x):
        return False
    s = str(x).strip()
    if not s:
        return False
    # optional: treat literal "None"/"nan" as missing
    if s.lower() in {"none", "nan"}:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = read_cfg(args.config)

    input_csv = cfg["input_csv"]
    output_npz = cfg["output_npz"]
    embeddings_dtype = _parse_dtype(cfg["embeddings_dtype"])

    cols_cfg = cfg["columns"]
    fen_before_col = str(cols_cfg["fen_before_col"])
    fen_prefix = str(cols_cfg["fen_prefix"])
    elo_oppo_col = str(cols_cfg["elo_oppo_col"])

    detector_cfg = cfg["detector"]
    elo_self = float(detector_cfg["elo_self"])

    maia_cfg = cfg["maia2"]
    maia_type = str(maia_cfg["type"])
    maia_device = str(maia_cfg["device"])
    save_root = maia_cfg.get("save_root", None)
    batch_size = int(maia_cfg["batch_size"])
    num_workers = int(maia_cfg["num_workers"])
    verbose = bool(maia_cfg["verbose"])

    out_npz_path = Path(output_npz)
    out_npz_path.parent.mkdir(parents=True, exist_ok=True)

    # Model load
    maia2_model = model.from_pretrained(type=maia_type, device=maia_device, save_root=save_root)

    df = pd.read_csv(input_csv)

    _require_columns(df, [fen_before_col, elo_oppo_col])
    _require_no_nulls(df, fen_before_col)  # core column should be present everywhere

    n_rows = len(df)
    log(f"Input rows: {n_rows}")
    log(f"Detector elo_self for embeddings: {elo_self}")
    log(f"Elo oppo column: {elo_oppo_col}")
    log(f"Maia2 type={maia_type} device={maia_device} batch_size={batch_size} num_workers={num_workers}")
    log(f"Embeddings dtype: {embeddings_dtype}")
    if save_root:
        log(f"Checkpoint(save_root): {save_root}")

    # FEN columns to embed
    fen_cols = [c for c in df.columns if str(c).startswith(fen_prefix)]
    if not fen_cols:
        raise ValueError(f"No columns found with prefix '{fen_prefix}'")

    log(f"FEN columns to embed ({len(fen_cols)}): {fen_cols}")

    # Opponent ELO: allow missing -> coerce to 0 with a warning
    elo_oppo_raw = df[elo_oppo_col]
    bad_elo_mask = elo_oppo_raw.isna()
    bad_elo_cnt = int(bad_elo_mask.sum())
    if bad_elo_cnt > 0:
        log(f"WARNING: '{elo_oppo_col}' has {bad_elo_cnt} nulls; coerced to 0.")
    elo_oppo = elo_oppo_raw.fillna(0).astype(float).to_numpy()

    # Build mapping: unique_key -> list of (fen_col, row_index)
    # key must include elo_oppo because Maia2 embedding input depends on it.
    # elo_self is constant (detector), so it does not need to be in the key.
    key_to_targets: Dict[Tuple[str, float], List[Tuple[str, int]]] = {}
    total_cells = n_rows * len(fen_cols)
    used_cells = 0

    # Iterate columns to keep code small; still O(N * #cols).
    for fen_col in fen_cols:
        col_vals = df[fen_col].to_numpy(dtype=object)
        for i in range(n_rows):
            v = col_vals[i]
            if not _is_valid_fen_cell(v):
                continue
            fen_str = str(v).strip()
            key = (fen_str, float(elo_oppo[i]))
            key_to_targets.setdefault(key, []).append((fen_col, i))
            used_cells += 1

    if not key_to_targets:
        raise ValueError(
            f"All fen_* columns are empty/null after filtering; nothing to embed. "
            f"(rows={n_rows}, fen_cols={len(fen_cols)})"
        )

    log(f"Total fen cells: {total_cells}, non-null used: {used_cells}")
    log(f"Unique (fen, elo_oppo) positions after dedup: {len(key_to_targets)}")

    # Build unique inference dataframe in a stable order
    # Maia2 TestDataset expects: [fen, move, elo_self, elo_oppo]
    keys = list(key_to_targets.keys())
    df_in = pd.DataFrame(
        {
            "fen": [k[0] for k in keys],
            "move": [None] * len(keys),          # placeholder
            "elo_self": float(elo_self),         # broadcast scalar
            "elo_oppo": [k[1] for k in keys],
        }
    )

    log("Running Maia2 embeddings on unique positions...")
    embs_torch = inference.inference_embs_batch(
        df_in,
        maia2_model,
        verbose=1 if verbose else 0,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    if embs_torch.shape[0] != len(df_in):
        raise ValueError(
            f"Embeddings rows != input rows: {embs_torch.shape[0]} vs {len(df_in)}"
        )

    # Convert once
    embs_np_unique = embs_torch.detach().cpu().numpy().astype(embeddings_dtype, copy=False)
    d = int(embs_np_unique.shape[1])
    log(f"Embeddings computed: shape={embs_np_unique.shape} dtype={embs_np_unique.dtype}")

    # Allocate outputs: per fen_col (N, D), filled with NaNs
    out: Dict[str, np.ndarray] = {
        c: np.full((n_rows, d), np.nan, dtype=embeddings_dtype) for c in fen_cols
    }

    # Scatter unique embeddings into all targets
    for j, key in enumerate(keys):
        vec = embs_np_unique[j]
        for fen_col, row_i in key_to_targets[key]:
            out[fen_col][row_i, :] = vec

    # Store column order inside the same file
    out["__fen_cols__"] = np.array(fen_cols, dtype=object)

    # Save
    np.savez_compressed(out_npz_path, **out)
    log(f"Wrote: {out_npz_path} keys={len(out)} (including __fen_cols__)")

    # Quick stats (optional, cheap)
    for fen_col in fen_cols:
        filled = int(np.isfinite(out[fen_col]).all(axis=1).sum())
        log(f"{fen_col}: filled rows = {filled} / {n_rows}")


if __name__ == "__main__":
    main()
