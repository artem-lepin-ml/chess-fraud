#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Augment a move-chain dataset with Maia2 top-1 moves and resulting successor FENs for multiple assumed self ratings.

For each row in the input CSV (one position per row, must include `fen_before` and `opponent_elo`), the script runs
Maia2 inference several times with different constant `elo_self` values. For each rating r it adds:

- `move_maia2_<r>`: Maia2 top-1 policy move (highest probability) from `fen_before` given (`elo_self`=r, `elo_oppo`=`opponent_elo`).
- `fen_maia2_<r>`: the successor FEN obtained by applying `move_maia2_<r>` (UCI) to `fen_before`.
- `maia2_move_probs_<r>`: full move probability dictionary produced by Maia2 (for debugging / analysis).
- `maia2_win_probs_<r>`: Maia2 win probability for the position.

Output is written as a CSV with the original columns plus the added per-rating columns.
"""


from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional

import chess
import pandas as pd
from omegaconf import OmegaConf

# pipelines/<stage>/<script>.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from external_models.maia2.maia2 import inference, model


def log(msg: str) -> None:
    print(f"[get_maia2_move] {msg}", flush=True)


def read_cfg(path: str) -> dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True) # type: ignore[return-value]


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def _require_no_nulls(df: pd.DataFrame, col: str) -> None:
    if df[col].isna().any():
        idx = df.index[df[col].isna()].tolist()[:10]
        raise ValueError(f"Column '{col}' contains nulls (showing up to 10 row indices): {idx}")


def _build_ratings_list(ratings_cfg: Dict[str, Any]) -> List[int]:
    values = ratings_cfg.get("values", None)
    if values is not None:
        vals = [int(x) for x in values]
        if not vals:
            raise ValueError("ratings.values is provided but empty")
        return vals

    lo = int(ratings_cfg["lo"])
    hi = int(ratings_cfg["hi"])
    step = int(ratings_cfg["step"])
    if step <= 0:
        raise ValueError("ratings.step must be > 0")
    if lo > hi:
        raise ValueError(f"ratings.lo must be <= ratings.hi, got lo={lo}, hi={hi}")

    out: List[int] = []
    r = lo
    while r <= hi:
        out.append(r)
        r += step
    if not out:
        raise ValueError("No ratings produced from lo/hi/step")
    return out


def _top1_from_move_probs(move_probs_col: List[Dict[str, float]]) -> List[str]:
    out: List[str] = []
    for mp in move_probs_col:
        highest_prob_move = max(mp, key=mp.get)
        out.append(highest_prob_move)
    return out


def _apply_uci_moves_to_fens(fens: List[str], uci_moves: List[str]) -> List[str]:
    if len(fens) != len(uci_moves):
        raise ValueError(f"fens and uci_moves length mismatch: {len(fens)} vs {len(uci_moves)}")

    out: List[str] = []
    for i, (fen, mv) in enumerate(zip(fens, uci_moves)):
        try:
            board = chess.Board(fen)
        except Exception as e:
            raise ValueError(f"Invalid FEN at row {i}: {fen}") from e

        try:
            board.push_uci(str(mv))
        except Exception as e:
            raise ValueError(f"Invalid UCI move at row {i}: move='{mv}' fen='{fen}'") from e

        out.append(board.fen())
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = read_cfg(args.config)

    input_csv = cfg["input_csv"]
    output_csv = cfg["output_csv"]
    if not input_csv or not output_csv:
        raise ValueError("Config must contain non-empty 'input_csv' and 'output_csv'")

    cols_cfg = cfg["columns"]
    fen_col = str(cols_cfg["fen_col"])
    move_col = str(cols_cfg["move_col"])
    opponent_elo_col = str(cols_cfg["opponent_elo_col"])

    ratings_cfg = cfg["ratings"]
    ratings = _build_ratings_list(ratings_cfg)
    log(f"Ratings to process: {ratings}")

    maia_cfg = cfg["maia2"]
    maia_type = str(maia_cfg["type"])
    maia_device = str(maia_cfg["device"])
    save_root = maia_cfg["save_root"]
    batch_size = int(maia_cfg["batch_size"])
    num_workers = int(maia_cfg["num_workers"])
    verbose = bool(maia_cfg["verbose"])

    # Model load
    maia2_model = model.from_pretrained(type=maia_type, device=maia_device, save_root=save_root)

    df = pd.read_csv(input_csv)

    _require_columns(df, [fen_col, move_col, opponent_elo_col])
    _require_no_nulls(df, fen_col)
    _require_no_nulls(df, move_col)
    _require_no_nulls(df, opponent_elo_col)

    opponent_elo = df[opponent_elo_col].astype(float)

    for r in ratings:
        log(f"Running Maia2 for elo_self={r}")

        # Maia2 TestDataset expects 4 columns in a fixed order: [fen, _, elo_self, elo_oppo]
        # We provide [fen_before, move, elo_self, opponent_elo]. The 'move' is a placeholder here.
        df_in = pd.DataFrame(
            {
                "fen": df[fen_col].astype(str),
                "move": None,
                "elo_self": float(r),
                "elo_oppo": opponent_elo,
            }
        )

        df_out, _ = inference.inference_batch(
            df_in,
            maia2_model,
            verbose=1 if verbose else 0,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        if "move_probs" not in df_out.columns:
            raise ValueError("Maia2 inference did not produce 'move_probs' column")
        if "win_probs" not in df_out.columns:
            raise ValueError("Maia2 inference did not produce 'win_probs' column")

        move_probs_list = df_out["move_probs"].tolist()
        win_probs_list = df_out["win_probs"].tolist()

        df[f"maia2_move_probs_{r}"] = move_probs_list
        df[f"maia2_win_probs_{r}"] = win_probs_list

        top1_moves = _top1_from_move_probs(move_probs_list)
        df[f"move_maia2_{r}"] = top1_moves

        fen_before_list = df[fen_col].astype(str).tolist()
        fen_after_list = _apply_uci_moves_to_fens(fen_before_list, top1_moves)
        df[f"fen_maia2_{r}"] = fen_after_list

    if len(df_out) != len(df):
        raise ValueError(f"Maia2 output rows != input rows: {len(df_out)} vs {len(df)}")

    df.to_csv(output_csv, index=False)
    log(f"Done. Wrote: {output_csv}")


if __name__ == "__main__":
    main()
