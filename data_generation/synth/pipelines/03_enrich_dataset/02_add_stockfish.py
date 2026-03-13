#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enrich chains with Stockfish evaluations and best-move rollouts (multiple depths).

This script reads a chains CSV with per-halfmove rows (must contain a FEN column),
runs Stockfish analysis for each configured depth (with shared multipv/time_limit),
and writes a new CSV with extra columns for every depth:

1) eval_cp_{depth}
   - White-perspective centipawn evaluation of the position.
   - If Stockfish returns mate (engineEval_mate != null), eval_cp is set to +/- mate_eval
     depending on which side is mating (White => +mate_eval, Black => -mate_eval).

2) move_stockfish_{depth}
   - Top-1 Stockfish move (best PV move).

3) fen_stockfish_{depth}
   - FEN after applying move_stockfish_{depth} to the original position.

Guaranteed behavior:
- required columns must exist and be non-null
- FENs must be valid
- engine outputs must have the same number of rows as input
- best move must exist and be legal in the given position
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import chess
import pandas as pd
from omegaconf import OmegaConf

# pipelines/<stage>/<script>.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import utils.stockfish_utils as stockfish_utils


def log(msg: str) -> None:
    print(f"[add_stockfish_eval] {msg}", flush=True)


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


def _white_perspective_eval_cp_from_cp(fen: str, cp_pov_turn: int) -> int:
    """
    stockfish_utils returns cp from score.pov(board.turn): positive means good for side to move.
    Convert to White perspective: positive means good for White.
    """
    board = chess.Board(fen)
    return cp_pov_turn if board.turn == chess.WHITE else -cp_pov_turn


def _white_perspective_eval_cp_from_mate(fen: str, mate_pov_turn: int, mate_eval: int) -> int:
    """
    mate_pov_turn sign is relative to side-to-move (pov(board.turn)).
    Map mate to +/- mate_eval in White perspective:
      +mate_eval if White is mating, -mate_eval if Black is mating.
    """
    board = chess.Board(fen)

    # With pov(board.turn):
    # - if mate_pov_turn > 0 => side to move is mating
    # - if mate_pov_turn < 0 => side to move is getting mated
    if board.turn == chess.WHITE:
        white_is_mating = mate_pov_turn > 0
    else:
        # side to move is Black
        white_is_mating = mate_pov_turn < 0

    return mate_eval if white_is_mating else -mate_eval


def _compute_eval_cp(fen: str, cp: Optional[int], mate: Optional[int], mate_eval: int) -> int:
    if (cp is None) == (mate is None):
        raise ValueError(f"Expected exactly one of (cp, mate) to be non-null. Got cp={cp}, mate={mate}")
    if cp is not None:
        return _white_perspective_eval_cp_from_cp(fen, cp)
    assert mate is not None
    return _white_perspective_eval_cp_from_mate(fen, mate, mate_eval)


def _best_move_and_fen_after(fen: str, best_move_uci: str) -> Tuple[str, str]:
    board = chess.Board(fen)
    try:
        mv = chess.Move.from_uci(best_move_uci)
    except Exception as e:
        raise ValueError(f"Invalid UCI move '{best_move_uci}' for fen='{fen}': {e}") from e

    if mv not in board.legal_moves:
        raise ValueError(f"Best move is not legal: move='{best_move_uci}' fen='{fen}'")

    board.push(mv)
    return best_move_uci, board.fen()


# -----------------------------
# Refactor: main helpers
# -----------------------------

def _load_input(input_csv: str, fen_col: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"Empty input CSV: {input_csv}")

    _require_columns(df, [fen_col])
    _require_no_nulls(df, fen_col)
    return df


def _run_stockfish(
    df: pd.DataFrame,
    fen_col: str,
    *,
    engine_path: str,
    depth: int,
    time_limit: float,
    multipv: int,
    num_engines: int,
) -> pd.DataFrame:
    df_eval = df[[fen_col]].copy()

    df_eval = asyncio.run(
        stockfish_utils.evaluate_dataframe_with_engine_pool(
            df=df_eval,
            depth=depth,
            multipv=multipv,
            num_engines=num_engines,
            fen_column=fen_col,
            engine_path=str(engine_path),
            time_limit=float(time_limit),
        )
    )

    cp_col = "engineEval_cp"
    mate_col = "engineEval_mate"
    top_col = f"top{multipv}"

    _require_columns(df_eval, [cp_col, mate_col, top_col])

    if len(df_eval) != len(df):
        raise ValueError(f"Stockfish output rows != input rows: {len(df_eval)} vs {len(df)}")

    return df_eval


def _normalize_optional_int(x: Any) -> Optional[int]:
    if pd.isna(x):
        return None
    return int(x)


def _extract_best_move_uci(top_list: Any, *, multipv: int, row_idx: int) -> str:
    if not isinstance(top_list, list):
        raise ValueError(f"Expected top{multipv} to be a list at row {row_idx}, got: {type(top_list)}")
    if not top_list:
        raise ValueError(f"Empty top{multipv} list at row {row_idx}")

    entry0 = top_list[0]
    if not isinstance(entry0, dict):
        raise ValueError(f"Expected top{multipv}[0] to be a dict at row {row_idx}, got: {type(entry0)}")

    mv = entry0.get("Move")
    if not mv:
        raise ValueError(f"Missing Move in top{multipv}[0] at row {row_idx}")

    return str(mv)


def _build_outputs_for_depth(
    df: pd.DataFrame,
    df_eval: pd.DataFrame,
    *,
    fen_col: str,
    depth: int,
    multipv: int,
    mate_eval: int,
) -> None:
    cp_col = "engineEval_cp"
    mate_col = "engineEval_mate"
    top_col = f"top{multipv}"

    eval_cp_out: List[int] = []
    move_out: List[str] = []
    fen_after_out: List[str] = []

    for i, (fen, cp_raw, mate_raw, top_list) in enumerate(
        zip(df[fen_col].astype(str), df_eval[cp_col], df_eval[mate_col], df_eval[top_col])
    ):
        cp_val = _normalize_optional_int(cp_raw)
        mate_val = _normalize_optional_int(mate_raw)

        ecp = _compute_eval_cp(fen, cp_val, mate_val, mate_eval)
        eval_cp_out.append(ecp)

        best_move = _extract_best_move_uci(top_list, multipv=multipv, row_idx=i)
        mv_uci, fen_after = _best_move_and_fen_after(fen, best_move)

        move_out.append(mv_uci)
        fen_after_out.append(fen_after)

    eval_col = f"eval_cp_{depth}"
    move_col = f"move_stockfish_{depth}"
    fen_after_col = f"fen_stockfish_{depth}"

    if eval_col in df.columns or move_col in df.columns or fen_after_col in df.columns:
        raise ValueError(
            f"Output columns already exist for depth={depth}: "
            f"{eval_col in df.columns=}, {move_col in df.columns=}, {fen_after_col in df.columns=}"
        )

    if len(eval_cp_out) != len(df) or len(move_out) != len(df) or len(fen_after_out) != len(df):
        raise ValueError(
            f"Length mismatch for depth={depth}: "
            f"eval={len(eval_cp_out)} move={len(move_out)} fen={len(fen_after_out)} vs n={len(df)}"
        )

    df[eval_col] = eval_cp_out
    df[move_col] = move_out
    df[fen_after_col] = fen_after_out


def _parse_depths(x: Any) -> List[int]:
    if not isinstance(x, list) or not x:
        raise ValueError(f"stockfish.depths must be a non-empty list, got: {type(x)}")
    depths = [int(d) for d in x]
    if any(d <= 0 for d in depths):
        raise ValueError(f"stockfish.depths must contain only positive ints, got: {depths}")
    # keep order, but ensure no duplicates
    seen = set()
    out: List[int] = []
    for d in depths:
        if d in seen:
            raise ValueError(f"Duplicate depth in stockfish.depths: {d}")
        seen.add(d)
        out.append(d)
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

    sf_cfg = cfg["stockfish"]
    engine_path = str(sf_cfg["engine_path"])
    depths = _parse_depths(sf_cfg["depths"])
    time_limit = float(sf_cfg["time_limit"])
    multipv = int(sf_cfg["multipv"])
    num_engines = int(sf_cfg["num_engines"])
    mate_eval = int(sf_cfg["mate_eval"])

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_input(input_csv, fen_col)

    log(f"Input rows: {len(df)}")
    log(f"FEN source: {fen_col}")
    log(f"Stockfish engine_path={engine_path}")
    log(f"Params: depths={depths} time_limit={time_limit} multipv={multipv} num_engines={num_engines} mate_eval={mate_eval}")

    for depth in depths:
        log(f"depth={depth}: start")
        df_eval = _run_stockfish(
            df,
            fen_col,
            engine_path=engine_path,
            depth=depth,
            time_limit=time_limit,
            multipv=multipv,
            num_engines=num_engines,
        )

        _build_outputs_for_depth(
            df,
            df_eval,
            fen_col=fen_col,
            depth=depth,
            multipv=multipv,
            mate_eval=mate_eval,
        )

    df.to_csv(out_path, index=False)
    log(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
