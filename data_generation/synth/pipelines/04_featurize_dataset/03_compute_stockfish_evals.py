#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Enrich a CSV that contains many fen_* columns with Stockfish evaluations.

For every column whose name starts with `fen_prefix` (default: "fen_"), we create a corresponding
`eval_*` column (prefix replaced: fen_XXX -> eval_XXX) *unless* that eval_* column already exists.

Each eval_* contains a single value: White-perspective centipawn evaluation at the configured depth.
- Stockfish cp is returned as score.pov(board.turn): positive means good for side to move.
- We convert to White perspective: positive means good for White.
- If mate is returned, we map it to +/- mate_eval in White perspective.

Row-level optimization:
- Within each row, many fen values may repeat across different fen_* columns.
  We evaluate each distinct FEN in that row only once, then copy the value to all matching columns.
- We DO NOT deduplicate across the entire table (no global cache).

Color / parity filter:
- We evaluate positions only for rows where the move belongs to the selected player color:
  - if player_color == "white": evaluate only rows with (half_move % 2 == 1)
  - else (player_color == "black"): evaluate only rows with (half_move % 2 == 0)
- For rows that do not pass this filter, newly created eval_* columns stay as NA.

Early-move empty FEN exception (requested behavior):
- If a required fen_* cell is empty/null on a row with half_move <= 20, we DO NOT raise.
  We simply skip evaluation for that (row, fen_col) and leave eval_* as NA (i.e., None).

Guaranteed behavior:
- input must be non-empty
- required columns must exist and be non-null: player_color_col, half_move_col
- for rows that are evaluated and half_move > 20, all required fen_* cells (for columns we compute)
  must be non-null/non-empty
- Stockfish output row count must match the input positions count for every batch
- every evaluated FEN must be a valid chess FEN (python-chess Board(fen) must parse)
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

import chess
import pandas as pd
from omegaconf import OmegaConf

# pipelines/<stage>/<script>.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import utils.stockfish_utils as stockfish_utils


def log(msg: str) -> None:
    print(f"[add_stockfish_manyfen_eval] {msg}", flush=True)


def read_cfg(path: str) -> dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]


def _require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in input CSV: {missing}")


def _require_no_nulls(df: pd.DataFrame, col: str) -> None:
    if df[col].isna().any():
        idx = df.index[df[col].isna()].tolist()[:10]
        raise ValueError(f"Column '{col}' contains nulls (showing up to 10 row indices): {idx}")


def _normalize_optional_int(x: Any) -> Optional[int]:
    if x is None or (isinstance(x, float) and pd.isna(x)) or pd.isna(x):
        return None
    return int(x)


def _is_empty_fen(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, float) and pd.isna(x):
        return True
    if pd.isna(x):
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False


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


def _load_input(input_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv).reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"Empty input CSV: {input_csv}")
    return df


def _parse_depth(sf_cfg: Dict[str, Any]) -> int:
    # Prefer single "depth". Allow legacy "depths: [15]" as fallback.
    if "depth" in sf_cfg:
        d = int(sf_cfg["depth"])
        if d <= 0:
            raise ValueError(f"stockfish.depth must be a positive int, got: {d}")
        return d

    if "depths" in sf_cfg:
        x = sf_cfg["depths"]
        if not isinstance(x, list) or len(x) != 1:
            raise ValueError(f"stockfish.depths must be a list with exactly one element, got: {x}")
        d = int(x[0])
        if d <= 0:
            raise ValueError(f"stockfish.depths[0] must be positive, got: {d}")
        return d

    raise ValueError("Config must contain 'stockfish.depth' (preferred) or 'stockfish.depths: [..]' (legacy)")


def _find_fen_columns(df: pd.DataFrame, fen_prefix: str) -> List[str]:
    fen_cols = [c for c in df.columns if isinstance(c, str) and c.startswith(fen_prefix)]
    if not fen_cols:
        raise ValueError(f"No columns found with prefix '{fen_prefix}'")
    return fen_cols


def _eval_col_for_fen_col(fen_col: str, fen_prefix: str) -> str:
    if not fen_col.startswith(fen_prefix):
        raise ValueError(f"fen_col does not start with fen_prefix: {fen_col} vs {fen_prefix}")
    return "eval_" + fen_col[len(fen_prefix) :]


def _normalize_player_color(x: Any) -> str:
    s = str(x).strip().lower()
    if s in ("white", "w"):
        return "white"
    if s in ("black", "b"):
        return "black"
    raise ValueError(f"Unexpected player_color value: {x!r} (expected 'white'/'black' or 'w'/'b')")


def _build_player_move_mask(df: pd.DataFrame, *, player_color_col: str, half_move_col: str) -> pd.Series:
    _require_columns(df, [player_color_col, half_move_col])
    _require_no_nulls(df, player_color_col)
    _require_no_nulls(df, half_move_col)

    colors = df[player_color_col].map(_normalize_player_color)
    halfmoves = df[half_move_col].astype(int)

    is_white_move = (colors == "white") & (halfmoves % 2 == 1)
    is_black_move = (colors == "black") & (halfmoves % 2 == 0)
    return is_white_move | is_black_move


def _run_stockfish_eval_for_fens(
    fens: List[str],
    *,
    engine_path: str,
    depth: int,
    time_limit: float,
    multipv: int,
    num_engines: int,
    mate_eval: int,
) -> List[int]:
    if not fens:
        return []

    df_eval_in = pd.DataFrame({"fen": fens})

    df_eval_out = asyncio.run(
        stockfish_utils.evaluate_dataframe_with_engine_pool(
            df=df_eval_in,
            depth=depth,
            multipv=multipv,
            num_engines=num_engines,
            fen_column="fen",
            engine_path=str(engine_path),
            time_limit=float(time_limit),
        )
    )

    cp_col = "engineEval_cp"
    mate_col = "engineEval_mate"
    _require_columns(df_eval_out, [cp_col, mate_col])

    if len(df_eval_out) != len(fens):
        raise ValueError(f"Stockfish output rows != input rows: {len(df_eval_out)} vs {len(fens)}")

    out: List[int] = []
    for fen, cp_raw, mate_raw in zip(df_eval_out["fen"].astype(str), df_eval_out[cp_col], df_eval_out[mate_col]):
        cp_val = _normalize_optional_int(cp_raw)
        mate_val = _normalize_optional_int(mate_raw)

        # also validates FEN via chess.Board inside _compute_eval_cp
        ecp = _compute_eval_cp(fen, cp_val, mate_val, mate_eval)
        out.append(ecp)

    if len(out) != len(fens):
        raise ValueError(f"Internal error: eval list length mismatch: {len(out)} vs {len(fens)}")

    return out


def _iter_row_batches(
    df: pd.DataFrame,
    *,
    row_indices: Iterable[int],
    fen_cols_to_compute: List[str],
    fen_prefix: str,
    half_move_col: str,
    max_positions_per_batch: int,
) -> Iterable[Tuple[List[str], List[Tuple[int, str, int]]]]:
    """
    Yields batches:
      - batch_fens: list[str] (NOT deduped across rows; only within each row)
      - assignments: list of (row_idx, eval_col, pos_idx_in_batch_fens)

    Note:
    - If a fen_* cell is empty/null and half_move <= 20, we skip that (row, col) (leave eval_* as NA).
    - If half_move > 20, empty/null fen_* is an error for evaluated rows.
    """
    if max_positions_per_batch <= 0:
        raise ValueError(f"max_positions_per_batch must be positive, got: {max_positions_per_batch}")

    batch_fens: List[str] = []
    assignments: List[Tuple[int, str, int]] = []

    for row_idx in row_indices:
        half_move = int(df.loc[row_idx, half_move_col])

        # Pull fen values for this row in a stable column order
        row_vals = df.loc[row_idx, fen_cols_to_compute].tolist()

        # Build per-row unique fen list + per-column local index
        local_map: Dict[str, int] = {}
        row_unique: List[str] = []
        col_local_idx: List[int] = []

        for fen_raw in row_vals:
            if _is_empty_fen(fen_raw):
                if half_move <= 20:
                    # Requested behavior: do not crash, just keep eval_* as NA for this cell.
                    col_local_idx.append(-1)
                    continue
                raise ValueError(
                    f"Empty/null FEN in row {row_idx} (half_move={half_move}) for evaluated row. "
                    f"Either fix data or ensure this row is excluded by the color/parity filter."
                )

            fen = str(fen_raw)
            j = local_map.get(fen)
            if j is None:
                j = len(row_unique)
                local_map[fen] = j
                row_unique.append(fen)
            col_local_idx.append(j)

        if not row_unique:
            # Either all fen_* are empty (allowed only for half_move <= 20),
            # or fen_cols_to_compute is empty (handled earlier).
            continue

        # flush if adding this row exceeds the batch limit
        if batch_fens and (len(batch_fens) + len(row_unique) > max_positions_per_batch):
            yield batch_fens, assignments
            batch_fens = []
            assignments = []

        base = len(batch_fens)
        batch_fens.extend(row_unique)

        for fen_col, j in zip(fen_cols_to_compute, col_local_idx):
            if j < 0:
                # empty fen (allowed only when half_move <= 20): skip assignment
                continue
            eval_col = _eval_col_for_fen_col(fen_col, fen_prefix)
            pos_idx = base + j
            assignments.append((row_idx, eval_col, pos_idx))

    if batch_fens:
        yield batch_fens, assignments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = read_cfg(args.config)

    input_csv = str(cfg["input_csv"])
    output_csv = str(cfg["output_csv"])

    cols_cfg = cfg["columns"]
    fen_prefix = str(cols_cfg["fen_prefix"])
    player_color_col = str(cols_cfg["player_color_col"])
    half_move_col = str(cols_cfg["half_move_col"])

    batching_cfg = cfg["batching"]
    max_positions_per_batch = int(batching_cfg["max_positions_per_batch"])

    sf_cfg = cfg["stockfish"]
    engine_path = str(sf_cfg["engine_path"])
    depth = _parse_depth(sf_cfg)
    time_limit = float(sf_cfg["time_limit"])
    multipv = int(sf_cfg["multipv"])
    num_engines = int(sf_cfg["num_engines"])
    mate_eval = int(sf_cfg["mate_eval"])

    out_path = Path(output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = _load_input(input_csv)

    _require_columns(df, [player_color_col, half_move_col])
    _require_no_nulls(df, player_color_col)
    _require_no_nulls(df, half_move_col)

    fen_cols = _find_fen_columns(df, fen_prefix)

    # Determine which fen_* need computation (skip if eval_* already exists)
    fen_cols_to_compute: List[str] = []
    skipped: List[str] = []
    new_eval_cols: List[str] = []

    for fen_col in fen_cols:
        eval_col = _eval_col_for_fen_col(fen_col, fen_prefix)
        if eval_col in df.columns:
            skipped.append(fen_col)
            continue
        fen_cols_to_compute.append(fen_col)
        new_eval_cols.append(eval_col)

    log(f"Input rows: {len(df)}")
    log(f"fen_prefix={fen_prefix!r} -> found fen cols: {len(fen_cols)}")
    log(f"To compute (missing eval_*): {len(fen_cols_to_compute)}  |  skipped (eval_* exists): {len(skipped)}")
    log(f"Filter: evaluate only rows where move color == player_color via parity on {half_move_col!r}")
    log(f"Stockfish: depth={depth} time_limit={time_limit} multipv={multipv} num_engines={num_engines} mate_eval={mate_eval}")
    log(f"Batching: max_positions_per_batch={max_positions_per_batch}")

    # Create new eval columns as NA (nullable int) for all rows
    for eval_col in new_eval_cols:
        df[eval_col] = pd.Series(pd.array([pd.NA] * len(df), dtype="Int32"))

    if not fen_cols_to_compute:
        log("Nothing to compute: all eval_* columns already exist. Writing output as-is.")
        df.to_csv(out_path, index=False)
        log(f"Done. Wrote: {out_path}")
        return

    mask = _build_player_move_mask(df, player_color_col=player_color_col, half_move_col=half_move_col)
    row_indices = df.index[mask].tolist()

    log(f"Rows passing player-color parity filter: {len(row_indices)} / {len(df)}")

    total_positions = 0
    total_rows_done = 0

    # Iterate batches (row-level dedupe only)
    for b, (batch_fens, assignments) in enumerate(
        _iter_row_batches(
            df,
            row_indices=row_indices,
            fen_cols_to_compute=fen_cols_to_compute,
            fen_prefix=fen_prefix,
            half_move_col=half_move_col,
            max_positions_per_batch=max_positions_per_batch,
        ),
        start=1,
    ):
        total_positions += len(batch_fens)

        # Evaluate all positions in this batch
        evals = _run_stockfish_eval_for_fens(
            batch_fens,
            engine_path=engine_path,
            depth=depth,
            time_limit=time_limit,
            multipv=multipv,
            num_engines=num_engines,
            mate_eval=mate_eval,
        )

        if len(evals) != len(batch_fens):
            raise ValueError(f"Internal error: evals length mismatch: {len(evals)} vs {len(batch_fens)}")

        # Group assignments by eval_col for vectorized setting
        rows_by_col: DefaultDict[str, List[int]] = defaultdict(list)
        vals_by_col: DefaultDict[str, List[int]] = defaultdict(list)

        for row_idx, eval_col, pos_idx in assignments:
            rows_by_col[eval_col].append(int(row_idx))
            vals_by_col[eval_col].append(int(evals[pos_idx]))

        for eval_col in rows_by_col.keys():
            df.loc[rows_by_col[eval_col], eval_col] = vals_by_col[eval_col]

        # progress logging (approx rows done: assignments / n_fen_cols_to_compute)
        if len(fen_cols_to_compute) > 0:
            rows_this_batch = len(assignments) // len(fen_cols_to_compute)
            total_rows_done += rows_this_batch

        log(
            f"batch={b}: fens={len(batch_fens)} assignments={len(assignments)} "
            f"(rows~{total_rows_done}/{len(row_indices)}) total_fens={total_positions}"
        )

    df.to_csv(out_path, index=False)
    log(f"Done. Wrote: {out_path}")


if __name__ == "__main__":
    main()
