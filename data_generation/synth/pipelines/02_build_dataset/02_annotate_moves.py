#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Annotate move-level dataset with:
- fen_before, fen_after: FEN position before/after each half_move
- move_uci: move in UCI format (derived from the applied move)
- clock_delta: clock(t-2) - clock(t) within each game_id (first 2 halfmoves -> 0)

Input CSV columns (moves.csv):
  game_id,date,white,black,white_elo,black_elo,time_control,termination,result,half_move,move,clock

Assumptions / guarantees (hard requirements to avoid silently wrong FEN):
- For each game_id, half_move sequence must start at 1 and be contiguous with no gaps.
- `move` is SAN (fallback to UCI is attempted if it looks like UCI).
- FEN reconstruction uses python-chess (import chess): Board.push_san(), Board.fen().

Output columns:
  game_id,date,white,black,white_elo,black_elo,time_control,termination,result,half_move,move,move_uci,
  fen_before,fen_after,clock,clock_delta
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

try:
    import chess  # python-chess
except ImportError as e:
    raise RuntimeError(
        "python-chess is required. Install: pip install python-chess"
    ) from e


def log(msg: str) -> None:
    print(f"[annotate_moves] {msg}", flush=True)


# ----------------------------
# CLI / params
# ----------------------------

@dataclass(frozen=True)
class Params:
    input_csv: Path
    output_csv: Path


def _parse_args() -> Params:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to moves.csv")
    parser.add_argument("--output", type=str, required=True, help="Path to output annotated CSV")
    args = parser.parse_args()
    return Params(input_csv=Path(args.input), output_csv=Path(args.output))


# ----------------------------
# Move parsing helpers
# ----------------------------

_UCI_RE = re.compile(r"^[a-h][1-8][a-h][1-8][qrbn]?$", re.IGNORECASE)


def _push_move(board: chess.Board, move_str: str) -> chess.Move:
    """
    Push a move onto the board. Primary: SAN. Fallback: UCI if it looks like UCI.
    Returns the applied chess.Move.
    Raises ValueError with context on failure.
    """
    s = str(move_str).strip()
    if not s:
        raise ValueError("Empty move string")

    try:
        mv = board.push_san(s)
        return mv
    except Exception as san_err:
        if _UCI_RE.match(s):
            try:
                mv = board.push_uci(s.lower())
                return mv
            except Exception as uci_err:
                raise ValueError(f"Failed to parse move as SAN or UCI: {s!r}") from uci_err
        raise ValueError(f"Failed to parse move as SAN: {s!r}") from san_err


# ----------------------------
# Core logic
# ----------------------------

def annotate_moves(input_csv: Path, output_csv: Path) -> None:
    df = pd.read_csv(input_csv)

    required = {
        "game_id", "date", "white", "black", "white_elo", "black_elo",
        "time_control", "termination", "result", "half_move", "move", "clock",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"moves.csv is missing required columns: {missing}")

    # Normalize types
    df["game_id"] = df["game_id"].astype(str)
    df["half_move"] = pd.to_numeric(df["half_move"], errors="raise").astype(int)
    df["move"] = df["move"].astype(str)
    df["clock"] = pd.to_numeric(df["clock"], errors="raise")

    # Sort for stable per-game processing
    df = df.sort_values(["game_id", "half_move"], kind="mergesort").reset_index(drop=True)

    # Compute clock_delta per game: clock(t) - clock(t-2), first 2 -> 0
    g_clock = df.groupby("game_id", sort=False)["clock"]
    df["clock_delta"] = -(df["clock"] - g_clock.shift(2)).fillna(0)

    # Compute FENs and move_uci per game
    fen_before_list: List[str] = []
    fen_after_list: List[str] = []
    move_uci_list: List[str] = []

    processed = 0

    for game_id, g in df.groupby("game_id", sort=False):
        half_moves = g["half_move"].to_numpy(dtype=int)
        if half_moves.size == 0:
            continue

        if int(half_moves[0]) != 1:
            raise ValueError(
                "Cannot reconstruct FEN from the standard initial position: half_move does not start at 1.\n"
                f"game_id={game_id}, first_half_move={int(half_moves[0])}"
            )

        expected = np.arange(1, int(half_moves[-1]) + 1, dtype=int)
        if half_moves.shape[0] != expected.shape[0] or not np.array_equal(half_moves, expected):
            raise ValueError(
                "Cannot reconstruct FEN: half_move sequence is not contiguous from 1.\n"
                f"game_id={game_id}, observed_range=[{int(half_moves[0])},{int(half_moves[-1])}], "
                f"observed_len={len(half_moves)}, expected_len={len(expected)}"
            )

        board = chess.Board()

        for _, r in g.iterrows():
            mv_str = r["move"]
            fen_before = board.fen()
            try:
                mv_obj = _push_move(board, mv_str)
            except Exception as e:
                hm = int(r["half_move"])
                raise ValueError(
                    "Failed to apply move while reconstructing FEN.\n"
                    f"game_id={game_id}, half_move={hm}, move={mv_str!r}"
                ) from e
            fen_after = board.fen()

            fen_before_list.append(fen_before)
            fen_after_list.append(fen_after)
            move_uci_list.append(mv_obj.uci())

            processed += 1
            if processed % 100000 == 0:
                log(f"Processed {processed:,} positions (last game_id={game_id})")

    if (
        len(fen_before_list) != len(df)
        or len(fen_after_list) != len(df)
        or len(move_uci_list) != len(df)
    ):
        raise ValueError(
            "Internal error: list length does not match dataframe length.\n"
            f"len_df={len(df)}, len_fen_before={len(fen_before_list)}, "
            f"len_fen_after={len(fen_after_list)}, len_move_uci={len(move_uci_list)}"
        )

    df["fen_before"] = fen_before_list
    df["fen_after"] = fen_after_list
    df["move_uci"] = move_uci_list

    out_cols = [
        "game_id", "date", "white", "black", "white_elo", "black_elo",
        "time_control", "termination", "result", "half_move", "move", "move_uci",
        "fen_before", "fen_after", "clock", "clock_delta",
    ]

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False, columns=out_cols)
    log(f"Wrote annotated moves to: {output_csv}")


def main() -> None:
    p = _parse_args()
    annotate_moves(p.input_csv, p.output_csv)


if __name__ == "__main__":
    main()
