#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Build a player-perspective move table.

Inputs (Stage 2):
- player_games.csv: one row per (player, game_id), includes split labels, rating_bin, player_color, plies_both_gt_thr
- moves.csv: one row per (game_id, half_move), includes SAN/UCI move + FENs + remaining clock + clock_delta

Output:
- player_moves.csv with columns:
  player, game_id, rating_bin, split_by_player, split_by_games, player_color, player_elo, opponent_elo, time_control,
  half_move, move, move_uci, fen_before, fen_after, clock, clock_delta,
  is_used, move_thinking_time

Filtering semantics (non-destructive):
- We DO NOT drop rows by half_move thresholds.
- Instead we compute is_used (bool) per row:
    is_used = (half_move >= drop_first_halfmoves + 1)
           & (half_move <= max_halfmove)
           & (half_move <= plies_both_gt_thr)

Sanity check:
- For each game_id, the number of rows with is_used=True must be >= min_segment_len_halfmoves.

Additional feature:
- move_thinking_time = clock_delta + increment
  where increment is parsed from time_control "base+increment" (e.g. "180+0", "300+2").
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def log(msg: str) -> None:
    print(f"[build_player_moves] {msg}", flush=True)


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


@dataclass(frozen=True)
class Params:
    player_games_csv: Path
    moves_csv: Path
    out_csv: Path

    drop_first_halfmoves: int
    max_halfmove: int
    min_segment_len_halfmoves: int

    # input column names
    pg_cols: Dict[str, str]
    mv_cols: Dict[str, str]


# -----------------------------
# Parsing
# -----------------------------

def _parse_params(cfg: Dict[str, Any]) -> Params:
    s = cfg["player_moves"]
    inp = s["input"]
    out = s["output"]
    flt = s["filters"]
    cols = s["columns"]

    pg = cols["player_games"]
    mv = cols["moves"]

    pg_cols = {
        "game_id": pg["game_id"],
        "player": pg["player"],
        "player_color": pg["player_color"],
        "rating_bin": pg["rating_bin"],
        "split_by_player": pg["split_by_player"],
        "split_by_games": pg["split_by_games"],
        "plies_both_gt_thr": pg["plies_both_gt_thr"],
        "player_elo": pg["player_elo"],
        "opponent_elo": pg["opponent_elo"],
        "time_control": pg["time_control"],
    }

    mv_cols = {
        "game_id": mv["game_id"],
        "half_move": mv["half_move"],
        "move": mv["move"],
        "move_uci": mv["move_uci"],
        "fen_before": mv["fen_before"],
        "fen_after": mv["fen_after"],
        "clock": mv["clock"],
        "clock_delta": mv["clock_delta"],
    }

    return Params(
        player_games_csv=Path(inp["player_games_csv"]),
        moves_csv=Path(inp["moves_csv"]),
        out_csv=Path(out["player_moves_csv"]),
        drop_first_halfmoves=int(flt["drop_first_halfmoves"]),
        max_halfmove=int(flt["max_halfmove"]),
        min_segment_len_halfmoves=int(flt["min_segment_len_halfmoves"]),
        pg_cols=pg_cols,
        mv_cols=mv_cols,
    )


# -----------------------------
# Loading / normalization
# -----------------------------

def _load_csv_select_rename(path: Path, mapping: Dict[str, str]) -> pd.DataFrame:
    """
    mapping: canonical_name -> input_column_name
    """
    usecols = list(mapping.values())
    df = pd.read_csv(path, usecols=usecols)

    rename_map = {src: dst for dst, src in mapping.items()}
    df = df.rename(columns=rename_map)

    _require_columns(df, list(mapping.keys()))
    return df


def _load_player_games(p: Params) -> pd.DataFrame:
    df = _load_csv_select_rename(p.player_games_csv, p.pg_cols)

    df["game_id"] = df["game_id"].astype(str)
    df["player"] = df["player"].astype(str)
    df["player_color"] = df["player_color"].astype(str).str.lower()
    df["rating_bin"] = df["rating_bin"].astype(str)

    df["split_by_player"] = df["split_by_player"].astype(str)
    df["split_by_games"] = df["split_by_games"].astype(str)

    df["plies_both_gt_thr"] = pd.to_numeric(df["plies_both_gt_thr"], errors="raise").astype(int)
    df["player_elo"] = pd.to_numeric(df["player_elo"], errors="raise")
    df["opponent_elo"] = pd.to_numeric(df["opponent_elo"], errors="raise")
    df["time_control"] = df["time_control"].astype(str).str.lower()

    bad_colors = sorted(set(df["player_color"].unique()) - {"white", "black"})
    if bad_colors:
        raise ValueError(f"Unexpected player_color values in player_games: {bad_colors}")

    # Uniqueness: (game_id, player) must be unique
    dup_gp = df.duplicated(subset=["game_id", "player"], keep=False)
    if dup_gp.any():
        sample = df.loc[dup_gp, ["game_id", "player", "player_color"]].head(10)
        raise ValueError(
            "player_games has duplicate (game_id, player). Sample:\n"
            f"{sample.to_string(index=False)}"
        )

    # Also enforce (game_id, player_color) uniqueness
    dup_gc = df.duplicated(subset=["game_id", "player_color"], keep=False)
    if dup_gc.any():
        sample = df.loc[dup_gc, ["game_id", "player_color", "player"]].head(10)
        raise ValueError(
            "player_games has duplicate (game_id, player_color). Sample:\n"
            f"{sample.to_string(index=False)}"
        )

    return df


def _load_moves(p: Params) -> pd.DataFrame:
    df = _load_csv_select_rename(p.moves_csv, p.mv_cols)

    df["game_id"] = df["game_id"].astype(str)
    df["half_move"] = pd.to_numeric(df["half_move"], errors="raise").astype(int)

    df["clock"] = pd.to_numeric(df["clock"], errors="raise")
    df["clock_delta"] = pd.to_numeric(df["clock_delta"], errors="raise")

    # Invariant: within each game_id, half_move keys must be unique
    dup = df.duplicated(subset=["game_id", "half_move"], keep=False)
    if dup.any():
        sample = df.loc[dup, ["game_id", "half_move"]].head(10)
        raise ValueError(
            "moves.csv has duplicate (game_id, half_move). Sample:\n"
            f"{sample.to_string(index=False)}"
        )

    _require_no_nulls(df, "game_id")
    _require_no_nulls(df, "half_move")
    _require_no_nulls(df, "fen_before")
    _require_no_nulls(df, "fen_after")
    _require_no_nulls(df, "move_uci")
    return df


def _parse_increment_seconds(tc: str) -> int:
    # Expected "base+inc"
    parts = str(tc).strip().split("+", 1)
    if len(parts) != 2:
        raise ValueError(f"Unexpected time_control format (expected 'base+inc'): {tc}")
    inc = parts[1].strip()
    if inc == "":
        raise ValueError(f"Empty increment in time_control: {tc}")
    try:
        return int(inc)
    except Exception as e:
        raise ValueError(f"Non-integer increment in time_control: {tc}") from e


# -----------------------------
# Main build
# -----------------------------

OUT_MOVE_COLS = [
    "game_id",
    "half_move",
    "move",
    "move_uci",
    "fen_before",
    "fen_after",
    "clock",
    "clock_delta",
]


OUT_FINAL_COLS = [
    "player",
    "game_id",
    "rating_bin",
    "split_by_player",
    "split_by_games",
    "player_color",
    "player_elo",
    "opponent_elo",
    "time_control",
    "half_move",
    "move",
    "move_uci",
    "fen_before",
    "fen_after",
    "clock",
    "clock_delta",
    "is_used",
    "move_thinking_time",
]


def build_player_moves(p: Params) -> None:
    pg = _load_player_games(p)
    mv = _load_moves(p)

    # Ensure moves only for games known in player_games
    known_games = set(pg["game_id"].unique())
    mv_games = set(mv["game_id"].unique())
    unknown_games = sorted(mv_games - known_games)
    if unknown_games:
        raise ValueError(
            "moves.csv contains game_id that are absent in player_games.csv. "
            f"Sample: {unknown_games[:10]} (total {len(unknown_games)})"
        )

    # max_keep per game: min(plies_both_gt_thr, max_halfmove)
    per_game_thr = pg.groupby("game_id", as_index=True)["plies_both_gt_thr"].min()
    max_keep_by_game = np.minimum(per_game_thr, p.max_halfmove).astype(int)
    max_keep_df = max_keep_by_game.rename("max_keep").reset_index()  # type: ignore

    mv = mv.merge(max_keep_df, on="game_id", how="inner", validate="many_to_one")

    min_keep_halfmove = p.drop_first_halfmoves + 1
    mv["is_used"] = (
        (mv["half_move"] >= min_keep_halfmove)
        & (mv["half_move"] <= mv["max_keep"])
    )

    used_len = mv.loc[mv["is_used"]].groupby("game_id")["half_move"].size()
    too_short = used_len[used_len < p.min_segment_len_halfmoves]
    if len(too_short) > 0:
        raise ValueError(
            "Some games become too short after filtering (is_used=True) "
            f"(min_segment_len_halfmoves={p.min_segment_len_halfmoves}).\n"
            f"Sample (game_id -> used_len):\n{too_short.head(10).to_string()}"
        )

    out = pg.merge(
        mv[OUT_MOVE_COLS + ["is_used"]],
        on="game_id",
        how="inner",
        validate="one_to_many",
    )

    if out.empty:
        raise RuntimeError("Internal error: output is empty after merge.")

    out["increment"] = out["time_control"].map(_parse_increment_seconds).astype(int)
    out["move_thinking_time"] = out["clock_delta"].astype(float) + out["increment"].astype(float)
    out = out.drop(columns=["increment"])

    out = out[OUT_FINAL_COLS].sort_values(["player", "game_id", "half_move"], kind="mergesort")

    p.out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(p.out_csv, index=False)
    log(f"Wrote {len(out):,} rows to: {p.out_csv}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = read_cfg(args.config)
    p = _parse_params(cfg)

    log(f"player_games_csv={p.player_games_csv}")
    log(f"moves_csv={p.moves_csv}")
    log(f"out_csv={p.out_csv}")
    log(
        "filters: "
        f"drop_first_halfmoves={p.drop_first_halfmoves} "
        f"max_halfmove={p.max_halfmove} "
        f"min_segment_len_halfmoves={p.min_segment_len_halfmoves}"
    )

    build_player_moves(p)


if __name__ == "__main__":
    main()
