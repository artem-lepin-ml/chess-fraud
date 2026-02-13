#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional, Tuple

import chess
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.lc0_utils import LeelaChessZeroEngine, LeelaChessZeroEngineConfig, SearchLimit


def log(msg: str) -> None:
    print(f"[maia1_char_moves] {msg}", flush=True)


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


def _apply_uci_move_to_fen(fen: str, uci_move: str, *, row_idx: int) -> str:
    try:
        board = chess.Board(fen)
    except Exception as e:
        raise ValueError(f"Invalid FEN at row {row_idx}: {fen}") from e

    try:
        board.push_uci(str(uci_move))
    except Exception as e:
        raise ValueError(f"Invalid/illegal UCI move at row {row_idx}: move='{uci_move}' fen='{fen}'") from e

    return board.fen()


def _parse_limit(limit_cfg: Dict[str, Any]) -> SearchLimit:
    nodes = limit_cfg.get("nodes", None)
    time_sec = limit_cfg.get("time_sec", None)

    nodes_val: Optional[int] = None
    time_val: Optional[float] = None

    if nodes is not None:
        nodes_val = int(nodes)
    if time_sec is not None:
        time_val = float(time_sec)

    return SearchLimit(nodes=nodes_val, time_sec=time_val)


def _character_name_from_weights_path(p: Path) -> str:
    # Expect: <name>.pb.gz
    suffixes = p.suffixes  # e.g. [".pb", ".gz"]
    if len(suffixes) >= 2 and suffixes[-2:] == [".pb", ".gz"]:
        name = p.name[: -len(".pb.gz")]
    else:
        # fallback: strip only the last suffix
        name = p.stem

    name = name.strip()
    if not name:
        raise ValueError(f"Could not derive a non-empty character name from weights file: {p}")
    return name


def _list_character_weights(weights_dir: Path, *, glob_pat: str) -> List[Tuple[str, Path]]:
    if not weights_dir.exists():
        raise ValueError(f"weights_dir does not exist: {weights_dir}")
    if not weights_dir.is_dir():
        raise ValueError(f"weights_dir is not a directory: {weights_dir}")

    paths = sorted(weights_dir.glob(glob_pat))
    if not paths:
        raise ValueError(f"No weights found in {weights_dir} with pattern '{glob_pat}'")

    items: List[Tuple[str, Path]] = []
    for p in paths:
        if not p.is_file():
            continue
        name = _character_name_from_weights_path(p)
        items.append((name, p))

    if not items:
        raise ValueError(f"No weight files resolved after filtering in {weights_dir} with pattern '{glob_pat}'")

    # Ensure unique names
    names = [n for n, _ in items]
    dup = sorted({n for n in names if names.count(n) > 1})
    if dup:
        raise ValueError(f"Duplicate character names after parsing filenames: {dup[:20]}")

    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = read_cfg(args.config)

    input_csv = str(cfg["input_csv"])
    output_csv = str(cfg["output_csv"])
    fen_col = str(cfg["fen_col"])

    lc0_path = str(cfg["lc0_path"])
    backend_args = cfg["backend_args"]
    threads = int(cfg["threads"])

    weights_dir = Path(str(cfg["weights_dir"]))
    weights_glob = str(cfg.get("weights_glob", "*.pb.gz"))

    limit = _parse_limit(cfg["limit"])

    df = pd.read_csv(input_csv)
    _require_columns(df, [fen_col])
    _require_no_nulls(df, fen_col)

    fens = df[fen_col].astype(str).tolist()
    n = len(fens)

    characters = _list_character_weights(weights_dir, glob_pat=weights_glob)
    log(f"Found {len(characters)} character weights in {weights_dir} (pattern='{weights_glob}')")
    log(f"Input rows: {n}")

    for name, weights_path in characters:
        move_col_out = f"move_maia_character_{name}"
        fen_col_out = f"fen_character_{name}"

        if move_col_out in df.columns or fen_col_out in df.columns:
            raise ValueError(
                f"Output columns already exist for character '{name}': "
                f"{move_col_out in df.columns=}, {fen_col_out in df.columns=}"
            )

        log(f"character='{name}': start ({weights_path.name})")
        eng = LeelaChessZeroEngine(
            LeelaChessZeroEngineConfig(
                lc0_path=lc0_path,
                backend_args=backend_args,
                threads=threads,
                timeout_sec=100.0,
                weights_path=str(weights_path),
            )
        )

        moves_out: List[str] = []
        fens_out: List[str] = []

        try:
            for i, fen in enumerate(tqdm(fens, desc=f"maia_char {name}", total=n, mininterval=0.5)):
                mv = eng.predict(chess.Board(fen), limit=limit)
                mv_uci = mv.uci()
                moves_out.append(mv_uci)
                fens_out.append(_apply_uci_move_to_fen(fen, mv_uci, row_idx=i))
        finally:
            eng.close()

        if len(moves_out) != n or len(fens_out) != n:
            raise ValueError(
                f"Length mismatch for character='{name}': {len(moves_out)} vs {len(fens_out)} vs {n}"
            )

        df[move_col_out] = moves_out
        df[fen_col_out] = fens_out

    df.to_csv(output_csv, index=False)
    log(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
