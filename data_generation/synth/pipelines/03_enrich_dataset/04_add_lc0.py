#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from pathlib import Path
import argparse
from typing import Any, Dict, List, Optional

import chess
import pandas as pd
from omegaconf import OmegaConf
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from utils.lc0_utils import LeelaChessZeroEngine, LeelaChessZeroEngineConfig, SearchLimit


def log(msg: str) -> None:
    print(f"[lc0_moves] {msg}", flush=True)


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


def _parse_nodes_list(limit_cfg: Dict[str, Any]) -> List[int]:
    nodes_list = limit_cfg["nodes_list"]
    if not isinstance(nodes_list, list) or not nodes_list:
        raise ValueError("limit.nodes_list must be a non-empty list[int]")

    out: List[int] = []
    for x in nodes_list:
        v = int(x)
        if v <= 0:
            raise ValueError(f"limit.nodes_list contains non-positive value: {v}")
        out.append(v)
    return out


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

    weights_path = Path(str(cfg["weights_path"]))

    limit_cfg = cfg["limit"]
    nodes_list = _parse_nodes_list(limit_cfg)

    df = pd.read_csv(input_csv)
    _require_columns(df, [fen_col])
    _require_no_nulls(df, fen_col)

    fens = df[fen_col].astype(str).tolist()
    n = len(fens)

    eng = LeelaChessZeroEngine(
        LeelaChessZeroEngineConfig(
            lc0_path=lc0_path,
            backend_args=backend_args,
            threads=threads,
            timeout_sec=100.0,
            weights_path=str(weights_path),
        )
    )

    try:
        for nodes in nodes_list:
            log(f"nodes={nodes}: start")

            limit = SearchLimit(nodes=int(nodes), time_sec=None)

            moves_out: List[str] = []
            fens_out: List[str] = []

            for i, fen in enumerate(tqdm(fens, desc=f"lc0 {nodes}", total=n, mininterval=0.5)):
                mv = eng.predict(chess.Board(fen), limit=limit)
                mv_uci = mv.uci()
                moves_out.append(mv_uci)
                fens_out.append(_apply_uci_move_to_fen(fen, mv_uci, row_idx=i))

            if len(moves_out) != n or len(fens_out) != n:
                raise ValueError(f"Length mismatch for nodes={nodes}: {len(moves_out)} vs {len(fens_out)} vs {n}")

            df[f"move_lc0_{nodes}"] = moves_out
            df[f"fen_lc0_{nodes}"] = fens_out
    finally:
        eng.close()

    df.to_csv(output_csv, index=False)
    log(f"Wrote: {output_csv}")


if __name__ == "__main__":
    main()
