#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add Allie predictions (top-1 move, time, and white-advantage / win-prob proxy) to an annotated
move-level CSV for multiple `cheat_elo` settings.

Multi-GPU / multi-process version:
- One worker process per `cheat_elo`.
- You may place multiple models on the same GPU via `cfg.per_gpu`.
- Progress bar is robust (no early-exit deadlock): workers send integer increments + a sentinel on finish.

Note: We use mp.Manager().Queue() so the queue can be passed to Pool workers safely.
"""


import argparse
import os
import json
import warnings
import multiprocessing as mp

import pandas as pd
import torch
from omegaconf import OmegaConf
import chess
from tqdm import tqdm

from evaluation.decode import Policy
from modeling.data import Game, UCITokenizer
from modeling.model import initialize_model


def chunk_elos(cfg, num_gpus: int):
    step = int(cfg.per_gpu) * int(num_gpus)
    for i in range(0, len(cfg.elos), step):
        yield cfg.elos[i : i + step]


def get_allie_tokenizer_and_model(cfg, device: str) -> tuple[UCITokenizer, torch.nn.Module]:
    ckpt = torch.load(cfg.checkpoint_path, map_location="cpu")
    hf_cfg = OmegaConf.load(cfg.hf_config_path)
    tokenizer = UCITokenizer(max_length=hf_cfg.data_config.tokenizer_config.max_length)

    model = initialize_model(tokenizer=tokenizer, **hf_cfg.model_config)
    model.load_state_dict(ckpt["model"])
    model.eval()
    model = model.to(device)

    return tokenizer, model


def fen_after_from_move(fen_before: str, move_uci: str) -> str:
    """Apply a legal UCI move to fen_before and return resulting FEN."""
    b = chess.Board(fen_before)
    b.push_uci(move_uci)
    return b.fen()


def move_probs_dict(legal_moves: list[str], probs_tensor) -> dict[str, float]:
    """Convert legal move probabilities tensor into a Python dict[uci -> prob]."""
    probs = probs_tensor.detach().cpu().tolist()
    return {m: float(p) for m, p in zip(legal_moves, probs)}


def get_game_samples(
    group: pd.DataFrame, cheat_elo: int, cfg
) -> tuple[list[Game], list[list[str]], list[int], list[str]]:
    row = group.iloc[0]

    if row["player_color"] == "white":
        white_elo = cheat_elo
        black_elo = int(row["opponent_elo"])
        offset = 0
    else:
        white_elo = int(row["opponent_elo"])
        black_elo = cheat_elo
        offset = 1

    move_list = group["move_uci"].to_list()
    time_list = group["move_thinking_time"].astype(int).to_list()

    games: list[Game] = []
    legal_moves_list: list[list[str]] = []
    row_idx_list: list[int] = []
    fen_before_list: list[str] = []

    # i is index inside the group (row position within this game_id slice)
    for i in range(cfg.debut_size + offset, len(group), 2):
        fen_before = group.iloc[i].loc["fen_before"]
        board = chess.Board(fen_before)
        legal_moves = [m.uci() for m in board.legal_moves]

        game = Game(
            time_control=row["time_control"],
            white_elo=white_elo,
            black_elo=black_elo,
            outcome=None,
            normal_termination=False,
            moves=move_list[:i],
            moves_seconds=time_list[:i],
            next_move_seconds=None,
        )

        games.append(game)
        legal_moves_list.append(legal_moves)
        row_idx_list.append(int(group.index[i]))  # global row index in the original df
        fen_before_list.append(fen_before)

    return games, legal_moves_list, row_idx_list, fen_before_list


def get_allie_results_for_elo(
    df_small: pd.DataFrame,
    cheat_elo: int,
    policy: Policy,
    cfg,
    progress_q,
) -> tuple[list[int], list[str], list[str], list[str], list[float]]:
    row_idx_all: list[int] = []
    fen_before_all: list[str] = []
    legal_moves_all: list[list[str]] = []
    game_all: list[Game] = []

    for _game_id, group in df_small.groupby("game_id", sort=False):
        games, legal_moves_list, row_idx_list, fen_before_list = get_game_samples(group, cheat_elo, cfg)
        game_all += games
        legal_moves_all += legal_moves_list
        row_idx_all += row_idx_list
        fen_before_all += fen_before_list

    assert len(game_all) == len(legal_moves_all) == len(row_idx_all) == len(fen_before_all)

    move_list: list[str] = []
    fen_list: list[str] = []
    move_probs_json_list: list[str] = []
    win_probs_list: list[float] = []

    # Batch progress updates to reduce IPC overhead.
    tick = 0
    tick_every = 512

    with torch.no_grad():
        for game, legal_moves, fen_before in zip(game_all, legal_moves_all, fen_before_all):
            score = policy.score_full(game, legal_moves)

            probs_leg = score.legal_move_probabilities
            top_i = int(probs_leg.argmax().item())
            move_uci = legal_moves[top_i]

            fen_after = fen_after_from_move(fen_before, move_uci)

            probs_map = move_probs_dict(legal_moves, probs_leg)
            probs_json = json.dumps(probs_map, ensure_ascii=False, separators=(",", ":"))

            # White-perspective value in [-1, 1]: positive means good for White.
            pov = 1.0 if game.board.turn == chess.WHITE else -1.0
            value_white = float(pov * (score.white_advantage if score.white_advantage is not None else 0.0))

            move_list.append(move_uci)
            fen_list.append(fen_after)
            move_probs_json_list.append(probs_json)
            win_probs_list.append(value_white)

            tick += 1
            if tick % tick_every == 0:
                progress_q.put(tick_every)

    rem = tick % tick_every
    if rem:
        progress_q.put(rem)

    return row_idx_all, move_list, fen_list, move_probs_json_list, win_probs_list


def process_score_to_columns(
    df: pd.DataFrame,
    row_idx_list: list[int],
    move_list: list[str],
    fen_list: list[str],
    move_probs_json_list: list[str],
    win_probs_list: list[float],
    cheat_elo: int,
) -> pd.DataFrame:
    c_move = f"move_allie_{cheat_elo}"
    c_fen = f"fen_allie_{cheat_elo}"
    c_probs = f"allie_move_probs_{cheat_elo}"
    c_win = f"allie_win_probs_{cheat_elo}"

    for c in (c_move, c_fen, c_probs, c_win):
        if c in df.columns:
            raise ValueError(f"Column already exists: {c}")

    df[c_move] = None
    df[c_fen] = None
    df[c_probs] = None
    df[c_win] = None

    df.loc[row_idx_list, c_move] = move_list
    df.loc[row_idx_list, c_fen] = fen_list
    df.loc[row_idx_list, c_probs] = move_probs_json_list
    df.loc[row_idx_list, c_win] = win_probs_list

    return df


def count_positions(df_small: pd.DataFrame, debut_size: int) -> int:
    """Count evaluated positions per single elo (matches get_game_samples() stride logic)."""
    total = 0
    for _game_id, group in df_small.groupby("game_id", sort=False):
        row0 = group.iloc[0]
        offset = 0 if row0["player_color"] == "white" else 1
        start = debut_size + offset
        if start >= len(group):
            continue
        total += 1 + (len(group) - 1 - start) // 2
    return total


def progress_worker(q, total: int, n_workers_total: int) -> None:
    """Consume progress increments until all workers send a sentinel (None)."""
    finished = 0
    with tqdm(total=total, desc="Allie progress (positions)") as pbar:
        while finished < n_workers_total:
            msg = q.get()
            if msg is None:
                finished += 1
            else:
                pbar.update(int(msg))


def run_one_elo(args):
    """
    Worker: process one cheat_elo on a specific GPU.
    Always sends a sentinel to progress queue on exit.
    """
    cfg, cheat_elo, gpu_id, df_small, progress_q = args
    try:
        torch.cuda.set_device(gpu_id)
        device = f"cuda:{gpu_id}"

        tokenizer, model = get_allie_tokenizer_and_model(cfg, device)
        policy = Policy(
            model=model,
            tokenizer=tokenizer,
            temperature=1.0,
            time_prediction=True,
            device=device,
        )

        row_idx_list, move_list, fen_list, move_probs_json_list, win_probs_list = get_allie_results_for_elo(
            df_small=df_small,
            cheat_elo=int(cheat_elo),
            policy=policy,
            cfg=cfg,
            progress_q=progress_q,
        )
        return int(cheat_elo), row_idx_list, move_list, fen_list, move_probs_json_list, win_probs_list
    finally:
        progress_q.put(None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    os.makedirs(os.path.dirname(os.path.abspath(cfg.output_csv)), exist_ok=True)

    df = pd.read_csv(cfg.input_csv)
    df_small = df[cfg.need_columns]

    num_gpus = torch.cuda.device_count()
    assert num_gpus >= 1, "No CUDA devices found"

    positions_per_elo = count_positions(df_small, int(cfg.debut_size))
    total_positions = positions_per_elo * len(cfg.elos)

    ctx = mp.get_context("spawn")

    manager = mp.Manager()
    progress_q = manager.Queue()

    progress_proc = ctx.Process(
        target=progress_worker,
        args=(progress_q, total_positions, len(cfg.elos)),
        daemon=True,
    )
    progress_proc.start()

    try:
        elo_chunks = chunk_elos(cfg, num_gpus)

        for chunk in elo_chunks:
            jobs = []
            for i, elo in enumerate(chunk):
                # Allow multiple workers on the same GPU.
                # Example: per_gpu=2 -> i=0,1 -> gpu0; i=2,3 -> gpu1; etc.
                gpu_id = (i // int(cfg.per_gpu)) % num_gpus
                jobs.append((cfg, int(elo), int(gpu_id), df_small, progress_q))

            results = []
            with ctx.Pool(processes=len(jobs)) as pool:
                for res in pool.imap_unordered(run_one_elo, jobs):
                    results.append(res)

            for cheat_elo, row_idx_list, move_list, fen_list, move_probs_json_list, win_probs_list in sorted(
                results, key=lambda x: x[0]
            ):
                df = process_score_to_columns(
                    df=df,
                    row_idx_list=row_idx_list,
                    move_list=move_list,
                    fen_list=fen_list,
                    move_probs_json_list=move_probs_json_list,
                    win_probs_list=win_probs_list,
                    cheat_elo=int(cheat_elo),
                )

        df.to_csv(cfg.output_csv, index=False)
        print("Done and saved to", cfg.output_csv)

    finally:
        if progress_proc.is_alive():
            progress_proc.join(timeout=5)
            if progress_proc.is_alive():
                progress_proc.terminate()
                progress_proc.join()


if __name__ == "__main__":
    main()
