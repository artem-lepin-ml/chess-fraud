#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute Allie embeddings for every move_* column in a move-level CSV and store them in a single NPZ.

Optimization:
- For each evaluated row (position), many move_* columns may contain the same UCI move.
  We deduplicate moves per-row: compute embedding once per unique move and write it
  into all move_* columns that contain that move for that row.

Evaluation window (same as inference-style logic):
- For each game_id group: skip first cfg.detector.debut_size plies,
- then evaluate only the perspective player's turns (stride=2, offset depends on player_color).

ELO logic:
- Opponent ELO is always taken from CSV (opponent_elo) for the evaluated row.
- Player ELO is fixed to cfg.detector.elo_self.
- time_control comes from CSV.
- move timing sequence always comes from CSV (move_thinking_time); we reuse it even if the move_* differs.

Output:
- A single NPZ file at cfg.output_npz.
- Keys are move_* column names.
- Each value is an (N, D) array aligned with the CSV row order.
  Rows that are not evaluated / missing move are filled with NaNs.

GPU memory note (important change vs previous version):
- We DO NOT request output_hidden_states=True. That was the main VRAM blow-up.
- We directly call the GPT2 backbone `transformer(...)` to get ONLY last_hidden_state.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from modeling.data import Game, UCITokenizer
from modeling.model import initialize_model


_LOG_PREFIX = "[allie_embs]"


def log(msg: str) -> None:
    print(f"{_LOG_PREFIX} {msg}", flush=True)


def read_cfg(path: str) -> DictConfig:
    cfg = OmegaConf.load(path)
    OmegaConf.resolve(cfg)  # resolve ${...} interpolations in-place
    return cfg


def require_columns(df: pd.DataFrame, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing required columns: {missing}")


def parse_embeddings_dtype(name: str) -> np.dtype:
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(f"Unsupported embeddings_dtype: {name} (expected float16|float32)")


def get_move_columns(df: pd.DataFrame, move_prefix: str) -> List[str]:
    return sorted([c for c in df.columns if c.startswith(move_prefix)])


def ensure_range_index(df: pd.DataFrame) -> Dict[Any, int] | None:
    if isinstance(df.index, pd.RangeIndex) and df.index.start == 0 and df.index.step == 1:
        return None
    log("WARNING: DataFrame index is not a simple RangeIndex; building an index->row_pos mapping.")
    return {idx: i for i, idx in enumerate(df.index.tolist())}


def get_tokenizer_and_model(cfg: DictConfig) -> Tuple[UCITokenizer, torch.nn.Module]:
    device = str(cfg.allie.device)

    ckpt_path = str(cfg.allie.checkpoint_path)
    hf_cfg_path = str(cfg.allie.hf_config_path)

    log(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")

    log(f"Loading HF config:  {hf_cfg_path}")
    hf_cfg = OmegaConf.load(hf_cfg_path)

    tokenizer = UCITokenizer(max_length=hf_cfg.data_config.tokenizer_config.max_length)

    model = initialize_model(tokenizer=tokenizer, **hf_cfg.model_config)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if device.startswith("cuda"):
        assert torch.cuda.is_available(), "CUDA requested but not available"

    model = model.to(device)
    return tokenizer, model


def extract_last_move_embeddings(
    last_hidden: torch.Tensor,
    lengths: torch.Tensor,
    n_special_tokens: int,
) -> torch.Tensor:
    """
    last_hidden: (B, T, D) last_hidden_state from transformer
    lengths: (B,) number of moves in each sample
    embedding index for the last move token: lengths + n_special_tokens - 1
    """
    idx = lengths + (n_special_tokens - 1)
    b = torch.arange(last_hidden.shape[0], device=last_hidden.device)
    return last_hidden[b, idx, :]


def _is_regression_wrapper(m: torch.nn.Module) -> bool:
    # CausalLMWithRegressionHead: has .value_head/.time_head and inner .model
    return hasattr(m, "value_head") and hasattr(m, "time_head") and hasattr(m, "model")


def _is_control_wrapper(m: torch.nn.Module) -> bool:
    # CausalLMWithControlToken: has .control_embeds/.vocab_size and inner .model
    return hasattr(m, "control_embeds") and hasattr(m, "vocab_size") and hasattr(m, "model")


def get_last_hidden_state_only(
    model: torch.nn.Module,
    input_ids: torch.LongTensor,
    attention_mask: torch.Tensor | None,
) -> torch.Tensor:
    """
    Return ONLY last_hidden_state (B, T, D) without requesting all hidden_states.

    Handles possible wrappers from modeling/model.py:
    - CausalLMWithRegressionHead (unwraps .model)
    - CausalLMWithControlToken (reconstructs inputs_embeds exactly like wrapper and calls base.transformer)
    """
    m = model

    # Unwrap regression head wrapper(s) if present
    while _is_regression_wrapper(m):
        m = m.model  # type: ignore[attr-defined]

    # Now m is either control wrapper or base HF CausalLM
    if _is_control_wrapper(m):
        # Reproduce CausalLMWithControlToken embedding substitution exactly
        ctrl = m  # type: ignore[assignment]
        base = ctrl.model  # underlying AutoModelForCausalLM
        if not hasattr(base, "transformer"):
            raise RuntimeError("Base model does not have .transformer; cannot extract last_hidden_state safely.")
        transformer = base.transformer

        vocab_size = int(ctrl.vocab_size)
        elo_min = int(ctrl.elo_min)
        elo_max = int(ctrl.elo_max)

        elo_mask = (input_ids >= vocab_size).unsqueeze(-1)

        elo_normed = ((input_ids - vocab_size).clamp(elo_min, elo_max) - elo_min) / (elo_max - elo_min)
        control_values = torch.stack((elo_normed, 1 - elo_normed), dim=2).to(ctrl.control_embeds.weight.dtype)
        control_embeds = control_values @ ctrl.control_embeds.weight  # (B, T, D)

        token_embeds = transformer.wte(input_ids.clamp(0, vocab_size - 1))
        inputs_embeds = torch.where(elo_mask, control_embeds, token_embeds)

        out = transformer(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
        )
        return out.last_hidden_state

    # Base HF CausalLM path
    base = m
    if not hasattr(base, "transformer"):
        raise RuntimeError("Model does not have .transformer; cannot extract last_hidden_state safely.")
    out = base.transformer(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True,
    )
    return out.last_hidden_state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    args = ap.parse_args()

    cfg = read_cfg(args.config)

    input_csv = str(cfg.input_csv)
    output_npz = str(cfg.output_npz)
    move_prefix = str(cfg.move_prefix)

    need_cols = list(cfg.need_columns)
    debut_size = int(cfg.detector.debut_size)
    elo_self = int(cfg.detector.elo_self)

    device = str(cfg.allie.device)
    batch_size = int(cfg.allie.batch_size)
    emb_dim = int(cfg.allie.embedding_size)
    n_special_tokens = int(cfg.allie.n_special_tokens)
    out_dtype = parse_embeddings_dtype(str(cfg.embeddings_dtype))

    Path(os.path.dirname(os.path.abspath(output_npz))).mkdir(parents=True, exist_ok=True)

    log(f"Reading CSV: {input_csv}")
    df = pd.read_csv(input_csv)

    require_columns(df, need_cols, where="main/read_csv")

    move_cols = get_move_columns(df, move_prefix)
    move_cols = [c for c in move_cols if c != "move_thinking_time"]  # exclude extra col
    move_cols = [c for c in move_cols if c != "move_label"]  # exclude extra col
    if not move_cols:
        raise ValueError(f"No columns found with prefix '{move_prefix}' in CSV.")

    log(f"Found {len(move_cols)} move columns: {move_cols}")

    index_to_pos = ensure_range_index(df)
    n_rows = len(df)

    arrays: Dict[str, np.ndarray] = {
        c: np.full((n_rows, emb_dim), np.nan, dtype=out_dtype) for c in move_cols
    }

    tokenizer, model = get_tokenizer_and_model(cfg)

    batch_games: List[Game] = []
    batch_lengths: List[int] = []
    batch_row_pos: List[int] = []
    batch_target_cols: List[List[str]] = []

    use_amp = bool(device.startswith("cuda") and out_dtype == np.float16)

    def flush_batch(pbar: tqdm) -> None:
        if not batch_games:
            return

        batch = tokenizer.pad_and_collate(batch_games)
        # keep attention_mask as-is (usually 0/1). Move tensors to device.
        batch = {k: v.to(device) for k, v in batch.items()}

        input_ids = batch["input_ids"]
        attention_mask = batch.get("attention_mask", None)

        # Critical change: only last_hidden_state, no output_hidden_states tuple.
        with torch.inference_mode():
            if use_amp:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    last_hidden = get_last_hidden_state_only(model, input_ids, attention_mask)
            else:
                last_hidden = get_last_hidden_state_only(model, input_ids, attention_mask)

        if last_hidden.shape[-1] != emb_dim:
            raise ValueError(f"Embedding dim mismatch: got {last_hidden.shape[-1]}, expected {emb_dim}")

        len_tensor = torch.tensor(batch_lengths, dtype=torch.long, device=last_hidden.device)
        embs = extract_last_move_embeddings(last_hidden, len_tensor, n_special_tokens)  # (B, D)
        embs = embs.detach().cpu()

        # Write to CPU arrays
        for j in range(embs.shape[0]):
            row_pos = batch_row_pos[j]
            vec = embs[j].to(dtype=torch.float16 if out_dtype == np.float16 else torch.float32).numpy()
            for col in batch_target_cols[j]:
                arrays[col][row_pos, :] = vec

        n = len(batch_games)
        batch_games.clear()
        batch_lengths.clear()
        batch_row_pos.clear()
        batch_target_cols.clear()
        pbar.update(n)

    pbar = tqdm(desc="Computing unique (row, move) embeddings", unit="samples")

    for game_id, group in df.groupby("game_id", sort=False):
        if group.empty:
            continue

        group = group.sort_values("half_move", kind="stable")

        half_moves = group["half_move"].to_numpy()
        expected = np.arange(1, len(group) + 1, dtype=half_moves.dtype)
        if not np.array_equal(half_moves, expected):
            raise ValueError(
                f"game_id={game_id}: half_move must be contiguous 1..N. "
                f"Got head={half_moves[:10].tolist()} (len={len(group)})."
            )

        colors = group["player_color"].dropna().unique().tolist()
        if len(colors) != 1:
            log(f"WARNING: game_id={game_id}: player_color is not constant: {colors}. Using first row value.")
        player_color = str(group.iloc[0]["player_color"])
        offset = 0 if player_color == "white" else 1

        time_control = str(group.iloc[0]["time_control"])
        move_list = group["move_uci"].astype(str).to_list()

        # Parse times robustly (used only for Game construction)
        time_list_raw = group["move_thinking_time"].to_list()
        time_list: List[int] = []
        bad_times = 0
        for t in time_list_raw:
            try:
                if pd.isna(t):
                    bad_times += 1
                    time_list.append(0)
                else:
                    time_list.append(int(t))
            except Exception:
                bad_times += 1
                time_list.append(0)
        if bad_times > 0:
            log(f"WARNING: game_id={game_id}: {bad_times} invalid move_thinking_time values; coerced to 0.")

        start_i = debut_size + offset
        if start_i >= len(group):
            continue

        for i in range(start_i, len(group), 2):
            row = group.iloc[i]
            opp_elo_val = row["opponent_elo"]
            if pd.isna(opp_elo_val):
                log(f"WARNING: game_id={game_id}, row={group.index[i]}: opponent_elo is NaN; using 0.")
                opp_elo = 0
            else:
                opp_elo = int(opp_elo_val)

            if player_color == "white":
                white_elo = elo_self
                black_elo = opp_elo
            else:
                white_elo = opp_elo
                black_elo = elo_self

            col_to_move: Dict[str, str] = {}
            for c in move_cols:
                v = row[c]
                if pd.isna(v) or v is None:
                    continue
                col_to_move[c] = str(v)

            if not col_to_move:
                continue

            move_to_cols: Dict[str, List[str]] = {}
            for c, mv in col_to_move.items():
                move_to_cols.setdefault(mv, []).append(c)

            length_in_moves = i + 1

            idx_val = group.index[i]
            row_pos = int(idx_val) if index_to_pos is None else index_to_pos[idx_val]

            for mv, cols_same_move in move_to_cols.items():
                game = Game(
                    time_control=time_control,
                    white_elo=white_elo,
                    black_elo=black_elo,
                    outcome=None,
                    normal_termination=False,
                    moves=move_list[:i] + [mv],
                    moves_seconds=time_list[: i + 1],
                    next_move_seconds=None,
                )

                batch_games.append(game)
                batch_lengths.append(length_in_moves)
                batch_row_pos.append(row_pos)
                batch_target_cols.append(cols_same_move)

                if len(batch_games) >= batch_size:
                    flush_batch(pbar)

    if batch_games:
        flush_batch(pbar)

    pbar.close()

    for c in move_cols:
        arr = arrays[c]
        filled = int(np.isfinite(arr).all(axis=1).sum())
        log(f"{c}: filled rows = {filled} / {n_rows}")

    log(f"Saving NPZ (may take time for large arrays): {output_npz}")
    np.savez(output_npz, **arrays)

    log("Done.")


if __name__ == "__main__":
    main()
