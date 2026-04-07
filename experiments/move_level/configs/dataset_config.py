from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DatasetConfig:
    """All dataset-level parameters for an experiment."""

    # --- Paths ---
    synth_csv_path: str | Path = ""
    synth_emb_npz_path: str | Path = ""
    tournament_csv_path: str | Path = ""
    tournament_emb_npz_path: str | Path = ""

    # --- Cheat models ---
    cheat_models: list[str] = field(default_factory=list)
    cheat_model_to_bin_idxs: dict[str, list[int]] = field(default_factory=dict)

    # --- Rating bins (synth) ---
    rating_bins: list[str] = field(
        default_factory=lambda: [
            "lt_1200", "1200-1400", "1400-1600",
            "1600-1800", "1800-2000", "2000-2200",
        ]
    )

    # --- Elo bins (tournament) ---
    tournament_elo_bins: list[tuple[str, float, float]] = field(
        default_factory=lambda: [
            ("<1.6k", float("-inf"), 1600),
            ("1.6-1.8k", 1600, 1800),
            ("1.8-2.0k", 1800, 2000),
            ("2.0-2.2k", 2000, 2200),
            (">2.2k", 2200, float("inf")),
        ]
    )

    # --- Split ---
    split_col: str = "split_by_player"
    player_col: str = "player"
    val_frac: float = 0.1

    # --- Embedding ---
    emb_key_human: str = "move_uci"  # NPZ key for human move embedding

    # --- DataLoader ---
    batch_size: int = 1024
    num_workers: int = 0

    # --- Misc ---
    seed: int = 42
    build_all_mixture: bool = True  # whether to build "ALL" pseudo-cheat
