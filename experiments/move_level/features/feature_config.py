from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FeatureConfig:
    """Toggle individual features on/off.

    Each boolean flag corresponds to a registered feature function in
    ``feature_builder.py``.  Adding a new feature requires:
      1. A new flag here (default ``False``).
      2. A builder function registered in ``FEATURE_REGISTRY``.
    """

    eval_delta: bool = False           # eval_after - eval_before
    maia2_win_prob: bool = False       # maia2_win_prob_2050
    maia2_move_prob: bool = False      # maia2_move_prob_nearest
    allie_win_prob: bool = False       # allie_win_prob_2500
    allie_move_prob: bool = False      # allie_move_prob_nearest
    move_thinking_time: bool = False   # log1p(move_thinking_time)

    # --- per-move / tabular ---
    sf15_match: bool = False           # (move == move_stockfish_15)
    player_elo: bool = False           # player Elo (optionally z-scored)
    opponent_elo: bool = False         # opponent Elo (optionally z-scored)

    # --- global switch ---
    standardize: bool = True           # z-score for elo/thinking_time features
