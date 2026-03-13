#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Dataset extraction pipeline (DuckDB-based).

This script builds a curated chess dataset from a raw move-level CSV export.

High-level logic:
- Stage 0: Load the raw moves CSV into a filtered DuckDB TEMP VIEW (optionally by month prefix and time control).
- Stage 1: Build or reuse a cached game-level table (games_meta) in Parquet. For each game we compute:
  - n_plies (max halfmove index)
  - plies_both_gt_thr: the number of halfmoves played while BOTH players still had clock > threshold (thr).
    Concretely, we find the earliest ply where White's clock <= thr (odd ply) and Black's clock <= thr (even ply),
    and define plies_both_gt_thr as (min(first_white_leq_thr_ply, first_black_leq_thr_ply) - 1), or max(ply) if none.
- Stage 2: Build a per-player view of games (one row per player per game).
- Stage 3: Compute per-player monthly stats (avg_elo, n_games) and assign each player to an avg rating bin.
- Stage 4: Deterministically preselect a limited number of players per bin for tractability.
- Stage 5: Filter eligible player-games for those players:
  - require plies_both_gt_thr >= min_plies
  - optionally enforce that the player's per-game rating bin equals their avg rating bin.
- Stage 6: Select final players per bin (enough eligible games), and split players into test vs train deterministically.
  Optionally, apply an opponent-based constraint ("selected vs selected" games are excluded) via a two-pass selection:
    - Pass A: pick a larger candidate pool per bin
    - Pass B: apply the opponent constraint, recompute eligibility counts, then pick final players per bin
- Stage 7: For each (bin, player) sample a fixed number of games deterministically, and assign:
  - split_by_player (per-player)
  - split_by_games (per-game split, consistent by game_id)
- Stage 8: Write outputs to CSV:
  - players.csv: selected players with rating_bin and split_by_player
  - games.csv: unique games with split_by_player and split_by_games
  - moves.csv: move-level rows ONLY up to cutoff ply (ply <= plies_both_gt_thr) for each selected game
  - player_games.csv: player-perspective index table for the final dataset (split_by_player, split_by_games)

Most intermediate artifacts are DuckDB TEMP VIEWs (in-memory). The only persistent cache is games_meta Parquet.
"""

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List

import duckdb
from omegaconf import OmegaConf
import yaml


def rating_bin_case(col: str, lo: int, hi: int, step: int) -> str:
    """
    Example bins:
      <lo -> 'lt_1100'
      [1100,1200) -> '1100_1200'
      ...
      [2100,2200) -> '2100_2200'
      >=hi -> 'gt_2200'
    """
    parts = [
        "CASE",
        f"  WHEN {col} < {lo} THEN 'lt_{lo}'",
        f"  WHEN {col} >= {hi} THEN 'gt_{hi}'",
    ]
    for start in range(lo, hi, step):
        end = start + step
        parts.append(f"  WHEN {col} >= {start} AND {col} < {end} THEN '{start}_{end}'")
    parts.append("END")
    return "\n".join(parts)

def read_cfg(path: str) -> dict:
    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True) # type: ignore[return-value]

def log(msg: str) -> None:
    print(f"[extract_games] {msg}", flush=True)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = read_cfg(args.config)

    # --- validate split params ---
    games_per_player = int(cfg["sampling"]["games_per_player"])
    final_players_per_bin = int(cfg["sampling"]["final_players_per_bin"])

    splits_cfg = cfg.get("splits", {})

    # by_player
    by_player_cfg = splits_cfg.get("by_player")
    test_frac = float(by_player_cfg["test_player_fraction"])
    player_salt = str(by_player_cfg.get("salt", "player_split_v1"))

    # by_games
    by_games_cfg = splits_cfg.get("by_games")
    test_game_frac = float(by_games_cfg["test_game_fraction"])
    game_salt = str(by_games_cfg.get("salt", "game_split_v1"))

    test_players_per_bin = int(round(final_players_per_bin * test_frac))
    train_players_per_bin = final_players_per_bin - test_players_per_bin
    if test_players_per_bin <= 0 or train_players_per_bin <= 0:
        raise ValueError("Bad test_player_fraction (empty split).")
    if not (0.0 <= test_game_frac <= 1.0):
        raise ValueError("splits.by_games.test_game_fraction must be in [0, 1].")
    

    # --- dedupe toggle (opponent constraint) ---
    drop_games_vs_selected_opponents = bool(cfg["sampling"].get("drop_games_vs_selected_opponents", False))
    candidate_players_multiplier = int(cfg["sampling"].get("candidate_players_multiplier", 3))
    if candidate_players_multiplier <= 0:
        raise ValueError("sampling.candidate_players_multiplier must be a positive integer.")

    # --- duckdb ---
    db_path = cfg["duckdb"]["db_path"]
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(db_path)

    con.execute(f"SET threads TO {int(cfg['duckdb'].get('threads', 8))};")
    if cfg["duckdb"].get("temp_directory"):
        con.execute(f"SET temp_directory='{cfg['duckdb']['temp_directory']}';")

    # --- columns ---
    col = cfg["columns"]
    G = col["game_id"]
    DATE = col["date"]
    W = col["white"]
    B = col["black"]
    WE = col["white_elo"]
    BE = col["black_elo"]
    TC = col["time_control"]
    TERM = col["termination"]
    RES = col["result"]
    PLY = col["halfmove"]
    CLK = col["clock"]
    MOVE = col["move"]

    # --- filters / bins ---
    lo = int(cfg["rating_bins"]["lo"])
    hi = int(cfg["rating_bins"]["hi"])
    step = int(cfg["rating_bins"]["step"])
    avg_bin_expr = rating_bin_case("avg_elo", lo, hi, step)
    game_bin_expr = rating_bin_case("player_elo", lo, hi, step)

    thr = float(cfg["filters"]["clock_threshold_seconds"])
    min_plies = int(cfg["filters"]["min_plies_while_both_gt_threshold"])

    min_games_month = int(cfg["sampling"]["min_games_per_player_month"])
    preselect_players = int(cfg["sampling"]["preselect_players_per_bin"])
    enforce_same_bin = bool(cfg["sampling"]["enforce_same_bin_for_game_elo"])

    # --- IO ---
    moves_path = cfg["input"]["moves_csv_path"]
    month_prefix = cfg["input"].get("filter_month_prefix")
    time_control = cfg["input"].get("filter_time_control")

    cache_path = cfg["cache"]["games_meta_parquet"]
    rebuild_cache = bool(cfg["cache"].get("rebuild_games_meta", True))

    # Output paths (CSVs)
    out_player_games_csv = cfg["output"]["player_games_csv"]
    out_players_csv = cfg["output"]["players_csv"]
    out_games_csv = cfg["output"]["games_csv"]
    out_moves_csv = cfg["output"]["moves_csv"]

    for p in [out_player_games_csv, out_players_csv, out_games_csv, out_moves_csv, cache_path]:
        os.makedirs(Path(p).parent, exist_ok=True)

    # --- build moves WHERE clause (optional) ---
    where = [f"{G} IS NOT NULL", f"{PLY} IS NOT NULL"]
    if month_prefix:
        # Date might be inferred as DATE; LIKE works only on strings.
        where.append(f"CAST({DATE} AS VARCHAR) LIKE '{month_prefix}%'")
    if time_control:
        where.append(f"{TC} = '{time_control}'")
    where_sql = " AND ".join(where)

    # --- Stage 0: define a base view over the raw moves file (filtered) ---
    log("Stage 0/8: creating filtered view over raw moves CSV")
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW raw_moves AS
    SELECT *
    FROM read_csv_auto('{moves_path}', header=true)
    WHERE {where_sql};
    """)

    # Warn (but do not drop) rows with missing clock + show affected game_ids (first 50)
    row = con.execute(f"SELECT COUNT(*) FROM raw_moves WHERE {CLK} IS NULL;").fetchone()
    n_missing_clock = int(row[0]) if row is not None else 0
    if n_missing_clock > 0:
        log(f"WARNING: raw_moves contains {n_missing_clock:,} rows with empty clock; keeping them (clock will be blank in output).")

        missing_game_ids = con.execute(f"""
            SELECT DISTINCT {G} AS game_id
            FROM raw_moves
            WHERE {CLK} IS NULL
            ORDER BY {G}
            LIMIT 50;
        """).fetchall()

        if not missing_game_ids:
            log("  (no game_ids found with missing clock; unexpected given the count above)")
        else:
            log("  First 50 game_ids with missing clock:")
            for (game_id,) in missing_game_ids:
                log(f"    {game_id}")

    # --- Stage 1: build or read games_meta (game-level table) ---
    log("Stage 1/8: building (or loading) game-level cache (games_meta)")
    if (not rebuild_cache) and Path(cache_path).exists():
        con.execute(f"CREATE OR REPLACE TEMP VIEW games_meta AS SELECT * FROM read_parquet('{cache_path}');")
        log(f"Loaded games_meta from cache: {cache_path}")
    else:
        log("Computing games_meta from raw moves (this can take a while on full month dumps)")
        sql_build_cache = f"""
        COPY (
          WITH moves AS (
            SELECT
              {G} AS game_id,
              {DATE} AS date,
              {W} AS white,
              {B} AS black,
              CAST({WE} AS INTEGER) AS white_elo,
              CAST({BE} AS INTEGER) AS black_elo,
              {TC} AS time_control,
              {TERM} AS termination,
              {RES} AS result,
              CAST({PLY} AS INTEGER) AS ply,
              CAST({CLK} AS DOUBLE) AS clock
            FROM raw_moves
          )
          SELECT
            game_id,
            ANY_VALUE(date) AS date,
            ANY_VALUE(white) AS white,
            ANY_VALUE(black) AS black,
            ANY_VALUE(white_elo) AS white_elo,
            ANY_VALUE(black_elo) AS black_elo,
            ANY_VALUE(time_control) AS time_control,
            ANY_VALUE(termination) AS termination,
            ANY_VALUE(result) AS result,
            MAX(ply) AS n_plies,

            -- first ply where WHITE clock <= thr (white moves are odd ply)
            MIN(ply) FILTER (WHERE (ply % 2)=1 AND clock <= {thr}) AS first_white_leq_thr_ply,
            -- first ply where BLACK clock <= thr (black moves are even ply)
            MIN(ply) FILTER (WHERE (ply % 2)=0 AND clock <= {thr}) AS first_black_leq_thr_ply,

            -- plies while BOTH players stayed >thr:
            CASE
              WHEN LEAST(
                     COALESCE(MIN(ply) FILTER (WHERE (ply % 2)=1 AND clock <= {thr}), 1000000000),
                     COALESCE(MIN(ply) FILTER (WHERE (ply % 2)=0 AND clock <= {thr}), 1000000000)
                   ) = 1000000000
                THEN MAX(ply)
              ELSE LEAST(
                     COALESCE(MIN(ply) FILTER (WHERE (ply % 2)=1 AND clock <= {thr}), 1000000000),
                     COALESCE(MIN(ply) FILTER (WHERE (ply % 2)=0 AND clock <= {thr}), 1000000000)
                   ) - 1
            END AS plies_both_gt_thr
          FROM moves
          GROUP BY game_id
        ) TO '{cache_path}' (FORMAT PARQUET);
        """
        con.execute(sql_build_cache)
        con.execute(f"CREATE OR REPLACE TEMP VIEW games_meta AS SELECT * FROM read_parquet('{cache_path}');")
        log(f"Built and cached games_meta: {cache_path}")

    # --- Stage 2: build a per-player view of games (one row per player per game) ---
    # For each game_id we create two rows:
    #   - White perspective: player=white, opponent=black, player_elo=white_elo, player_color='white'
    #   - Black perspective: player=black, opponent=white, player_elo=black_elo, player_color='black'
    # This makes later steps (player stats, binning, sampling N games per player) uniform and straightforward.
    log("Stage 2/8: creating player_games view (player perspective, 2 rows per game)")
    con.execute("""
    CREATE OR REPLACE TEMP VIEW player_games AS
    SELECT
      game_id,
      date,
      time_control,
      termination,
      result,
      n_plies,
      plies_both_gt_thr,
      white AS player,
      black AS opponent,
      white_elo AS player_elo,
      black_elo AS opponent_elo,
      'white' AS player_color
    FROM games_meta
    UNION ALL
    SELECT
      game_id,
      date,
      time_control,
      termination,
      result,
      n_plies,
      plies_both_gt_thr,
      black AS player,
      white AS opponent,
      black_elo AS player_elo,
      white_elo AS opponent_elo,
      'black' AS player_color
    FROM games_meta;
    """)

    # --- Stage 3: month stats per player (avg elo + n games >= threshold), then bin by avg elo ---
    log("Stage 3/8: computing per-player monthly stats and avg-bin assignment")
    sql_players_stats = f"""
    CREATE OR REPLACE TEMP VIEW player_month_stats AS
    SELECT
      player,
      AVG(player_elo) AS avg_elo,
      COUNT(*) AS n_games
    FROM player_games
    GROUP BY player
    HAVING COUNT(*) >= {min_games_month};

    CREATE OR REPLACE TEMP VIEW players_binned AS
    SELECT
      player,
      avg_elo,
      n_games,
      {avg_bin_expr} AS avg_bin
    FROM player_month_stats;
    """
    con.execute(sql_players_stats)

    # --- Stage 4: preselect players per bin (deterministic) ---
    log("Stage 4/8: preselecting players per bin")
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW preselected_players AS
    SELECT avg_bin, player, avg_elo, n_games
    FROM (
      SELECT
        avg_bin, player, avg_elo, n_games,
        ROW_NUMBER() OVER (PARTITION BY avg_bin ORDER BY hash(player)) AS rn
      FROM players_binned
    )
    WHERE rn <= {preselect_players};
    """)

    # --- Stage 5: eligible games for those players (plies constraint + same-bin-in-game constraint) ---
    log("Stage 5/8: filtering eligible games for preselected players")
    same_bin_pred = "TRUE"
    if enforce_same_bin:
        same_bin_pred = "player_bin = avg_bin"

    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW eligible_player_games AS
    SELECT
      pg.*,
      psp.avg_elo,
      psp.avg_bin,
      psp.n_games AS n_games,
      {game_bin_expr} AS player_bin
    FROM player_games pg
    JOIN preselected_players psp
      ON psp.player = pg.player
    WHERE pg.plies_both_gt_thr >= {min_plies}
      AND {same_bin_pred};
    """)

    # --- Stage 6: select players per bin and assign split_by_player ---
    if not drop_games_vs_selected_opponents:
        log("Stage 6/8: selecting final players per bin and splitting players into test/train (split_by_player)")
        con.execute(f"""
        CREATE OR REPLACE TEMP VIEW players_with_enough_games AS
        SELECT
          avg_bin,
          player,
          ANY_VALUE(avg_elo) AS avg_elo,
          ANY_VALUE(n_games) AS n_games,
          COUNT(*) AS n_eligible
        FROM eligible_player_games
        GROUP BY avg_bin, player
        HAVING COUNT(*) >= {games_per_player};

        CREATE OR REPLACE TEMP VIEW final_players AS
        SELECT avg_bin, player, avg_elo, n_games, n_eligible
        FROM (
          SELECT
            avg_bin, player, avg_elo, n_games, n_eligible,
            ROW_NUMBER() OVER (PARTITION BY avg_bin ORDER BY hash('{player_salt}:final:' || player)) AS rn
          FROM players_with_enough_games
        )
        WHERE rn <= {final_players_per_bin};

        CREATE OR REPLACE TEMP VIEW player_split AS
        SELECT
          avg_bin,
          player,
          avg_elo,
          n_games,
          n_eligible,
          CASE
            WHEN ROW_NUMBER() OVER (PARTITION BY avg_bin ORDER BY hash('{player_salt}:test:' || player)) <= {test_players_per_bin}
              THEN 'test'
            ELSE 'train'
          END AS split_by_player
        FROM final_players;

        CREATE OR REPLACE TEMP VIEW eligible_player_games_for_sampling AS
        SELECT epg.*
        FROM eligible_player_games epg
        JOIN player_split ps
          ON ps.avg_bin = epg.avg_bin
         AND ps.player = epg.player;
        """)
    else:
        candidate_players_per_bin = final_players_per_bin * candidate_players_multiplier
        log("Stage 6/8: selecting final players per bin under opponent constraint (two-pass)")

        con.execute(f"""
        -- Pass A: start with players who have enough eligible games (pre-constraint),
        -- then pick a larger deterministic candidate pool per bin.
        CREATE OR REPLACE TEMP VIEW players_with_enough_games_pre AS
        SELECT
          avg_bin,
          player,
          ANY_VALUE(avg_elo) AS avg_elo,
          ANY_VALUE(n_games) AS n_games,
          COUNT(*) AS n_eligible_pre
        FROM eligible_player_games
        GROUP BY avg_bin, player
        HAVING COUNT(*) >= {games_per_player};

        CREATE OR REPLACE TEMP VIEW candidate_players AS
        SELECT avg_bin, player, avg_elo, n_games, n_eligible_pre
        FROM (
          SELECT
            avg_bin, player, avg_elo, n_games, n_eligible_pre,
            ROW_NUMBER() OVER (
              PARTITION BY avg_bin
              ORDER BY hash('{player_salt}:candidate:' || player)
            ) AS rn
          FROM players_with_enough_games_pre
        )
        WHERE rn <= {candidate_players_per_bin};

        -- Pass B: keep only games where the opponent is NOT in the selected player pool,
        -- then recompute per-player eligibility counts and select final players.
        CREATE OR REPLACE TEMP VIEW eligible_player_games_dedup AS
        SELECT epg.*
        FROM eligible_player_games epg
        JOIN candidate_players cp
          ON cp.avg_bin = epg.avg_bin
         AND cp.player  = epg.player
        LEFT JOIN candidate_players cp_opp
          ON cp_opp.player = epg.opponent
        WHERE cp_opp.player IS NULL;

        CREATE OR REPLACE TEMP VIEW players_with_enough_games_post AS
        SELECT
          cp.avg_bin AS avg_bin,
          cp.player  AS player,
          ANY_VALUE(cp.avg_elo) AS avg_elo,
          ANY_VALUE(cp.n_games) AS n_games,
          COUNT(*) AS n_eligible
        FROM eligible_player_games_dedup epg
        JOIN candidate_players cp
          ON cp.avg_bin = epg.avg_bin
         AND cp.player  = epg.player
        GROUP BY cp.avg_bin, cp.player
        HAVING COUNT(*) >= {games_per_player};

        CREATE OR REPLACE TEMP VIEW final_players AS
        SELECT avg_bin, player, avg_elo, n_games, n_eligible
        FROM (
          SELECT
            avg_bin, player, avg_elo, n_games, n_eligible,
            ROW_NUMBER() OVER (PARTITION BY avg_bin ORDER BY hash('{player_salt}:final:' || player)) AS rn
          FROM players_with_enough_games_post
        )
        WHERE rn <= {final_players_per_bin};

        CREATE OR REPLACE TEMP VIEW player_split AS
        SELECT
          avg_bin,
          player,
          avg_elo,
          n_games,
          n_eligible,
          CASE
            WHEN ROW_NUMBER() OVER (PARTITION BY avg_bin ORDER BY hash('{player_salt}:test:' || player)) <= {test_players_per_bin}
              THEN 'test'
            ELSE 'train'
          END AS split_by_player
        FROM final_players;

        CREATE OR REPLACE TEMP VIEW eligible_player_games_for_sampling AS
        SELECT epg.*
        FROM eligible_player_games_dedup epg
        JOIN player_split ps
          ON ps.avg_bin = epg.avg_bin
         AND ps.player = epg.player;
        """)

    # --- Stage 7: split_by_games (by game_id), sample games per player, attach both splits ---
    log("Stage 7/8: sampling games per player and assigning split_by_player + split_by_games")
    con.execute(f"""
    CREATE OR REPLACE TEMP VIEW game_split AS
    WITH g AS (
      SELECT DISTINCT game_id
      FROM eligible_player_games_for_sampling
    )
    SELECT
      game_id,
      CASE
        WHEN (abs(hash(game_id || '{game_salt}')) % 1000000) < CAST({test_game_frac} * 1000000 AS BIGINT)
          THEN 'test'
        ELSE 'train'
      END AS split_by_games
    FROM g;

    CREATE OR REPLACE TEMP VIEW sampled_player_games AS
    SELECT
      epg.avg_bin AS rating_bin,
      epg.player,
      epg.opponent,
      epg.player_color,
      epg.game_id,
      epg.player_elo,
      epg.opponent_elo,
      epg.avg_elo,
      epg.date,
      epg.time_control,
      epg.termination,
      epg.result,
      epg.n_plies,
      epg.plies_both_gt_thr,
      ps.n_games AS player_n_games_month,
      ps.n_eligible AS player_n_eligible,
      ps.split_by_player,
      gs.split_by_games,
      ROW_NUMBER() OVER (
        PARTITION BY epg.avg_bin, epg.player
        ORDER BY hash(epg.game_id || epg.player)
      ) AS game_rn
    FROM eligible_player_games_for_sampling epg
    JOIN player_split ps
      ON ps.avg_bin = epg.avg_bin
     AND ps.player = epg.player
    JOIN game_split gs
      ON gs.game_id = epg.game_id;

    CREATE OR REPLACE TEMP VIEW dataset_player_games AS
    SELECT
      split_by_player,
      split_by_games,
      rating_bin,
      player,
      opponent,
      player_color,
      game_id,
      player_elo,
      opponent_elo,
      avg_elo,
      date,
      time_control,
      termination,
      result,
      n_plies,
      plies_both_gt_thr,
      player_n_games_month,
      player_n_eligible
    FROM sampled_player_games
    WHERE game_rn <= {games_per_player};
    """)

    # --- Stage 8: write outputs (CSV) ---
    log("Stage 8/8: writing outputs (CSV)")

    # 1) players dataset with full required per-player stats
    con.execute(f"""
    COPY (
      SELECT
        avg_bin AS rating_bin,
        player,
        split_by_player,
        avg_elo,
        n_games AS n_games_month,
        n_eligible AS n_eligible_games
      FROM player_split
      ORDER BY rating_bin, split_by_player, player
    ) TO '{out_players_csv}' (HEADER, DELIMITER ',');
    """)
    log(f"Wrote players CSV: {out_players_csv}")

    # 2) games dataset with required per-game stats + split_by_player + split_by_games
    con.execute(f"""
    COPY (
      WITH pg AS (SELECT * FROM dataset_player_games),
      selected_games AS (
        SELECT DISTINCT game_id FROM pg
      ),
      split_by_player_game AS (
        SELECT
          game_id,
          CASE
            WHEN MAX(CASE WHEN split_by_player='test' THEN 1 ELSE 0 END)=1 THEN 'test'
            ELSE 'train'
          END AS split_by_player
        FROM pg
        GROUP BY game_id
      )
      SELECT
        sbpg.split_by_player,
        gs.split_by_games,
        gm.game_id,
        gm.date,
        gm.white,
        gm.black,
        gm.white_elo,
        gm.black_elo,
        gm.time_control,
        gm.termination,
        gm.result,
        gm.n_plies,
        gm.plies_both_gt_thr
      FROM games_meta gm
      JOIN selected_games sg ON sg.game_id = gm.game_id
      JOIN split_by_player_game sbpg ON sbpg.game_id = gm.game_id
      JOIN game_split gs            ON gs.game_id = gm.game_id
      ORDER BY sbpg.split_by_player, gs.split_by_games, gm.game_id
    ) TO '{out_games_csv}' (HEADER, DELIMITER ',');
    """)
    log(f"Wrote games CSV: {out_games_csv}")

    # 3) moves dataset: move-level rows for selected games, truncated at cutoff ply per game
    #    Export columns in snake_case (not raw source column names).
    con.execute(f"""
    COPY (
      WITH selected_games AS (
        SELECT DISTINCT game_id
        FROM dataset_player_games
      ),
      cutoffs AS (
        SELECT
          gm.game_id,
          gm.plies_both_gt_thr AS cutoff_ply
        FROM games_meta gm
        JOIN selected_games sg ON sg.game_id = gm.game_id
      )
      SELECT
        rm.{G}    AS game_id,
        rm.{DATE} AS date,
        rm.{W}    AS white,
        rm.{B}    AS black,
        rm.{WE}   AS white_elo,
        rm.{BE}   AS black_elo,
        rm.{TC}   AS time_control,
        rm.{TERM} AS termination,
        rm.{RES}  AS result,
        rm.{PLY}  AS half_move,
        rm.{MOVE} AS move,
        rm.{CLK}  AS clock
      FROM raw_moves rm
      JOIN cutoffs c ON rm.{G} = c.game_id
      WHERE CAST(rm.{PLY} AS INTEGER) <= c.cutoff_ply
      ORDER BY rm.{G}, CAST(rm.{PLY} AS INTEGER)
    ) TO '{out_moves_csv}' (HEADER, DELIMITER ',');
    """)
    log(f"Wrote moves CSV: {out_moves_csv}")

    # 4) player-games dataset (index): drop player_n_games_month, player_n_eligible from final output
    con.execute(f"""
    COPY (
      SELECT
        split_by_player,
        split_by_games,
        rating_bin,
        player,
        opponent,
        player_color,
        game_id,
        player_elo,
        opponent_elo,
        avg_elo,
        date,
        time_control,
        termination,
        result,
        n_plies,
        plies_both_gt_thr
      FROM dataset_player_games
    ) TO '{out_player_games_csv}' (HEADER, DELIMITER ',');
    """)
    log(f"Wrote player-games CSV: {out_player_games_csv}")

    con.close()

    log("Done.")
    log(f"games_meta cache: {cache_path}")


if __name__ == "__main__":
    main()
