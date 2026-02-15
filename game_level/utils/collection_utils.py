# collection_utils.py
# ============================================================
# Legacy (2-class) code is kept as-is.
# New 3-class pipeline is added in a way that:
#   - does NOT pass df into "from_indices" builders
#   - does NOT change p / selection logic when assigning class=2
#   - produces balanced number of fair/cheat collections per player
#   - stores unimprovable_indices inside CollectionIndices
#   - returns move-level labels for BOTH fair and cheat collections
# ============================================================

import random
from dataclasses import dataclass, field
from tqdm import tqdm

import numpy as np  # kept (often used elsewhere)
import pandas as pd  # kept (often used elsewhere)

from .consts import (
    ALLIE_CHEAT_COLUMNS,
    MAIA2_CHEAT_COLUMNS,
    K_MIN,
    K_MAX,
    MIN_HMOVE,
    MAX_HMOVE,
    MIN_CHEAT_P,
    MAX_CHEAT_P,
)  # expects: ALLIE_CHEAT_COLUMNS, K_MIN, K_MAX, MIN_HMOVE, MIN_CHEAT_P, MAX_CHEAT_P, etc.


# =========================
#       DATA STRUCTS
# =========================

@dataclass
class CollectionIndices:
    fair_indices: list
    cheat_indices: list
    # 3cls extension: indices that should be labeled as class=2
    unimprovable_indices: list = field(default_factory=list)


@dataclass
class CollectionSample:
    # for legacy: always both exist (fair + cheat variant)
    # for tournament legacy: only one may exist
    fair_collection: CollectionIndices | None
    cheat_collection: CollectionIndices | None


@dataclass
class DatasetIndicesData:
    # list over players, for each player list of CollectionSample
    collection_indices: list[list[CollectionSample]]
    elo_list: list
    p_list: list
    cheat_column_list: list | None = None


# =========================
#       EXISTING (KEEP)
# =========================

def choose_cheat_column():
    cheat_column = random.choice(ALLIE_CHEAT_COLUMNS)
    return cheat_column


def make_cheat_move_index(df, games):
    """
    Legacy cheat move selector: avoids unimprovable during selection.
    """
    possible_stops = []
    while not possible_stops:
        game_id = random.choice(games)
        group = df.loc[df["game_id"] == game_id]
        unimprovable_list = group["unimprovable"].to_list()
        row = group.iloc[0]
        if row["player_color"] == "white":
            possible_stops = [i for i in range(MIN_HMOVE, len(group), 2)
                              if not unimprovable_list[i - 1]]
        else:
            possible_stops = [i for i in range(MIN_HMOVE + 1, len(group), 2)
                              if not unimprovable_list[i - 1]]
    stop_hmove = random.choice(possible_stops)
    move_idx = group.index[stop_hmove - 1]
    return move_idx


def make_fair_move_index(df, games):
    """
    Legacy fair move selector: can include unimprovable.
    """
    game_id = random.choice(games)
    group = df.loc[df["game_id"] == game_id]
    row = group.iloc[0]
    if row["player_color"] == "white":
        possible_stops = [i for i in range(MIN_HMOVE, len(group), 2)]
    else:
        possible_stops = [i for i in range(MIN_HMOVE + 1, len(group), 2)]
    stop_hmove = random.choice(possible_stops)
    move_idx = group.index[stop_hmove - 1]
    return move_idx


def get_one_collection_embs_indices(df, collection_len, p):
    """Legacy 2-class indices: p is share of cheat-side indices."""
    games = df["game_id"].unique()
    assert len(games) > 0, f"Expected at least 1 game, got {len(games)}"

    n_cheat_samples = int(collection_len * p)
    n_fair_samples = collection_len - n_cheat_samples
    cheat_move_indices = []
    fair_move_indices = []

    for _ in range(n_fair_samples):
        fair_move_indices.append(make_fair_move_index(df, games))

    for _ in range(n_cheat_samples):
        cheat_move_indices.append(make_cheat_move_index(df, games))

    return CollectionIndices(fair_move_indices, cheat_move_indices)


def make_player_collections_indices(df, n_collections):
    """Legacy: per collection creates both fair(p=0) and cheat(p~U)."""
    all_collections_indices = []
    p_list = []
    row = df.iloc[0]
    elo = row["player_elo"]

    for _ in range(n_collections):
        collection_len = random.randint(K_MIN, K_MAX)
        p = random.uniform(MIN_CHEAT_P, MAX_CHEAT_P)
        p_list.append(p)

        cheat_collection_indices = get_one_collection_embs_indices(df, collection_len, p=p)
        fair_collection_indices = get_one_collection_embs_indices(df, collection_len, p=0.0)

        all_collections_indices.append(CollectionSample(fair_collection_indices, cheat_collection_indices))

    return all_collections_indices, [elo] * len(p_list), p_list


# ========================================
#       BUILDING FROM INDICES (KEEP)
# ========================================

def get_one_collection_embs_from_indices(fair_arr, cheat_arr, collection_indices, fen_before=None):
    """
    Legacy 2-class: returns embs and {0,1} labels based on side.
    """
    fair_emb_list = []
    cheat_emb_list = []

    for fair_move_idx in collection_indices.fair_indices:
        fair_emb = fair_arr[fair_move_idx]
        if fen_before is not None:
            fair_emb = fair_emb + fen_before[fair_move_idx]
        fair_emb_list.append(fair_emb)

    for cheat_move_idx in collection_indices.cheat_indices:
        cheat_emb = cheat_arr[cheat_move_idx]
        if fen_before is not None:
            cheat_emb = cheat_emb + fen_before[cheat_move_idx]
        cheat_emb_list.append(cheat_emb)

    return (
        fair_emb_list + cheat_emb_list,
        [0.0] * len(fair_emb_list) + [1.0] * len(cheat_emb_list),
    )


def make_player_collections_from_indices(fair_arr, cheat_arr, player_indices, fen_before=None):
    """
    Legacy 2-class: returns fair_collections, cheat_collections, cheat_label_list(only for cheat side).
    """
    fair_collections = []
    cheat_collections = []
    cheat_label_list = []

    for collection_indices in player_indices:
        fair_emb_list, _ = get_one_collection_embs_from_indices(
            fair_arr, cheat_arr, collection_indices.fair_collection, fen_before
        )
        fair_collections.append(fair_emb_list)

        cheat_emb_list, cheat_labels = get_one_collection_embs_from_indices(
            fair_arr, cheat_arr, collection_indices.cheat_collection, fen_before
        )
        cheat_label_list.append(cheat_labels)
        cheat_collections.append(cheat_emb_list)

    return fair_collections, cheat_collections, cheat_label_list


def make_collections_indices_for_all_players(df, cheat_column=None, n_collections=1):
    """
    Legacy indices builder.
    cheat_column is None => per-player cheat_column_list sampled.
    """
    collection_indices = []
    elo_list = []
    p_list = []
    cheat_column_list = []

    for _, group in tqdm(df.groupby("player"), total=df["player"].nunique(), desc="Building dataset"):
        if cheat_column is None:
            cheat_column_list.append(choose_cheat_column())

        player_collections_indices, player_elo_list, player_p_list = make_player_collections_indices(group, n_collections)
        collection_indices.append(player_collections_indices)
        elo_list += player_elo_list
        p_list += player_p_list

    dataset_data = DatasetIndicesData(collection_indices, elo_list, p_list)
    if cheat_column is None:
        dataset_data.cheat_column_list = cheat_column_list
    return dataset_data


def make_tournament_collections_from_indices(fair_arr, player_indices: list[CollectionSample], fen_before=None):
    """
    Legacy tournament path:
      - each CollectionSample has ONLY fair_collection OR cheat_collection.
      - for fair: put all indices into fair side
      - for cheat: uses fair_arr for both (because tournament stores only real moves)
    """
    fair_collections = []
    cheat_collections = []
    cheat_labels = []

    for collection_sample in player_indices:
        if collection_sample.fair_collection is not None:
            assert collection_sample.cheat_collection is None, "only one variant: cheat OR fair"
            assert not collection_sample.fair_collection.cheat_indices, "tournament: all moves are in fair_indices"
            fair_emb_list, _ = get_one_collection_embs_from_indices(
                fair_arr, None, collection_sample.fair_collection, fen_before
            )
            fair_collections.append(fair_emb_list)

        elif collection_sample.cheat_collection is not None:
            assert collection_sample.fair_collection is None, "only one variant: cheat OR fair"
            cheat_emb_list, cheat_labels_list = get_one_collection_embs_from_indices(
                fair_arr, fair_arr, collection_sample.cheat_collection, fen_before
            )
            cheat_collections.append(cheat_emb_list)
            cheat_labels.append(cheat_labels_list)

        else:
            raise AssertionError("both collections are None")

    return fair_collections, cheat_collections, cheat_labels


def make_collections_from_indices(emb_dict, dataset_data, cheat_column, tournament=False):
    """
    Legacy builder.
    """
    fair_emb_list = []
    cheat_emb_list = []
    cheat_label_list = []

    fair_arr = emb_dict.get("move_uci")
    if fair_arr is None:
        fair_arr = emb_dict["fen_after"]

    if cheat_column is not None and not tournament:
        cheat_arr = emb_dict.get(cheat_column)
        if cheat_arr is None:
            cheat_column = "_".join(["fen"] + cheat_column.split("_")[1:])
            cheat_arr = emb_dict[cheat_column]

    for i, player_indices in tqdm(enumerate(dataset_data.collection_indices), desc="Building dataset"):
        if cheat_column is None and not tournament:
            player_cheat_column = dataset_data.cheat_column_list[i]
            cheat_arr = emb_dict.get(player_cheat_column)
            if cheat_arr is None:
                player_cheat_column = "_".join(["fen"] + player_cheat_column.split("_")[1:])
                cheat_arr = emb_dict[player_cheat_column]

        if tournament:
            player_fair_emb_list, player_cheat_emb_list, player_cheat_label_list = make_tournament_collections_from_indices(
                fair_arr, player_indices, emb_dict.get("fen_before")
            )
            fair_emb_list += player_fair_emb_list
            cheat_emb_list += player_cheat_emb_list
            cheat_label_list += player_cheat_label_list
            continue

        player_fair_emb_list, player_cheat_emb_list, player_cheat_label_list = make_player_collections_from_indices(
            fair_arr, cheat_arr, player_indices, emb_dict.get("fen_before")
        )
        fair_emb_list += player_fair_emb_list
        cheat_emb_list += player_cheat_emb_list
        cheat_label_list += player_cheat_label_list

    if tournament:
        return fair_emb_list, cheat_emb_list, cheat_label_list, dataset_data.p_list, dataset_data.elo_list

    return fair_emb_list, cheat_emb_list, cheat_label_list, dataset_data.p_list, dataset_data.elo_list


# ============================================================
#                     NEW: 3-CLASS PIPELINE
# ============================================================
# Goal (per your spec):
#   - indices selection logic is unchanged (no df checks for unimprovable during selection)
#   - only AFTER indices are drawn we mark which of them are unimprovable (class=2)
#   - p refers to the share of "cheat-side indices" in the collection, regardless of unimprovable status
#   - per player we generate pairs: fair collection first (p=0), then cheat collection (p~U)
#   - from_indices does NOT take df; it uses collection_indices.unimprovable_indices only
# ============================================================

def get_one_collection_embs_indices_3cls(df, collection_len, p):
    """
    3cls indices builder.

    Selection:
      - fair_indices: drawn via make_fair_move_index (legacy)
      - cheat_indices: ALSO drawn via make_fair_move_index (legacy) [IMPORTANT per your spec]
        => unimprovable may appear in cheat side, and that must NOT affect p.

    Labeling:
      - unimprovable_indices is computed AFTER selection, from df.loc[idx, "unimprovable"].
      - This does not change membership of fair/cheat indices => p remains intact.
    """
    games = df["game_id"].unique()
    assert len(games) > 0, f"Expected at least 1 game, got {len(games)}"
    assert "unimprovable" in df.columns, "df must contain 'unimprovable'"

    n_cheat = int(collection_len * p)
    n_fair = collection_len - n_cheat

    fair_indices = [make_fair_move_index(df, games) for _ in range(n_fair)]
    cheat_indices = [make_fair_move_index(df, games) for _ in range(n_cheat)]

    all_idx = fair_indices + cheat_indices
    unimprovable_indices = [idx for idx in all_idx if bool(df.loc[idx, "unimprovable"])]

    return CollectionIndices(
        fair_indices=fair_indices,
        cheat_indices=cheat_indices,
        unimprovable_indices=unimprovable_indices,
    )


def make_player_collections_indices_3cls(df, n_collections):
    """
    Balanced per-player:
      for each k in [1..n_collections]:
        1) fair collection (p=0)
        2) cheat collection (p~U[min,max])
    => equal number of fair/cheat collections per player.
    """
    all_collections = []
    p_list = []
    elo = df.iloc[0]["player_elo"]

    for _ in range(n_collections):
        K = random.randint(K_MIN, K_MAX)
        p = random.uniform(MIN_CHEAT_P, MAX_CHEAT_P)
        p_list.append(p)

        fair_col = get_one_collection_embs_indices_3cls(df, K, p=0.0)
        cheat_col = get_one_collection_embs_indices_3cls(df, K, p=p)

        all_collections.append(CollectionSample(fair_collection=fair_col, cheat_collection=cheat_col))

    return all_collections, [elo] * len(p_list), p_list


def make_collections_indices_for_all_players_3cls(df, cheat_column=None, n_collections=1):
    """
    3cls indices builder (exported name you import from scripts).

    If cheat_column is None => we keep legacy behaviour: store per-player cheat column.
    """
    collection_indices = []
    elo_list = []
    p_list = []
    cheat_column_list = []

    for _, group in tqdm(
        df.groupby("player"),
        total=df["player"].nunique(),
        desc="Building dataset (3cls indices)",
    ):
        if cheat_column is None:
            cheat_column_list.append(choose_cheat_column())

        player_collections_indices, player_elo_list, player_p_list = make_player_collections_indices_3cls(
            group, n_collections
        )

        collection_indices.append(player_collections_indices)
        elo_list += player_elo_list
        p_list += player_p_list

    dataset_data = DatasetIndicesData(
        collection_indices=collection_indices,
        elo_list=elo_list,
        p_list=p_list,
    )
    if cheat_column is None:
        dataset_data.cheat_column_list = cheat_column_list
    return dataset_data


def get_one_collection_embs_from_indices_3cls(
    fair_arr,
    cheat_arr,
    collection_indices: CollectionIndices,
    fen_before=None,
):
    """
    Build embeddings + 3-class labels WITHOUT df.

    labels:
      - fair_indices: 0 (fair) or 2 (unimprovable)
      - cheat_indices: 1 (cheat) or 2 (unimprovable)
    where "unimprovable" is taken from collection_indices.unimprovable_indices.
    """
    unimpr = set(collection_indices.unimprovable_indices)

    embs = []
    labels = []

    # fair side
    for idx in collection_indices.fair_indices:
        emb = fair_arr[idx]
        if fen_before is not None:
            emb = emb + fen_before[idx]
        embs.append(emb)
        labels.append(2 if idx in unimpr else 0)

    # cheat side
    for idx in collection_indices.cheat_indices:
        emb = cheat_arr[idx]
        if fen_before is not None:
            emb = emb + fen_before[idx]
        embs.append(emb)
        labels.append(2 if idx in unimpr else 1)

    return embs, labels


def make_player_collections_from_indices_3cls(
    fair_arr,
    cheat_arr,
    player_indices: list[CollectionSample],
    fen_before=None,
):
    """
    Returns:
      fair_collections: List[List[emb]]
      cheat_collections: List[List[emb]]
      fair_label_list:  List[List[int]]   in {0,2} (but we keep {0,1,2} formally)
      cheat_label_list: List[List[int]]   in {0,1,2} (cheat side includes 1/2, fair side includes 0/2)
    """
    fair_collections = []
    cheat_collections = []
    fair_label_list = []
    cheat_label_list = []

    for cs in player_indices:
        assert cs.fair_collection is not None and cs.cheat_collection is not None, \
            "3cls non-tournament path expects both fair and cheat collections per sample"

        fair_embs, fair_labels = get_one_collection_embs_from_indices_3cls(
            fair_arr, cheat_arr, cs.fair_collection, fen_before
        )
        cheat_embs, cheat_labels = get_one_collection_embs_from_indices_3cls(
            fair_arr, cheat_arr, cs.cheat_collection, fen_before
        )

        fair_collections.append(fair_embs)
        cheat_collections.append(cheat_embs)
        fair_label_list.append(fair_labels)
        cheat_label_list.append(cheat_labels)

    return fair_collections, cheat_collections, fair_label_list, cheat_label_list


def make_collections_from_indices_3cls(emb_dict, dataset_data, cheat_column, tournament=False):
    """
    3cls builder (non-tournament).

    Returns:
      fair_emb_list, cheat_emb_list,
      fair_label_list, cheat_label_list,
      p_list, elo_list
    """
    assert tournament is False, "3cls helper currently targets non-tournament augmentation path"

    fair_emb_list = []
    cheat_emb_list = []
    fair_label_list = []
    cheat_label_list = []

    fair_arr = emb_dict.get("move_uci")
    if fair_arr is None:
        fair_arr = emb_dict["fen_after"]

    if cheat_column is not None:
        cheat_arr = emb_dict.get(cheat_column)
        if cheat_arr is None:
            cheat_column = "_".join(["fen"] + cheat_column.split("_")[1:])
            cheat_arr = emb_dict[cheat_column]

    for i, player_indices in tqdm(enumerate(dataset_data.collection_indices), desc="Building dataset (3cls)"):
        if cheat_column is None:
            player_cheat_column = dataset_data.cheat_column_list[i]
            cheat_arr = emb_dict.get(player_cheat_column)
            if cheat_arr is None:
                player_cheat_column = "_".join(["fen"] + player_cheat_column.split("_")[1:])
                cheat_arr = emb_dict[player_cheat_column]

        pf_emb, pc_emb, pf_lbl, pc_lbl = make_player_collections_from_indices_3cls(
            fair_arr, cheat_arr, player_indices, emb_dict.get("fen_before")
        )

        fair_emb_list += pf_emb
        cheat_emb_list += pc_emb
        fair_label_list += pf_lbl
        cheat_label_list += pc_lbl

    return (
        fair_emb_list,
        cheat_emb_list,
        fair_label_list,
        cheat_label_list,
        dataset_data.p_list,
        dataset_data.elo_list,
    )


# Optional: make exports explicit (helps with ImportError confusion)
__all__ = [
    # dataclasses
    "CollectionIndices",
    "CollectionSample",
    "DatasetIndicesData",
    # legacy (2-class)
    "choose_cheat_column",
    "make_cheat_move_index",
    "make_fair_move_index",
    "get_one_collection_embs_indices",
    "make_player_collections_indices",
    "get_one_collection_embs_from_indices",
    "make_player_collections_from_indices",
    "make_collections_indices_for_all_players",
    "make_tournament_collections_from_indices",
    "make_collections_from_indices",
    # new (3-class)
    "get_one_collection_embs_indices_3cls",
    "make_player_collections_indices_3cls",
    "make_collections_indices_for_all_players_3cls",
    "get_one_collection_embs_from_indices_3cls",
    "make_player_collections_from_indices_3cls",
    "make_collections_from_indices_3cls",
]
