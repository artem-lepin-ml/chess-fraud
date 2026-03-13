import random
from dataclasses import dataclass, field
from tqdm import tqdm

from consts import *

@dataclass
class CollectionIndices:
    fair_indices: list
    cheat_indices: list
@dataclass
class CollectionSample:
    fair_collection: CollectionIndices | None
    cheat_collection: CollectionIndices | None


@dataclass
class DatasetIndicesData:
    # list over players, for each player list of CollectionSample
    collection_indices: list[list[CollectionSample]]
    elo_list: list
    p_list: list
    cheat_column_list: list | None = None


def choose_cheat_column():
    cheat_column = random.choice(ALLIE_CHEAT_COLUMNS)
    return cheat_column

# ========================================
#           CREATING INDICES
# ========================================

def make_cheat_move_index(df, games):
    """
    Avoids unimprovable during selection
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
    Can include unimprovable
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
    """
    Creates CollectionIndices for one collection
    """
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
    """
    Creates collections indices for a given player
    """
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


def make_collections_indices_for_all_players(df, cheat_column=None, n_collections=1):
    """
    Creates all collections indices for all players
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


# ========================================
#       BUILDING FROM INDICES
# ========================================

def get_one_collection_embs_from_indices(fair_arr, cheat_arr, collection_indices, fen_before=None):
    """
    Returns embs and labels
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
    Returns fair_collections, cheat_collections, cheat_label_list(only for cheat side)
    for a given player
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


def make_tournament_collections_from_indices(fair_arr, player_indices: list[CollectionSample], fen_before=None):
    """
    Builds all collections from the tournament games
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
    Builds all collections for all players from indices
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
