import pandas as pd
import pickle
from sklearn.model_selection import GroupShuffleSplit
from humanlike.collection_cls_experiment.utils.collection_utils import make_collections_indices_for_all_players
from consts import (
    TRAIN_N_COLLECTIONS,
    TEST_N_COLLECTIONS,
    TRAIN_INDICES_PATH,
    VAL_INDICES_PATH,
    TEST_INDICES_PATH,
    MAX_HIGH_ELO,
    SYNT_DATA_PATH
    )


def make_train_dataset_indices(train_df):
    train_indices_dataset_data = make_collections_indices_for_all_players(
        train_df,
        cheat_column=None,
        n_collections=TRAIN_N_COLLECTIONS,
    )
    with open(TRAIN_INDICES_PATH, mode="wb") as file:
        pickle.dump(train_indices_dataset_data, file)


def make_val_and_test_dataset_indices(val_df, test_df):
    val_indices_dataset_data =\
        make_collections_indices_for_all_players(
            val_df, cheat_column=None, n_collections=TEST_N_COLLECTIONS
        )
    test_indices_dataset_data =\
        make_collections_indices_for_all_players(
            test_df, cheat_column=None, n_collections=TEST_N_COLLECTIONS
        )

    with open(VAL_INDICES_PATH, mode="wb") as file:
        pickle.dump(val_indices_dataset_data, file)
    
    with open(TEST_INDICES_PATH, mode="wb") as file:
        pickle.dump(test_indices_dataset_data, file)


def make_collections_data():
    df = pd.read_csv(SYNT_DATA_PATH)
    df["unimprovable"] = False
    df.loc[df["move_uci"] == df["move_stockfish_15"], "unimprovable"] = True
    df = df[df["player_elo"] <= MAX_HIGH_ELO]
    train_val_df = df[df["split_by_player"] == "train"]
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42,
    )
    train_idx, val_idx = next(gss.split(train_val_df, groups=train_val_df["player"]))
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    test_df = df[df["split_by_player"] == "test"]
    make_train_dataset_indices(train_df)
    make_val_and_test_dataset_indices(val_df, test_df)


if __name__ == "__main__":
    make_collections_data()
