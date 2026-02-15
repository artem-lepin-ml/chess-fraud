import pandas as pd
import pickle
from sklearn.model_selection import GroupShuffleSplit

from utils import load_config, CollectionConfig
from utils.collection_utils import make_collections_indices_for_all_players


def make_train_dataset_indices(train_df, train_n_collections, data_output_dir):
    """Create train dataset indices."""
    train_indices_dataset_data = make_collections_indices_for_all_players(
        train_df,
        cheat_column=None,
        n_collections=train_n_collections,
    )
    with open(f"{data_output_dir}/train_dataset_indices.pickle", mode="wb") as file:
        pickle.dump(train_indices_dataset_data, file)


def make_val_and_test_dataset_indices(val_df, test_df, test_n_collections, data_output_dir):
    """Create validation and test dataset indices."""
    val_indices_dataset_data =\
        make_collections_indices_for_all_players(
            val_df, cheat_column=None, n_collections=test_n_collections
        )
    test_indices_dataset_data =\
        make_collections_indices_for_all_players(
            test_df, cheat_column=None, n_collections=test_n_collections
        )

    with open(f"{data_output_dir}/val_dataset_indices.pickle", mode="wb") as file:
        pickle.dump(val_indices_dataset_data, file)

    with open(f"{data_output_dir}/test_dataset_indices.pickle", mode="wb") as file:
        pickle.dump(test_indices_dataset_data, file)


def make_collections_data(cfg):
    """
    Main function to create collections data from the dataset.

    Args:
        cfg: Configuration object containing data paths and collection settings
    """
    # Load the dataset
    data_path = cfg.data.csv_path
    df = pd.read_csv(data_path)

    # Mark unimprovable moves (moves that match Stockfish top line)
    df["unimprovable"] = False
    df.loc[df["move_uci"] == df["move_stockfish_15"], "unimprovable"] = True

    # Filter by ELO
    max_elo = cfg.elo.bounds[-1]
    df = df[df["player_elo"] <= max_elo]

    # Split data into train, validation, and test
    train_val_df = df[df["split_by_player"] == "train"]
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=cfg.training.seed,
    )
    train_idx, val_idx = next(gss.split(train_val_df, groups=train_val_df["player"]))
    train_df = train_val_df.iloc[train_idx]
    val_df = train_val_df.iloc[val_idx]
    test_df = df[df["split_by_player"] == "test"]

    # Get collection configuration
    collection_cfg = CollectionConfig.from_cfg(cfg)
    data_output_dir = cfg.data.data_output_dir

    # Create dataset indices
    make_train_dataset_indices(train_df, collection_cfg.train_n_collections, data_output_dir)
    make_val_and_test_dataset_indices(
        val_df, test_df, collection_cfg.test_n_collections, data_output_dir
    )


if __name__ == "__main__":
    # Load configuration
    cfg = load_config("config.yaml")

    # Create collections data
    make_collections_data(cfg)
