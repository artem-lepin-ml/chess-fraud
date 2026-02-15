"""
Collection method for game-level cheating detection.

Our approach uses the Chess Encoder model trained on move collections
to predict whether a player is cheating at the game level.
"""

import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from utils import (
    load_config,
    load_embeddings,
    load_pickle,
    ModelConfig,
    TrainingConfig,
    CollectionConfig,
    AllCheatColumns,
)
from utils.collection_utils import choose_cheat_column, make_collections_from_indices
from utils.train_and_eval_utils import eval
from model import ChessEncoder
from dataset import ChessEncoderDataset, collate_fn


def run_collection_method(cfg, cheat_engine="maia2_2050", p_value=0.5):
    """
    Run the collection method for game-level cheating detection.

    Args:
        cfg: Configuration object
        cheat_engine: Which cheat engine to use ("maia2_2050" or "allie_2500")
        p_value: Fixed p value to use for collections (0 to 1)

    Returns:
        Dictionary containing evaluation metrics
    """
    # Get configurations
    model_cfg = ModelConfig.from_cfg(cfg)
    training_cfg = TrainingConfig.from_cfg(cfg)
    collection_cfg = CollectionConfig.from_cfg(cfg)
    all_columns = AllCheatColumns.from_cfg(cfg)

    # Set device
    device = torch.device(training_cfg.device if training_cfg.device.startswith("cuda") and torch.cuda.is_available() else "cpu")

    # Load embeddings
    if cheat_engine == "maia2_2050":
        embeddings_path = cfg.data.maia_embeddings_path
        columns = all_columns.maia2 + ["fen_before", "fen_after"]
    else:  # allie_2500
        embeddings_path = cfg.data.allie_embeddings_path
        columns = all_columns.allie + ["move_uci"]
        # Add fen_before if needed
        if "fen_before" in cfg.data:
            columns.append("fen_before")

    emb_dict = load_embeddings(embeddings_path, columns)

    # Load dataset indices
    data_dir = cfg.data.data_output_dir
    train_indices = load_pickle(f"{data_dir}/train_dataset_indices.pickle")
    val_indices = load_pickle(f"{data_dir}/val_dataset_indices.pickle")
    test_indices = load_pickle(f"{data_dir}/test_dataset_indices.pickle")

    # Select cheat column
    cheat_column = choose_cheat_column()

    # Build datasets with fixed p_value
    # Note: In this version, we use a fixed p value specified by the caller
    # This differs from training where p varies from MIN_CHEAT_P to MAX_CHEAT_P

    # For training/val/test, we use the original indices with varying p
    train_data = make_collections_from_indices(emb_dict, train_indices, cheat_column)
    val_data = make_collections_from_indices(emb_dict, val_indices, cheat_column)
    test_data = make_collections_from_indices(emb_dict, test_indices, cheat_column)

    # Create dataloaders
    train_loader = DataLoader(
        ChessEncoderDataset(*train_data),
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )

    val_loader = DataLoader(
        ChessEncoderDataset(*val_data),
        batch_size=training_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    test_loader = DataLoader(
        ChessEncoderDataset(*test_data),
        batch_size=training_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Create model
    model = ChessEncoder(model_cfg)

    # Train model
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    # Import training function
    from utils.train_and_eval_utils import train_and_eval

    # Setup checkpoint directory
    checkpoint_dir = cfg.data.checkpoints_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"{cheat_engine}_collection_method.pt")

    # Save training configuration info for reference
    training_info = {
        "cheat_engine": cheat_engine,
        "p_value": p_value,
        "batch_size": training_cfg.batch_size,
        "learning_rate": training_cfg.learning_rate,
        "n_epochs": training_cfg.n_epochs,
    }

    # Train and evaluate
    results = train_and_eval(
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        test_loader,
        n_epochs=training_cfg.n_epochs,
        verbose=True,
        file=None,
        model_name=model_path,
        device=device,
        eval_train=False,
        eval_test=True,
    )

    results["cheat_engine"] = cheat_engine
    results["p_value"] = p_value
    results.update(training_info)

    return results


if __name__ == "__main__":
    import yaml

    # For testing - should use the main script in production
    cfg = load_config(os.path.join(parent_dir, "config.yaml"))

    # Run collection method with default p_value from config
    p_value = cfg.experiments.game_level_experiment.collection_method.p_value

    print(f"Running collection method with p_value={p_value}")

    results = run_collection_method(cfg, cheat_engine="maia2_2050", p_value=p_value)

    print("\nCollection Method Results:")
    for key, value in results.items():
        if isinstance(value, list) and len(value) > 0:
            print(f"{key}: [length={len(value)}, last={value[-1]}]")
        else:
            print(f"{key}: {value}")
