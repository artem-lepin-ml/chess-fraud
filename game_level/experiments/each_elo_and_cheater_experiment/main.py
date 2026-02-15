"""
Main script for the each ELO and cheater experiment.

This experiment:
1. Trains a separate model for each cheat engine / AI chess engine
2. Evaluates each model across different ELO bins
3. Saves results to logs/results.csv
"""

import os
import sys
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from utils import (
    load_config,
    load_embeddings,
    load_pickle,
    save_pickle,
    ModelConfig,
    TrainingConfig,
    CollectionConfig,
    AllCheatColumns,
    seed_everything,
)
from utils.collection_utils import make_collections_from_indices
from utils.train_and_eval_utils import train_and_eval, eval
from model import ChessEncoder
from dataset import ChessEncoderDataset, collate_fn


def train_and_eval_on_one_cheater(cfg, npz_name, cheat_column, device):
    """
    Train and evaluate a model on one specific cheat engine.

    Args:
        cfg: Configuration object
        npz_name: Name of the embeddings file (e.g., "maia2_2050", "allie_2500")
        cheat_column: Which cheat column to use
        device: Device to use for training

    Returns:
        Dictionary containing evaluation results across ELO bins
    """
    # Get configurations
    model_cfg = ModelConfig.from_cfg(cfg)
    training_cfg = TrainingConfig.from_cfg(cfg)
    all_columns = AllCheatColumns.from_cfg(cfg)

    # Set seed
    seed_everything(training_cfg.seed)

    # Load embeddings
    if npz_name == "maia2_2050":
        embeddings_path = cfg.data.maia_embeddings_path
    else:
        embeddings_path = cfg.data.allie_embeddings_path

    emb_df = np.load(embeddings_path)
    emb_dict = {
        cheat_column: emb_df[cheat_column],
    }

    # Add additional columns based on embedding type
    if npz_name == "maia2_2050":
        emb_dict["fen_before"] = emb_df["fen_before"]
        emb_dict["fen_after"] = emb_df["fen_after"]
    else:
        emb_dict["move_uci"] = emb_df["move_uci"]
        if "fen_before" in emb_df.files:
            emb_dict["fen_before"] = emb_df["fen_before"]

    # Load dataset indices
    data_dir = cfg.data.data_output_dir
    train_indices = load_pickle(f"{data_dir}/train_dataset_indices.pickle")
    val_indices = load_pickle(f"{data_dir}/val_dataset_indices.pickle")
    test_indices = load_pickle(f"{data_dir}/test_dataset_indices.pickle")

    # Set device
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    # Build datasets
    train_data = make_collections_from_indices(emb_dict, train_indices, cheat_column)
    val_data = make_collections_from_indices(emb_dict, val_indices, cheat_column)
    test_data = make_collections_from_indices(emb_dict, test_indices, cheat_column)

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

    test_loader_all = DataLoader(
        ChessEncoderDataset(*test_data),
        batch_size=training_cfg.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )

    # Create model
    model = ChessEncoder(model_cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    # Setup checkpoint directory
    checkpoint_dir = cfg.data.checkpoints_dir
    experiment_dir = os.path.join(checkpoint_dir, "each_elo_and_cheater")
    os.makedirs(experiment_dir, exist_ok=True)

    exp_subdir = os.path.join(experiment_dir, npz_name)
    os.makedirs(exp_subdir, exist_ok=True)

    model_name = os.path.join(exp_subdir, f"{cheat_column}.pt")

    # Train and evaluate
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    training_logs_dir = os.path.join(logs_dir, npz_name)
    os.makedirs(training_logs_dir, exist_ok=True)

    training_log_path = os.path.join(training_logs_dir, f"{cheat_column}_training.csv")
    with open(training_log_path, 'w') as f:
        # Redirect classification_report output
        pass

    results_per_epoch = train_and_eval(
        model,
        optimizer,
        loss_fn,
        train_loader,
        val_loader,
        test_loader_all,
        n_epochs=training_cfg.n_epochs,
        verbose=False,
        model_name=model_name,
        device=device,
        eval_train=False,
        eval_test=True,
    )

    # Save training logs
    pd.DataFrame(results_per_epoch).to_csv(
        os.path.join(training_logs_dir, f"{cheat_column}_metrics.csv"),
        index=False
    )

    # Evaluate on ELO bins
    elo_bounds = cfg.elo.bounds
    results_dict = {
        "cheat_engine": npz_name,
        "cheat_column": cheat_column,
    }

    # For each ELO bin
    for i, elo_upper in enumerate(elo_bounds):
        elo_lower = elo_bounds[i-1] if i > 0 else 0

        test_loader = DataLoader(
            ChessEncoderDataset(*test_data, elo_bounds=(elo_lower, elo_upper)),
            batch_size=training_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        if len(test_loader.dataset) == 0:
            # No samples in this ELO bin
            results_dict[f"f1_{elo_lower}_{elo_upper}"] = np.nan
            results_dict[f"precision_{elo_lower}_{elo_upper}"] = np.nan
            results_dict[f"recall_{elo_lower}_{elo_upper}"] = np.nan
            results_dict[f"f1_macro_{elo_lower}_{elo_upper}"] = np.nan
            results_dict[f"f1_weighted_{elo_lower}_{elo_upper}"] = np.nan
            continue

        # Evaluate
        loss, f1, precision, recall, f1_macro, f1_weighted = eval(
            model, loss_fn, test_loader, verbose=False, device=device
        )

        results_dict[f"f1_{elo_lower}_{elo_upper}"] = f1
        results_dict[f"precision_{elo_lower}_{elo_upper}"] = precision
        results_dict[f"recall_{elo_lower}_{elo_upper}"] = recall
        results_dict[f"f1_macro_{elo_lower}_{elo_upper}"] = f1_macro
        results_dict[f"f1_weighted_{elo_lower}_{elo_upper}"] = f1_weighted

    # Also evaluate on all data
    loss, f1, precision, recall, f1_macro, f1_weighted = eval(
        model, loss_fn, test_loader_all, verbose=False, device=device
    )
    results_dict["f1_all"] = f1
    results_dict["precision_all"] = precision
    results_dict["recall_all"] = recall
    results_dict["f1_macro_all"] = f1_macro
    results_dict["f1_weighted_all"] = f1_weighted

    # Best iteration metrics
    results_dict["best_iteration"] = results_per_epoch.get("best_iteration", -1)
    results_dict["best_val_f1_macro"] = results_per_epoch.get("best_val_f1_macro", np.nan)

    return results_dict


def main():
    """Main function to run each ELO and cheater experiment."""
    cfg = load_config(os.path.join(parent_dir, "config.yaml"))

    # Get training config for device
    training_cfg = TrainingConfig.from_cfg(cfg)
    device = training_cfg.device

    print("=" * 60)
    print("EACH ELO AND CHEATER EXPERIMENT")
    print("=" * 60)

    results_list = []
    all_columns = AllCheatColumns.from_cfg(cfg)

    # Get cheater lists from config
    cheaters = cfg.experiments.each_elo_and_cheater_experiment.cheaters

    # Run for allie models
    print("\nRunning for Allie models...")
    for cheat_column in cheaters.allie_models:
        try:
            print(f"\n  - Training with cheat engine: allie_2500, column: {cheat_column}")
            result = train_and_eval_on_one_cheater(
                cfg, "allie_2500", cheat_column, device
            )
            results_list.append(result)
            print(f"    F1 (all): {result['f1_all']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    # Run for maia2 models
    print("\nRunning for Maia2 models...")
    for cheat_column in cheaters.maia2_models:
        try:
            print(f"\n  - Training with cheat engine: maia2_2050, column: {cheat_column}")
            result = train_and_eval_on_one_cheater(
                cfg, "maia2_2050", cheat_column, device
            )
            results_list.append(result)
            print(f"    F1 (all): {result['f1_all']:.4f}")
        except Exception as e:
            print(f"    Error: {e}")
            import traceback
            traceback.print_exc()

    # Save results
    if results_list:
        results_df = pd.DataFrame(results_list)

        logs_dir = "logs"
        results_path = os.path.join(logs_dir, "results.csv")
        results_df.to_csv(results_path, index=False)

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {results_path}")
        print(f"\nNumber of models trained: {len(results_list)}")
        print("=" * 60)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
