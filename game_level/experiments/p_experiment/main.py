"""
Main script for the p experiment (varying proportion of cheating moves).

This experiment evaluates model performance across different values of p
(the proportion of cheating moves in a collection).
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
    ModelConfig,
    TrainingConfig,
    CollectionConfig,
    AllCheatColumns,
)
from utils.collection_utils import make_collections_from_indices
from utils.train_and_eval_utils import eval
from model import ChessEncoder
from dataset import ChessEncoderDataset, collate_fn


def run_p_experiment(cfg, cheat_engine="maia2_2050"):
    """
    Run the p experiment - evaluate model performance across different p values.

    Args:
        cfg: Configuration object
        cheat_engine: Which cheat engine to use

    Returns:
        DataFrame containing results for each p value
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
    else:
        embeddings_path = cfg.data.allie_embeddings_path
        columns = all_columns.allie + ["move_uci", "fen_before"]

    emb_dict = load_embeddings(embeddings_path, columns)

    # Load dataset indices
    data_dir = cfg.data.data_output_dir
    train_indices = load_pickle(f"{data_dir}/train_dataset_indices.pickle")
    test_indices = load_pickle(f"{data_dir}/test_dataset_indices.pickle")

    # Select cheat column (for simplicity, use the first one from the list)
    if cheat_engine == "maia2_2050":
        cheat_column = all_columns.maia2[0]
    else:
        cheat_column = all_columns.allie[0]

    # Build test dataset
    test_data = make_collections_from_indices(emb_dict, test_indices, cheat_column)

    p_cfg = cfg.experiments.p_experiment
    p_min = p_cfg.p_min
    p_max = p_cfg.p_max
    p_step = p_cfg.p_step

    # Define p values to test
    p_values = np.arange(p_min, p_max + p_step, p_step)

    results = []

    # Option 1: Train from scratch for each p value
    # Option 2: Use a single pre-trained model and evaluate with different p filters
    # Here we implement Option 2 (using pre-trained model)

    # First, train a model on full dataset
    print(f"Training model on full dataset ({cheat_engine})...")

    train_data = make_collections_from_indices(emb_dict, train_indices, cheat_column)

    train_loader = DataLoader(
        ChessEncoderDataset(*train_data),
        batch_size=training_cfg.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )

    # Create model
    model = ChessEncoder(model_cfg).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    # Import training function
    from utils.train_and_eval_utils import train_one_epoch

    # Train
    n_epochs = training_cfg.n_epochs
    for epoch in range(n_epochs):
        train_one_epoch(model, optimizer, loss_fn, train_loader, device)
        print(f"Epoch {epoch + 1}/{n_epochs} completed")

    # Save checkpoint
    checkpoint_dir = cfg.data.checkpoints_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f"{cheat_engine}_p_experiment.pt")
    torch.save(model.state_dict(), model_path)

    # Now evaluate for each p value
    print("\nEvaluating for different p values...")

    for p in p_values:
        p_bounds = (p - p_step/2, p + p_step/2) if p > 0 and p < 1 else (p - p_step/2, p + p_step/2)

        # Create test loader with specific p bounds
        test_dataset = ChessEncoderDataset(*test_data, p_bounds=p_bounds)
        test_loader = DataLoader(
            test_dataset,
            batch_size=training_cfg.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        if len(test_dataset) == 0:
            print(f"Skipping p={p:.2f}: no samples in this range")
            continue

        # Evaluate
        loss, f1, precision, recall, f1_macro, f1_weighted = eval(
            model, loss_fn, test_loader, verbose=False, device=device
        )

        results.append({
            "cheat_engine": cheat_engine,
            "p_value": p,
            "f1": f1,
            "f1_macro": f1_macro,
            "precision": precision,
            "recall": recall,
            "num_samples": len(test_dataset),
        })

        print(f"p={p:.2f}: F1={f1:.4f}, F1_macro={f1_macro:.4f}, "
              f"Precision={precision:.4f}, Recall={recall:.4f}, N={len(test_dataset)}")

    return pd.DataFrame(results)


def main():
    """Main function to run p experiment."""
    cfg = load_config(os.path.join(parent_dir, "config.yaml"))

    print("=" * 60)
    print("P EXPERIMENT - Varying Proportion of Cheating Moves")
    print("=" * 60)

    all_results = []

    # Run for both cheat engines
    for cheat_engine in ["maia2_2050", "allie_2500"]:
        try:
            print(f"\n{'=' * 60}")
            print(f"Running p experiment with {cheat_engine}")
            print("=" * 60)

            results_df = run_p_experiment(cfg, cheat_engine=cheat_engine)
            all_results.append(results_df)

        except Exception as e:
            print(f"Error running p experiment for {cheat_engine}: {e}")
            import traceback
            traceback.print_exc()

    # Combine results
    if all_results:
        combined_results = pd.concat(all_results, ignore_index=True)

        # Save results
        logs_dir = "logs"
        os.makedirs(logs_dir, exist_ok=True)
        results_path = os.path.join(logs_dir, "results.csv")
        combined_results.to_csv(results_path, index=False)

        print("\n" + "=" * 60)
        print("EXPERIMENT COMPLETE")
        print("=" * 60)
        print(f"\nResults saved to: {results_path}")
        print("\nSummary:")
        print(combined_results.to_string(index=False))
        print("=" * 60)
    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
