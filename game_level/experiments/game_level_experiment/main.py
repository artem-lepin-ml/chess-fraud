"""
Main script for the game-level experiment.

This experiment:
1. Runs all baselines (Stockfish first-line, constant, human accusation, Irwin)
2. Runs the collection method (our approach)
3. Saves results to logs/results.csv
"""

import os
import sys
import pandas as pd
import numpy as np

parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from utils import load_config
from baselines import run_all_baselines
from collection_method import run_collection_method
from irwin import run_irwin_baseline_optimized_threshold


def main():
    # Load configuration
    cfg = load_config(os.path.join(parent_dir, "config.yaml"))

    # Load test data
    data_path = cfg.data.csv_path
    df = pd.read_csv(data_path)

    print("=" * 60)
    print("GAME LEVEL EXPERIMENT")
    print("=" * 60)

    # Create sample test data for evaluation
    # Use a subset of games for faster testing
    test_games = df["game_id"].unique()[:500]
    test_df = df[df["game_id"].isin(test_games)]

    # Create true labels based on cheat_game annotations
    # We need game-level true labels
    y_true = []
    for game_id in test_df["game_id"].unique():
        game_df = test_df[test_df["game_id"] == game_id]
        # For White
        if len(game_df) > 0:
            white_cheat = game_df["white_cheat_game"].iloc[0]
            y_true.append(int(white_cheat))
            # For Black
            black_cheat = game_df["black_cheat_game"].iloc[0]
            y_true.append(int(black_cheat))

    y_true = np.array(y_true)

    results = []

    # ============================================
    # BASELINES
    # ============================================
    print("\n[1/3] Running baselines...")

    baseline_results = run_all_baselines(test_df, y_true)

    for baseline_name, metrics in baseline_results.items():
        result_row = {
            "method": baseline_name,
            "f1": metrics["f1"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
        }
        results.append(result_row)
        print(f"\n{baseline_name}: F1={metrics['f1']:.4f}, Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}")

    # Irwin baseline
    print("\n  - Irwin baseline...")
    try:
        irwin_results = run_irwin_baseline_optimized_threshold(data_path=cfg.data.csv_path)
        results.append({
            "method": f"irwin",
            "f1": irwin_results["f1"],
            "precision": irwin_results["precision"],
            "recall": irwin_results["recall"],
        })
        print(f"Irwin: F1={irwin_results['f1']:.4f}, Precision={irwin_results['precision']:.4f}, Recall={irwin_results['recall']:.4f}")
    except Exception as e:
        print(f"Error running Irwin baseline: {e}")
        import traceback
        traceback.print_exc()

    # ============================================
    # COLLECTION METHOD (OUR APPROACH)
    # ============================================
    print("\n[2/3] Running collection method...")

    p_value = cfg.experiments.game_level_experiment.collection_method.p_value
    print(f"Using p_value = {p_value}")

    # Run collection method for both cheats engines
    for cheat_engine in ["maia2_2050", "allie_2500"]:
        try:
            print(f"\nTraining and evaluating with {cheat_engine}...")
            collection_results = run_collection_method(cfg, cheat_engine=cheat_engine, p_value=p_value)

            # Extract key metrics from results
            # The results dict contains lists for each epoch
            # We want the best test metrics
            if "best_test_f1" in collection_results:
                result_row = {
                    "method": f"collection_method_{cheat_engine}",
                    "f1": collection_results["best_test_f1"],
                    "precision": collection_results.get("best_test_precision", np.nan),
                    "recall": collection_results.get("best_test_recall", np.nan),
                }
                results.append(result_row)
                print(f"{cheat_engine}: F1={result_row['f1']:.4f}, Precision={result_row['precision']:.4f}, Recall={result_row['recall']:.4f}")
            else:
                print(f"Warning: Could not extract test metrics for {cheat_engine}")
        except Exception as e:
            print(f"Error running collection method for {cheat_engine}: {e}")

    print("\n[3/3] Done!")

    # ============================================
    # SAVE RESULTS
    # ============================================
    results_df = pd.DataFrame(results)

    # Ensure logs directory exists
    logs_dir = "logs"
    os.makedirs(logs_dir, exist_ok=True)

    # Save results
    results_path = os.path.join(logs_dir, "results.csv")
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)
    print(f"\nResults saved to: {results_path}")
    print("\nSummary:")
    print(results_df.to_string(index=False))
    print("=" * 60)


if __name__ == "__main__":
    main()
