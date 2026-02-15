"""
Main entry point for the chess fraud collection classification experiments.

This script:
1. Runs make_collections.py to create the dataset
2. Runs all experiments sequentially according to config.yaml
"""

import argparse
import os
import subprocess
import sys

from utils import load_config


def main():
    parser = argparse.ArgumentParser(description="Run chess fraud experiments")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--skip-make-collections",
        action="store_true",
        help="Skip make_collections.py step if indices already exist",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=["all", "game-level", "p", "each-elo-cheater"],
        default=["all"],
        help="Which experiments to run",
    )
    args = parser.parse_args()

    # Load configuration
    cfg = load_config(args.config)

    print("=" * 60)
    print("CHESS FRAUD COLLECTION CLASSIFICATION EXPERIMENTS")
    print("=" * 60)

    # Step 1: Make collections
    if not args.skip_make_collections:
        print("\n[1/3] Making collections...")
        subprocess.run([sys.executable, "make_collections.py"], check=True)
    else:
        print("\n[1/3] Skipping make_collections.py (requested)")

    # Step 2 Determine which experiments to run
    run_all = "all" in args.experiments
    run_game_level = run_all or "game-level" in args.experiments
    run_p_experiment = run_all or "p" in args.experiments
    run_each_elo_cheater = run_all or "each-elo-cheater" in args.experiments

    # Step 3: Run experiments
    print("\n[2/3] Running experiments...")

    if run_game_level:
        print("\n  - Game Level Experiment...")
        exp_dir = "experiments/game_level_experiment"
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(f"{exp_dir}/logs", exist_ok=True)
        subprocess.run([sys.executable, f"{exp_dir}/main.py"], check=True)

    if run_p_experiment:
        print("\n  - P Experiment...")
        exp_dir = "experiments/p_experiment"
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(f"{exp_dir}/logs", exist_ok=True)
        subprocess.run([sys.executable, f"{exp_dir}/main.py"], check=True)

    if run_each_elo_cheater:
        print("\n  - Each ELO and Cheater Experiment...")
        exp_dir = "experiments/each_elo_and_cheater_experiment"
        os.makedirs(exp_dir, exist_ok=True)
        os.makedirs(f"{exp_dir}/logs", exist_ok=True)
        subprocess.run([sys.executable, f"{exp_dir}/main.py"], check=True)

    print("\n[3/3] Done!")
    print("=" * 60)
    print("\nResults saved to experiment logs directories:")
    if run_game_level:
        print("  - experiments/game_level_experiment/logs/results.csv")
    if run_p_experiment:
        print("  - experiments/p_experiment/logs/results.csv")
    if run_each_elo_cheater:
        print("  - experiments/each_elo_and_cheater_experiment/logs/results.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
