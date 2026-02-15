"""
Baseline methods for game-level cheating detection.

This module implements simple baselines:
1. Stockfish first-line matching
2. Constant classification (always cheat)
3. Human accusation-based classification
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report


def stockfish_baseline_predictions(df, threshold=0.5):
    """
    Predict cheating based on Stockfish first-line matching ratio.

    Args:
        df: DataFrame containing move data with stockfish columns
        threshold: Threshold for ratio of Stockfish matches to classify as cheat

    Returns:
        numpy array of predictions (0 for fair, 1 for cheat)
        Dictionary mapping (game_id, player) to prediction
    """
    predictions = []
    prediction_map = {}

    # Group by both game_id and color (via half_move % 2)
    # half_move % 2 == 1 means it's White's move, 2 means Black's move
    df_with_color = df.copy()
    df_with_color["color"] = (df_with_color["half_move"] % 2).map({1: "white", 2: "black"})

    for game_id in df_with_color["game_id"].unique():
        game_df = df_with_color[df_with_color["game_id"] == game_id]

        for color in ["white", "black"]:
            player_df = game_df[game_df["color"] == color]

            # Calculate ratio of moves that match Stockfish's first line
            total_moves = len(player_df)
            if total_moves == 0:
                predictions.append(0)
                prediction_map[(game_id, color)] = 0
                continue

            stockfish_matches = player_df["move_uci"] == player_df["move_stockfish_15"]
            match_ratio = stockfish_matches.sum() / total_moves

            # Higher match ratio suggests cheating (engine play)
            if match_ratio >= threshold:
                predictions.append(1)
                prediction_map[(game_id, color)] = 1
            else:
                predictions.append(0)
                prediction_map[(game_id, color)] = 0

    return np.array(predictions), prediction_map


def constant_clasifier_baseline(num_samples, constant_class=1):
    """
    Always predict the same class.

    Args:
        num_samples: Number of samples to predict
        constant_class: Class to always predict (0=fair, 1=cheat)

    Returns:
        numpy array of predictions
    """
    return np.full(num_samples, constant_class, dtype=int)


def human_accusation_baseline(df):
    """
    Predict cheating based on human accusations.

    Args:
        df: DataFrame containing accusation data

    Returns:
        numpy array of predictions
        Dictionary mapping (game_id, color) to prediction
    """
    predictions = []
    prediction_map = {}

    df_with_color = df.copy()
    df_with_color["color"] = (df_with_color["half_move"] % 2).map({1: "white", 2: "black"})

    for game_id in df_with_color["game_id"].unique():
        game_df = df_with_color[df_with_color["game_id"] == game_id]

        for color in ["white", "black"]:
            player_df = game_df[game_df["color"] == color]

            # Get the player name based on color
            if color == "white":
                player = player_df["white_player"].iloc[0] if len(player_df) > 0 else None
                accused_col = "black_accused_white"  # Black accused White
            else:
                player = player_df["black_player"].iloc[0] if len(player_df) > 0 else None
                accused_col = "white_accused_black"  # White accused Black

            # If player exists and has been accused, predict cheat
            if player is not None and len(player_df) > 0:
                is_accused = player_df[accused_col].iloc[0]
                predictions.append(1 if is_accused else 0)
                prediction_map[(game_id, color)] = 1 if is_accused else 0
            else:
                predictions.append(0)
                prediction_map[(game_id, color)] = 0

    return np.array(predictions), prediction_map


def evaluate_baseline(y_true, y_pred):
    """
    Evaluate baseline predictions.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary containing evaluation metrics
    """
    return {
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }


def run_all_baselines(df, y_true):
    """
    Run all baseline methods and collect results.

    Args:
        df: DataFrame containing game data
        y_true: True labels for evaluation

    Returns:
        Dictionary containing results from all baselines
    """
    results = {}

    # Stockfish baseline
    stockfish_pred, _ = stockfish_baseline_predictions(df, threshold=0.5)
    if len(stockfish_pred) == len(y_true):
        results["stockfish"] = evaluate_baseline(y_true, stockfish_pred)
        print("Stockfish baseline results:")
        print(classification_report(y_true, stockfish_pred, zero_division=0))
    else:
        print(f"Warning: Stockfish prediction count ({len(stockfish_pred)}) "
              f"does not match true labels count ({len(y_true)})")

    # Constant baseline (always cheat)
    constant_pred = constant_clasifier_baseline(len(y_true), constant_class=1)
    results["constant_cheat"] = evaluate_baseline(y_true, constant_pred)
    print("\nConstant cheat baseline results:")
    print(classification_report(y_true, constant_pred, zero_division=0))

    # Human accusation baseline
    human_pred, _ = human_accusation_baseline(df)
    # Truncate or pad to match y_true length if needed
    if len(human_pred) > len(y_true):
        human_pred = human_pred[:len(y_true)]
    elif len(human_pred) < len(y_true):
        human_pred = np.pad(human_pred, (0, len(y_true) - len(human_pred)), constant_values=0)

    results["human_accusation"] = evaluate_baseline(y_true, human_pred)
    print("\nHuman accusation baseline results:")
    print(classification_report(y_true, human_pred, zero_division=0))

    return results


if __name__ == "__main__":
    import pandas as pd
    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Test with sample data
    df = pd.read_csv(os.path.join(parent_dir, "../data/processed/tournament/chess_fraud_dataset.csv"))

    # Sample some games for testing
    sample_games = df["game_id"].unique()[:100]
    sample_df = df[df["game_id"].isin(sample_games)]

    # Create dummy true labels for testing (one per game/color pair)
    y_true = np.random.randint(0, 2, size=len(sample_games) * 2)

    print("Testing baseline methods...")
    results = run_all_baselines(sample_df, y_true)
