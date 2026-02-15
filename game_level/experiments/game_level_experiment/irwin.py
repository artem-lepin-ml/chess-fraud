"""
Irwin baseline for game-level cheating detection.

This module implements the Irwin cheating detection algorithm
using the irwin library in the irwin/ subfolder.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

# Add irwin directory to path
irwin_dir = os.path.join(os.path.dirname(__file__), "irwin")
sys.path.insert(0, irwin_dir)

# Irwin imports
from conf.config import config
from modules.game.Colour import Colour, White, Black
from modules.game.Game import Game, GameID, PlayerID, Emt
from modules.game.AnalysedGame import AnalysedGame, GameAnalysedGame
from modules.game.AnalysedMove import AnalysedMove, Analysis, UCI
from modules.game.EngineEval import EngineEval
from modules.irwin.AnalysedGameModel import AnalysedGameModel
import chess


def build_game_from_df(df_game: pd.DataFrame, game_id: str,
                       white_player: str, black_player: str) -> Game:
    """
    Create a Game from DataFrame rows for one game.

    Args:
        df_game: DataFrame containing move data for one game
        game_id: Game identifier
        white_player: White player name
        black_player: Black player name

    Returns:
        Game object with SAN moves, timestamps, and engine evaluations
    """
    board = chess.Board()
    san_moves, emts, analysis = [], [], []

    for i in range(len(df_game)):
        row = df_game.iloc[i]

        # SAN move and time
        move = chess.Move.from_uci(row["uci"])
        san_moves.append(board.san(move))
        board.push(move)
        emts.append(abs(int(row["timestamp"])) * 100)

        # Determine who makes the move: even move_index = white, odd = black
        cur_white = (row["half_move"] % 2 == 1)
        next_white = not cur_white

        # Engine evaluation BEFORE the move
        cp_before = row["engineEval_cp"]
        mate_before = row["engineEval_mate"]
        cp_before = cp_before if cur_white else -cp_before
        mate_before = None if pd.isna(mate_before) else (
            mate_before if cur_white else -mate_before)

        # Engine evaluation AFTER the move
        cp_after = row["engineEval_cp_after"]
        mate_after = row["engineEval_mate_after"]
        cp_after = cp_after if next_white else -cp_after
        mate_after = None if pd.isna(mate_after) else (
            mate_after if next_white else -mate_after)

        # Add EngineEval for before and after move
        analysis.append(EngineEval(cp=cp_before, mate=mate_before))
        analysis.append(EngineEval(cp=cp_after, mate=mate_after))

    board = chess.Board()
    return Game(
        id=game_id,
        white=white_player,
        black=black_player,
        pgn=san_moves,
        emts=emts,       # times for moves
        analysis=analysis  # 2*(n-1) evaluations: before and after each half-move
    )


def get_analysis_list(top5, color):
    """
    Parse top5 analysis string into list of Analysis objects.

    Args:
        top5: JSON string containing top 5 moves with engine evaluations
        color: 1 for WHITE, -1 for BLACK

    Returns:
        List of Analysis objects
    """
    import ast
    analysis_list = []
    top5 = ast.literal_eval(top5)
    for move in top5:
        if move["Mate"] is None:
            analysis = Analysis(
                uci=UCI(move["Move"]),
                engineEval=EngineEval(
                    cp=move["Centipawn"] * color,
                    mate=move["Mate"]
                )
            )
        else:
            analysis = Analysis(
                uci=UCI(move["Move"]),
                engineEval=EngineEval(
                    cp=move["Centipawn"],
                    mate=move["Mate"] * color
                )
            )
        analysis_list.append(analysis)
    return analysis_list


def get_irwin_predictions(df, model):
    """
    Run Irwin model on DataFrame to get cheating predictions.

    Args:
        df: DataFrame with game/move data
        model: Loaded AnalysedGameModel

    Returns:
        Tuple of (main_predictions, move_predictions)
        main_predictions: Game-level predictions (score 0-100)
        move_predictions: Move-level predictions for each game
    """
    white_game_analysed_games = []
    black_game_analysed_games = []

    for game_id, group in df.groupby("game_id"):
        white_player = group["white_player"].iloc[0]
        black_player = group["black_player"].iloc[0]
        assert len(group["white_player"].unique()) == 1
        assert len(group["black_player"].unique()) == 1

        moves = []
        for i, row in group.reset_index().iterrows():
            engine_eval = EngineEval(
                cp=row["engineEval_cp"],
                mate=row["engineEval_mate"]
            )
            color = 1 if i % 2 == 0 else -1
            top5 = row["top5"] if pd.notna(row["top5"]) else "[]"
            move = AnalysedMove(
                uci=row["uci"],
                move=row["move_index"],
                emt=row["timestamp"],
                engineEval=engine_eval,
                analyses=get_analysis_list(top5, color) if top5 != "[]" else []
            )
            moves.append(move)

        game = build_game_from_df(group, game_id, white_player, black_player)
        white_analysed_game = AnalysedGame.new(
            gameId=game_id, playerId=white_player, colour=White, analysedMoves=moves[:-1]
        )
        black_analysed_game = AnalysedGame.new(
            gameId=game_id, playerId=black_player, colour=Black, analysedMoves=moves[:-1]
        )
        white_game_analysed_game = GameAnalysedGame(
            analysedGame=white_analysed_game,
            game=game
        )
        black_game_analysed_game = GameAnalysedGame(
            analysedGame=black_analysed_game,
            game=game
        )
        white_game_analysed_games.append(white_game_analysed_game)
        black_game_analysed_games.append(black_game_analysed_game)

    predictions = model.predict(white_game_analysed_games + black_game_analysed_games)
    main_predictions = [pred.game[0] if pred is not None else None for pred in predictions]
    sum_predictions = [pred.weightedGamePrediction() if pred is not None else None for pred in predictions]
    move_predictions = [pred.weightedMovePredictions() for pred in predictions]

    return sum_predictions, move_predictions


def get_game_labels(df):
    """
    Get game-level labels from DataFrame.

    Args:
        df: DataFrame with move labels

    Returns:
        Series indicating if game is cheating (True) for each game/color pair
    """
    agg = df.groupby(["game_id", "is_white_move"])["move_label"].mean() > 0
    return agg


def remove_nans(main, labels):
    """
    Remove samples with None predictions.

    Args:
        main: List of predictions
        labels: Series/array of labels

    Returns:
        Tuple of (main_clean, labels_clean) with Nones removed
    """
    main = np.array(main)
    labels = np.array(labels.values) if hasattr(labels, 'values') else np.array(labels)
    mask = ~pd.isnull(main)
    print(f"Nans removed: {sum(~mask)} / {len(mask)}")
    return main[mask], labels[mask]


def find_best_threshold_simple(scores, labels):
    """
    Find optimal threshold for predictions.

    Args:
        scores: Prediction scores
        labels: True labels
        metric: 'accuracy' or 'f_score'

    Returns:
        Tuple of (best_threshold, best_score)
    """
    thresholds = np.arange(101)
    best_metric = 0
    best_threshold = 50

    for thresh in thresholds:
        predictions = (scores >= thresh).astype(int)

        current_metric = f1_score(labels, predictions, average='macro', zero_division=0)

        if current_metric > best_metric:
            best_metric = current_metric
            best_threshold = thresh
    print(f"Optimal threshold: {best_threshold}, F1 macro: {best_metric:.4f}")
    return best_threshold, best_metric


def evaluate_irwin_baseline(df, threshold=50):
    """
    Run Irwin baseline and evaluate predictions.

    Args:
        df: DataFrame with game/move data
        threshold: Threshold for cheat prediction (0-100)

    Returns:
        Dictionary containing evaluation metrics
    """
    print("Loading Irwin model...")
    model = AnalysedGameModel(config=config)

    print("Preparing data...")
    # Set move_index for reference
    df["move_index"] = df["half_move"] - 1

    # Filter out moves before half_move 20 (debunking threshold from original paper)
    df_filtered = df[df["half_move"] > 20].copy()

    # Get game labels (average move label for each game/color)
    game_labels = df_filtered.groupby(["game_id", "is_white_move"])["move_label"].mean() > 0

    print("Running Irwin predictions...")
    main, move = get_irwin_predictions(df, model)

    # Remove NaN predictions
    main_clean, game_labels_clean = remove_nans(main, game_labels)

    # Convert predictions
    preds = (main_clean >= threshold).astype(int)
    labels = game_labels_clean.astype(int)

    # Calculate metrics
    f1 = f1_score(labels, preds, average='macro', zero_division=0)
    f1_binary = f1_score(labels, preds, zero_division=0)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)

    print("\nIrwin Baseline Results:")
    print("====================")
    print(f"Predictions: {len(preds)}")
    print(f"Positive predictions: {preds.sum()} ({preds.sum()/len(preds)*100:.1f}%)")
    print(f"True labels: {labels.sum()} ({labels.sum()/len(labels)*100:.1f}%)")
    print(f"F1 (macro): {f1:.4f}")
    print(f"F1 (binary): {f1_binary:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    print("\nDetailed classification report:")
    print(classification_report(labels, preds, zero_division=0))

    return {
        "method": "irwin",
        "threshold": threshold,
        "f1": f1,
        "f1_binary": f1_binary,
        "precision": precision,
        "recall": recall,
        "num_samples": len(preds),
    }


def run_irwin_baseline_optimized_threshold(data_path=None, threshold=None):
    """
    Run Irwin baseline with optimal threshold discovery.

    Args:
        data_path: Path to CSV file (default: from config)
        threshold: If None, finds optimal threshold; otherwise uses given threshold

    Returns:
        Dictionary containing evaluation metrics
    """
    # Load data
    if data_path is None:
        from utils import load_config
        cfg = load_config(os.path.join(parent_dir, "config.yaml"))
        data_path = cfg.data.csv_path

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Find optimal threshold if not provided
    if threshold is None:
        model = AnalysedGameModel(config=config)
        df["move_index"] = df["half_move"] - 1
        df_filtered = df[df["half_move"] > 20].copy()
        game_labels = df_filtered.groupby(["game_id", "is_white_move"])["move_label"].mean() > 0
        main, move = get_irwin_predictions(df, model)
        main_clean, game_labels_clean = remove_nans(main, game_labels)
        threshold, best_score = find_best_threshold_simple(main_clean, game_labels_clean)
        print(f"\nUsing optimal threshold: {threshold}")

    return evaluate_irwin_baseline(df, threshold=threshold)


if __name__ == "__main__":
    # Test with data
    run_irwin_baseline_optimized_threshold()
