import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def st_top_match(tournament_input):
    """ === Baseline 1: Stockfish top-1 match rate > 50% === """
    game_top1_stats = []
    for game_id in tournament_input['game_id'].unique():
        game_moves = tournament_input[tournament_input['game_id'] == game_id]
        white_moves = game_moves[game_moves['is_white_move']]
        black_moves = game_moves[game_moves['is_black_move']]
        
        white_top1_pct = (white_moves['move_rank'] == 1).sum() / len(white_moves) if len(white_moves) > 0 else 0
        black_top1_pct = (black_moves['move_rank'] == 1).sum() / len(black_moves) if len(black_moves) > 0 else 0
        
        game_info = tournament_input[tournament_input['game_id'] == game_id].iloc[0]
        game_top1_stats.append({
            'game_id': game_id,
            'white_top1_pct': white_top1_pct,
            'black_top1_pct': black_top1_pct,
            'white_cheated': game_info['white_cheated'],
            'black_cheated': game_info['black_cheated']
        })

    game_top1_df = pd.DataFrame(game_top1_stats)

    # Predict cheating if top-1 percentage > 50%
    threshold = 0.5
    y_true = pd.concat([game_top1_df['white_cheated'], game_top1_df['black_cheated']]).astype(int)
    y_pred = pd.concat([(game_top1_df['white_top1_pct'] > threshold), 
                        (game_top1_df['black_top1_pct'] > threshold)]).astype(int)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return {
        "method" : "Stockfish_top1_match",
        "f1" : f1,
        "precision" : prec,
        "recall" : rec,
        "f1_macro" : f1_macro,
        "acc" : acc
    }


def human_accusation(tournament_input):
    """ === Baseline 2: Human accusations === """

    game_accusations = []
    for _, game in tournament_input.iterrows():
        game_accusations.append({
            'accused': game['black_accused_white'],
            'actually_cheated': game['white_cheated']
        })
        game_accusations.append({
            'accused': game['white_accused_black'],
            'actually_cheated': game['black_cheated']
        })

    accusations_df = pd.DataFrame(game_accusations)
    y_true = accusations_df['actually_cheated'].astype(int)
    y_pred = accusations_df['accused'].astype(int)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return {
        "method" : "human_accuasition",
        "f1" : f1,
        "precision" : prec,
        "recall" : rec,
        "f1_macro" : f1_macro,
        "acc" : acc
    }


def always_cheat(tournament_input):
    """ === Baseline 3: Always cheat === """

    game_accusations = []
    for _, game in tournament_input.iterrows():
        game_accusations.append({
            'accused': game['black_accused_white'],
            'actually_cheated': game['white_cheated']
        })
        game_accusations.append({
            'accused': game['white_accused_black'],
            'actually_cheated': game['black_cheated']
        })

    accusations_df = pd.DataFrame(game_accusations)
    y_true = accusations_df['actually_cheated'].astype(int)
    y_pred = [1] * len(y_true)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    return {
        "method" : "always_cheat",
        "f1" : f1,
        "precision" : prec,
        "recall" : rec,
        "f1_macro" : f1_macro,
        "acc" : acc
    }