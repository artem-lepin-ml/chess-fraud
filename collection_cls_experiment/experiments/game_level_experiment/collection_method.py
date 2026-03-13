import torch
import torch.nn as nn
import numpy as np
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import pandas as pd

from collection_utils import (
    make_collections_from_indices, 
    DatasetIndicesData, 
    CollectionSample, 
    CollectionIndices)
from utils.train_and_eval_utils import train_and_eval, eval
from model import ChessEncoder
from dataset import ChessEncoderDataset, collate_fn
from utils.common_utils import seed_everything
from consts import (
    DEVICE,
    LR,
    N_EPOCHS,
    BATCH_SIZE,
    VAL_INDICES_PATH,
    TEST_INDICES_PATH,
    TRAIN_INDICES_PATH,
    SYNT_MAIA2_EMBS_PATH,
    SYNT_ALLIE_EMBS_PATH,
    TOURNAMENT_MAIA2_EMBS_PATH,
    TOURNAMENT_ALLIE_EMBS_PATH,
    MAIA2_CHEAT_COLUMNS,
    ALLIE_CHEAT_COLUMNS,
    PROCESSED_TOURNAMENT_PATH,
    K_MAX)


def get_collections_method_results_on_tournament(model, model_name):
    emb_dict = {}
    if model_name == "maia2_2050":
        emb_df = np.load(TOURNAMENT_MAIA2_EMBS_PATH)
        for column in tqdm(emb_df.files, desc="Loading .npz embeddings"):
            if column in MAIA2_CHEAT_COLUMNS or column in ["fen_before", "fen_after"]:
                emb_dict[column] = emb_df[column]
        emb_dict = {
            "fen_before" : emb_df["fen_before"],
            "fen_after" : emb_df["fen_after"]
        }
    else:
        emb_df = np.load(TOURNAMENT_ALLIE_EMBS_PATH)
        for column in tqdm(emb_df.files, desc="Loading .npz embeddings"):
            if column in ALLIE_CHEAT_COLUMNS or column == "move_uci":
                emb_dict[column] = emb_df[column]
        emb_dict = {
            "move_uci" : emb_df["move_uci"]
        }
    df = pd.read_csv(PROCESSED_TOURNAMENT_PATH)
    print(f"UNIQUE GAMES: {df['game_id'].nunique()}")
    emb_df = np.load()
    
    collections = []
    p_list = []
    fair_elo_list = []
    cheat_elo_list = []
    for game_id, group in df.groupby("game_id"):
        if game_id.endswith("w"):
            indices = group.index[20 : 20 + 2 * K_MAX: 2]
        else:
            indices = group.index[21 : 20 + 2 * K_MAX + 1: 2]
        if len(indices) < 10:
            continue
        collection = df.loc[indices]
        fair_indices = collection.loc[collection["move_label"] == 0].index.to_list()
        cheat_indices = collection.loc[collection["move_label"] == 1].index.to_list()
        col_indices = CollectionIndices(fair_indices, cheat_indices)
        if sum(df["move_label"].loc[indices]) == 0:
            fair_elo_list.append(group.iloc[0]["player_elo"])
            collections.append([CollectionSample(col_indices, None)])
        else:
            cheat_elo_list.append(group.iloc[0]["player_elo"])
            p_list.append(len(cheat_indices) / len(indices))
            collections.append([CollectionSample(None, col_indices)])

    dataset_data = DatasetIndicesData(collections, [], p_list, None)
    fair_emb_list, cheat_emb_list, cheat_label_list, p_list, elo_list = make_collections_from_indices(emb_dict, dataset_data, None, True)
    dataset_ = ChessEncoderDataset(fair_emb_list, cheat_emb_list, cheat_label_list, p_list,[], tournament=True, cheat_elo_list=cheat_elo_list, fair_elo_list=fair_elo_list)
    loader = DataLoader(
                dataset_,
                BATCH_SIZE,
                shuffle=False,
                collate_fn=collate_fn
            )
    _, f1, precision, recall, f1_macro, acc = eval(model, loader, verbose=False)
    return f1, precision, recall, f1_macro, acc


def get_collection_method_results(model_name):
    seed_everything()

    emb_dict = {}
    if model_name == "maia2_2050":
        emb_df = np.load(SYNT_MAIA2_EMBS_PATH)
        for column in tqdm(emb_df.files, desc="Loading .npz embeddings"):
            if column in MAIA2_CHEAT_COLUMNS or column in ["fen_before", "fen_after"]:
                emb_dict[column] = emb_df[column]
    else:
        emb_df = np.load(SYNT_ALLIE_EMBS_PATH)
        for column in tqdm(emb_df.files, desc="Loading .npz embeddings"):
            if column in ALLIE_CHEAT_COLUMNS or column == "move_uci":
                emb_dict[column] = emb_df[column]
    cheat_column = None

    with open(VAL_INDICES_PATH, mode="rb") as file:
        val_data = pickle.load(file)
        val_data = make_collections_from_indices(emb_dict, val_data, cheat_column)
        val_loader = DataLoader(
            ChessEncoderDataset(*val_data),
            BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=-1
        )

    with open(TEST_INDICES_PATH, mode="rb") as file:
        test_data = pickle.load(file)
        test_data = make_collections_from_indices(emb_dict, test_data, cheat_column)
        test_loader = DataLoader(
            ChessEncoderDataset(*test_data),
            BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=-1
        )

    with open(TRAIN_INDICES_PATH, mode="rb") as file:
        train_data = pickle.load(file)
        train_data = make_collections_from_indices(emb_dict, train_data, cheat_column)
        train_loader = DataLoader(
            ChessEncoderDataset(*train_data),
            BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=-1
        )

    model = ChessEncoder()
    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    (Path(__file__).parent / "checkpoints").mkdir(exist_ok=True)
    model_path = Path(__file__).parent / "checkpoints" / f"{model_name}.pt"
    train_and_eval(
        model, 
        optimizer,
        train_loader, 
        val_loader, 
        test_loader, 
        n_epochs=N_EPOCHS,
        eval_test=False, 
        verbose=False,
        model_path=model_path
        )
    # load the best model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    #loss, f1, precision, recall, f1_macro, acc
    _, f1, precision, recall, f1_macro, acc = eval(
        model,
        test_loader,
        verbose=False,
        eval_moves=False
    )
    synt_results = {
        "dataset" : "Synth",
        "method" : model_name,
        "f1" : f1,
        "precision" : precision,
        "recall" : recall,
        "f1_macro" : f1_macro,
        "accuracy" : acc
        }
    f1, precision, recall, f1_macro, acc = get_collections_method_results_on_tournament(model, model_name)
    tournament_results = {
        "dataset" : "Tournament",
        "method" : model_name,
        "f1" : f1,
        "precision" : precision,
        "recall" : recall,
        "f1_macro" : f1_macro,
        "accuracy" : acc
    }
    results = pd.DataFrame([synt_results, tournament_results])
    return results
