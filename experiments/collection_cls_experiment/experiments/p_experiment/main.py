import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt

from collection_utils import make_collections_from_indices
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
    MAIA2_CHEAT_COLUMNS,
    ALLIE_CHEAT_COLUMNS,
    MIN_CHEAT_P,
    MAX_CHEAT_P)


def get_results_for_a_model(model_name):
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
        model_path=model_path,
        eval_test=False, 
        verbose=False
        )
    # load the best model
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    results = defaultdict(list)
    for p_bound in np.arange(MIN_CHEAT_P, MAX_CHEAT_P, 0.1):
        test_loader = DataLoader(
            ChessEncoderDataset(*test_data, p_bounds=(p_bound, p_bound + 0.1)),
            BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn
        )
        #loss, f1, precision, recall, f1_macro, acc
        _, f1, precision, recall, f1_macro, acc = eval(model, test_loader, verbose=False)
        results["f1"].append(f1)
        results["precision"].append(precision)
        results["recall"].append(recall)
        results["f1_macro"].append(f1_macro)
        results["accuracy"].append(acc)
    df = pd.DataFrame(results)
    (Path(__file__).parent / "results").mkdir(exist_ok=True)
    df.to_csv(f"results/{model_name}.csv")
    return results


def make_p_experiment():
    maia_results = get_results_for_a_model("maia2_2050")
    allie_results = get_results_for_a_model("allie_2500")
    x = np.arange(MIN_CHEAT_P, MAX_CHEAT_P, 0.1)
    bin_labels = [f"({int((v)*100)}%-{int((v+0.1)*100)}%]" for v in x]
    figures_dir = Path(__file__) / "figures"
    figures_dir.mkdir(exist_ok=True)

    for metric in maia_results:
        sns.lineplot(x=x, y=maia_results[metric], label="maia2_2050")
        sns.lineplot(x=x, y=allie_results[metric], label="allie_2500")

        plt.title(f"{metric} Performance vs. Ratio of Cheating Moves in Game Collections")
        plt.xlabel("Ratio of Cheating Moves in Game Collections")
        plt.ylabel(metric)

        plt.xticks(ticks=x, labels=bin_labels, rotation=30)
        plt.legend()

        plt.savefig(
            figures_dir / f"{metric}.pdf",
            format="pdf",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()


if __name__ == "__main__":
    make_p_experiment()