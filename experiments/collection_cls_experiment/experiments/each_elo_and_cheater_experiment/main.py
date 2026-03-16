import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import pickle
from pathlib import Path
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

from humanlike.collection_cls_experiment.utils.collection_utils import make_collections_from_indices
from humanlike.collection_cls_experiment.utils.train_and_eval_utils import train_and_eval, eval
from humanlike.collection_cls_experiment.collection_method.dataset import ChessEncoderDataset, collate_fn
from humanlike.collection_cls_experiment.collection_method.model import ChessEncoder
from consts import *


def build_heatmap_df(df, model_name, elo_bins, cheaters_order):
    model_df = df[df["model"] == model_name]
    heatmap_df = (
        model_df
        .set_index("cheater_id")[elo_bins]
        .loc[cheaters_order]
    )
    return heatmap_df


def make_figures(df):
    need_columns = [c for c in df.columns if c.startswith("f1_macro")]
    df = df[need_columns + ["model", "cheat_column"]]

    df = df.rename(
        columns=dict(
            zip(
                need_columns,
                ["-".join(c.split("_")[2:]) for c in need_columns],
            )
        )
    )

    # =========================
    # Normalize cheating strategy names
    # =========================
    def normalize_cheater(name: str) -> str:
        if name.startswith("fen_"):
            return name[len("fen_"):]
        if name.startswith("move_"):
            return name[len("move_"):]
        return name

    df["cheater_id"] = df["cheat_column"].apply(normalize_cheater)

    elo_bins = [
        "0-1200",
        "1200-1400",
        "1400-1600",
        "1600-1800",
        "1800-2000",
        "2000-2200",
    ]


    # =========================
    # Determine global order (by quality)
    # =========================
    allie_df = df[df["model"] == "allie_2500"]

    cheater_quality = (
        allie_df
        .set_index("cheater_id")[elo_bins]
        .mean(axis=1)
    )

    cheaters_order = (
        cheater_quality
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    # =========================
    # Build heatmaps
    # =========================
    heatmap_allie = build_heatmap_df(
        df,
        model_name="allie_2500",
        elo_bins=elo_bins,
        cheaters_order=cheaters_order,
    )

    heatmap_maia = build_heatmap_df(
        df,
        model_name="maia2_2050",
        elo_bins=elo_bins,
        cheaters_order=cheaters_order,
    )

    # =========================
    # Plot
    # =========================
    vmin = min(heatmap_allie.min().min(), heatmap_maia.min().min())
    vmax = max(heatmap_allie.max().max(), heatmap_maia.max().max())

    fig = plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(
        nrows=1,
        ncols=3,
        width_ratios=[1, 1, 0.05],
        wspace=0.25,
    )

    ax_left = fig.add_subplot(gs[0, 0])
    ax_right = fig.add_subplot(gs[0, 1], sharey=ax_left)
    cax = fig.add_subplot(gs[0, 2])

    sns.heatmap(
        heatmap_allie,
        ax=ax_left,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        cbar=False,
    )

    sns.heatmap(
        heatmap_maia,
        ax=ax_right,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        annot=True,
        fmt=".2f",
        cbar=True,
        cbar_ax=cax,
        cbar_kws={"label": "F1-macro"},
    )

    ax_left.set_title("Allie detector")
    ax_right.set_title("MAIA2 detector")

    ax_left.set_xlabel("Player Elo bin")
    ax_right.set_xlabel("Player Elo bin")

    ax_left.set_ylabel("Cheating strategy")
    ax_right.set_ylabel("")
    ax_right.tick_params(axis="y", left=False, labelleft=False)

    for ax in (ax_left, ax_right):
        ax.set_xticklabels(
            ax.get_xticklabels(),
            rotation=30,
            ha="right",
        )

    plt.tight_layout()
    figures_dir = (Path(__file__).parent / "figures")
    figures_dir.mkdir(exist_ok=True)
    plt.savefig(figures_dir / "elo_heatmap.pdf", format='pdf', dpi=300, bbox_inches='tight')
    plt.show()


def evaluate_on_each_bin(model_path, test_data, model_name):
    model = ChessEncoder()
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    results_dict = {"model" : model_name}
    for elo_bin in ELO_BOUNDS:
        low_bound = elo_bin - BIN_WIDTH
        if elo_bin == MIN_ELO:
            low_bound = 0
        test_loader = DataLoader(
            ChessEncoderDataset(*test_data, elo_bounds=(low_bound, elo_bin)),
            BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=-1
        )
        #loss, f1, precision, recall, f1_macro, acc 
        _, f1, precision, recall, f1_macro, acc = eval(model, test_loader, verbose=False)
        results_dict[f"f1_{low_bound}_{elo_bin}"] = f1
        results_dict[f"precision_{low_bound}_{elo_bin}"] = precision
        results_dict[f"recall_{low_bound}_{elo_bin}"] = recall
        results_dict[f"f1_macro_{low_bound}_{elo_bin}"] = f1_macro
        results_dict[f"accuracy_{low_bound}_{elo_bin}"] = acc
    # across the entire set of elos
    test_loader = DataLoader(
        ChessEncoderDataset(*test_data),
        BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=-1
    )
    #loss, f1, precision, recall, f1_macro, accuracy 
    _, f1, precision, recall, f1_macro, acc = eval(model, test_loader, verbose=False)
    results_dict[f"f1_all"] = f1
    results_dict[f"precision_all"] = precision
    results_dict[f"recall_all"] = recall
    results_dict[f"f1_macro_all"] = f1_macro
    results_dict[f"accuracy_all"] = acc

    return results_dict      


def train_and_eval_on_one_cheater(model_name, cheat_column):
    emb_dict = {}
    if model_name == "maia2_2050":
        emb_df = np.load(SYNT_MAIA2_EMBS_PATH)
        emb_dict["fen_before"] = emb_df["fen_before"]
        emb_dict["fen_after"] = emb_df["fen_after"]
    else:
        emb_df = np.load(SYNT_ALLIE_EMBS_PATH)
        emb_dict["move_uci"] = emb_df["move_uci"]
    emb_dict[cheat_column] = emb_df[cheat_column]

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
    model_path = Path(__file__).parent / "checkpoints" / model_name / f"{cheat_column}.pt"
    train_and_eval(model, optimizer, train_loader, val_loader, test_loader, N_EPOCHS, model_path=model_path)
    for_elo_dict = evaluate_on_each_bin(model_path, test_data, model_name)
    for_elo_dict["cheat_column"] = cheat_column
    return for_elo_dict

def make_table_for_all_cheaters_and_elos():
    results_list = []
    for cheat_column in ALLIE_CHEAT_COLUMNS:
        results_list.append(train_and_eval_on_one_cheater("allie_2500", cheat_column))

    for cheat_column in MAIA2_CHEAT_COLUMNS:
        results_list.append(train_and_eval_on_one_cheater("maia2_2050", cheat_column))
    
    (Path(__file__).parent / "results").mkdir(exist_ok=True)
    results = pd.DataFrame(results_list)
    results.to_csv("results/each_elo_and_cheater_results.csv")
    make_figures(results)


if __name__ == "__main__":
    make_table_for_all_cheaters_and_elos()