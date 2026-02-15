import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score
from collections import defaultdict
from omegaconf import OmegaConf
import chess
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from sklearn.dummy import DummyClassifier

import matplotlib.pyplot as plt
import seaborn as sns
from pprint import pprint

# Default device and epochs for backward compatibility
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
N_EPOCHS = 5


def optimal_threshold_f1_macro(y_true, y_probs):
    thresholds = np.linspace(0.0, 1.0, 100)
    best_f1 = 0.
    best_t = -1
    for t in thresholds:
        y_pred = (y_probs >= t)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    return best_t, best_f1


def train_one_epoch(model, optimizer, loss_fn, loader, device):
    model.train()
    total_loss = 0.
    move_loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.], dtype=torch.float32, device=device))
    for collections, move_labels, mask, collection_labels in loader:
        collections = collections.to(device)
        move_labels = move_labels.to(device)
        mask = mask.to(device)
        collection_labels = collection_labels.to(device)
        optimizer.zero_grad()
        move_logits, collection_logits = model(collections, mask)
        valid = ~mask  # mask=True на padding
        valid = valid[:, 1:] # не берем CLS
        valid = valid.reshape(-1)
        move_loss = move_loss_fn(
            move_logits.view(-1)[valid],
            move_labels.view(-1)[valid],
        )
        collection_loss = loss_fn(collection_logits.squeeze(), collection_labels)
        loss = move_loss + collection_loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(loader)

from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt


def eval(model, loss_fn, loader, verbose=True, file=None, device=None):
    model.eval()
    total_loss = 0.
    all_preds = []
    all_true = []

    # === NEW: move-level storage ===
    all_move_probs = []
    all_move_true = []

    with torch.no_grad():
        for collections, move_labels, mask, collection_labels in loader:
            collections = collections.to(device)
            move_labels = move_labels.to(device)
            mask = mask.to(device)
            collection_labels = collection_labels.to(device)

            move_logits, collection_logits = model(collections, mask)

            valid = ~mask
            valid = valid[:, 1:]          # drop CLS
            valid_flat = valid.reshape(-1)

            move_logits_flat = move_logits.view(-1)
            move_labels_flat = move_labels.view(-1)

            move_loss = loss_fn(
                move_logits_flat[valid_flat],
                move_labels_flat[valid_flat],
            )

            collection_loss = loss_fn(
                collection_logits.squeeze(),
                collection_labels,
            )

            loss = move_loss + collection_loss
            total_loss += loss.item()

            # === game-level ===
            bin_preds = (torch.sigmoid(collection_logits) > 0.5)
            all_preds.append(bin_preds)
            all_true.append(collection_labels)

            # === NEW: move-level ===
            move_probs = torch.sigmoid(move_logits_flat[valid_flat])
            move_true = move_labels_flat[valid_flat]

            all_move_probs.append(move_probs)
            all_move_true.append(move_true)

    # ===============================
    # Game-level metrics
    # ===============================
    all_true = torch.cat(all_true).cpu()
    all_preds = torch.cat(all_preds).cpu()

    if verbose:
        print("GAME-LEVEL".center(60, "="))
        print(classification_report(all_true, all_preds))

    if file:
        print(classification_report(all_true, all_preds), file=file)

    # ===============================
    # Move-level metrics (NEW)
    # ===============================
    all_move_probs = torch.cat(all_move_probs).cpu()
    all_move_true = torch.cat(all_move_true).cpu()
    all_move_preds = (all_move_probs > 0.5).long()
    
    if verbose:
        print("MOVE-LEVEL".center(60, "="))
        print(classification_report(all_move_true, all_move_preds))

        # ROC-AUC
        roc_auc = roc_auc_score(all_move_true, all_move_probs)
        fpr, tpr, _ = roc_curve(all_move_true, all_move_probs)

        print(f"MOVE-LEVEL ROC AUC: {roc_auc:.4f}")

        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Move-level ROC")
        plt.legend()
        plt.show()

    if file:
        print("MOVE-LEVEL", file=file)
        print(classification_report(all_move_true, all_move_preds), file=file)

    return (
        total_loss / len(loader),
        f1_score(all_true, all_preds),
        precision_score(all_true, all_preds),
        recall_score(all_true, all_preds),
        f1_score(all_true, all_preds, average="macro"),
        f1_score(all_true, all_preds, average="weighted"),
    )


def train_and_eval(model, optimizer, loss_fn, train_loader, val_loader, test_loader, n_epochs=N_EPOCHS, verbose=False, plots=False, save_cls_reports_file=None, model_name=None, device=DEVICE, eval_train=False, eval_test=False):
    results_dict = defaultdict(list)
    file = None
    best_f_macro = 0.
    if save_cls_reports_file is not None:
        file = open(save_cls_reports_file, mode="w")
    for epoch in tqdm(range(n_epochs), desc="Model training: "):    
        train_one_epoch(model, optimizer, loss_fn, train_loader, device=device)
        
        if verbose: print("TRAIN".center(50, "="))
        if file: print(f"EPOCH: {epoch}. TRAIN.".center(50, "="), file=file)

        if eval_train:
            loss, f1, precision, recall, f1_macro, f1_weighted = eval(model, loss_fn, train_loader, verbose=verbose, file=file, device=device)
            results_dict["train_loss"].append(loss)
            results_dict["train_f1"].append(f1)
            results_dict["train_f1_macro"].append(f1_macro)
            results_dict["train_precision"].append(precision)
            results_dict["train_recall"].append(recall)
            results_dict["train_f1_weighted"].append(f1_weighted)
        
        if verbose: print("VAL".center(50, "="))
        if file: print("VAL.".center(50, "="), file=file)

        loss, f1, precision, recall, f1_macro, f1_weighted = eval(model, loss_fn, val_loader, verbose=verbose, file=file, device=device)
        results_dict["val_loss"].append(loss)
        results_dict["val_f1"].append(f1)
        results_dict["val_f1_macro"].append(f1_macro)
        results_dict["val_precision"].append(precision)
        results_dict["val_recall"].append(recall)
        results_dict["val_f1_weighted"].append(f1_weighted)
        # сохраняем модель с лучшим f1 на валидации
        if f1_macro > best_f_macro and model_name is not None:
            torch.save(model.state_dict(), model_name)
            best_f_macro = f1_macro

        if verbose: print("TEST".center(50, "="))
        if file: print("VAL.".center(50, "="), file=file)

        if eval_test:
            loss, f1, precision, recall, f1_macro, f1_weighted = eval(model, loss_fn, test_loader, verbose=verbose, file=file, device=device)
            results_dict["test_f1"].append(f1)   
            results_dict["test_f1_macro"].append(f1_macro)
            results_dict["test_precision"].append(precision)
            results_dict["test_recall"].append(recall)
            results_dict["test_f1_weighted"].append(f1_weighted)

    best_iteration = np.argmax(results_dict["val_f1_macro"])
    if eval_test: test_f1_macro = results_dict["test_f1_macro"][best_iteration]
    results_dict["best_iteration"] = best_iteration
    if eval_test: results_dict["best_test_f1_macro"] = test_f1_macro
    results_dict["best_val_f1_macro"] = max(results_dict["val_f1_macro"])

    if plots and eval_train:
        plt.plot(results_dict["train_loss"], label="train")
        plt.plot(results_dict["val_loss"], label="val")
        plt.title("Loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        plt.show()

        plt.plot(results_dict["train_f1"], label="train")
        plt.plot(results_dict["val_f1"], label="val")
        plt.title("F1 score of cheat games")
        plt.xlabel("epoch")
        plt.ylabel("f1")
        plt.legend()
        plt.show()

    if file: file.close()    
    return results_dict













def elo_train_one_epoch(model, optimizer, loss_fn, loader, device):
    model.train()
    total_elo_loss = 0.
    elo_loss_fn = nn.CrossEntropyLoss()
    for collections, mask, elo_target in loader:
        collections = collections.to(device)
        mask = mask.to(device)
        elo_target = elo_target.to(device)
        optimizer.zero_grad()
        elo_logits = model(collections, mask)
        loss = elo_loss_fn(elo_logits, elo_target)
        total_elo_loss += loss.item()
        loss.backward()
        optimizer.step()
    print(f"Elo loss: {total_elo_loss / len(loader)}")


def elo_eval(model, loss_fn, loader, verbose=True, file=None, device=DEVICE, verbose_dummy=False, eval_moves=True):
    model.eval()
    all_elo_true = []
    all_elo_pred = []

    with torch.no_grad():
        for collections, mask, elo_target in loader:
            collections = collections.to(device)
            mask = mask.to(device)
            elo_target = elo_target.to(device)
            
            elo_logits = model(collections, mask)
                     
            elo_pred = torch.argmax(elo_logits, dim=1).cpu()
            all_elo_true.append(elo_target.cpu())
            all_elo_pred.append(elo_pred.cpu())

    all_elo_true = torch.cat(all_elo_true).numpy()
    all_elo_pred = torch.cat(all_elo_pred).numpy()

    print("ELO-LEVEL".center(50, "="))
    print("ELO class counts (true):", np.bincount(all_elo_true, minlength=6))
    print("ELO class counts (pred):", np.bincount(all_elo_pred, minlength=6))
    print("ELO accuracy:", accuracy_score(all_elo_true, all_elo_pred))
    print("ELO macro-F1:", f1_score(all_elo_true, all_elo_pred, average="macro", zero_division=0))
    print("ELO classification report:")
    print(classification_report(all_elo_true, all_elo_pred, zero_division=0))

    elo_cm = confusion_matrix(all_elo_true, all_elo_pred, labels=list(range(6)))
    print("ELO confusion matrix (rows=true, cols=pred):")
    print(elo_cm)

    if file:
        print(classification_report(all_elo_true, all_elo_pred), file=file)
    #loss, f1, precision, recall, f1_macro, f1_weighted 
    return precision_score(all_elo_true, all_elo_pred, average="macro",), \
        recall_score(all_elo_true, all_elo_pred, average="macro",), \
        f1_score(all_elo_true, all_elo_pred, average="macro"), \
        f1_score(all_elo_true, all_elo_pred, average="weighted")


def elo_train_and_eval(model, optimizer, loss_fn, 
                   train_loader, val_loader, test_loader, 
                   n_epochs=N_EPOCHS, verbose=False, plots=False, 
                   save_cls_reports_file=None, model_name=None, 
                   device=DEVICE, eval_train=False, eval_test=False, 
                   eval_moves=True):
    results_dict = defaultdict(list)
    file = None
    best_f_macro = 0.
    if save_cls_reports_file is not None:
        file = open(save_cls_reports_file, mode="w")
    for epoch in tqdm(range(n_epochs), desc="Model training: "):    
        elo_train_one_epoch(model, optimizer, loss_fn, train_loader, device=device)
        
        if verbose: print("TRAIN".center(50, "="))
        if file: print(f"EPOCH: {epoch}. TRAIN.".center(50, "="), file=file)

        if eval_train:
            precision, recall, f1_macro, f1_weighted = elo_eval(model, loss_fn, train_loader, verbose=verbose, file=file, device=device, eval_moves=eval_moves)
            results_dict["train_f1_macro"].append(f1_macro)
            results_dict["train_precision"].append(precision)
            results_dict["train_recall"].append(recall)
            results_dict["train_f1_weighted"].append(f1_weighted)
        
        if verbose: print("VAL".center(50, "="))
        if file: print("VAL.".center(50, "="), file=file)

        precision, recall, f1_macro, f1_weighted = elo_eval(model, loss_fn, val_loader, verbose=verbose, file=file, device=device, eval_moves=eval_moves)
        results_dict["val_f1_macro"].append(f1_macro)
        results_dict["val_precision"].append(precision)
        results_dict["val_recaall"].append(recall)
        results_dict["val_f1_weighted"].append(f1_weighted)
        # сохраняем модель с лучшим f1 на валидации
        if f1_macro > best_f_macro and model_name is not None:
            torch.save(model.state_dict(), model_name)
            best_f_macro = f1_macro

        if verbose: print("TEST".center(50, "="))
        if file: print("VAL.".center(50, "="), file=file)

        if eval_test:
            precision, recall, f1_macro, f1_weighted = elo_eval(model, loss_fn, test_loader, verbose=verbose, file=file, device=device, eval_moves=eval_moves)
            results_dict["test_f1_macro"].append(f1_macro)
            results_dict["test_precision"].append(precision)
            results_dict["test_recall"].append(recall)
            results_dict["test_f1_weighted"].append(f1_weighted)

    best_iteration = np.argmax(results_dict["val_f1_macro"])
    if eval_test: test_f1_macro = results_dict["test_f1_macro"][best_iteration]
    results_dict["best_iteration"] = best_iteration
    if eval_test: results_dict["best_test_f1_macro"] = test_f1_macro
    results_dict["best_val_f1_macro"] = max(results_dict["val_f1_macro"])

    if file: file.close()    
    return results_dict









def anticheat_train_one_epoch(
    model,
    optimizer,
    loader,
    *,
    move_loss_pos_weight=3.,
    device=DEVICE,
):
    model.train()

    if move_loss_pos_weight is None:
        move_loss_fn = nn.BCEWithLogitsLoss()
    else:
        move_loss_fn = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(
                [move_loss_pos_weight],
                device=device,
                dtype=torch.float32,
            )
        )

    game_loss_fn = nn.BCEWithLogitsLoss(
        # pos_weight=torch.tensor(
        #         [1.5],
        #         device=device,
        #         dtype=torch.float32,
        #     )
    )

    total_move_loss = 0.0
    total_game_loss = 0.0

    for collections, move_labels, padding_mask, game_labels, elo_target in loader:
        collections = collections.to(device)
        move_labels = move_labels.to(device)
        padding_mask = padding_mask.to(device)
        game_labels = game_labels.to(device)
        elo_target = elo_target.to(device)

        optimizer.zero_grad()

        move_logits, game_logits = model(
            collections,
            padding_mask,
            elo_target=elo_target,
        )

        # ---- move loss (mask padding) ----
        valid = ~padding_mask                      # (B, S)
        flat_valid = valid.view(-1)

        move_loss = move_loss_fn(
            move_logits.view(-1)[flat_valid],
            move_labels.view(-1)[flat_valid],
        )

        # ---- game loss ----
        game_loss = 2. * game_loss_fn(
            game_logits,
            game_labels,
        )

        loss = move_loss + game_loss
        loss.backward()
        optimizer.step()

        total_move_loss += move_loss.item()
        total_game_loss += game_loss.item()

    n = len(loader)
    print(
        f"Move loss: {total_move_loss / n:.4f}, "
        f"Game loss: {total_game_loss / n:.4f}"
    )


def anticheat_eval(
    model,
    loader,
    *,
    device=DEVICE,
    verbose=True,
    eval_moves=True,
):
    model.eval()

    all_game_probs = []
    all_game_true = []

    all_move_probs = []
    all_move_true = []

    with torch.no_grad():
        for collections, move_labels, padding_mask, game_labels, elo_target in loader:
            collections = collections.to(device)
            move_labels = move_labels.to(device)
            padding_mask = padding_mask.to(device)
            game_labels = game_labels.to(device)
            elo_target = elo_target.to(device)

            move_logits, game_logits = model(
                collections,
                padding_mask,
                elo_target,
            )

            # ---- game-level ----
            game_probs = torch.sigmoid(game_logits)
            all_game_probs.append(game_probs.cpu())
            all_game_true.append(game_labels.cpu())

            # ---- move-level ----
            if eval_moves:
                move_logits = move_logits.squeeze(-1)
                move_probs = torch.sigmoid(move_logits)
                valid = ~padding_mask
                all_move_probs.append(move_probs[valid].cpu())
                all_move_true.append(move_labels[valid].cpu())

    metrics = {}

    # ================= GAME LEVEL =================
    all_game_true = torch.cat(all_game_true).numpy()
    all_game_probs = torch.cat(all_game_probs).numpy()

    game_thr, game_f1_macro = optimal_threshold_f1_macro(
        all_game_true, all_game_probs
    )
    game_preds = (all_game_probs >= 0.5).astype(int)

    metrics.update({
        "game_f1_macro": game_f1_macro,
        "game_precision_cheat": precision_score(
            all_game_true, game_preds, pos_label=1, zero_division=0
        ),
        "game_recall_cheat": recall_score(
            all_game_true, game_preds, pos_label=1, zero_division=0
        ),
        "game_f1_cheat": f1_score(
            all_game_true, game_preds, pos_label=1, zero_division=0
        ),
    })

    if verbose:
        print("GAME-LEVEL".center(50, "="))
        print(f"Best threshold: {game_thr:.3f}, best f1-macro {game_f1_macro}")
        print(classification_report(all_game_true, game_preds, zero_division=0))

    # ================= MOVE LEVEL =================
    if eval_moves:
        all_move_true = torch.cat(all_move_true).numpy()
        all_move_probs = torch.cat(all_move_probs).numpy()

        move_thr, move_f1_macro = optimal_threshold_f1_macro(
            all_move_true, all_move_probs
        )
        move_preds = (all_move_probs >= 0.5).astype(int)

        metrics.update({
            "move_f1_macro": move_f1_macro,
            "move_precision_cheat": precision_score(
                all_move_true, move_preds, pos_label=1, zero_division=0
            ),
            "move_recall_cheat": recall_score(
                all_move_true, move_preds, pos_label=1, zero_division=0
            ),
            "move_f1_cheat": f1_score(
                all_move_true, move_preds, pos_label=1, zero_division=0
            ),
        })

        if verbose:
            print("MOVE-LEVEL".center(50, "="))
            print(f"Best threshold: {move_thr:.3f}, best f1-macro {move_f1_macro}")
            print(classification_report(all_move_true, move_preds, zero_division=0))

    return metrics




def anticheat_train_and_eval(
    model,
    optimizer,
    *,
    train_loader,
    val_loader,
    test_loader=None,
    n_epochs=N_EPOCHS,
    device=DEVICE,
    verbose=False,
    eval_moves=True,
    eval_train=False,
    eval_test=False,
    model_name=None,
):
    results_dict = defaultdict(list)
    best_val_f1 = -1.0

    for epoch in tqdm(range(n_epochs), desc="Model training"):
        # -------- TRAIN --------
        anticheat_train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
        )

        # -------- TRAIN EVAL --------
        if eval_train:
            train_metrics = anticheat_eval(
                model,
                train_loader,
                device=device,
                verbose=verbose,
                eval_moves=eval_moves,
            )
            for k, v in train_metrics.items():
                results_dict[f"train_{k}"].append(v)

        # -------- VAL EVAL --------
        val_metrics = anticheat_eval(
            model,
            val_loader,
            device=device,
            verbose=verbose,
            eval_moves=eval_moves,
        )
        for k, v in val_metrics.items():
            results_dict[f"val_{k}"].append(v)

        # -------- CHECKPOINT --------
        if model_name is not None:
            cur_f1 = val_metrics["game_f1_macro"]
            if cur_f1 > best_val_f1:
                best_val_f1 = cur_f1
                torch.save(model.state_dict(), model_name)

        # -------- TEST EVAL --------
        if eval_test:
            assert test_loader is not None
            test_metrics = anticheat_eval(
                model,
                test_loader,
                device=device,
                verbose=verbose,
                eval_moves=eval_moves,
            )
            for k, v in test_metrics.items():
                results_dict[f"test_{k}"].append(v)

    results_dict["best_val_game_f1_macro"] = max(
        results_dict["val_game_f1_macro"]
    )

    return results_dict



#=============================================
# expects: DEVICE, N_EPOCHS
# expects: collapse_move_labels_to_binary, optimal_threshold_f1_macro

def collapse_move_labels_to_binary(move_labels: torch.Tensor) -> torch.Tensor:
    """
    move_labels: Tensor with values in {0,1,2} or padded with -100
    returns binary labels:
      1 -> cheat
      0,2 -> honest
      -100 stays -100
    """
    out = move_labels.clone()
    valid = out != -100
    out[valid] = (out[valid] == 1).long()
    return out


def train_one_epoch_3cls(
    model,
    optimizer,
    loader,
    *,
    device=DEVICE,
    lambda_game=1.0,   # можно начать с 1.0, потом тюнить
):
    model.train()
    move_loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    game_loss_fn = nn.BCEWithLogitsLoss()

    total_move_loss = 0.0
    total_game_loss = 0.0
    total_loss = 0.0
    n_batches = 0

    for collections, move_labels, padding_mask, collection_labels in loader:
        collections = collections.to(device)
        move_labels = move_labels.to(device)
        padding_mask = padding_mask.to(device)
        collection_labels = collection_labels.to(device)  # float (B,)

        optimizer.zero_grad(set_to_none=True)

        move_logits, game_logits = model(collections, padding_mask)  # (B,S,3), (B,1)

        loss_move = move_loss_fn(
            move_logits.reshape(-1, 3),
            move_labels.reshape(-1),
        )

        # game_logits: (B,1) -> (B,)
        loss_game = game_loss_fn(
            game_logits.squeeze(-1),
            collection_labels,
        )

        loss = loss_move + lambda_game * loss_game
        loss.backward()
        optimizer.step()

        total_move_loss += float(loss_move.item())
        total_game_loss += float(loss_game.item())
        total_loss += float(loss.item())
        n_batches += 1

    out = {
        "move_loss": total_move_loss / max(1, n_batches),
        "game_loss": total_game_loss / max(1, n_batches),
        "loss": total_loss / max(1, n_batches),
    }
    print(f"Loss: {out['loss']:.4f} | move: {out['move_loss']:.4f} | game: {out['game_loss']:.4f}")
    return out




def eval_3cls(model, loader, *, device=DEVICE, verbose=True):
    model.eval()

    all_game_probs = []
    all_game_true = []

    with torch.no_grad():
        for collections, move_labels, padding_mask, collection_labels in loader:
            collections = collections.to(device)
            padding_mask = padding_mask.to(device)
            collection_labels = collection_labels.to(device)

            _, game_logits = model(collections, padding_mask)  # (B,1)
            game_probs = torch.sigmoid(game_logits.squeeze(-1))  # (B,)

            all_game_probs.append(game_probs.detach().cpu())
            all_game_true.append(collection_labels.detach().cpu().long())

    all_game_probs = torch.cat(all_game_probs).numpy()
    all_game_true = torch.cat(all_game_true).numpy()

    best_thr, best_f1_macro = optimal_threshold_f1_macro(all_game_true, all_game_probs)
    preds = (all_game_probs >= 0.5).astype(int)

    if verbose:
        print("GAME-LEVEL (CLS token)".center(60, "="))
        print(f"Best threshold (search): {best_thr:.3f}")
        print(f"Best f1-macro (at best thr): {best_f1_macro:.4f}")
        print("PREDICTIONS AT THRESHOLD = 0.5".center(60, "-"))
        print(classification_report(all_game_true, preds, zero_division=0))

        cm = confusion_matrix(all_game_true, preds)
        print("CONFUSION MATRIX")
        print(cm)

        fpr, tpr, _ = roc_curve(all_game_true, all_game_probs)
        auc = roc_auc_score(all_game_true, all_game_probs)

        plt.figure(figsize=(5, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], "--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Game-level ROC (CLS)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    return {
        "game_f1_macro_best_thr": float(best_f1_macro),
        "game_f1_cheat@0.5": float(f1_score(all_game_true, preds, pos_label=1, zero_division=0)),
        "game_precision_cheat@0.5": float(precision_score(all_game_true, preds, pos_label=1, zero_division=0)),
        "game_recall_cheat@0.5": float(recall_score(all_game_true, preds, pos_label=1, zero_division=0)),
        "game_auc": float(roc_auc_score(all_game_true, all_game_probs)),
    }



def train_and_eval_3cls(
    model,
    optimizer,
    *,
    train_loader,
    val_loader,
    test_loader=None,
    n_epochs=N_EPOCHS,
    device=DEVICE,
    verbose=False,
    eval_train=False,
    eval_test=False,
    model_name=None,
):
    results = defaultdict(list)
    best_val_f1 = -1.0

    for epoch in tqdm(range(n_epochs), desc="Training 3cls"):
        print(f"\nEPOCH {epoch}".center(60, "="))

        train_out = train_one_epoch_3cls(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
        )
        results["train_move_loss"].append(train_out["move_loss"])

        if eval_train:
            train_metrics = eval_3cls(model, train_loader, device=device, verbose=verbose)
            for k, v in train_metrics.items():
                results[f"train_{k}"].append(v)

        val_metrics = eval_3cls(model, val_loader, device=device, verbose=verbose)
        for k, v in val_metrics.items():
            results[f"val_{k}"].append(v)

        cur_f1 = val_metrics["game_f1_macro_best_thr"]
        if model_name is not None and cur_f1 > best_val_f1:
            best_val_f1 = cur_f1
            torch.save(model.state_dict(), model_name)

        if eval_test:
            assert test_loader is not None
            test_metrics = eval_3cls(model, test_loader, device=device, verbose=verbose)
            for k, v in test_metrics.items():
                results[f"test_{k}"].append(v)

    results["best_val_game_f1_macro_best_thr"] = best_val_f1
    return results



#====================================
def st_train_one_epoch(
    model,
    optimizer,
    loader,
    *,
    device=DEVICE,
    lambda_game=1.0,
):
    model.train()

    move_loss_fn = nn.CrossEntropyLoss(ignore_index=-100, weight=torch.tensor([1., 10.], device=device))
    game_loss_fn = nn.BCEWithLogitsLoss()

    total_move_loss = 0.0
    total_game_loss = 0.0
    total_loss = 0.0
    n_batches = 0

    for (
        embeddings,
        stockfish_match,
        move_labels,
        padding_mask,
        collection_labels,
    ) in loader:

        embeddings = embeddings.to(device)
        stockfish_match = stockfish_match.to(device)
        move_labels = move_labels.to(device)
        padding_mask = padding_mask.to(device)
        collection_labels = collection_labels.to(device)

        # ---- collapse move labels to binary ----
        move_labels_bin = collapse_move_labels_to_binary(move_labels)
        assert torch.isfinite(collection_labels).all(), (
            "NaN or inf in collection_labels"
        )
        assert torch.isfinite(move_labels_bin[move_labels_bin != -100]).all(), (
            "NaN in move labels"
        )
        optimizer.zero_grad(set_to_none=True)

        move_logits, game_logits = model(
            embeddings,
            stockfish_match,
            padding_mask,
        )  # (B,S,2), (B,1)

        # ---- move loss ----
        loss_move = move_loss_fn(
            move_logits.reshape(-1, 2),
            move_labels_bin.reshape(-1),
        )

        # ---- game loss ----
        loss_game = game_loss_fn(
            game_logits.squeeze(-1),
            collection_labels,
        )

        loss = loss_move + lambda_game * loss_game
        loss.backward()
        optimizer.step()

        total_move_loss += float(loss_move.item())
        total_game_loss += float(loss_game.item())
        total_loss += float(loss.item())
        n_batches += 1

    out = {
        "move_loss": total_move_loss / max(1, n_batches),
        "game_loss": total_game_loss / max(1, n_batches),
        "loss": total_loss / max(1, n_batches),
    }

    print(
        f"Loss: {out['loss']:.4f} | "
        f"move: {out['move_loss']:.4f} | "
        f"game: {out['game_loss']:.4f}"
    )
    return out


def st_eval(
    model,
    loader,
    *,
    device=DEVICE,
    verbose=True,
):
    model.eval()

    all_game_probs = []
    all_game_true = []

    all_move_probs = []
    all_move_true = []

    with torch.no_grad():
        for (
            embeddings,
            stockfish_match,
            move_labels,
            padding_mask,
            collection_labels,
        ) in loader:

            embeddings = embeddings.to(device)
            stockfish_match = stockfish_match.to(device)
            move_labels = move_labels.to(device)
            padding_mask = padding_mask.to(device)
            collection_labels = collection_labels.to(device)

            move_logits, game_logits = model(
                embeddings,
                stockfish_match,
                padding_mask,
            )  # (B,S,2), (B,1)

            # -------------------------
            # game-level
            # -------------------------
            game_probs = torch.sigmoid(game_logits.squeeze(-1))  # (B,)
            all_game_probs.append(game_probs.detach().cpu())
            all_game_true.append(collection_labels.detach().cpu().long())

            # -------------------------
            # move-level (binary) with ignore_index=-100
            # -------------------------
            move_labels_bin = collapse_move_labels_to_binary(move_labels)  # (B,S), {-100,0,1}
            valid = (move_labels_bin != -100)  # (B,S)

            # prob of class=1 (cheat)
            move_probs = torch.softmax(move_logits, dim=-1)[..., 1]  # (B,S)

            if valid.any():
                all_move_probs.append(move_probs[valid].detach().cpu())
                all_move_true.append(move_labels_bin[valid].detach().cpu().long())

    # ============================================================
    # aggregate
    # ============================================================
    all_game_probs = torch.cat(all_game_probs).numpy()
    all_game_true = torch.cat(all_game_true).numpy()

    if len(all_move_probs) > 0:
        all_move_probs = torch.cat(all_move_probs).numpy()
        all_move_true = torch.cat(all_move_true).numpy()
    else:
        all_move_probs = None
        all_move_true = None

    # ============================================================
    # game metrics
    # ============================================================
    best_thr, best_f1_macro = optimal_threshold_f1_macro(all_game_true, all_game_probs)
    game_preds_05 = (all_game_probs >= 0.5).astype(int)

    out = {
        "game_f1_macro_best_thr": float(best_f1_macro),
        "game_best_thr": float(best_thr) if best_thr is not None else -1.0,
        "game_f1_cheat@0.5": float(f1_score(all_game_true, game_preds_05, pos_label=1, zero_division=0)),
        "game_precision_cheat@0.5": float(precision_score(all_game_true, game_preds_05, pos_label=1, zero_division=0)),
        "game_recall_cheat@0.5": float(recall_score(all_game_true, game_preds_05, pos_label=1, zero_division=0)),
    }

    # ROC AUC only if finite and both classes present
    if np.isfinite(all_game_probs).all() and (np.unique(all_game_true).size == 2):
        out["game_auc"] = float(roc_auc_score(all_game_true, all_game_probs))
    else:
        out["game_auc"] = float("nan")

    # ============================================================
    # move metrics
    # ============================================================
    if all_move_probs is not None:
        move_preds_05 = (all_move_probs >= 0.5).astype(int)

        out.update(
            {
                "move_f1_cheat@0.5": float(f1_score(all_move_true, move_preds_05, pos_label=1, zero_division=0)),
                "move_precision_cheat@0.5": float(precision_score(all_move_true, move_preds_05, pos_label=1, zero_division=0)),
                "move_recall_cheat@0.5": float(recall_score(all_move_true, move_preds_05, pos_label=1, zero_division=0)),
            }
        )

        if np.isfinite(all_move_probs).all() and (np.unique(all_move_true).size == 2):
            out["move_auc"] = float(roc_auc_score(all_move_true, all_move_probs))
        else:
            out["move_auc"] = float("nan")
    else:
        out.update(
            {
                "move_f1_cheat@0.5": float("nan"),
                "move_precision_cheat@0.5": float("nan"),
                "move_recall_cheat@0.5": float("nan"),
                "move_auc": float("nan"),
            }
        )

    if verbose:
        # -------------------------
        # GAME REPORT
        # -------------------------
        print("GAME-LEVEL (CLS token)".center(60, "="))
        if best_thr is None:
            print("Best threshold (search): None")
            print(f"Best f1-macro (at best thr): {best_f1_macro:.4f}")
        else:
            print(f"Best threshold (search): {best_thr:.3f}")
            print(f"Best f1-macro (at best thr): {best_f1_macro:.4f}")

        print("PREDICTIONS AT THRESHOLD = 0.5".center(60, "-"))
        print(classification_report(all_game_true, game_preds_05, zero_division=0))

        cm = confusion_matrix(all_game_true, game_preds_05)
        print("CONFUSION MATRIX")
        print(cm)

        if np.isfinite(all_game_probs).all() and (np.unique(all_game_true).size == 2):
            fpr, tpr, _ = roc_curve(all_game_true, all_game_probs)
            auc = roc_auc_score(all_game_true, all_game_probs)

            plt.figure(figsize=(5, 5))
            plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
            plt.plot([0, 1], [0, 1], "--", color="gray")
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Game-level ROC (CLS)")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        else:
            print("ROC/AUC skipped for game-level (need finite probs and both classes present).")

        # -------------------------
        # MOVE REPORT
        # -------------------------
        print("MOVE-LEVEL (token)".center(60, "="))
        if all_move_probs is None:
            print("No valid move labels found (all -100). Skipping move-level metrics.")
        else:
            print("PREDICTIONS AT THRESHOLD = 0.5".center(60, "-"))
            print(classification_report(all_move_true, move_preds_05, zero_division=0))

            cm_m = confusion_matrix(all_move_true, move_preds_05)
            print("CONFUSION MATRIX")
            print(cm_m)

            if np.isfinite(all_move_probs).all() and (np.unique(all_move_true).size == 2):
                fpr_m, tpr_m, _ = roc_curve(all_move_true, all_move_probs)
                auc_m = roc_auc_score(all_move_true, all_move_probs)

                plt.figure(figsize=(5, 5))
                plt.plot(fpr_m, tpr_m, label=f"AUC = {auc_m:.3f}")
                plt.plot([0, 1], [0, 1], "--", color="gray")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("Move-level ROC (token)")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
            else:
                print("ROC/AUC skipped for move-level (need finite probs and both classes present).")

    return out



def st_train_and_eval(
    model,
    optimizer,
    *,
    train_loader,
    val_loader,
    test_loader=None,
    n_epochs=N_EPOCHS,
    device=DEVICE,
    verbose=False,
    eval_train=False,
    eval_test=False,
    model_name=None,
):
    results = defaultdict(list)
    best_val_f1 = -1.0

    for epoch in tqdm(range(n_epochs), desc="Training (Stockfish-conditioned)"):
        print(f"\nEPOCH {epoch}".center(60, "="))

        train_out = st_train_one_epoch(
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            device=device,
        )
        results["train_loss"].append(train_out["loss"])
        results["train_move_loss"].append(train_out["move_loss"])
        results["train_game_loss"].append(train_out["game_loss"])

        if eval_train:
            train_metrics = st_eval(
                model,
                train_loader,
                device=device,
                verbose=verbose,
            )
            for k, v in train_metrics.items():
                results[f"train_{k}"].append(v)

        val_metrics = st_eval(
            model,
            val_loader,
            device=device,
            verbose=verbose,
        )
        for k, v in val_metrics.items():
            results[f"val_{k}"].append(v)

        cur_f1 = val_metrics["game_f1_macro_best_thr"]
        if model_name is not None and cur_f1 > best_val_f1:
            best_val_f1 = cur_f1
            torch.save(model.state_dict(), model_name)

        if eval_test:
            assert test_loader is not None
            test_metrics = st_eval(
                model,
                test_loader,
                device=device,
                verbose=verbose,
            )
            for k, v in test_metrics.items():
                results[f"test_{k}"].append(v)

    results["best_val_game_f1_macro_best_thr"] = best_val_f1
    return results

