import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve)
from collections import defaultdict
from tqdm import tqdm
import matplotlib.pyplot as plt
from consts import *


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


def train_one_epoch(model, optimizer, loss_fn, loader, device=DEVICE):
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
        valid = ~mask  # mask=True on padding
        valid = valid[:, 1:] # drop CLS
        valid = valid.reshape(-1)
        move_loss = move_loss_fn(
            move_logits.view(-1)[valid],
            move_labels.view(-1)[valid],
        )
        collection_loss = loss_fn(collection_logits.squeeze(), collection_labels)
        loss = MOVE_LOSS_COEFF * move_loss + collection_loss
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    return total_loss / len(loader)


def eval(model, loader, verbose=True, file=None, device=DEVICE):
    loss_fn = nn.BCEWithLogitsLoss()
    model.eval()
    total_loss = 0.
    all_preds = []
    all_true = []

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
            valid = valid[:, 1:] # drop CLS
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

            loss = MOVE_LOSS_COEFF * move_loss + collection_loss
            total_loss += loss.item()

            # === game-level ===
            bin_preds = (torch.sigmoid(collection_logits) > 0.5)
            all_preds.append(bin_preds)
            all_true.append(collection_labels)

            # === move-level ===
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
    # Move-level metrics
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
        accuracy_score(all_true, all_preds),
    )


def train_and_eval(
        model, 
        optimizer,
        train_loader, 
        val_loader, 
        test_loader, 
        n_epochs=N_EPOCHS, 
        verbose=False, 
        plots=False, 
        save_cls_reports_file=None, 
        model_path=None, 
        device=DEVICE, 
        eval_train=False, 
        eval_test=False
    ):
    loss_fn = nn.BCEWithLogitsLoss()
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
        # save model with the best validation f1
        if f1_macro > best_f_macro and model_path is not None:
            torch.save(model.state_dict(), model_path)
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
