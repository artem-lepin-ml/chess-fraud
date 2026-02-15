import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np


class ChessEncoderDataset(Dataset):
    """
    PyTorch Dataset for chess move collections.

    Args:
        fair_emb_list: List of fair move embeddings for each player/collection
        cheat_emb_list: List of cheat move embeddings for each player/collection
        cheat_label_list: List of labels for cheat sequences
        p_list: List of p values (ratio of cheat moves) for each collection
        player_elo_list: List of player ELOs for each collection
        p_bounds: Tuple of (min_p, max_p) to filter collections by p
        elo_bounds: Tuple of (min_elo, max_elo) to filter collections by ELO
        tournament: If True, use tournament mode (separate fair/cheat elo lists)
        fair_elo_list: List of ELOs for fair collections (tournament mode only)
        cheat_elo_list: List of ELOs for cheat collections (tournament mode only)
    """

    def __init__(self, fair_emb_list, cheat_emb_list, cheat_label_list, p_list, player_elo_list, p_bounds=(0., 1.), elo_bounds=(0., 5000.), tournament=False, fair_elo_list=None, cheat_elo_list=None):
        self.cheat_emb_list = []
        self.cheat_label_list = []
        self.fair_emb_list = []

        if tournament:
            # Tournament mode: separate fair/cheat collections from real games
            for player_fair_emb_list, elo in zip(fair_emb_list, fair_elo_list, strict=True):
                if not elo_bounds[0] < elo <= elo_bounds[1]:
                    continue
                fair_emb_arr = np.stack(player_fair_emb_list)
                assert not np.isnan(fair_emb_arr).any(), "nan encountered"
                self.fair_emb_list.append(
                    torch.as_tensor(fair_emb_arr, dtype=torch.float32)
                )
            assert len(cheat_emb_list) == len(cheat_label_list) == len(p_list) == len(cheat_elo_list), f"{len(cheat_emb_list)}, {len(cheat_label_list)}, {len(p_list)}, {len(cheat_elo_list)}"
            for player_cheat_emb_list, player_labels, p, elo in zip(cheat_emb_list, cheat_label_list, p_list, cheat_elo_list):
                if not elo_bounds[0] < elo <= elo_bounds[1]:
                    continue
                if not p_bounds[0] < p <= p_bounds[1]:
                    continue
                player_cheat_arr = np.stack(player_cheat_emb_list)
                assert not np.isnan(player_cheat_arr).any(), "nan encountered"
                self.cheat_emb_list.append(
                    torch.as_tensor(player_cheat_arr, dtype=torch.float32)
                )
                self.cheat_label_list.append(
                    torch.as_tensor(player_labels, dtype=torch.float32)
                )
            return

        # Regular mode: paired fair/cheat collections
        assert len(cheat_emb_list) == len(fair_emb_list) == len(cheat_label_list) == len(player_elo_list) == len(p_list), f"{len(cheat_emb_list)}, {len(fair_emb_list)}, {len(cheat_label_list)}, {len(player_elo_list)}, {len(p_list)}"
        for player_fair_emb_list, player_cheat_emb_list, player_labels, player_elo, p in \
            zip(fair_emb_list, cheat_emb_list, cheat_label_list, player_elo_list, p_list):
            assert len(player_fair_emb_list) == len(player_cheat_emb_list) == len(player_labels)
            # Filter by p and ELO bounds
            if not p_bounds[0] < p <= p_bounds[1]:
                continue
            if not elo_bounds[0] < player_elo <= elo_bounds[1]:
                continue
            self.cheat_emb_list.append(
                torch.as_tensor(np.stack(player_cheat_emb_list), dtype=torch.float32)
            )
            self.cheat_label_list.append(
                torch.as_tensor(player_labels, dtype=torch.float32)
            )
            self.fair_emb_list.append(
                torch.as_tensor(np.stack(player_fair_emb_list), dtype=torch.float32)
            )

    def __len__(self):
        return len(self.fair_emb_list) + len(self.cheat_emb_list)

    def __getitem__(self, idx):
        """Get a single collection sample."""
        if idx < len(self.cheat_emb_list):
            cheat_example = self.cheat_emb_list[idx]
            cheat_labels = self.cheat_label_list[idx]
            # Randomly shuffle cheat moves (common in training)
            perm = torch.randperm(len(cheat_example))
            cheat_example = cheat_example[perm]
            cheat_labels = cheat_labels[perm]
            return cheat_example, cheat_labels, 1.
        fair_example = self.fair_emb_list[idx - len(self.cheat_emb_list)]
        return fair_example, torch.zeros(len(fair_example)), 0.


def collate_fn(batch):
    """
    Collate function for DataLoader to create padded batches.

    Args:
        batch: List of (collections, move_labels, collection_labels) tuples

    Returns:
        Tuple of (padded_collections, padded_move_labels, src_key_padding_mask, labels)
    """
    collections, move_labels, collection_labels = zip(*batch)
    lengths = torch.as_tensor([len(collection) for collection in collections])
    # Pad to maximum length in batch
    padded_collections = pad_sequence(collections, batch_first=True)
    padded_move_labels = pad_sequence(move_labels, batch_first=True)
    labels = torch.as_tensor(collection_labels)

    # Batch, Seq, Emb lengths
    B, S, E = padded_collections.shape
    # Create padding mask (True for padding positions)
    # Note: -1 start because model adds CLS token at position 0
    src_key_padding_mask = (torch.arange(-1, S).expand(B, S + 1) >= lengths.unsqueeze(1))
    return padded_collections, padded_move_labels, src_key_padding_mask, labels
