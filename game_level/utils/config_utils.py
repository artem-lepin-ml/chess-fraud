import pickle
import numpy as np
from typing import Dict
from omegaconf import OmegaConf
from dataclasses import dataclass
import yaml


def load_config(config_path: str = "./config.yaml"):
    """Load configuration from YAML file."""
    cfg = OmegaConf.load(config_path)
    return cfg


def load_embeddings(embeddings_path: str, columns: list = None) -> Dict[str, np.ndarray]:
    """
    Load embeddings from .npz file.

    Args:
        embeddings_path: Path to the .npz file
        columns: List of column names to load. If None, loads all columns.

    Returns:
        Dictionary mapping column names to numpy arrays
    """
    emb_df = np.load(embeddings_path)
    emb_dict = {}

    if columns is not None:
        for column in columns:
            if column in emb_df.files:
                emb_dict[column] = emb_df[column]
    else:
        for column in emb_df.files:
            emb_dict[column] = emb_df[column]

    return emb_dict


def load_pickle(pickle_path: str):
    """Load a pickle file."""
    with open(pickle_path, "rb") as f:
        return pickle.load(f)


def save_pickle(obj, pickle_path: str):
    """Save an object to a pickle file."""
    with open(pickle_path, "wb") as f:
        pickle.dump(obj, f)


@dataclass
class ModelConfig:
    """Configuration for the Chess Encoder model."""
    embedding_dim: int
    encoder_hidden_dim: int
    num_attention_heads: int
    num_encoder_layers: int
    last_layer_dim: int

    @classmethod
    def from_cfg(cls, cfg):
        """Create ModelConfig from the global configuration."""
        return cls(
            embedding_dim=cfg.model.embedding_dim,
            encoder_hidden_dim=cfg.model.encoder_hidden_dim,
            num_attention_heads=cfg.model.num_attention_heads,
            num_encoder_layers=cfg.model.num_encoder_layers,
            last_layer_dim=cfg.model.last_layer_dim,
        )


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int
    learning_rate: float
    n_epochs: int
    move_loss_coeff: float
    seed: int
    gpu_id: int
    device: str

    @classmethod
    def from_cfg(cls, cfg):
        """Create TrainingConfig from the global configuration."""
        return cls(
            batch_size=cfg.training.batch_size,
            learning_rate=cfg.training.learning_rate,
            n_epochs=cfg.training.n_epochs,
            move_loss_coeff=cfg.training.move_loss_coeff,
            seed=cfg.training.seed,
            gpu_id=cfg.device.gpu_id,
            device=cfg.device.device,
        )


@dataclass
class CollectionConfig:
    """Configuration for collections."""
    k_min: int
    k_max: int
    min_hmove: int
    max_hmove: int
    min_cheat_p: float
    max_cheat_p: float
    train_n_collections: int
    test_n_collections: int
    optimal_n_cols: int

    @classmethod
    def from_cfg(cls, cfg):
        """Create CollectionConfig from the global configuration."""
        return cls(
            k_min=cfg.collections.k_min,
            k_max=cfg.collections.k_max,
            min_hmove=cfg.collections.min_hmove,
            max_hmove=cfg.collections.max_hmove,
            min_cheat_p=cfg.collections.min_cheat_p,
            max_cheat_p=cfg.collections.max_cheat_p,
            train_n_collections=cfg.collections.train_n_collections,
            test_n_collections=cfg.collections.test_n_collections,
            optimal_n_cols=cfg.collections.optimal_n_cols,
        )


@dataclass
class AllCheatColumns:
    """Data class holding all cheat column configurations."""
    allie: list
    maia2: list

    @classmethod
    def from_cfg(cls, cfg):
        """Create AllCheatColumns from the global configuration."""
        return cls(
            allie=list(cfg.cheat_columns.allie),
            maia2=list(cfg.cheat_columns.maia2),
        )
