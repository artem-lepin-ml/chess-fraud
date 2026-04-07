"""Generic MLP builder with multi-layer support."""

from __future__ import annotations

from torch import nn

from ..configs.model_config import ModelConfig


def _get_activation(name: str) -> nn.Module:
    name_lower = name.lower()
    if name_lower == "relu":
        return nn.ReLU()
    if name_lower == "gelu":
        return nn.GELU()
    if name_lower in ("leakyrelu", "leaky_relu"):
        return nn.LeakyReLU(0.1)
    raise ValueError(f"Unknown activation: {name}")


def _get_norm(name: str | None, dim: int) -> nn.Module | None:
    if name is None:
        return None
    name_lower = name.lower()
    if name_lower == "batch":
        return nn.BatchNorm1d(dim)
    if name_lower == "layer":
        return nn.LayerNorm(dim)
    raise ValueError(f"Unknown normalization: {name}")


def build_mlp(
    input_dim: int,
    out_dim: int = 1,
    *,
    hidden_dims: list[int] | None = None,
    activation: str = "GELU",
    normalization: str | None = "batch",
    dropout: float = 0.0,
) -> nn.Sequential:
    """Build an MLP with N hidden layers.

    Per hidden layer: ``Linear -> [Norm] -> Activation -> [Dropout]``.
    Final layer: ``Linear(last_hidden, out_dim)`` with no activation.
    """
    if hidden_dims is None:
        hidden_dims = [256]

    layers: list[nn.Module] = []
    prev_dim = input_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(prev_dim, h_dim))
        norm = _get_norm(normalization, h_dim)
        if norm is not None:
            layers.append(norm)
        layers.append(_get_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        prev_dim = h_dim

    layers.append(nn.Linear(prev_dim, out_dim))
    return nn.Sequential(*layers)


def build_model_from_config(
    input_dim: int,
    cfg: ModelConfig,
    out_dim: int = 1,
) -> nn.Sequential:
    """Convenience: build model from a ``ModelConfig`` dataclass."""
    if cfg.arch != "mlp":
        raise ValueError(f"Unknown architecture: {cfg.arch}")
    return build_mlp(
        input_dim,
        out_dim,
        hidden_dims=cfg.hidden_dims,
        activation=cfg.activation,
        normalization=cfg.normalization,
        dropout=cfg.dropout,
    )
