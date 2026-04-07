from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model architecture parameters."""

    arch: str = "mlp"
    hidden_dims: list[int] = field(default_factory=lambda: [256])
    activation: str = "GELU"        # ReLU | GELU | LeakyReLU
    normalization: str | None = "batch"  # batch | layer | None
    dropout: float = 0.0


@dataclass
class TrainConfig:
    """Training hyper-parameters."""

    lr: float = 3e-4
    weight_decay: float = 3e-4
    epochs: int = 500
    early_stop_patience: int = 20
    scheduler_patience: int = 10
    scheduler_factor: float = 0.1
    threshold_grid: list[float] = field(default_factory=lambda: [0.5])
