"""Deterministic paired inputs for the published move-level detection models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import os
import random

import numpy as np
import pandas as pd
import torch
from torch import nn

CLASSICAL_ASSISTANCE_SOURCES = (
    "stockfish_1",
    "stockfish_5",
    "stockfish_9",
    "stockfish_11",
    "stockfish_15",
    "lc0_1",
    "lc0_10",
    "lc0_100",
)

_FEATURE_NAMES = (
    "player_elo",
    "opponent_elo",
    "thinking_time",
    "centipawn_loss",
    "normalized_centipawn_loss",
    "allie_win_probability",
    "stockfish_1_top1_match",
    "stockfish_9_top1_match",
    "stockfish_15_top1_match",
)
_EMBEDDING_DIMENSION = 1024
_WIN_PROBABILITY_SCALE = 0.00368208
_FLOAT32_MAX = float(np.finfo(np.float32).max)


@dataclass(frozen=True)
class MoveLevelPreprocessor:
    """Train-split statistics used to standardize the four scalar features."""

    player_elo_mean: float
    player_elo_std: float
    opponent_elo_mean: float
    opponent_elo_std: float
    thinking_time_mean: float
    thinking_time_std: float
    centipawn_loss_mean: float
    centipawn_loss_std: float


@dataclass(frozen=True)
class MoveLevelTrainConfig:
    """Published optimization settings for the move-level detection FFN."""

    seed: int = 42
    learning_rate: float = 3e-4
    weight_decay: float = 3e-4
    max_epochs: int = 2000
    scheduler_factor: float = 0.1
    scheduler_patience: int = 10
    early_stopping_patience: int = 40


@dataclass(frozen=True)
class MoveLevelCandidates:
    """Aligned candidate inputs for one side of each paired position."""

    row_ids: np.ndarray
    moves: np.ndarray
    evaluations: np.ndarray
    features: np.ndarray
    embeddings: np.ndarray


@dataclass(frozen=True)
class MoveLevelPairs:
    """Fair and assistance candidates aligned by source position."""

    fair: MoveLevelCandidates
    assistance: MoveLevelCandidates
    assistance_sources: np.ndarray
    feature_names: tuple[str, ...] = _FEATURE_NAMES


def _set_all_seeds(seed: int) -> None:
    """Seed Python, NumPy, and Torch for deterministic detector training."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


def fit_move_level_preprocessor(train_rows: pd.DataFrame) -> MoveLevelPreprocessor:
    """Fit population feature statistics on observed training rows only."""

    _require_columns(
        train_rows,
        (
            "player_elo",
            "opponent_elo",
            "move_thinking_time",
            "centipawn_loss",
        ),
    )
    if train_rows.empty:
        raise ValueError("Cannot fit the move-level detection preprocessor on zero rows.")

    player_elo = _finite_column(train_rows, "player_elo")
    opponent_elo = _finite_column(train_rows, "opponent_elo")
    thinking_time = np.log1p(
        np.maximum(_finite_column(train_rows, "move_thinking_time"), 0.0)
    )
    centipawn_loss = _finite_column(train_rows, "centipawn_loss")
    return MoveLevelPreprocessor(
        player_elo_mean=float(player_elo.mean()),
        player_elo_std=_usable_std(player_elo),
        opponent_elo_mean=float(opponent_elo.mean()),
        opponent_elo_std=_usable_std(opponent_elo),
        thinking_time_mean=float(thinking_time.mean()),
        thinking_time_std=_usable_std(thinking_time),
        centipawn_loss_mean=float(centipawn_loss.mean()),
        centipawn_loss_std=_usable_std(centipawn_loss),
    )


def build_move_level_detector(input_dim: int) -> nn.Sequential:
    """Build the published single-hidden-layer binary classifier."""

    if input_dim <= 0:
        raise ValueError("move-level detection FFN input_dim must be positive.")
    return nn.Sequential(
        nn.Linear(input_dim, 256),
        nn.BatchNorm1d(256),
        nn.GELU(),
        nn.Linear(256, 1),
    )


def train_move_level_detector(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
    *,
    config: MoveLevelTrainConfig | None = None,
    device: torch.device | str | None = None,
) -> nn.Module:
    """Train one full-batch FFN and restore the best validation Macro-F1 state.

    The returned module carries ``training_history``,
    ``best_epoch``, and ``best_validation_macro_f1`` diagnostics.
    """

    settings = config or MoveLevelTrainConfig()
    _validate_train_config(settings)
    train_array = _float32_matrix(train_inputs, name="train_inputs")
    validation_array = _float32_matrix(
        validation_inputs, name="validation_inputs"
    )
    if len(train_array) == 0:
        raise ValueError("train_inputs must contain at least one row.")
    if len(validation_array) == 0:
        raise ValueError("validation_inputs must contain at least one row.")
    if train_array.shape[1] != validation_array.shape[1]:
        raise ValueError("Training and validation inputs must have equal width.")
    train_targets = _binary_labels(
        train_labels, expected_rows=len(train_array), name="train_labels"
    ).astype(np.float32, copy=False)
    validation_targets = _binary_labels(
        validation_labels,
        expected_rows=len(validation_array),
        name="validation_labels",
    )

    resolved_device = torch.device(
        device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    _set_all_seeds(settings.seed)
    model = build_move_level_detector(train_array.shape[1]).to(resolved_device)
    train_tensor = torch.from_numpy(train_array).to(resolved_device)
    train_target_tensor = torch.from_numpy(train_targets).to(resolved_device)
    validation_tensor = torch.from_numpy(validation_array).to(resolved_device)
    validation_target_tensor = torch.from_numpy(
        validation_targets.astype(np.float32, copy=False)
    ).to(resolved_device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=settings.learning_rate,
        weight_decay=settings.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=settings.scheduler_factor,
        patience=settings.scheduler_patience,
    )
    loss_function = nn.BCEWithLogitsLoss()
    best_macro_f1 = float("-inf")
    best_epoch = -1
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    training_history: list[dict[str, float | int]] = []

    for _epoch in range(settings.max_epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_tensor).reshape(-1)
        loss = loss_function(logits, train_target_tensor)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_tensor).reshape(-1)
            validation_loss = loss_function(
                validation_logits, validation_target_tensor
            )
            validation_probabilities = torch.sigmoid(validation_logits)
            validation_predictions = (
                validation_probabilities >= 0.5
            ).to(dtype=torch.int64).cpu().numpy()
        metrics = compute_detection_metrics(
            validation_targets, validation_predictions
        )
        macro_f1 = metrics["macro_f1"]
        learning_rate = float(optimizer.param_groups[0]["lr"])
        training_history.append(
            {
                "epoch": _epoch,
                "train_loss": float(loss.detach().item()),
                "validation_loss": float(validation_loss.item()),
                "validation_macro_f1": float(macro_f1),
                "learning_rate": learning_rate,
            }
        )
        scheduler.step(macro_f1)

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            best_epoch = _epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= settings.early_stopping_patience:
                break

    if best_state is None:
        raise RuntimeError("move-level detection training did not produce a model checkpoint.")
    model.load_state_dict(best_state)
    model.training_history = training_history
    model.best_epoch = best_epoch
    model.best_validation_macro_f1 = float(best_macro_f1)
    return model


def predict_move_level_detector(
    model: nn.Module,
    inputs: np.ndarray,
    *,
    device: torch.device | str | None = None,
) -> np.ndarray:
    """Predict binary labels with the fixed published probability cutoff of 0.5."""

    input_array = _float32_matrix(inputs, name="inputs")
    resolved_device = _model_device(model) if device is None else torch.device(device)
    model.to(resolved_device)
    model.eval()
    with torch.no_grad():
        input_tensor = torch.from_numpy(input_array).to(resolved_device)
        probabilities = torch.sigmoid(model(input_tensor).reshape(-1))
    return (probabilities >= 0.5).to(dtype=torch.int64).cpu().numpy()


def compute_detection_metrics(
    labels: np.ndarray, predictions: np.ndarray
) -> dict[str, float]:
    """Compute specificity, recall, and two-class Macro-F1."""

    true_labels = _binary_labels(labels, name="labels")
    predicted_labels = _binary_labels(
        predictions, expected_rows=len(true_labels), name="predictions"
    )
    true_negative = int(np.sum((true_labels == 0) & (predicted_labels == 0)))
    false_positive = int(np.sum((true_labels == 0) & (predicted_labels == 1)))
    true_positive = int(np.sum((true_labels == 1) & (predicted_labels == 1)))
    false_negative = int(np.sum((true_labels == 1) & (predicted_labels == 0)))

    specificity = _safe_ratio(true_negative, true_negative + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    negative_f1 = _safe_ratio(
        2 * true_negative, 2 * true_negative + false_positive + false_negative
    )
    positive_f1 = _safe_ratio(
        2 * true_positive, 2 * true_positive + false_positive + false_negative
    )
    return {
        "specificity": specificity,
        "recall": recall,
        "macro_f1": (negative_f1 + positive_f1) / 2.0,
    }


def build_observed_move_inputs(
    rows: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    preprocessor: MoveLevelPreprocessor,
    *,
    embedding_index_column: str = "embedding_row",
) -> MoveLevelCandidates:
    """Build observed human candidates using fitted Synth preprocessing."""

    if rows.index.has_duplicates:
        raise ValueError("move-level detection observed row IDs must be unique.")
    _validate_observed_columns(rows)
    human_embeddings = _validate_embedding_metadata(embeddings, "move_player")
    embedding_rows = _embedding_rows(
        rows,
        {"move_player": human_embeddings},
        embedding_index_col=embedding_index_column,
        allow_positional=False,
        require_monotonic=False,
    )

    moves = rows["move_player"].to_numpy(copy=True)
    features = _base_features(rows, preprocessor)
    features[:, 3] = _centipawn_loss_feature(
        _finite_column(rows, "centipawn_loss"), preprocessor
    )
    features[:, 4] = _finite_column(rows, "normalized_centipawn_loss")
    features[:, 6:9] = _top_move_matches(rows, moves)
    selected_embeddings = np.empty(
        (len(rows), _EMBEDDING_DIMENSION), dtype=np.float32
    )
    for row_position, embedding_row in enumerate(embedding_rows):
        selected_row = human_embeddings[embedding_row]
        minimum = float(np.min(selected_row))
        maximum = float(np.max(selected_row))
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(
                f"Selected embedding row {embedding_row} contains non-finite values."
            )
        if minimum < -_FLOAT32_MAX or maximum > _FLOAT32_MAX:
            raise ValueError(
                f"Selected embedding row {embedding_row} contains values outside "
                "the float32 range."
            )
        selected_embeddings[row_position] = selected_row

    return MoveLevelCandidates(
        row_ids=rows.index.to_numpy(copy=True),
        moves=moves,
        evaluations=_finite_column(rows, "eval_after").astype(
            np.float32, copy=False
        ),
        features=features,
        embeddings=selected_embeddings,
    )


def build_move_level_pairs(
    rows: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    *,
    embedding_index_col: str = "embedding_row",
    allow_positional_embeddings: bool = False,
    preprocessor: MoveLevelPreprocessor | None = None,
) -> MoveLevelPairs:
    """Build deterministic fair/assistance candidates in the input row order."""

    if rows.index.has_duplicates:
        raise ValueError("move-level detection position row IDs must be unique.")
    _validate_input_columns(rows)
    embedding_arrays = _validate_embeddings(embeddings)
    embedding_rows = _embedding_rows(
        rows,
        embedding_arrays,
        embedding_index_col=embedding_index_col,
        allow_positional=allow_positional_embeddings,
    )

    n_rows = len(rows)
    rng = np.random.default_rng(42)
    assistance_sources = rng.choice(CLASSICAL_ASSISTANCE_SOURCES, size=n_rows)

    fair_moves = rows["move_player"].to_numpy(copy=True)
    fair_evaluations = _finite_column(rows, "eval_after").astype(
        np.float32, copy=False
    )
    fair_features = _base_features(rows, preprocessor)
    fair_features[:, 3] = _centipawn_loss_feature(
        _finite_column(rows, "centipawn_loss"), preprocessor
    )
    fair_features[:, 4] = _finite_column(rows, "normalized_centipawn_loss")
    fair_features[:, 6:9] = _top_move_matches(rows, fair_moves)

    assistance_moves = np.empty(n_rows, dtype=object)
    assistance_evaluations = np.empty(n_rows, dtype=np.float32)
    assistance_evaluations_for_features = np.empty(n_rows, dtype=np.float64)
    fair_embeddings = np.empty(
        (n_rows, _EMBEDDING_DIMENSION), dtype=np.float32
    )
    assistance_embeddings = np.empty(
        (n_rows, _EMBEDDING_DIMENSION), dtype=np.float32
    )
    for row_position, source in enumerate(assistance_sources):
        move_key = f"move_{source}"
        evaluation_key = f"eval_{source}"
        candidate_evaluation = _finite_float32_scalar(
            rows[evaluation_key].iloc[row_position],
            label=f"Selected assistance evaluation {evaluation_key!r}",
        )
        assistance_moves[row_position] = rows[move_key].iloc[row_position]
        assistance_evaluations[row_position] = candidate_evaluation
        assistance_evaluations_for_features[row_position] = candidate_evaluation
        fair_embeddings[row_position] = embedding_arrays["move_player"][
            embedding_rows[row_position]
        ]
        assistance_embeddings[row_position] = embedding_arrays[move_key][
            embedding_rows[row_position]
        ]

    eval_before = _finite_column(rows, "eval_before")
    assistance_loss = np.maximum(
        0.0, eval_before - assistance_evaluations_for_features
    )
    assistance_normalized_loss = np.maximum(
        0.0,
        _sigmoid(eval_before) - _sigmoid(assistance_evaluations_for_features),
    )
    assistance_features = _base_features(rows, preprocessor)
    assistance_features[:, 3] = _centipawn_loss_feature(
        assistance_loss, preprocessor
    )
    assistance_features[:, 4] = assistance_normalized_loss
    assistance_features[:, 6:9] = _top_move_matches(rows, assistance_moves)

    row_ids = rows.index.to_numpy(copy=True)
    fair = MoveLevelCandidates(
        row_ids=row_ids.copy(),
        moves=fair_moves,
        evaluations=fair_evaluations,
        features=fair_features,
        embeddings=fair_embeddings,
    )
    assistance = MoveLevelCandidates(
        row_ids=row_ids.copy(),
        moves=assistance_moves,
        evaluations=assistance_evaluations,
        features=assistance_features,
        embeddings=assistance_embeddings,
    )
    return MoveLevelPairs(
        fair=fair,
        assistance=assistance,
        assistance_sources=assistance_sources,
    )


def _validate_input_columns(rows: pd.DataFrame) -> None:
    required = [
        "move_player",
        "eval_before",
        "eval_after",
        "centipawn_loss",
        "normalized_centipawn_loss",
        "player_elo",
        "opponent_elo",
        "move_thinking_time",
        "allie_win_prob_2500",
    ]
    for source in CLASSICAL_ASSISTANCE_SOURCES:
        required.extend((f"move_{source}", f"eval_{source}"))
    _require_columns(rows, required)


def _validate_observed_columns(rows: pd.DataFrame) -> None:
    _require_columns(
        rows,
        [
            "move_player",
            "eval_after",
            "centipawn_loss",
            "normalized_centipawn_loss",
            "player_elo",
            "opponent_elo",
            "move_thinking_time",
            "allie_win_prob_2500",
            "move_stockfish_1",
            "move_stockfish_9",
            "move_stockfish_15",
        ],
    )


def _require_columns(rows: pd.DataFrame, columns: tuple[str, ...] | list[str]) -> None:
    missing = [column for column in columns if column not in rows.columns]
    if missing:
        raise KeyError(f"Missing required move-level detection columns: {missing!r}.")


def _finite_column(rows: pd.DataFrame, column: str) -> np.ndarray:
    try:
        values = rows[column].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"move-level detection column {column!r} must be numeric.") from exc
    if not np.isfinite(values).all():
        raise ValueError(f"move-level detection column {column!r} contains non-finite values.")
    if np.any(np.abs(values) > _FLOAT32_MAX):
        raise ValueError(
            f"move-level detection column {column!r} contains values outside the float32 range."
        )
    return values


def _usable_std(values: np.ndarray) -> float:
    standard_deviation = float(values.std())
    return standard_deviation if standard_deviation >= 1e-12 else 1.0


def _finite_float32_scalar(value: object, *, label: str) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{label} must be a real number.") from exc
    if not np.isfinite(numeric_value):
        raise ValueError(f"{label} must be finite.")
    if abs(numeric_value) > _FLOAT32_MAX:
        raise ValueError(f"{label} is outside the float32 range.")
    return numeric_value


def _validate_train_config(config: MoveLevelTrainConfig) -> None:
    if config.seed < 0:
        raise ValueError("move-level detection training seed must be non-negative.")
    if config.learning_rate <= 0.0 or config.weight_decay < 0.0:
        raise ValueError("Learning rate must be positive and weight decay non-negative.")
    if config.max_epochs <= 0:
        raise ValueError("move-level detection max_epochs must be positive.")
    if not 0.0 < config.scheduler_factor < 1.0:
        raise ValueError("move-level detection scheduler_factor must lie within (0, 1).")
    if config.scheduler_patience < 0 or config.early_stopping_patience <= 0:
        raise ValueError("move-level detection patience values are invalid.")


def _float32_matrix(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional matrix.")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numbers.")
    if array.size:
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(f"{name} contains non-finite values.")
        if minimum < -_FLOAT32_MAX or maximum > _FLOAT32_MAX:
            raise ValueError(f"{name} contains values outside the float32 range.")
    return np.ascontiguousarray(array, dtype=np.float32)


def _binary_labels(
    values: np.ndarray,
    *,
    expected_rows: int | None = None,
    name: str,
) -> np.ndarray:
    labels = np.asarray(values)
    if labels.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional.")
    if expected_rows is not None and len(labels) != expected_rows:
        raise ValueError(f"{name} must contain exactly {expected_rows} labels.")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError(f"{name} must contain only binary labels 0 and 1.")
    return labels.astype(np.int64, copy=False)


def _model_device(model: nn.Module) -> torch.device:
    parameter = next(model.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(model.buffers(), None)
    return buffer.device if buffer is not None else torch.device("cpu")


def _safe_ratio(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _validate_embeddings(
    embeddings: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    keys = ["move_player", *[f"move_{source}" for source in CLASSICAL_ASSISTANCE_SOURCES]]
    return {key: _validate_embedding_array(embeddings, key) for key in keys}


def _validate_embedding_array(
    embeddings: Mapping[str, np.ndarray], key: str
) -> np.ndarray:
    array = _validate_embedding_metadata(embeddings, key)
    if array.size:
        minimum = float(np.min(array))
        maximum = float(np.max(array))
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(f"Embedding array {key!r} contains non-finite values.")
        if minimum < -_FLOAT32_MAX or maximum > _FLOAT32_MAX:
            raise ValueError(
                f"Embedding array {key!r} contains values outside the float32 range."
            )
    return array


def _validate_embedding_metadata(
    embeddings: Mapping[str, np.ndarray], key: str
) -> np.ndarray:
    if key not in embeddings:
        raise KeyError(f"Missing required embedding array {key!r}.")
    array = np.asarray(embeddings[key])
    if array.ndim != 2 or array.shape[1] != _EMBEDDING_DIMENSION:
        raise ValueError(
            f"Embedding array {key!r} must have shape (N, {_EMBEDDING_DIMENSION}); "
            f"received {array.shape!r}."
        )
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"Embedding array {key!r} must contain real numbers.")
    return array


def _embedding_rows(
    rows: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    *,
    embedding_index_col: str,
    allow_positional: bool,
    require_monotonic: bool = True,
) -> np.ndarray:
    if embedding_index_col not in rows.columns:
        if not allow_positional:
            raise KeyError(
                f"Missing embedding index column {embedding_index_col!r}; "
                "set allow_positional_embeddings=True only for verified positional data."
            )
        wrong_lengths = {
            key: len(array) for key, array in embeddings.items() if len(array) != len(rows)
        }
        if wrong_lengths:
            raise ValueError(
                "Positional embedding arrays must have exactly one row per input row; "
                f"received lengths {wrong_lengths!r}."
            )
        return np.arange(len(rows), dtype=np.int64)

    raw_indices = rows[embedding_index_col].to_numpy()
    if not all(
        isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_))
        for value in raw_indices
    ):
        raise TypeError(f"Embedding indices in {embedding_index_col!r} must be integers.")
    indices = raw_indices.astype(np.int64, copy=False)
    if np.any(indices < 0):
        raise ValueError("Embedding indices must be non-negative.")
    if len(np.unique(indices)) != len(indices):
        raise ValueError("Embedding indices must be unique.")
    if require_monotonic and len(indices) > 1 and np.any(np.diff(indices) <= 0):
        raise ValueError("Embedding indices must preserve canonical row order.")
    maximum_index = int(indices.max()) if len(indices) else -1
    for key, array in embeddings.items():
        if maximum_index >= len(array):
            raise ValueError(
                f"Embedding index {maximum_index} is out of range for {key!r} "
                f"with {len(array)} rows."
            )
    return indices


def _base_features(
    rows: pd.DataFrame, preprocessor: MoveLevelPreprocessor | None
) -> np.ndarray:
    features = np.zeros((len(rows), len(_FEATURE_NAMES)), dtype=np.float32)
    player_elo = _finite_column(rows, "player_elo")
    opponent_elo = _finite_column(rows, "opponent_elo")
    thinking_time = _finite_column(rows, "move_thinking_time")
    if preprocessor is not None:
        player_elo = (
            player_elo - preprocessor.player_elo_mean
        ) / preprocessor.player_elo_std
        opponent_elo = (
            opponent_elo - preprocessor.opponent_elo_mean
        ) / preprocessor.opponent_elo_std
        thinking_time = np.log1p(np.maximum(thinking_time, 0.0))
        thinking_time = (
            thinking_time - preprocessor.thinking_time_mean
        ) / preprocessor.thinking_time_std
    features[:, 0] = player_elo
    features[:, 1] = opponent_elo
    features[:, 2] = thinking_time
    allie_win_probability = _finite_column(rows, "allie_win_prob_2500")
    if np.any((allie_win_probability < 0.0) | (allie_win_probability > 1.0)):
        raise ValueError("Allie win probabilities must lie within [0, 1].")
    features[:, 5] = allie_win_probability
    return features


def _centipawn_loss_feature(
    values: np.ndarray, preprocessor: MoveLevelPreprocessor | None
) -> np.ndarray:
    if preprocessor is None:
        return values
    return (
        values - preprocessor.centipawn_loss_mean
    ) / preprocessor.centipawn_loss_std


def _top_move_matches(rows: pd.DataFrame, candidate_moves: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            candidate_moves == rows[f"move_stockfish_{depth}"].to_numpy()
            for depth in (1, 9, 15)
        ]
    ).astype(np.float32, copy=False)


def _sigmoid(centipawns: np.ndarray) -> np.ndarray:
    values = np.asarray(centipawns, dtype=np.float64)
    scaled = np.clip(_WIN_PROBABILITY_SCALE * values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-scaled))
