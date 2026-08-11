"""Behavioral contract for the reproducible move-level move-level detection pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import fields
import inspect

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import experiments.move_level.detection as detection
from experiments.move_level.detection import (
    CLASSICAL_ASSISTANCE_SOURCES,
    build_move_level_pairs,
    fit_move_level_preprocessor,
)


EXPECTED_SOURCES = (
    "stockfish_1",
    "stockfish_5",
    "stockfish_9",
    "stockfish_11",
    "stockfish_15",
    "lc0_1",
    "lc0_10",
    "lc0_100",
)

EXPECTED_FEATURE_NAMES = (
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


def _rows(n_rows: int, embedding_rows: list[object] | None = None) -> pd.DataFrame:
    row_number = np.arange(n_rows)
    data: dict[str, object] = {
        "move_player": [f"human-{i}" for i in row_number],
        "eval_before": np.full(n_rows, 100.0),
        # Deliberately inconsistent with the published loss columns. The fair
        # branch must consume the published values rather than recompute them.
        "eval_after": 900.0 + row_number,
        "centipawn_loss": 7.0 + row_number,
        "normalized_centipawn_loss": 0.07 + row_number / 100.0,
        "player_elo": 1000.0 + 100.0 * row_number,
        "opponent_elo": 1200.0 + 100.0 * row_number,
        "move_thinking_time": 5.0 + row_number,
        "allie_win_prob_2500": 0.20 + row_number / 100.0,
    }
    if embedding_rows is not None:
        data["embedding_row"] = embedding_rows

    for source_position, source in enumerate(EXPECTED_SOURCES):
        data[f"move_{source}"] = [f"{source}-move-{i}" for i in row_number]
        data[f"eval_{source}"] = np.full(
            n_rows, 100.0 - 10.0 * (source_position + 1)
        )

    rows = pd.DataFrame(data, index=[f"position-{i}" for i in row_number])
    if n_rows:
        rows.loc[rows.index[0], "move_player"] = rows.loc[
            rows.index[0], "move_stockfish_1"
        ]
    if n_rows > 1:
        rows.loc[rows.index[1], "move_player"] = rows.loc[
            rows.index[1], "move_stockfish_9"
        ]
    if n_rows > 2:
        rows.loc[rows.index[2], "move_player"] = rows.loc[
            rows.index[2], "move_stockfish_15"
        ]
    return rows


def _embeddings(n_rows: int) -> dict[str, np.ndarray]:
    embeddings = {
        "move_player": np.repeat(
            (-1000.0 + np.arange(n_rows))[:, None], 1024, axis=1
        ).astype(np.float32)
    }
    for source_position, source in enumerate(EXPECTED_SOURCES):
        row_signatures = 100.0 * (source_position + 1) + np.arange(n_rows)
        embeddings[f"move_{source}"] = np.repeat(
            row_signatures[:, None], 1024, axis=1
        ).astype(np.float32)
    return embeddings


class _MaterializationTrackingArray(np.ndarray):
    """Record dtype casts that materialize an entire source array."""

    full_shape: tuple[int, ...]
    full_materializations: list[np.dtype]

    def __new__(cls, values: np.ndarray):
        array = np.asarray(values).view(cls)
        array.full_shape = array.shape
        array.full_materializations = []
        return array

    def __array_finalize__(self, source: np.ndarray | None) -> None:
        if source is None:
            return
        self.full_shape = getattr(source, "full_shape", self.shape)
        self.full_materializations = getattr(source, "full_materializations", [])

    def astype(self, dtype: object, *args: object, **kwargs: object) -> np.ndarray:
        result = super().astype(dtype, *args, **kwargs)
        if self.shape == self.full_shape and not np.shares_memory(self, result):
            self.full_materializations.append(np.dtype(dtype))
        return result


def _build(
    rows: pd.DataFrame,
    embeddings: Mapping[str, np.ndarray],
    **kwargs: object,
):
    return build_move_level_pairs(
        rows,
        embeddings,
        embedding_index_col="embedding_row",
        **kwargs,
    )


def test_classical_assistance_sources_use_the_published_order() -> None:
    assert CLASSICAL_ASSISTANCE_SOURCES == EXPECTED_SOURCES


def test_fair_candidate_uses_published_values_and_exact_feature_order() -> None:
    rows = _rows(2, [0, 1])
    pairs = _build(rows, _embeddings(2))

    assert pairs.feature_names == EXPECTED_FEATURE_NAMES
    assert pairs.fair.moves.tolist() == ["stockfish_1-move-0", "stockfish_9-move-1"]
    np.testing.assert_array_equal(pairs.fair.evaluations, [900.0, 901.0])
    np.testing.assert_allclose(
        pairs.fair.features,
        [
            [1000.0, 1200.0, 5.0, 7.0, 0.07, 0.20, 1.0, 0.0, 0.0],
            [1100.0, 1300.0, 6.0, 8.0, 0.08, 0.21, 0.0, 1.0, 0.0],
        ],
        rtol=1e-6,
    )
    np.testing.assert_array_equal(pairs.fair.embeddings, _embeddings(2)["move_player"])


def test_all_mixture_uses_one_seeded_source_for_every_candidate_component() -> None:
    rows = _rows(8, list(range(8)))
    train_rows = _rows(2, [0, 1])
    train_rows["centipawn_loss"] = [10.0, 30.0]
    preprocessor = fit_move_level_preprocessor(train_rows)
    # The first selected candidate is intentionally better than eval_before,
    # making both assistance loss features exercise their zero clipping.
    rows.loc["position-0", "eval_stockfish_1"] = 150.0
    pairs = _build(rows, _embeddings(8), preprocessor=preprocessor)

    expected_sources = [
        "stockfish_1",
        "lc0_10",
        "lc0_1",
        "stockfish_11",
        "stockfish_11",
        "lc0_10",
        "stockfish_1",
        "lc0_1",
    ]
    assert pairs.assistance_sources.tolist() == expected_sources
    assert pairs.assistance.moves.tolist() == [
        "stockfish_1-move-0",
        "lc0_10-move-1",
        "lc0_1-move-2",
        "stockfish_11-move-3",
        "stockfish_11-move-4",
        "lc0_10-move-5",
        "stockfish_1-move-6",
        "lc0_1-move-7",
    ]
    np.testing.assert_array_equal(
        pairs.assistance.evaluations,
        [150.0, 30.0, 40.0, 60.0, 60.0, 30.0, 90.0, 40.0],
    )
    np.testing.assert_array_equal(
        pairs.assistance.embeddings[:, 0],
        [100.0, 701.0, 602.0, 403.0, 404.0, 705.0, 106.0, 607.0],
    )
    np.testing.assert_allclose(
        pairs.assistance.features[:, 3],
        [-2.0, 5.0, 4.0, 2.0, 2.0, 5.0, -1.0, 4.0],
    )
    np.testing.assert_allclose(
        pairs.assistance.features[:, 4],
        [
            0.0,
            0.06343834329910869,
            0.054271513901072765,
            0.03601824877915916,
            0.03601824877915916,
            0.06343834329910869,
            0.008929031551888178,
            0.054271513901072765,
        ],
        rtol=1e-6,
        atol=1e-8,
    )
    np.testing.assert_array_equal(
        pairs.assistance.features[:, 6:9],
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
    )


def test_preprocessor_fits_train_observed_population_stats_after_time_transform() -> None:
    train_rows = pd.DataFrame(
        {
            "player_elo": [1000.0, 2000.0],
            "opponent_elo": [1100.0, 2100.0],
            "move_thinking_time": [-50.0, np.expm1(2.0)],
            "centipawn_loss": [10.0, 30.0],
        }
    )

    preprocessor = fit_move_level_preprocessor(train_rows)

    assert {field.name for field in fields(preprocessor)} == {
        "player_elo_mean",
        "player_elo_std",
        "opponent_elo_mean",
        "opponent_elo_std",
        "thinking_time_mean",
        "thinking_time_std",
        "centipawn_loss_mean",
        "centipawn_loss_std",
    }
    assert vars(preprocessor) == pytest.approx(
        {
            "player_elo_mean": 1500.0,
            "player_elo_std": 500.0,
            "opponent_elo_mean": 1600.0,
            "opponent_elo_std": 500.0,
            "thinking_time_mean": 1.0,
            "thinking_time_std": 1.0,
            "centipawn_loss_mean": 20.0,
            "centipawn_loss_std": 10.0,
        }
    )


@pytest.mark.parametrize(
    (
        "player_elo",
        "opponent_elo",
        "thinking_time",
        "human_cpl",
        "normalized_cpl",
        "allie_probability",
        "assistance_evaluation",
        "expected_fair",
        "expected_assistance",
    ),
    [
        (
            100_000.0,
            100_500.0,
            -99.0,
            120.0,
            0.31,
            0.71,
            60.0,
            [197.0, 197.8, -1.0, 10.0, 0.31, 0.71, 1.0, 0.0, 0.0],
            [197.0, 197.8, -1.0, 2.0, 0.036018248779159157, 0.71, 1.0, 0.0, 0.0],
        ),
        (
            80_000.0,
            80_500.0,
            np.expm1(5.0),
            220.0,
            0.41,
            0.81,
            80.0,
            [157.0, 157.8, 4.0, 20.0, 0.41, 0.81, 1.0, 0.0, 0.0],
            [157.0, 157.8, 4.0, 0.0, 0.017912203289730644, 0.81, 1.0, 0.0, 0.0],
        ),
    ],
    ids=["validation-sentinel", "test-sentinel"],
)
def test_pair_preprocessing_reuses_train_stats_without_held_out_leakage(
    player_elo: float,
    opponent_elo: float,
    thinking_time: float,
    human_cpl: float,
    normalized_cpl: float,
    allie_probability: float,
    assistance_evaluation: float,
    expected_fair: list[float],
    expected_assistance: list[float],
) -> None:
    train_rows = _rows(2, [0, 1])
    train_rows["player_elo"] = [1000.0, 2000.0]
    train_rows["opponent_elo"] = [1100.0, 2100.0]
    train_rows["move_thinking_time"] = [-50.0, np.expm1(2.0)]
    train_rows["centipawn_loss"] = [10.0, 30.0]
    preprocessor = fit_move_level_preprocessor(train_rows)
    held_out = _rows(1, [0])
    held_out.loc["position-0", "player_elo"] = player_elo
    held_out.loc["position-0", "opponent_elo"] = opponent_elo
    held_out.loc["position-0", "move_thinking_time"] = thinking_time
    held_out.loc["position-0", "centipawn_loss"] = human_cpl
    held_out.loc["position-0", "normalized_centipawn_loss"] = normalized_cpl
    held_out.loc["position-0", "allie_win_prob_2500"] = allie_probability
    held_out.loc["position-0", "eval_stockfish_1"] = assistance_evaluation

    pairs = _build(held_out, _embeddings(1), preprocessor=preprocessor)

    np.testing.assert_allclose(pairs.fair.features[0], expected_fair, rtol=1e-6)
    np.testing.assert_allclose(
        pairs.assistance.features[0], expected_assistance, rtol=1e-6
    )


def _observed_rows() -> pd.DataFrame:
    human_rows = pd.DataFrame(
        {
            "position_key": ["game-c", "game-a", "game-b"],
            "move_player": ["e2e4", "d2d4", "g1f3"],
            "eval_after": [10.0, 20.0, 30.0],
            "centipawn_loss": [2.0, 3.0, 4.0],
            "normalized_centipawn_loss": [0.02, 0.03, 0.04],
            "player_elo": [1050.0, 1150.0, 950.0],
            "opponent_elo": [1250.0, 1350.0, 1150.0],
            "move_thinking_time": [11.0, 12.0, 13.0],
            "allie_win_prob_2500": [0.4, 0.5, 0.6],
            "move_stockfish_1": ["e2e4", "a2a3", "a2a3"],
            "move_stockfish_9": ["a2a3", "d2d4", "a2a3"],
            "move_stockfish_15": ["a2a3", "a2a3", "g1f3"],
        },
        index=["hf-row-c", "hf-row-a", "hf-row-b"],
    )
    embedding_index = pd.DataFrame(
        {
            "position_key": ["game-a", "game-b", "game-c"],
            "embedding_row": [0, 1, 2],
        }
    )
    return (
        human_rows.rename_axis("hf_row_id")
        .reset_index()
        .merge(
            embedding_index,
            on="position_key",
            how="left",
            sort=False,
            validate="one_to_one",
        )
        .set_index("hf_row_id")
    )


def _observed_embeddings(n_rows: int = 3) -> dict[str, np.ndarray]:
    row_signatures = 100.0 * np.arange(1, n_rows + 1)
    return {
        "move_player": np.repeat(row_signatures[:, None], 1024, axis=1)
    }


def test_observed_candidates_use_human_fields_and_preserve_joined_hf_order() -> None:
    build_observed_move_inputs = _trainer_api("build_observed_move_inputs")
    preprocessor = fit_move_level_preprocessor(_rows(2, [0, 1]))
    original_preprocessor = detection.MoveLevelPreprocessor(**vars(preprocessor))
    rows = _observed_rows()

    observed = build_observed_move_inputs(
        rows,
        _observed_embeddings(),
        preprocessor,
    )

    assert preprocessor == original_preprocessor
    assert observed.row_ids.tolist() == ["hf-row-c", "hf-row-a", "hf-row-b"]
    assert observed.moves.tolist() == ["e2e4", "d2d4", "g1f3"]
    np.testing.assert_array_equal(observed.evaluations, [10.0, 20.0, 30.0])
    np.testing.assert_array_equal(observed.embeddings[:, 0], [300.0, 100.0, 200.0])
    np.testing.assert_allclose(
        observed.features,
        [
            [0.0, 0.0, 7.993112211203847, -11.0, 0.02, 0.4, 1.0, 0.0, 0.0],
            [2.0, 2.0, 9.031611785298916, -9.0, 0.03, 0.5, 0.0, 1.0, 0.0],
            [-2.0, -2.0, 9.993112211203846, -7.0, 0.04, 0.6, 0.0, 0.0, 1.0],
        ],
        rtol=1e-6,
    )
    assert observed.embeddings.dtype == np.dtype(np.float32)
    assert observed.features.dtype == np.dtype(np.float32)


def test_observed_embedding_index_column_can_be_named_explicitly() -> None:
    build_observed_move_inputs = _trainer_api("build_observed_move_inputs")
    rows = _observed_rows().rename(columns={"embedding_row": "joined_embedding_row"})

    observed = build_observed_move_inputs(
        rows,
        _observed_embeddings(),
        fit_move_level_preprocessor(_rows(2, [0, 1])),
        embedding_index_column="joined_embedding_row",
    )

    np.testing.assert_array_equal(observed.embeddings[:, 0], [300.0, 100.0, 200.0])


@pytest.mark.parametrize(
    ("embedding_rows", "error_type"),
    [
        (None, (KeyError, ValueError)),
        ([2, 2, 1], ValueError),
        ([2.0, 0.0, 1.0], (TypeError, ValueError)),
        ([2, -1, 1], ValueError),
        ([3, 0, 1], ValueError),
        ([2, 0, pd.NA], (TypeError, ValueError)),
    ],
    ids=[
        "missing",
        "duplicate",
        "non-integer",
        "negative",
        "out-of-range",
        "incomplete-one-to-one-join",
    ],
)
def test_observed_embedding_indices_are_validated_without_synth_monotonicity(
    embedding_rows: list[object] | None,
    error_type: type[Exception] | tuple[type[Exception], ...],
) -> None:
    build_observed_move_inputs = _trainer_api("build_observed_move_inputs")
    rows = _observed_rows()
    if embedding_rows is None:
        rows = rows.drop(columns="embedding_row")
    else:
        rows["embedding_row"] = embedding_rows

    with pytest.raises(error_type):
        build_observed_move_inputs(
            rows,
            _observed_embeddings(),
            fit_move_level_preprocessor(_rows(2, [0, 1])),
        )


def test_observed_human_embeddings_must_have_1024_dimensions() -> None:
    build_observed_move_inputs = _trainer_api("build_observed_move_inputs")

    with pytest.raises(ValueError):
        build_observed_move_inputs(
            _observed_rows(),
            {"move_player": np.zeros((3, 16), dtype=np.float32)},
            fit_move_level_preprocessor(_rows(2, [0, 1])),
        )


def test_observed_ignores_unreferenced_nonfinite_embeddings_but_synth_remains_strict() -> None:
    observed_embeddings = _observed_embeddings(5)
    observed_embeddings["move_player"][3, 0] = np.nan
    observed_embeddings["move_player"][4, 0] = np.inf

    observed = detection.build_observed_move_inputs(
        _observed_rows(),
        observed_embeddings,
        fit_move_level_preprocessor(_rows(2, [0, 1])),
    )

    np.testing.assert_array_equal(observed.embeddings[:, 0], [300.0, 100.0, 200.0])
    assert np.isfinite(observed.embeddings).all()

    synth_embeddings = _embeddings(3)
    synth_embeddings["move_lc0_100"][2, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _build(_rows(1, [0]), synth_embeddings)


@pytest.mark.parametrize("nonfinite_value", [np.nan, np.inf], ids=["nan", "infinity"])
def test_observed_rejects_nonfinite_selected_embedding_rows(
    nonfinite_value: float,
) -> None:
    embeddings = _observed_embeddings()
    # Joined embedding row 0 is selected by the second observed/HF row.
    embeddings["move_player"][0, 0] = nonfinite_value

    with pytest.raises(ValueError, match="non-finite"):
        detection.build_observed_move_inputs(
            _observed_rows(),
            embeddings,
            fit_move_level_preprocessor(_rows(2, [0, 1])),
        )


def test_embedding_mapping_is_required_unless_positional_mapping_is_explicit() -> None:
    rows = _rows(2)

    with pytest.raises((KeyError, ValueError)):
        _build(rows, _embeddings(2))

    pairs = _build(rows, _embeddings(2), allow_positional_embeddings=True)
    np.testing.assert_array_equal(pairs.fair.embeddings[:, 0], [-1000.0, -999.0])


@pytest.mark.parametrize(
    ("embedding_rows", "error_type"),
    [
        ([0.0, 1.0], (TypeError, ValueError)),
        ([0, 0], ValueError),
        ([-1, 0], ValueError),
        ([0, 3], ValueError),
        ([1, 0], ValueError),
    ],
    ids=["non-integer", "duplicate", "negative", "out-of-range", "non-monotonic"],
)
def test_invalid_embedding_row_indices_are_rejected(
    embedding_rows: list[object],
    error_type: type[Exception] | tuple[type[Exception], ...],
) -> None:
    rows = _rows(2, embedding_rows)

    with pytest.raises(error_type):
        _build(rows, _embeddings(3))


def test_positional_embedding_mapping_rejects_wrong_row_count() -> None:
    rows = _rows(2)

    with pytest.raises(ValueError):
        _build(rows, _embeddings(3), allow_positional_embeddings=True)


def test_embeddings_must_have_1024_dimensions() -> None:
    rows = _rows(2, [0, 1])
    embeddings = _embeddings(2)
    embeddings["move_lc0_100"] = np.zeros((2, 16), dtype=np.float32)

    with pytest.raises(ValueError):
        _build(rows, embeddings)


def test_complex_embedding_arrays_are_rejected_even_when_not_selected() -> None:
    rows = _rows(2, [0, 1])
    embeddings = _embeddings(2)
    embeddings["move_lc0_100"] = embeddings["move_lc0_100"].astype(np.complex64)
    embeddings["move_lc0_100"][0, 0] += 1j

    with pytest.raises((TypeError, ValueError)):
        _build(rows, embeddings)


def test_finite_float64_embeddings_outside_float32_range_are_rejected() -> None:
    rows = _rows(2, [0, 1])
    embeddings = _embeddings(2)
    embeddings["move_player"] = embeddings["move_player"].astype(np.float64)
    embeddings["move_player"][0, 0] = 1.0e100

    assert np.isfinite(embeddings["move_player"][0, 0])
    with pytest.raises(ValueError):
        _build(rows, embeddings)


def test_valid_canonical_embedding_mapping_indexes_every_candidate_array() -> None:
    rows = _rows(2, [0, 2])
    pairs = _build(rows, _embeddings(3))

    np.testing.assert_array_equal(pairs.fair.embeddings[:, 0], [-1000.0, -998.0])
    assert pairs.assistance_sources.tolist() == ["stockfish_1", "lc0_10"]
    np.testing.assert_array_equal(pairs.assistance.embeddings[:, 0], [100.0, 702.0])


def test_validation_does_not_materialize_full_source_embedding_arrays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(2, [1, 3])
    embeddings = _embeddings(4)
    tracked_arrays: list[_MaterializationTrackingArray] = []
    for position, (key, values) in enumerate(embeddings.items()):
        dtype = np.float16 if position % 2 == 0 else np.float32
        tracked = _MaterializationTrackingArray(values.astype(dtype))
        embeddings[key] = tracked
        tracked_arrays.append(tracked)

    original_asarray = np.asarray

    def preserve_tracking_subclass(value: object, *args: object, **kwargs: object):
        if isinstance(value, _MaterializationTrackingArray) and not args and not kwargs:
            return value
        return original_asarray(value, *args, **kwargs)

    monkeypatch.setattr(detection.np, "asarray", preserve_tracking_subclass)
    pairs = _build(rows, embeddings)

    assert all(not array.full_materializations for array in tracked_arrays)
    assert pairs.fair.embeddings.dtype == np.dtype(np.float32)
    assert pairs.assistance.embeddings.dtype == np.dtype(np.float32)


@pytest.mark.parametrize(
    "candidate_evaluation",
    [np.nan, np.inf, -np.inf, 1.0e100, -1.0e100],
    ids=["nan", "positive-inf", "negative-inf", "too-large", "too-small"],
)
def test_selected_assistance_evaluation_must_fit_finite_float32(
    candidate_evaluation: float,
) -> None:
    rows = _rows(1, [0])
    # default_rng(42) selects stockfish_1 for the first canonical row.
    rows.loc["position-0", "eval_stockfish_1"] = candidate_evaluation

    with pytest.raises(ValueError):
        _build(rows, _embeddings(1))


def test_duplicate_position_row_ids_are_rejected() -> None:
    rows = _rows(2, [0, 1])
    rows.index = ["same-position", "same-position"]

    with pytest.raises(ValueError):
        _build(rows, _embeddings(2))


@pytest.mark.parametrize("probability", [-0.01, 1.01], ids=["below-zero", "above-one"])
def test_allie_win_probability_must_be_bounded(probability: float) -> None:
    rows = _rows(2, [0, 1])
    rows.loc["position-0", "allie_win_prob_2500"] = probability

    with pytest.raises(ValueError):
        _build(rows, _embeddings(2))


def test_feature_matrices_remain_float32_when_joined_with_embeddings() -> None:
    pairs = _build(_rows(2, [0, 1]), _embeddings(2))

    assert pairs.fair.features.dtype == np.dtype(np.float32)
    assert pairs.assistance.features.dtype == np.dtype(np.float32)
    fair_inputs = np.concatenate([pairs.fair.embeddings, pairs.fair.features], axis=1)
    assistance_inputs = np.concatenate(
        [pairs.assistance.embeddings, pairs.assistance.features], axis=1
    )
    assert fair_inputs.dtype == np.dtype(np.float32)
    assert assistance_inputs.dtype == np.dtype(np.float32)


def test_paired_candidates_preserve_position_identity_and_input_order() -> None:
    rows = _rows(3, [0, 1, 2])
    # Canonical embedding rows stay aligned while position identities are
    # deliberately not lexicographically ordered.
    rows.index = ["position-z", "position-a", "position-m"]
    pairs = _build(rows, _embeddings(3))

    expected_row_ids = ["position-z", "position-a", "position-m"]
    assert pairs.fair.row_ids.tolist() == expected_row_ids
    assert pairs.assistance.row_ids.tolist() == expected_row_ids


def _trainer_api(name: str):
    assert hasattr(detection, name), f"Missing public move-level detection API: {name}"
    return getattr(detection, name)


def test_move_level_train_config_has_published_defaults_without_batch_or_threshold_tuning() -> None:
    config_type = _trainer_api("MoveLevelTrainConfig")
    config = config_type()

    assert config.seed == 42
    assert config.learning_rate == pytest.approx(3e-4)
    assert config.weight_decay == pytest.approx(3e-4)
    assert config.max_epochs == 2000
    assert config.scheduler_factor == pytest.approx(0.1)
    assert config.scheduler_patience == 10
    assert config.early_stopping_patience == 40

    config_fields = {field.name for field in fields(config)}
    assert config_fields.isdisjoint({"batch_size", "threshold", "threshold_grid"})
    train_parameters = inspect.signature(_trainer_api("train_move_level_detector")).parameters
    assert "batch_size" not in train_parameters
    assert "threshold" not in train_parameters
    assert "threshold_grid" not in train_parameters


def test_move_level_ffn_uses_one_256_unit_batch_normalized_gelu_layer_and_one_logit() -> None:
    build_move_level_detector = _trainer_api("build_move_level_detector")

    model = build_move_level_detector(input_dim=1033)

    assert isinstance(model, nn.Sequential)
    assert [type(layer) for layer in model] == [
        nn.Linear,
        nn.BatchNorm1d,
        nn.GELU,
        nn.Linear,
    ]
    assert model[0].in_features == 1033
    assert model[0].out_features == 256
    assert model[1].num_features == 256
    assert model[3].in_features == 256
    assert model[3].out_features == 1
    assert model(torch.zeros((3, 1033), dtype=torch.float32)).shape == (3, 1)


class _SingleLogitModel(nn.Module):
    """Tiny trainable model used to expose each epoch's checkpoint state."""

    def __init__(self) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.logit.expand(len(inputs), 1)


def test_training_is_full_batch_adamw_and_restores_best_validation_macro_f1(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_type = _trainer_api("MoveLevelTrainConfig")
    train_move_level_detector = _trainer_api("train_move_level_detector")
    model = _SingleLogitModel()
    epoch_states: list[float] = []
    optimizer_steps: list[None] = []
    scheduler_metrics: list[float] = []
    optimizer_kwargs: dict[str, object] = {}
    scheduler_kwargs: dict[str, object] = {}
    macro_f1_values = iter((0.9, 0.8, 0.7))

    real_adamw = torch.optim.AdamW

    def adamw_spy(parameters, **kwargs):
        optimizer_kwargs.update(kwargs)
        optimizer = real_adamw(parameters, **kwargs)
        real_step = optimizer.step

        def counted_step(*args, **step_kwargs):
            optimizer_steps.append(None)
            return real_step(*args, **step_kwargs)

        optimizer.step = counted_step
        return optimizer

    class SchedulerSpy:
        def step(self, metric: float) -> None:
            scheduler_metrics.append(float(metric))

    def scheduler_spy(_optimizer, **kwargs):
        scheduler_kwargs.update(kwargs)
        return SchedulerSpy()

    def controlled_metrics(_labels: np.ndarray, _predictions: np.ndarray):
        epoch_states.append(float(model.logit.detach()))
        macro_f1 = next(macro_f1_values)
        return {"specificity": 1.0, "recall": 1.0, "macro_f1": macro_f1}

    monkeypatch.setattr(detection, "build_move_level_detector", lambda _input_dim: model)
    monkeypatch.setattr(detection, "compute_detection_metrics", controlled_metrics)
    monkeypatch.setattr(torch.optim, "AdamW", adamw_spy)
    monkeypatch.setattr(torch.optim.lr_scheduler, "ReduceLROnPlateau", scheduler_spy)

    train_inputs = np.ones((5, 2), dtype=np.float32)
    train_labels = np.ones(5, dtype=np.float32)
    validation_inputs = np.ones((2, 2), dtype=np.float32)
    validation_labels = np.ones(2, dtype=np.int64)
    trained = train_move_level_detector(
        train_inputs,
        train_labels,
        validation_inputs,
        validation_labels,
        config=config_type(
            learning_rate=0.1,
            max_epochs=20,
            early_stopping_patience=2,
        ),
    )

    assert trained is model
    assert len(optimizer_steps) == 3  # one full-batch update per epoch
    assert optimizer_kwargs["lr"] == pytest.approx(0.1)
    assert optimizer_kwargs["weight_decay"] == pytest.approx(3e-4)
    assert scheduler_kwargs == {"mode": "max", "factor": 0.1, "patience": 10}
    assert scheduler_metrics == pytest.approx([0.9, 0.8, 0.7])
    assert float(trained.logit.detach()) == pytest.approx(epoch_states[0])

    diagnostic_attributes = {
        "training_history",
        "best_epoch",
        "best_validation_macro_f1",
    }
    missing_attributes = sorted(
        name for name in diagnostic_attributes if not hasattr(trained, name)
    )
    assert not missing_attributes, f"Missing trainer diagnostics: {missing_attributes}"

    history = trained.training_history
    assert len(history) == 3
    required_record_fields = {
        "epoch",
        "train_loss",
        "validation_loss",
        "validation_macro_f1",
        "learning_rate",
    }
    assert all(isinstance(record, Mapping) for record in history)
    assert all(required_record_fields <= record.keys() for record in history)
    assert [record["epoch"] for record in history] == [0, 1, 2]
    assert [record["validation_macro_f1"] for record in history] == pytest.approx(
        [0.9, 0.8, 0.7]
    )
    assert [record["learning_rate"] for record in history] == pytest.approx(
        [0.1, 0.1, 0.1]
    )
    assert all(
        np.isfinite(record[metric]) and record[metric] >= 0.0
        for record in history
        for metric in ("train_loss", "validation_loss")
    )
    assert trained.best_epoch == 0
    assert trained.best_validation_macro_f1 == pytest.approx(0.9)


@pytest.mark.parametrize(
    (
        "train_inputs",
        "train_labels",
        "validation_inputs",
        "validation_labels",
        "error_pattern",
    ),
    [
        (
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            np.ones((2, 2), dtype=np.float32),
            np.array([0, 1]),
            r"(?i)train",
        ),
        (
            np.ones((2, 2), dtype=np.float32),
            np.array([0, 1]),
            np.empty((0, 2), dtype=np.float32),
            np.empty(0, dtype=np.int64),
            r"(?i)validation",
        ),
        (
            np.ones((2, 2), dtype=np.float32),
            np.array([0]),
            np.ones((2, 2), dtype=np.float32),
            np.array([0, 1]),
            "train_labels",
        ),
        (
            np.ones((2, 2), dtype=np.float32),
            np.array([0, 1]),
            np.ones((2, 2), dtype=np.float32),
            np.array([0]),
            "validation_labels",
        ),
    ],
    ids=[
        "empty-training-split",
        "empty-validation-split",
        "training-length-mismatch",
        "validation-length-mismatch",
    ],
)
def test_training_rejects_empty_or_misaligned_splits(
    train_inputs: np.ndarray,
    train_labels: np.ndarray,
    validation_inputs: np.ndarray,
    validation_labels: np.ndarray,
    error_pattern: str,
) -> None:
    with pytest.raises(ValueError, match=error_pattern):
        detection.train_move_level_detector(
            train_inputs,
            train_labels,
            validation_inputs,
            validation_labels,
            config=detection.MoveLevelTrainConfig(
                max_epochs=1,
                early_stopping_patience=1,
            ),
            device="cpu",
        )


def test_prediction_uses_a_fixed_half_probability_threshold() -> None:
    predict_move_level_detector = _trainer_api("predict_move_level_detector")

    class FixedLogits(nn.Module):
        def forward(self, _inputs: torch.Tensor) -> torch.Tensor:
            return torch.tensor([[-0.01], [0.0], [0.01]], dtype=torch.float32)

    predictions = predict_move_level_detector(
        FixedLogits(), np.zeros((3, 2), dtype=np.float32)
    )

    np.testing.assert_array_equal(predictions, np.array([0, 1, 1], dtype=np.int64))
    assert "threshold" not in inspect.signature(predict_move_level_detector).parameters


@pytest.mark.parametrize(
    ("labels", "predictions", "expected"),
    [
        ([0, 0, 1, 1], [0, 1, 1, 0], (0.5, 0.5, 0.5)),
        ([0, 0], [0, 0], (1.0, 0.0, 0.5)),
        ([1, 1], [1, 1], (0.0, 1.0, 0.5)),
        ([0, 0], [0, 1], (0.5, 0.0, 1.0 / 3.0)),
        ([1, 1], [1, 0], (0.0, 0.5, 1.0 / 3.0)),
    ],
    ids=[
        "both-classes",
        "no-positive-labels",
        "no-negative-labels",
        "no-positive-labels-with-false-positive",
        "no-negative-labels-with-false-negative",
    ],
)
def test_move_level_metrics_define_specificity_recall_and_two_class_macro_f1(
    labels: list[int],
    predictions: list[int],
    expected: tuple[float, float, float],
) -> None:
    compute_detection_metrics = _trainer_api("compute_detection_metrics")

    metrics = compute_detection_metrics(np.array(labels), np.array(predictions))

    assert metrics["specificity"] == pytest.approx(expected[0])
    assert metrics["recall"] == pytest.approx(expected[1])
    assert metrics["macro_f1"] == pytest.approx(expected[2])
