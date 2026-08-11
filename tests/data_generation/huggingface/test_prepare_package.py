"""Contract tests for additive combined Hugging Face package assembly."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import data_generation.huggingface.prepare_package as package_builder
import data_generation.huggingface.prepare_allie_embeddings as embedding_builder
from data_generation.huggingface.prepare_package import prepare_package
from data_generation.huggingface.prepare_allie_embeddings import (
    SYNTH_ARRAY_FILENAMES,
    prepare_allie_embeddings,
)
from experiments.move_level.allie_embeddings import resolve_allie_embedding_root


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_component_release(directory: Path, files: dict[str, bytes]) -> None:
    directory.mkdir()
    inventory: dict[str, dict[str, object]] = {}
    for filename, content in files.items():
        path = directory / filename
        path.write_bytes(content)
        inventory[filename] = {"bytes": len(content), "sha256": _sha256(path)}
    (directory / "manifest.json").write_text(
        json.dumps({"dataset_version": "2.0.0-rc1", "files": inventory}),
        encoding="utf-8",
    )


def _build_allie_embeddings(root: Path) -> Path:
    root.mkdir()
    arrays = root / "arrays"
    arrays.mkdir()
    for filename in SYNTH_ARRAY_FILENAMES:
        np.save(arrays / filename, np.zeros((1, 2), dtype=np.float32), allow_pickle=False)
    synth = root / "synth.csv"
    pd.DataFrame(
        {
            "player_id": ["synth-player"],
            "game_id": ["synth-game"],
            "half_move": [2],
            "move_player": ["e2e4"],
            "move_thinking_time": [1.0],
        }
    ).to_csv(synth, index=False)
    tournament = root / "tournament.csv"
    pd.DataFrame(
        {
            "game_id": ["tournament-game"],
            "player_id": [7],
            "half_move": [3],
            "move_player": ["d2d4"],
        }
    ).to_csv(tournament, index=False)
    archive = root / "embs_allie_2500.npz"
    np.savez(archive, embeddings=np.ones((1, 2), dtype=np.float32))
    output = root / "artifacts" / "allie_embeddings"
    prepare_allie_embeddings(
        synth_manifest_csv=synth,
        synth_arrays_dir=arrays,
        tournament_manifest_csv=tournament,
        tournament_archive=archive,
        output_dir=output,
    )
    return output


def _refresh_embedding_file_record(root: Path, relative_path: str) -> None:
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = root / relative_path
    manifest["files"][relative_path]["bytes"] = path.stat().st_size
    manifest["files"][relative_path]["sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_adds_validated_allie_embeddings_tree_without_changing_core_mappings(tmp_path: Path) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(
        synth, {"train.parquet": b"train", "test.parquet": b"test"}
    )
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    output = tmp_path / "package"

    manifest = prepare_package(
        tournament_release_dir=tournament,
        synth_release_dir=synth,
        card_path=card,
        allie_embeddings_dir=allie_embeddings,
        output_dir=output,
        dataset_version="2.0.0-rc1",
    )

    assert manifest["configurations"] == {
        "chess_fraud": {"full": "data/chess_fraud/full.parquet"},
        "chess_fraud_synth": {
            "train": "data/chess_fraud_synth/train.parquet",
            "test": "data/chess_fraud_synth/test.parquet",
        },
    }
    for relative_path in (
        "data/chess_fraud/full.parquet",
        "data/chess_fraud_synth/train.parquet",
        "data/chess_fraud_synth/test.parquet",
    ):
        source = {
            "data/chess_fraud/full.parquet": tournament / "chess_fraud.parquet",
            "data/chess_fraud_synth/train.parquet": synth / "train.parquet",
            "data/chess_fraud_synth/test.parquet": synth / "test.parquet",
        }[relative_path]
        assert _sha256(output / relative_path) == _sha256(source)
    for source in allie_embeddings.rglob("*"):
        if source.is_file():
            copied = output / "artifacts" / "allie_embeddings" / source.relative_to(allie_embeddings)
            assert _sha256(copied) == _sha256(source)
    assert "artifacts/allie_embeddings/manifest.json" in manifest["files"]
    assert manifest["allie_embedding_contract_version"] == "allie-embeddings-v1"


def test_rejects_invalid_artifact_checksum_and_existing_output(tmp_path: Path) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(synth, {"train.parquet": b"train", "test.parquet": b"test"})
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    (allie_embeddings / "synth" / "move_uci.npy").write_bytes(b"tampered")
    output = tmp_path / "package"

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_package(
            tournament_release_dir=tournament,
            synth_release_dir=synth,
            card_path=card,
            allie_embeddings_dir=allie_embeddings,
            output_dir=output,
            dataset_version="2.0.0-rc1",
        )
    assert not output.exists()

    allie_embeddings = _build_allie_embeddings(tmp_path / "another-allie_embeddings-source")
    output.mkdir()
    with pytest.raises(FileExistsError, match="output directory already exists"):
        prepare_package(
            tournament_release_dir=tournament,
            synth_release_dir=synth,
            card_path=card,
            allie_embeddings_dir=allie_embeddings,
            output_dir=output,
            dataset_version="2.0.0-rc1",
        )


def test_rejects_allie_embeddings_payload_mutated_after_validation_during_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(synth, {"train.parquet": b"train", "test.parquet": b"test"})
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    output = tmp_path / "package"
    original_copy = package_builder._copy_file

    def mutate_after_manifest_copy(source: Path, destination: Path) -> None:
        original_copy(source, destination)
        if source == allie_embeddings / "manifest.json":
            (allie_embeddings / "synth" / "move_uci.npy").write_bytes(b"changed during copy")

    monkeypatch.setattr(package_builder, "_copy_file", mutate_after_manifest_copy)

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_package(
            tournament_release_dir=tournament,
            synth_release_dir=synth,
            card_path=card,
            allie_embeddings_dir=allie_embeddings,
            output_dir=output,
            dataset_version="2.0.0-rc1",
        )

    assert not output.exists()
    assert not [
        path
        for path in output.parent.glob(f".{output.name}.*")
        if path.is_dir()
    ]


def test_package_refuses_a_concurrent_builder_lock(tmp_path: Path) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(synth, {"train.parquet": b"train", "test.parquet": b"test"})
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    output = tmp_path / "package"
    lock = output.parent / f".{output.name}.lock"
    with lock.open("w", encoding="utf-8") as owner:
        owner.write(json.dumps({"pid": os.getpid(), "token": "live-builder"}))
        owner.flush()
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)

        with pytest.raises(FileExistsError, match="another build"):
            prepare_package(
                tournament_release_dir=tournament,
                synth_release_dir=synth,
                card_path=card,
                allie_embeddings_dir=allie_embeddings,
                output_dir=output,
                dataset_version="2.0.0-rc1",
            )

    assert not output.exists()


def test_rejects_any_uninventoried_file_inside_the_allie_embeddings_tree(tmp_path: Path) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(synth, {"train.parquet": b"train", "test.parquet": b"test"})
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    (allie_embeddings / "synth" / "manifest.json").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="missing or unexpected files"):
        prepare_package(
            tournament_release_dir=tournament,
            synth_release_dir=synth,
            card_path=card,
            allie_embeddings_dir=allie_embeddings,
            output_dir=tmp_path / "package",
            dataset_version="2.0.0-rc1",
        )


@pytest.mark.parametrize(
    ("embedding_rows", "player_ids", "expected_error"),
    [
        ([0, 0], ["synth-player", "synth-player"], "embedding_row"),
        ([0, 1], ["synth-player", "synth-player"], "duplicate join key"),
    ],
)
def test_rejects_self_consistent_invalid_synth_index_contract(
    tmp_path: Path,
    embedding_rows: list[int],
    player_ids: list[str],
    expected_error: str,
) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(synth, {"train.parquet": b"train", "test.parquet": b"test"})
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    pq.write_table(
        pa.table(
            {
                "embedding_row": pa.array(embedding_rows, type=pa.int64()),
                "player_id": pa.array(player_ids, type=pa.string()),
                "game_id": pa.array(["synth-game", "synth-game"], type=pa.string()),
                "half_move": pa.array([2, 2], type=pa.int64()),
                "move_player": pa.array(["e2e4", "e2e4"], type=pa.string()),
                "move_thinking_time": pa.array([1.0, 1.0], type=pa.float64()),
            }
        ),
        allie_embeddings / "synth" / "row_index.parquet",
    )
    _refresh_embedding_file_record(allie_embeddings, "synth/row_index.parquet")
    manifest_path = allie_embeddings / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["row_indexes"]["synth"]["count"] = 2
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        prepare_package(
            tournament_release_dir=tournament,
            synth_release_dir=synth,
            card_path=card,
            allie_embeddings_dir=allie_embeddings,
            output_dir=tmp_path / "package",
            dataset_version="2.0.0-rc1",
        )


def test_rejects_self_consistent_array_row_count_and_numpy_metadata(
    tmp_path: Path,
) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(synth, {"train.parquet": b"train", "test.parquet": b"test"})
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    np.save(allie_embeddings / "synth" / "move_uci.npy", np.zeros((3, 2), dtype=np.float64))
    _refresh_embedding_file_record(allie_embeddings, "synth/move_uci.npy")
    manifest_path = allie_embeddings / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["synth/move_uci.npy"]["numpy"] = {
        "dtype": "float64",
        "shape": [3, 2],
    }
    manifest["source_provenance"]["synth_arrays"]["move_uci.npy"]["sha256"] = (
        manifest["files"]["synth/move_uci.npy"]["sha256"]
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="row count"):
        prepare_package(
            tournament_release_dir=tournament,
            synth_release_dir=synth,
            card_path=card,
            allie_embeddings_dir=allie_embeddings,
            output_dir=tmp_path / "package",
            dataset_version="2.0.0-rc1",
        )


@pytest.mark.parametrize(
    ("mutate_manifest", "expected_error"),
    [
        (lambda manifest: manifest.pop("source_provenance"), "source provenance"),
        (lambda manifest: manifest.__setitem__("source_provenance", {}), "source provenance"),
        (
            lambda manifest: manifest.__setitem__(
                "preparation_command",
                "python -m data_generation.huggingface.prepare_allie_embeddings "
                "--synth-manifest-csv <synth-manifest.csv>",
            ),
            "preparation command",
        ),
        (lambda manifest: manifest.__setitem__("preparation_command", False), "preparation command"),
    ],
)
def test_rejects_missing_or_semantically_invalid_reproducibility_metadata(
    tmp_path: Path, mutate_manifest: object, expected_error: str
) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(synth, {"train.parquet": b"train", "test.parquet": b"test"})
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    manifest_path = allie_embeddings / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mutate_manifest(manifest)  # type: ignore[operator]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=expected_error):
        prepare_package(
            tournament_release_dir=tournament,
            synth_release_dir=synth,
            card_path=card,
            allie_embeddings_dir=allie_embeddings,
            output_dir=tmp_path / "package",
            dataset_version="2.0.0-rc1",
        )


@pytest.mark.parametrize(
    "source_path",
    [
        ("synth_arrays", "move_uci.npy"),
        ("tournament_archive", None),
    ],
)
def test_rejects_well_formed_provenance_that_disagrees_with_copied_payload(
    tmp_path: Path, source_path: tuple[str, str | None]
) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(synth, {"train.parquet": b"train", "test.parquet": b"test"})
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    manifest_path = allie_embeddings / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    group, filename = source_path
    record = manifest["source_provenance"][group]
    if filename is not None:
        record = record[filename]
    record["sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="source provenance"):
        prepare_package(
            tournament_release_dir=tournament,
            synth_release_dir=synth,
            card_path=card,
            allie_embeddings_dir=allie_embeddings,
            output_dir=tmp_path / "package",
            dataset_version="2.0.0-rc1",
        )


def test_rejects_row_index_that_disagrees_with_canonical_source_provenance(
    tmp_path: Path,
) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(synth, {"train.parquet": b"train", "test.parquet": b"test"})
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    index_path = allie_embeddings / "synth" / "row_index.parquet"
    index = pq.read_table(index_path).set_column(
        5,
        "move_thinking_time",
        pa.array([9.0], type=pa.float64()),
    )
    pq.write_table(index, index_path)
    _refresh_embedding_file_record(allie_embeddings, "synth/row_index.parquet")

    with pytest.raises(ValueError, match="source provenance"):
        prepare_package(
            tournament_release_dir=tournament,
            synth_release_dir=synth,
            card_path=card,
            allie_embeddings_dir=allie_embeddings,
            output_dir=tmp_path / "package",
            dataset_version="2.0.0-rc1",
        )


def test_builder_package_output_is_accepted_by_the_resolver(tmp_path: Path) -> None:
    tournament = tmp_path / "tournament"
    synth = tmp_path / "synth"
    _write_component_release(tournament, {"chess_fraud.parquet": b"tournament"})
    _write_component_release(synth, {"train.parquet": b"train", "test.parquet": b"test"})
    card = tmp_path / "README.md"
    card.write_text("# ChessFraud\n", encoding="utf-8")
    allie_embeddings = _build_allie_embeddings(tmp_path / "allie_embeddings-source")
    package = tmp_path / "package"
    prepare_package(
        tournament_release_dir=tournament,
        synth_release_dir=synth,
        card_path=card,
        allie_embeddings_dir=allie_embeddings,
        output_dir=package,
        dataset_version="2.0.0-rc1",
    )

    resolved = resolve_allie_embedding_root(
        revision="1" * 40,
        environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(package)},
    )

    assert resolved == package / "artifacts" / "allie_embeddings"


def test_row_index_join_columns_are_materialized_at_most_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    table = pa.table(
        {
            "embedding_row": pa.array([0, 1, 2], type=pa.int64()),
            "player_id": pa.array(["player-a", "player-b", "player-c"], type=pa.string()),
            "game_id": pa.array(["game-a", "game-b", "game-c"], type=pa.string()),
            "half_move": pa.array([4, 8, 12], type=pa.int64()),
            "move_player": pa.array(["e2e4", "d2d4", "g1f3"], type=pa.string()),
            "move_thinking_time": pa.array([1.0, 2.0, 3.0], type=pa.float64()),
        }
    )
    calls = {name: 0 for name in ("player_id", "game_id", "half_move", "move_player")}

    class CountingColumn:
        def __init__(self, name: str) -> None:
            self.name = name

        def to_pylist(self) -> list[object]:
            if self.name in calls:
                calls[self.name] += 1
            return table.column(self.name).to_pylist()

    class CountingTable:
        schema = table.schema
        num_rows = table.num_rows

        def column(self, name: str) -> CountingColumn:
            return CountingColumn(name)

        def group_by(self, keys: list[str]) -> object:
            return table.group_by(keys)

    monkeypatch.setattr(embedding_builder.pq, "read_table", lambda _: CountingTable())
    row_index = {
        "path": "synth/row_index.parquet",
        "count": table.num_rows,
        "schema": [
            {"name": "embedding_row", "type": "int64"},
            {"name": "player_id", "type": "string"},
            {"name": "game_id", "type": "string"},
            {"name": "half_move", "type": "int64"},
            {"name": "move_player", "type": "string"},
            {"name": "move_thinking_time", "type": "float64"},
        ],
        "join_key": ["player_id", "game_id", "half_move", "move_player"],
    }

    assert embedding_builder._validate_row_index(tmp_path, "synth", row_index) == table.num_rows
    assert all(count <= 1 for count in calls.values())
