"""Contract tests for the fail-closed public Allie embedding artifact builder."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

import data_generation.huggingface.prepare_allie_embeddings as builder
from data_generation.huggingface.prepare_allie_embeddings import (
    SYNTH_ARRAY_FILENAMES,
    _npz_metadata,
    prepare_allie_embeddings,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _semantic_row_index_sha256(path: Path, columns: tuple[str, ...]) -> str:
    """Compute the test-side canonical row digest from public scalar values."""

    table = pq.read_table(path, columns=list(columns))
    digest = hashlib.sha256()
    digest.update(b"allie-embedding-row-index-canonical-json-v1\n")
    digest.update(
        json.dumps(
            [(field.name, str(field.type)) for field in table.schema],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    materialized = [table.column(column).to_pylist() for column in columns]
    for row_number in range(table.num_rows):
        row = []
        for column in materialized:
            value = column[row_number]
            row.append({"float64_hex": value.hex()} if isinstance(value, float) else value)
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


class ArtifactFixtures:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.synth_arrays_dir = root / "synth-arrays"
        self.synth_manifest_csv = root / "synth-manifest.csv"
        self.tournament_manifest_csv = root / "tournament-manifest.csv"
        self.tournament_archive = root / "embs_allie_2500.npz"
        self.output_dir = root / "artifacts" / "allie_embeddings"
        self.synth_arrays_dir.mkdir()
        self._write_sources()

    def _write_sources(self) -> None:
        for position, filename in enumerate(SYNTH_ARRAY_FILENAMES):
            np.save(
                self.synth_arrays_dir / filename,
                np.full((2, 3), position, dtype=np.float32),
                allow_pickle=False,
            )
        pd.DataFrame(
            {
                "player_id": ["player-1", "player-2"],
                "game_id": ["game-a", "game-b"],
                "half_move": [4, 8],
                "move_player": ["e2e4", "d2d4"],
                "move_thinking_time": [1.25, 2.5],
            }
        ).to_csv(self.synth_manifest_csv, index=False)
        np.savez(
            self.tournament_archive,
            layer=np.arange(6, dtype=np.float32).reshape(2, 3),
            pooled=np.ones((2, 2), dtype=np.float16),
        )
        pd.DataFrame(
            {
                "game_id": ["tournament-a", "tournament-b"],
                "player_id": [101, 102],
                "half_move": [5, 6],
                "move_player": ["g1f3", "c2c4"],
            }
        ).to_csv(self.tournament_manifest_csv, index=False)

    def build(self) -> dict[str, object]:
        return prepare_allie_embeddings(
            synth_manifest_csv=self.synth_manifest_csv,
            synth_arrays_dir=self.synth_arrays_dir,
            tournament_manifest_csv=self.tournament_manifest_csv,
            tournament_archive=self.tournament_archive,
            output_dir=self.output_dir,
        )


def test_builds_exact_byte_preserving_layout_indexes_and_manifest(tmp_path: Path) -> None:
    fixture = ArtifactFixtures(tmp_path)

    manifest = fixture.build()

    expected_files = {
        "manifest.json",
        "synth/row_index.parquet",
        *(f"synth/{filename}" for filename in SYNTH_ARRAY_FILENAMES),
        "tournament/row_index.parquet",
        "tournament/embs_allie_2500.npz",
    }
    actual_files = {
        str(path.relative_to(fixture.output_dir))
        for path in fixture.output_dir.rglob("*")
        if path.is_file()
    }
    assert actual_files == expected_files
    assert SYNTH_ARRAY_FILENAMES == (
        "move_uci.npy",
        "move_stockfish_1.npy",
        "move_stockfish_5.npy",
        "move_stockfish_9.npy",
        "move_stockfish_11.npy",
        "move_stockfish_15.npy",
        "move_lc0_1.npy",
        "move_lc0_10.npy",
        "move_lc0_100.npy",
        "move_maia2_2050.npy",
        "move_allie_2500.npy",
    )

    for filename in SYNTH_ARRAY_FILENAMES:
        source = fixture.synth_arrays_dir / filename
        copied = fixture.output_dir / "synth" / filename
        assert _sha256(copied) == _sha256(source)
        source_array = np.load(source, mmap_mode="r", allow_pickle=False)
        copied_array = np.load(copied, mmap_mode="r", allow_pickle=False)
        assert copied_array.dtype == source_array.dtype
        assert copied_array.shape == source_array.shape
    assert _sha256(fixture.output_dir / "tournament" / "embs_allie_2500.npz") == _sha256(
        fixture.tournament_archive
    )

    synth_schema = pq.read_schema(fixture.output_dir / "synth" / "row_index.parquet")
    tournament_schema = pq.read_schema(
        fixture.output_dir / "tournament" / "row_index.parquet"
    )
    assert [(field.name, str(field.type)) for field in synth_schema] == [
        ("embedding_row", "int64"),
        ("player_id", "string"),
        ("game_id", "string"),
        ("half_move", "int64"),
        ("move_player", "string"),
        ("move_thinking_time", "double"),
    ]
    assert [(field.name, str(field.type)) for field in tournament_schema] == [
        ("embedding_row", "int64"),
        ("game_id", "string"),
        ("player_id", "int64"),
        ("half_move", "int64"),
        ("move_player", "string"),
    ]
    synth_index = pd.read_parquet(fixture.output_dir / "synth" / "row_index.parquet")
    tournament_index = pd.read_parquet(
        fixture.output_dir / "tournament" / "row_index.parquet"
    )
    assert synth_index["embedding_row"].tolist() == [0, 1]
    assert tournament_index["embedding_row"].tolist() == [0, 1]
    assert not synth_index.duplicated(
        ["player_id", "game_id", "half_move", "move_player"]
    ).any()
    assert not tournament_index.duplicated(
        ["game_id", "player_id", "half_move", "move_player"]
    ).any()

    saved_manifest = json.loads((fixture.output_dir / "manifest.json").read_text())
    assert manifest == saved_manifest
    assert saved_manifest["artifact_contract_version"] == "allie-embeddings-v1"
    assert saved_manifest["dataset_repository"] == "artemlepin/chess-fraud"
    assert saved_manifest["row_indexes"]["synth"]["count"] == 2
    assert saved_manifest["row_indexes"]["tournament"]["count"] == 2
    assert saved_manifest["files"]["synth/move_uci.npy"]["numpy"] == {
        "dtype": "float32",
        "shape": [2, 3],
    }
    assert saved_manifest["files"]["tournament/embs_allie_2500.npz"]["numpy"] == {
        "members": {
            "layer": {"dtype": "float32", "shape": [2, 3]},
            "pooled": {"dtype": "float16", "shape": [2, 2]},
        }
    }
    assert saved_manifest["source_provenance"]["synth_manifest_csv"]["filename"] == (
        "synth-manifest.csv"
    )
    assert saved_manifest["row_index_digest_format"] == (
        "allie-embedding-row-index-canonical-json-v1"
    )
    for name, columns in {
        "synth": (
            "player_id",
            "game_id",
            "half_move",
            "move_player",
            "move_thinking_time",
        ),
        "tournament": ("game_id", "player_id", "half_move", "move_player"),
    }.items():
        expected_digest = _semantic_row_index_sha256(
            fixture.output_dir / name / "row_index.parquet", columns
        )
        assert saved_manifest["row_indexes"][name]["semantic_sha256"] == expected_digest
        source_name = f"{name}_manifest_csv"
        assert (
            saved_manifest["source_provenance"][source_name]["normalized_rows_sha256"]
            == expected_digest
        )
    assert str(tmp_path) not in json.dumps(saved_manifest)


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        (lambda fixture: (fixture.synth_arrays_dir / "move_lc0_100.npy").unlink(), "missing"),
        (
            lambda fixture: np.save(
                fixture.synth_arrays_dir / "unexpected.npy",
                np.zeros((2, 3), dtype=np.float32),
            ),
            "unexpected",
        ),
        (
            lambda fixture: np.save(
                fixture.synth_arrays_dir / "move_lc0_100.npy",
                np.zeros((3, 3), dtype=np.float32),
            ),
            "row count",
        ),
        (
            lambda fixture: fixture.synth_manifest_csv.write_text(
                "player_id,game_id,half_move,move_player,move_thinking_time\n"
                "player-1,game-a,4,e2e4,1.25\n"
                "player-1,game-a,4,e2e4,2.5\n",
                encoding="utf-8",
            ),
            "duplicate",
        ),
        (
            lambda fixture: fixture.synth_manifest_csv.write_text(
                "player_id,game_id,half_move,move_player,move_thinking_time\n"
                "player-1,game-a,4,e2e4,nan\n"
                "player-2,game-b,8,d2d4,2.5\n",
                encoding="utf-8",
            ),
            "finite",
        ),
    ],
)
def test_rejects_invalid_synth_sources_without_leaving_output(
    tmp_path: Path, mutation: object, expected_error: str
) -> None:
    fixture = ArtifactFixtures(tmp_path)
    mutation(fixture)  # type: ignore[operator]

    with pytest.raises((ValueError, FileNotFoundError), match=expected_error):
        fixture.build()

    assert not fixture.output_dir.exists()
    assert not [
        path
        for path in fixture.output_dir.parent.glob(f".{fixture.output_dir.name}.*")
        if path.is_dir()
    ]


def test_rejects_tournament_count_mismatch_and_existing_output(tmp_path: Path) -> None:
    fixture = ArtifactFixtures(tmp_path)
    fixture.tournament_manifest_csv.write_text(
        "game_id,player_id,half_move,move_player\n"
        "tournament-a,101,5,g1f3\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="row count"):
        fixture.build()
    assert not fixture.output_dir.exists()

    fixture = ArtifactFixtures(tmp_path / "existing")
    fixture.output_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError, match="output directory already exists"):
        fixture.build()


def test_preserves_string_identifier_lexemes_and_records_a_rerunnable_command(
    tmp_path: Path,
) -> None:
    fixture = ArtifactFixtures(tmp_path)
    fixture.synth_manifest_csv.write_text(
        "player_id,game_id,half_move,move_player,move_thinking_time\n"
        "001,0007,4,e2e4,1.25\n"
        "002,0008,8,d2d4,2.5\n",
        encoding="utf-8",
    )
    fixture.tournament_manifest_csv.write_text(
        "game_id,player_id,half_move,move_player\n"
        "0009,101,5,g1f3\n"
        "0010,102,6,c2c4\n",
        encoding="utf-8",
    )

    manifest = fixture.build()

    synth_index = pq.read_table(fixture.output_dir / "synth" / "row_index.parquet")
    tournament_index = pq.read_table(
        fixture.output_dir / "tournament" / "row_index.parquet"
    )
    assert synth_index.column("player_id").to_pylist() == ["001", "002"]
    assert synth_index.column("game_id").to_pylist() == ["0007", "0008"]
    assert tournament_index.column("game_id").to_pylist() == ["0009", "0010"]
    command = manifest["preparation_command"]
    assert isinstance(command, str)
    for required_argument in (
        "--synth-manifest-csv",
        "--synth-arrays-dir",
        "--tournament-manifest-csv",
        "--tournament-archive",
        "--output-dir",
    ):
        assert required_argument in command
    assert str(tmp_path) not in command


def test_rejects_a_noncanonical_tournament_archive_filename(tmp_path: Path) -> None:
    fixture = ArtifactFixtures(tmp_path)
    renamed_archive = fixture.root / "arbitrary-input-name.npz"
    fixture.tournament_archive.rename(renamed_archive)
    fixture.tournament_archive = renamed_archive

    with pytest.raises(ValueError, match="filename"):
        fixture.build()


def test_reads_npz_member_metadata_without_loading_member_arrays(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "embs_allie_2500.npz"
    np.savez(archive, embeddings=np.zeros((2, 3), dtype=np.float32))

    def fail_if_np_load_is_used(*args: object, **kwargs: object) -> object:
        raise AssertionError("NPZ member payload must not be materialized")

    monkeypatch.setattr(
        "data_generation.huggingface.prepare_allie_embeddings.np.load",
        fail_if_np_load_is_used,
    )

    assert _npz_metadata(archive, expected_rows=2) == {
        "members": {"embeddings": {"dtype": "float32", "shape": [2, 3]}}
    }


def test_refuses_a_concurrent_builder_lock(tmp_path: Path) -> None:
    fixture = ArtifactFixtures(tmp_path)
    fixture.output_dir.parent.mkdir(parents=True, exist_ok=True)
    ready_read, ready_write = os.pipe()
    release_read, release_write = os.pipe()
    child = os.fork()
    if child == 0:
        os.close(ready_read)
        os.close(release_write)
        try:
            with builder._output_lock(fixture.output_dir):
                os.write(ready_write, b"1")
                os.read(release_read, 1)
        finally:
            os._exit(0)
    os.close(ready_write)
    os.close(release_read)
    try:
        assert os.read(ready_read, 1) == b"1"
        with pytest.raises(FileExistsError, match="another build"):
            fixture.build()
        assert not fixture.output_dir.exists()
    finally:
        os.write(release_write, b"1")
        os.close(release_write)
        os.close(ready_read)
        os.waitpid(child, 0)


def test_recovers_when_owner_crashes_before_lock_metadata_is_written(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = ArtifactFixtures(tmp_path)
    fixture.output_dir.parent.mkdir(parents=True, exist_ok=True)
    lock = fixture.output_dir.parent / f".{fixture.output_dir.name}.lock"
    child = os.fork()
    if child == 0:
        monkeypatch.setattr(builder.os, "fdopen", lambda *_args, **_kwargs: os._exit(73))
        fixture.build()
        os._exit(1)
    _, status = os.waitpid(child, 0)

    assert os.waitstatus_to_exitcode(status) == 73
    assert lock.is_file()
    assert lock.stat().st_size == 0

    fixture.build()

    assert fixture.output_dir.is_dir()


def test_lock_rejects_a_symlink_without_clobbering_its_target(tmp_path: Path) -> None:
    fixture = ArtifactFixtures(tmp_path)
    fixture.output_dir.parent.mkdir(parents=True, exist_ok=True)
    victim = tmp_path / "lock-target.txt"
    original = b"must remain unchanged\n"
    victim.write_bytes(original)
    lock = fixture.output_dir.parent / f".{fixture.output_dir.name}.lock"
    lock.symlink_to(victim)

    with pytest.raises(ValueError, match="lock path must be a regular file"):
        fixture.build()

    assert victim.read_bytes() == original
    assert lock.is_symlink()
    assert not fixture.output_dir.exists()


def test_cleans_temporary_directory_when_copy_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = ArtifactFixtures(tmp_path)
    import data_generation.huggingface.prepare_allie_embeddings as builder

    def fail_copy(source: Path, destination: Path) -> None:
        raise OSError("simulated copy failure")

    monkeypatch.setattr(builder, "_copy_file", fail_copy)

    with pytest.raises(OSError, match="simulated copy failure"):
        fixture.build()

    assert not fixture.output_dir.exists()
    assert not [
        path
        for path in fixture.output_dir.parent.glob(f".{fixture.output_dir.name}.*")
        if path.is_dir()
    ]
