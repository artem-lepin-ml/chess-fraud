"""Contract tests for deterministic acquisition of the Allie embedding bundle."""

from __future__ import annotations

import hashlib
import importlib
import json
from collections.abc import Callable
from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SYNTH_ARRAYS = (
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
ROW_INDEX_DIGEST_FORMAT = "allie-embedding-row-index-canonical-json-v1"
PREPARATION_COMMAND = (
    "python -m data_generation.huggingface.prepare_allie_embeddings "
    "--synth-manifest-csv <synth-manifest.csv> "
    "--synth-arrays-dir <synth-arrays-dir> "
    "--tournament-manifest-csv <tournament-manifest.csv> "
    "--tournament-archive <embs_allie_2500.npz> "
    "--output-dir <artifacts/allie_embeddings>"
)
SYNTH_SCHEMA = (
    ("embedding_row", pa.int64()),
    ("player_id", pa.string()),
    ("game_id", pa.string()),
    ("half_move", pa.int64()),
    ("move_player", pa.string()),
    ("move_thinking_time", pa.float64()),
)
TOURNAMENT_SCHEMA = (
    ("embedding_row", pa.int64()),
    ("game_id", pa.string()),
    ("player_id", pa.int64()),
    ("half_move", pa.int64()),
    ("move_player", pa.string()),
)


def _artifacts_module() -> ModuleType:
    """Load the resolver so its absence is a RED test failure, not collection error."""
    try:
        return importlib.import_module("experiments.move_level.allie_embeddings")
    except ModuleNotFoundError as error:
        if error.name == "experiments.move_level.allie_embeddings":
            pytest.fail("Allie embedding artifact resolver module is missing")
        raise


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_parquet(
    path: Path,
    schema: tuple[tuple[str, pa.DataType], ...],
    *,
    columns: dict[str, list[object]] | None = None,
) -> None:
    if columns is not None:
        pq.write_table(pa.table(columns, schema=pa.schema(schema)), path)
        return
    columns: dict[str, list[object]] = {}
    for name, data_type in schema:
        if pa.types.is_string(data_type):
            columns[name] = ["row-0", "row-1"]
        elif pa.types.is_floating(data_type):
            columns[name] = [1.5, 2.5]
        else:
            columns[name] = [0, 1]
    pq.write_table(pa.table(columns, schema=pa.schema(schema)), path)


def _manifest_type(data_type: pa.DataType) -> str:
    return "float64" if pa.types.is_float64(data_type) else str(data_type)


def _array_metadata(path: Path) -> dict[str, object]:
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    return {"dtype": str(array.dtype), "shape": list(array.shape)}


def _archive_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        return {
            "members": {
                name: {"dtype": str(archive[name].dtype), "shape": list(archive[name].shape)}
                for name in sorted(archive.files)
            }
        }


def _file_metadata(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _semantic_row_index_sha256(path: Path, columns: tuple[str, ...]) -> str:
    table = pq.read_table(path, columns=list(columns))
    digest = hashlib.sha256()
    digest.update(f"{ROW_INDEX_DIGEST_FORMAT}\n".encode("ascii"))
    digest.update(
        json.dumps(
            [(field.name, str(field.type)) for field in table.schema],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    columns_data = [table.column(column).to_pylist() for column in columns]
    for row_number in range(table.num_rows):
        row = []
        for column in columns_data:
            value = column[row_number]
            row.append({"float64_hex": value.hex()} if isinstance(value, float) else value)
        digest.update(
            json.dumps(row, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            + b"\n"
        )
    return digest.hexdigest()


def _write_artifact_snapshot(tmp_path: Path) -> Path:
    snapshot_root = tmp_path / "snapshot"
    artifact_root = snapshot_root / "artifacts" / "allie_embeddings"
    synth_dir = artifact_root / "synth"
    tournament_dir = artifact_root / "tournament"
    synth_dir.mkdir(parents=True)
    tournament_dir.mkdir()

    for index, filename in enumerate(SYNTH_ARRAYS):
        np.save(synth_dir / filename, np.array([index, index + 1], dtype=np.int64))
    np.savez(tournament_dir / "embs_allie_2500.npz", embeddings=np.eye(2, dtype=np.float32))
    _write_parquet(synth_dir / "row_index.parquet", SYNTH_SCHEMA)
    _write_parquet(tournament_dir / "row_index.parquet", TOURNAMENT_SCHEMA)

    files: dict[str, dict[str, object]] = {}
    for path in sorted(artifact_root.rglob("*")):
        if not path.is_file():
            continue
        relative_path = path.relative_to(artifact_root).as_posix()
        files[relative_path] = _file_metadata(path)
    for filename in SYNTH_ARRAYS:
        files[f"synth/{filename}"]["numpy"] = _array_metadata(synth_dir / filename)
    files["tournament/embs_allie_2500.npz"]["numpy"] = _archive_metadata(
        tournament_dir / "embs_allie_2500.npz"
    )
    synth_digest = _semantic_row_index_sha256(
        synth_dir / "row_index.parquet",
        ("player_id", "game_id", "half_move", "move_player", "move_thinking_time"),
    )
    tournament_digest = _semantic_row_index_sha256(
        tournament_dir / "row_index.parquet",
        ("game_id", "player_id", "half_move", "move_player"),
    )
    manifest = {
        "artifact_contract_version": "allie-embeddings-v1",
        "dataset_repository": "artemlepin/chess-fraud",
        "generated_at_utc": "2026-08-11T00:00:00+00:00",
        "row_index_digest_format": ROW_INDEX_DIGEST_FORMAT,
        "files": files,
        "row_indexes": {
            "synth": {
                "path": "synth/row_index.parquet",
                "count": 2,
                "schema": [
                    {"name": name, "type": _manifest_type(data_type)}
                    for name, data_type in SYNTH_SCHEMA
                ],
                "join_key": ["player_id", "game_id", "half_move", "move_player"],
                "semantic_sha256": synth_digest,
            },
            "tournament": {
                "path": "tournament/row_index.parquet",
                "count": 2,
                "schema": [
                    {"name": name, "type": _manifest_type(data_type)}
                    for name, data_type in TOURNAMENT_SCHEMA
                ],
                "join_key": ["game_id", "player_id", "half_move", "move_player"],
                "semantic_sha256": tournament_digest,
            },
        },
        "source_provenance": {
            "synth_manifest_csv": {
                "filename": "synth-manifest.csv",
                "sha256": "a" * 64,
                "normalized_rows_sha256": synth_digest,
            },
            "synth_arrays": {
                filename: {
                    "filename": filename,
                    "sha256": files[f"synth/{filename}"]["sha256"],
                }
                for filename in SYNTH_ARRAYS
            },
            "tournament_manifest_csv": {
                "filename": "tournament-manifest.csv",
                "sha256": "c" * 64,
                "normalized_rows_sha256": tournament_digest,
            },
            "tournament_archive": {
                "filename": "embs_allie_2500.npz",
                "sha256": files["tournament/embs_allie_2500.npz"]["sha256"],
            },
        },
        "preparation_command": PREPARATION_COMMAND,
    }
    (artifact_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot_root


def _rewrite_manifest(snapshot_root: Path, transform: Callable[[dict[str, object]], None]) -> None:
    manifest_path = snapshot_root / "artifacts" / "allie_embeddings" / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    transform(manifest)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _replace_payload(snapshot_root: Path, relative_path: str) -> None:
    artifact_root = snapshot_root / "artifacts" / "allie_embeddings"
    payload_path = artifact_root / relative_path
    _rewrite_manifest(
        snapshot_root,
        lambda manifest: manifest["files"].__setitem__(relative_path, _file_metadata(payload_path)),
    )


def _refresh_payload_checksum(snapshot_root: Path, relative_path: str) -> None:
    artifact_root = snapshot_root / "artifacts" / "allie_embeddings"
    payload_path = artifact_root / relative_path

    def update(manifest: dict[str, object]) -> None:
        record = manifest["files"][relative_path]
        record["bytes"] = payload_path.stat().st_size
        record["sha256"] = _sha256(payload_path)
        if relative_path.startswith("synth/") and relative_path.endswith(".npy"):
            filename = Path(relative_path).name
            manifest["source_provenance"]["synth_arrays"][filename]["sha256"] = record[
                "sha256"
            ]
        elif relative_path == "tournament/embs_allie_2500.npz":
            manifest["source_provenance"]["tournament_archive"]["sha256"] = record[
                "sha256"
            ]

    _rewrite_manifest(snapshot_root, update)


def test_local_override_wins_without_calling_the_downloader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()

    def unexpected_download(**_: object) -> str:
        raise AssertionError("a valid local override must prevent a download")

    monkeypatch.setattr(artifacts, "snapshot_download", unexpected_download)

    resolved = artifacts.resolve_allie_embedding_root(
        revision="a" * 40,
        environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(snapshot_root)},
    )

    assert resolved == snapshot_root / "artifacts" / "allie_embeddings"



def test_rejects_a_non_immutable_revision_before_downloading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifacts = _artifacts_module()
    def unexpected_download(**_: object) -> str:
        raise AssertionError("an invalid revision must not trigger a download")

    monkeypatch.setattr(artifacts, "snapshot_download", unexpected_download)

    with pytest.raises(ValueError, match="40-character.*SHA"):
        artifacts.resolve_allie_embedding_root(revision="main", environ={})


def test_download_uses_the_exact_immutable_dataset_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()
    received: dict[str, object] = {}

    def download(**kwargs: object) -> str:
        received.update(kwargs)
        return str(snapshot_root)

    monkeypatch.setattr(artifacts, "snapshot_download", download)

    resolved = artifacts.resolve_allie_embedding_root(
        revision="b" * 40,
        environ={"CHESSFRAUD_ALLIE_CACHE_DIR": "/tmp/allie_embeddings-cache"},
    )

    assert resolved == snapshot_root / "artifacts" / "allie_embeddings"
    assert received == {
        "repo_id": "artemlepin/chess-fraud",
        "repo_type": "dataset",
        "revision": "b" * 40,
        "allow_patterns": ["artifacts/allie_embeddings/**"],
        "cache_dir": "/tmp/allie_embeddings-cache",
    }


def test_default_cache_emits_the_resource_warning_and_uses_huggingface_default(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()
    received: dict[str, object] = {}

    def download(**kwargs: object) -> str:
        received.update(kwargs)
        return str(snapshot_root)

    monkeypatch.setattr(artifacts, "snapshot_download", download)

    artifacts.resolve_allie_embedding_root(revision="b" * 40, environ={})

    warning = capsys.readouterr().out
    assert "9.56 GB" in warning
    assert "CHESSFRAUD_ALLIE_EMBEDDING_DIR" in warning
    assert "CHESSFRAUD_ALLIE_CACHE_DIR" in warning
    assert "cache_dir" not in received


@pytest.mark.parametrize("mutation", ["missing", "unexpected", "checksum", "schema"])
def test_rejects_invalid_local_artifact_contract(
    tmp_path: Path, mutation: str
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()
    artifact_root = snapshot_root / "artifacts" / "allie_embeddings"
    if mutation == "missing":
        (artifact_root / "synth" / SYNTH_ARRAYS[0]).unlink()
    elif mutation == "unexpected":
        (artifact_root / "synth" / "unapproved.npy").write_bytes(b"not an artifact")
    elif mutation == "checksum":
        (artifact_root / "synth" / SYNTH_ARRAYS[0]).write_bytes(b"corrupted")
    else:
        wrong_schema = tuple((name, pa.string()) for name, _ in SYNTH_SCHEMA)
        _write_parquet(artifact_root / "synth" / "row_index.parquet", wrong_schema)
        _rewrite_manifest(
            snapshot_root,
            lambda manifest: manifest["files"].__setitem__(
                "synth/row_index.parquet",
                {
                    "bytes": (artifact_root / "synth" / "row_index.parquet").stat().st_size,
                    "sha256": _sha256(artifact_root / "synth" / "row_index.parquet"),
                },
            ),
        )

    with pytest.raises(ValueError, match="Allie embedding"):
        artifacts.resolve_allie_embedding_root(
            revision="c" * 40,
            environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(snapshot_root)},
        )


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        (
            "duplicate_embedding_row",
            "embedding_row values must be unique and contiguous starting at zero",
        ),
        (
            "gapped_embedding_row",
            "embedding_row values must be unique and contiguous starting at zero",
        ),
        ("duplicate_join_key", "join keys must be unique"),
    ],
)
def test_rejects_semantically_invalid_row_indexes(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()
    parquet_path = snapshot_root / "artifacts" / "allie_embeddings" / "synth" / "row_index.parquet"
    columns = {
        "embedding_row": (
            [0, 0]
            if mutation == "duplicate_embedding_row"
            else [0, 1]
            if mutation == "duplicate_join_key"
            else [0, 2]
        ),
        "player_id": ["player-a", "player-a"] if mutation == "duplicate_join_key" else ["player-a", "player-b"],
        "game_id": ["game-a", "game-a"] if mutation == "duplicate_join_key" else ["game-a", "game-b"],
        "half_move": [4, 4] if mutation == "duplicate_join_key" else [4, 8],
        "move_player": ["e2e4", "e2e4"] if mutation == "duplicate_join_key" else ["e2e4", "d2d4"],
        "move_thinking_time": [1.5, 2.5],
    }
    _write_parquet(parquet_path, SYNTH_SCHEMA, columns=columns)
    _replace_payload(snapshot_root, "synth/row_index.parquet")

    with pytest.raises(ValueError, match=expected_error):
        artifacts.resolve_allie_embedding_root(
            revision="e" * 40,
            environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(snapshot_root)},
        )


@pytest.mark.parametrize(
    "invalid_value",
    [None, np.nan, np.inf, -np.inf],
    ids=["null", "nan", "positive-infinity", "negative-infinity"],
)
def test_rejects_nonfinite_synth_thinking_time_with_self_consistent_metadata(
    tmp_path: Path, invalid_value: float | None
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()
    parquet_path = snapshot_root / "artifacts" / "allie_embeddings" / "synth" / "row_index.parquet"
    table = pq.read_table(parquet_path).set_column(
        5,
        "move_thinking_time",
        pa.array([invalid_value, 2.5], type=pa.float64()),
    )
    pq.write_table(table, parquet_path)
    semantic_digest = _semantic_row_index_sha256(
        parquet_path,
        ("player_id", "game_id", "half_move", "move_player", "move_thinking_time"),
    )

    def refresh_all_bindings(manifest: dict[str, object]) -> None:
        manifest["files"]["synth/row_index.parquet"] = _file_metadata(parquet_path)
        manifest["row_indexes"]["synth"]["semantic_sha256"] = semantic_digest
        manifest["source_provenance"]["synth_manifest_csv"][
            "normalized_rows_sha256"
        ] = semantic_digest

    _rewrite_manifest(snapshot_root, refresh_all_bindings)

    with pytest.raises(ValueError, match="move_thinking_time.*finite"):
        artifacts.resolve_allie_embedding_root(
            revision="6" * 40,
            environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(snapshot_root)},
        )


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        ("generated_at_utc", "generated_at_utc"),
        ("source_provenance", "source provenance"),
        ("source_checksum", "source provenance"),
        ("source_filename", "source provenance"),
        ("preparation_command", "preparation command"),
        ("npy_metadata", "NumPy metadata"),
        ("npz_metadata", "NumPy metadata"),
    ],
)
def test_rejects_incomplete_manifest_reproducibility_metadata(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()

    def mutate(manifest: dict[str, object]) -> None:
        if mutation in {"generated_at_utc", "source_provenance", "preparation_command"}:
            manifest.pop(mutation)
        elif mutation == "source_checksum":
            manifest["source_provenance"]["synth_arrays"][SYNTH_ARRAYS[0]].pop("sha256")
        elif mutation == "source_filename":
            manifest["source_provenance"]["synth_arrays"][SYNTH_ARRAYS[0]]["filename"] = "other.npy"
        elif mutation == "npy_metadata":
            manifest["files"][f"synth/{SYNTH_ARRAYS[0]}"].pop("numpy")
        else:
            manifest["files"]["tournament/embs_allie_2500.npz"].pop("numpy")

    _rewrite_manifest(snapshot_root, mutate)

    with pytest.raises(ValueError, match=expected_error):
        artifacts.resolve_allie_embedding_root(
            revision="f" * 40,
            environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(snapshot_root)},
        )


def test_rejects_well_formed_provenance_that_disagrees_with_payload(
    tmp_path: Path,
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()
    _rewrite_manifest(
        snapshot_root,
        lambda manifest: manifest["source_provenance"]["synth_arrays"][
            SYNTH_ARRAYS[0]
        ].__setitem__("sha256", "0" * 64),
    )

    with pytest.raises(ValueError, match="source provenance"):
        artifacts.resolve_allie_embedding_root(
            revision="4" * 40,
            environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(snapshot_root)},
        )


def test_rejects_semantically_changed_row_index_with_refreshed_file_checksum(
    tmp_path: Path,
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()
    index_path = snapshot_root / "artifacts" / "allie_embeddings" / "synth" / "row_index.parquet"
    table = pq.read_table(index_path).set_column(
        5,
        "move_thinking_time",
        pa.array([9.5, 10.5], type=pa.float64()),
    )
    pq.write_table(table, index_path)
    _refresh_payload_checksum(snapshot_root, "synth/row_index.parquet")

    with pytest.raises(ValueError, match="source provenance"):
        artifacts.resolve_allie_embedding_root(
            revision="5" * 40,
            environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(snapshot_root)},
        )


@pytest.mark.parametrize(
    "mutation, relative_path, expected_error",
    [
        ("npy", "synth/move_uci.npy", "NumPy metadata"),
        ("npz", "tournament/embs_allie_2500.npz", "NumPy metadata"),
    ],
)
def test_rejects_payloads_that_disagree_with_manifest_numpy_metadata(
    tmp_path: Path, mutation: str, relative_path: str, expected_error: str
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()
    payload_path = snapshot_root / "artifacts" / "allie_embeddings" / relative_path
    if mutation == "npy":
        np.save(payload_path, np.array([0, 1], dtype=np.float32), allow_pickle=False)
    else:
        np.savez(payload_path, embeddings=np.eye(2, dtype=np.float64))
    _refresh_payload_checksum(snapshot_root, relative_path)

    with pytest.raises(ValueError, match=expected_error):
        artifacts.resolve_allie_embedding_root(
            revision="1" * 40,
            environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(snapshot_root)},
        )


@pytest.mark.parametrize(
    "mutation, relative_path",
    [
        ("npy", "synth/move_uci.npy"),
        ("npz", "tournament/embs_allie_2500.npz"),
    ],
)
def test_rejects_array_metadata_with_a_row_count_mismatch(
    tmp_path: Path, mutation: str, relative_path: str
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()
    payload_path = snapshot_root / "artifacts" / "allie_embeddings" / relative_path
    if mutation == "npy":
        np.save(payload_path, np.array([0, 1, 2], dtype=np.int64), allow_pickle=False)
        metadata = _array_metadata(payload_path)
    else:
        np.savez(payload_path, embeddings=np.eye(3, dtype=np.float32))
        metadata = _archive_metadata(payload_path)
    _refresh_payload_checksum(snapshot_root, relative_path)
    _rewrite_manifest(
        snapshot_root,
        lambda manifest: manifest["files"][relative_path].__setitem__("numpy", metadata),
    )

    with pytest.raises(ValueError, match="row count"):
        artifacts.resolve_allie_embedding_root(
            revision="3" * 40,
            environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(snapshot_root)},
        )


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        ("invalid_utf8", "manifest is unreadable"),
        ("invalid_json", "manifest is unreadable"),
        ("invalid_parquet", "synth/row_index.parquet.*unreadable"),
    ],
)
def test_normalizes_malformed_contract_files_to_actionable_errors(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    snapshot_root = _write_artifact_snapshot(tmp_path)
    artifacts = _artifacts_module()
    artifact_root = snapshot_root / "artifacts" / "allie_embeddings"
    if mutation == "invalid_utf8":
        (artifact_root / "manifest.json").write_bytes(b"\xff")
    elif mutation == "invalid_json":
        (artifact_root / "manifest.json").write_text("{", encoding="utf-8")
    else:
        parquet_path = artifact_root / "synth" / "row_index.parquet"
        parquet_path.write_bytes(b"not parquet")
        _replace_payload(snapshot_root, "synth/row_index.parquet")

    with pytest.raises(ValueError, match=expected_error):
        artifacts.resolve_allie_embedding_root(
            revision="2" * 40,
            environ={"CHESSFRAUD_ALLIE_EMBEDDING_DIR": str(snapshot_root)},
        )


def test_download_errors_are_actionable_and_never_use_historical_local_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifacts = _artifacts_module()
    legacy_path = tmp_path / "data" / "processed" / "synth"
    legacy_path.mkdir(parents=True)
    (legacy_path / "move_uci.npy").write_bytes(b"historical local artifact")

    def unavailable(**_: object) -> str:
        raise OSError("offline")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(artifacts, "snapshot_download", unavailable)

    with pytest.raises(RuntimeError, match="download.*CHESSFRAUD_ALLIE_EMBEDDING_DIR"):
        artifacts.resolve_allie_embedding_root(revision="d" * 40, environ={})
