"""Build the byte-preserving public Allie embedding artifact bundle."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import tempfile
from typing import Any, Sequence
import zipfile

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


SYNTH_ARRAY_FILENAMES = (
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
TOURNAMENT_ARCHIVE_FILENAME = "embs_allie_2500.npz"
ARTIFACT_CONTRACT_VERSION = "allie-embeddings-v1"
DATASET_REPOSITORY = "artemlepin/chess-fraud"

_SYNTH_COLUMNS = (
    "player_id",
    "game_id",
    "half_move",
    "move_player",
    "move_thinking_time",
)
_TOURNAMENT_COLUMNS = ("game_id", "player_id", "half_move", "move_player")
_SYNTH_INDEX_SCHEMA = (
    ("embedding_row", "int64"),
    ("player_id", "string"),
    ("game_id", "string"),
    ("half_move", "int64"),
    ("move_player", "string"),
    ("move_thinking_time", "float64"),
)
_TOURNAMENT_INDEX_SCHEMA = (
    ("embedding_row", "int64"),
    ("game_id", "string"),
    ("player_id", "int64"),
    ("half_move", "int64"),
    ("move_player", "string"),
)
_ALLIE_EMBEDDING_OUTPUT_FILES = {
    *(f"synth/{filename}" for filename in SYNTH_ARRAY_FILENAMES),
    "synth/row_index.parquet",
    f"tournament/{TOURNAMENT_ARCHIVE_FILENAME}",
    "tournament/row_index.parquet",
}
_ROW_INDEX_CONTRACTS = {
    "synth": {
        "path": "synth/row_index.parquet",
        "schema": _SYNTH_INDEX_SCHEMA,
        "join_key": ("player_id", "game_id", "half_move", "move_player"),
    },
    "tournament": {
        "path": "tournament/row_index.parquet",
        "schema": _TOURNAMENT_INDEX_SCHEMA,
        "join_key": ("game_id", "player_id", "half_move", "move_player"),
    },
}
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_ROW_INDEX_DIGEST_FORMAT = "allie-embedding-row-index-canonical-json-v1"
_PREPARATION_COMMAND = (
    "python -m data_generation.huggingface.prepare_allie_embeddings "
    "--synth-manifest-csv <synth-manifest.csv> "
    "--synth-arrays-dir <synth-arrays-dir> "
    "--tournament-manifest-csv <tournament-manifest.csv> "
    "--tournament-archive <embs_allie_2500.npz> "
    "--output-dir <artifacts/allie_embeddings>"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


@contextmanager
def _output_lock(output: Path) -> Any:
    """Serialize builders with ownership that the kernel releases on process exit."""

    lock = output.parent / f".{output.name}.lock"
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    except OSError as error:
        if error.errno == errno.ELOOP:
            raise ValueError(f"lock path must be a regular file: {lock}") from error
        raise
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise ValueError(f"lock path must be a regular file: {lock}")
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise FileExistsError(f"another build is already preparing: {output}") from error
        os.ftruncate(descriptor, 0)
        with os.fdopen(descriptor, "w", encoding="utf-8", closefd=False) as stream:
            json.dump({"pid": os.getpid()}, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(descriptor)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _require_columns(frame: pd.DataFrame, columns: Sequence[str], source: Path) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns in {source.name}: {missing}")


def _string_column(frame: pd.DataFrame, column: str, source: Path) -> list[str]:
    values = frame[column]
    if values.isna().any():
        raise ValueError(f"missing {column} value in {source.name}")
    strings = values.astype(str)
    if (strings.str.len() == 0).any():
        raise ValueError(f"empty {column} value in {source.name}")
    return strings.tolist()


def _int64_column(frame: pd.DataFrame, column: str, source: Path) -> list[int]:
    values: list[int] = []
    for value in _string_column(frame, column, source):
        if value.startswith(("+", "-")):
            digits = value[1:]
        else:
            digits = value
        if not digits.isdecimal():
            raise ValueError(f"non-integral {column} value in {source.name}")
        integer = int(value)
        if not _INT64_MIN <= integer <= _INT64_MAX:
            raise ValueError(f"{column} value is outside signed int64 range in {source.name}")
        values.append(integer)
    return values


def _finite_float64_column(frame: pd.DataFrame, column: str, source: Path) -> list[float]:
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{column} values must be finite in {source.name}")
    return values.astype(np.float64).tolist()


def _validate_unique_values(
    values: dict[str, list[object]], keys: Sequence[str], source: Path
) -> None:
    rows = [tuple(values[key][position] for key in keys) for position in range(len(values[keys[0]]))]
    if len(rows) != len(set(rows)):
        raise ValueError(f"duplicate join key in {source.name}")


def _load_synth_index(source: Path) -> pa.Table:
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, dtype="string", keep_default_na=False, na_filter=False)
    _require_columns(frame, _SYNTH_COLUMNS, source)
    values: dict[str, list[object]] = {
        "player_id": _string_column(frame, "player_id", source),
        "game_id": _string_column(frame, "game_id", source),
        "half_move": _int64_column(frame, "half_move", source),
        "move_player": _string_column(frame, "move_player", source),
        "move_thinking_time": _finite_float64_column(frame, "move_thinking_time", source),
    }
    _validate_unique_values(values, _SYNTH_COLUMNS[:-1], source)
    return pa.table(
        {
            "embedding_row": pa.array(np.arange(len(frame), dtype=np.int64), type=pa.int64()),
            "player_id": pa.array(values["player_id"], type=pa.string()),
            "game_id": pa.array(values["game_id"], type=pa.string()),
            "half_move": pa.array(values["half_move"], type=pa.int64()),
            "move_player": pa.array(values["move_player"], type=pa.string()),
            "move_thinking_time": pa.array(values["move_thinking_time"], type=pa.float64()),
        }
    )


def _load_tournament_index(source: Path) -> pa.Table:
    if not source.is_file():
        raise FileNotFoundError(source)
    frame = pd.read_csv(source, dtype="string", keep_default_na=False, na_filter=False)
    _require_columns(frame, _TOURNAMENT_COLUMNS, source)
    values: dict[str, list[object]] = {
        "game_id": _string_column(frame, "game_id", source),
        "player_id": _int64_column(frame, "player_id", source),
        "half_move": _int64_column(frame, "half_move", source),
        "move_player": _string_column(frame, "move_player", source),
    }
    _validate_unique_values(values, _TOURNAMENT_COLUMNS, source)
    return pa.table(
        {
            "embedding_row": pa.array(np.arange(len(frame), dtype=np.int64), type=pa.int64()),
            "game_id": pa.array(values["game_id"], type=pa.string()),
            "player_id": pa.array(values["player_id"], type=pa.int64()),
            "half_move": pa.array(values["half_move"], type=pa.int64()),
            "move_player": pa.array(values["move_player"], type=pa.string()),
        }
    )


def _array_metadata(path: Path, expected_rows: int) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            version = np.lib.format.read_magic(stream)
            shape, _fortran_order, dtype = np.lib.format._read_array_header(stream, version)
    except Exception as error:
        raise ValueError(f"invalid NumPy array {path.name}: {error}") from error
    if dtype.hasobject:
        raise ValueError(f"array {path.name} must not use object dtype")
    if not shape:
        raise ValueError(f"array {path.name} must have a row dimension")
    if shape[0] != expected_rows:
        raise ValueError(f"array row count mismatch for {path.name}")
    return {"dtype": str(dtype), "shape": list(shape)}


def _npz_metadata(path: Path, expected_rows: int) -> dict[str, object]:
    try:
        with zipfile.ZipFile(path) as archive:
            members: dict[str, dict[str, object]] = {}
            member_filenames = archive.namelist()
            if not member_filenames:
                raise ValueError("archive has no named arrays")
            for member_filename in sorted(member_filenames):
                if not member_filename.endswith(".npy") or "/" in member_filename:
                    raise ValueError(f"invalid archive member name: {member_filename}")
                name = member_filename.removesuffix(".npy")
                if not name or name in members:
                    raise ValueError(f"duplicate or empty archive member name: {member_filename}")
                with archive.open(member_filename) as stream:
                    version = np.lib.format.read_magic(stream)
                    shape, _fortran_order, dtype = np.lib.format._read_array_header(stream, version)
                if dtype.hasobject:
                    raise ValueError(f"archive member {name} must not use object dtype")
                if not shape:
                    raise ValueError(f"archive member {name} must have a row dimension")
                if shape[0] != expected_rows:
                    raise ValueError(f"archive row count mismatch for member {name}")
                members[name] = {"dtype": str(dtype), "shape": list(shape)}
    except ValueError:
        raise
    except Exception as error:
        raise ValueError(f"invalid tournament archive {path.name}: {error}") from error
    return {"members": members}


def _validate_synth_arrays(directory: Path, expected_rows: int) -> dict[str, dict[str, object]]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    actual_names = {path.name for path in directory.iterdir()}
    expected_names = set(SYNTH_ARRAY_FILENAMES)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing:
        raise ValueError(f"missing required Synth arrays: {missing}")
    if unexpected:
        raise ValueError(f"unexpected Synth arrays: {unexpected}")
    metadata: dict[str, dict[str, object]] = {}
    for filename in SYNTH_ARRAY_FILENAMES:
        path = directory / filename
        if not path.is_file():
            raise ValueError(f"missing required Synth array: {filename}")
        metadata[filename] = _array_metadata(path, expected_rows)
    return metadata


def _file_record(path: Path, numpy_metadata: dict[str, object] | None = None) -> dict[str, object]:
    record: dict[str, object] = {"bytes": path.stat().st_size, "sha256": _sha256(path)}
    if numpy_metadata is not None:
        record["numpy"] = numpy_metadata
    return record


def _schema_record(schema: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
    return [{"name": name, "type": type_name} for name, type_name in schema]


def _semantic_row_index_sha256(table: pa.Table, columns: Sequence[str]) -> str:
    """Hash normalized row values independently of Parquet bytes and host paths."""

    selected = table.select(columns)
    digest = hashlib.sha256()
    digest.update(f"{_ROW_INDEX_DIGEST_FORMAT}\n".encode("ascii"))
    digest.update(
        json.dumps(
            [(field.name, str(field.type)) for field in selected.schema],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    for batch in selected.to_batches(max_chunksize=65536):
        materialized = [batch.column(column).to_pylist() for column in columns]
        for row_number in range(batch.num_rows):
            row: list[object] = []
            for column in materialized:
                value = column[row_number]
                row.append(
                    {"float64_hex": value.hex()}
                    if isinstance(value, float)
                    else value
                )
            digest.update(
                json.dumps(
                    row, ensure_ascii=False, separators=(",", ":")
                ).encode("utf-8")
                + b"\n"
            )
    return digest.hexdigest()


def _load_artifact_manifest(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must contain a JSON object: {manifest_path}")
    return manifest


def _observed_schema(table: pa.Table) -> list[dict[str, str]]:
    return [
        {"name": field.name, "type": "float64" if str(field.type) == "double" else str(field.type)}
        for field in table.schema
    ]


def _validate_reproducibility_metadata(manifest: dict[str, Any]) -> None:
    generated_at = manifest.get("generated_at_utc")
    if not isinstance(generated_at, str):
        raise ValueError("Allie embedding manifest generated_at_utc is required")
    try:
        generated_at_value = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Allie embedding manifest generated_at_utc must be ISO-8601 UTC") from error
    if generated_at_value.tzinfo is None or generated_at_value.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("Allie embedding manifest generated_at_utc must use UTC")
    if manifest.get("preparation_command") != _PREPARATION_COMMAND:
        raise ValueError("Allie embedding manifest has an invalid preparation command")
    if manifest.get("row_index_digest_format") != _ROW_INDEX_DIGEST_FORMAT:
        raise ValueError("Allie embedding manifest has an invalid row-index digest format")
    _validate_source_provenance(manifest.get("source_provenance"))


def _validate_source_provenance(value: object) -> None:
    required_keys = {
        "synth_manifest_csv",
        "synth_arrays",
        "tournament_manifest_csv",
        "tournament_archive",
    }
    if not isinstance(value, dict) or set(value) != required_keys:
        raise ValueError("Allie embedding manifest has incomplete source provenance")
    _validate_source_file_record(
        value["synth_manifest_csv"], "Synth manifest", normalized_rows=True
    )
    _validate_source_file_record(
        value["tournament_manifest_csv"], "tournament manifest", normalized_rows=True
    )
    _validate_source_file_record(
        value["tournament_archive"],
        "tournament archive",
        expected_filename=TOURNAMENT_ARCHIVE_FILENAME,
    )
    synth_arrays = value["synth_arrays"]
    if not isinstance(synth_arrays, dict) or set(synth_arrays) != set(SYNTH_ARRAY_FILENAMES):
        raise ValueError("Allie embedding manifest has incomplete source provenance for Synth arrays")
    for filename in SYNTH_ARRAY_FILENAMES:
        _validate_source_file_record(
            synth_arrays[filename], f"Synth array {filename}", expected_filename=filename
        )


def _validate_source_file_record(
    value: object,
    name: str,
    *,
    expected_filename: str | None = None,
    normalized_rows: bool = False,
) -> None:
    expected_keys = {"filename", "sha256"}
    if normalized_rows:
        expected_keys.add("normalized_rows_sha256")
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError(f"Allie embedding source provenance is invalid for {name}")
    filename = value.get("filename")
    checksum = value.get("sha256")
    if (
        not isinstance(filename, str)
        or not filename
        or filename in {".", ".."}
        or Path(filename).name != filename
        or "\\" in filename
        or (expected_filename is not None and filename != expected_filename)
        or not isinstance(checksum, str)
        or _SHA256_PATTERN.fullmatch(checksum) is None
    ):
        raise ValueError(f"Allie embedding source provenance is invalid for {name}")
    if normalized_rows:
        normalized_checksum = value.get("normalized_rows_sha256")
        if (
            not isinstance(normalized_checksum, str)
            or _SHA256_PATTERN.fullmatch(normalized_checksum) is None
        ):
            raise ValueError(f"Allie embedding source provenance is invalid for {name}")


def _validate_row_index(
    root: Path,
    name: str,
    actual: object,
    source_semantic_sha256: object | None = None,
) -> int:
    expected = _ROW_INDEX_CONTRACTS[name]
    expected_schema = _schema_record(expected["schema"])
    if not isinstance(actual, dict) or actual.get("path") != expected["path"]:
        raise ValueError(f"Allie embedding manifest has invalid {name} row-index path")
    if actual.get("schema") != expected_schema:
        raise ValueError(f"Allie embedding manifest has invalid {name} row-index schema")
    if actual.get("join_key") != list(expected["join_key"]):
        raise ValueError(f"Allie embedding manifest has invalid {name} join key")
    table = pq.read_table(root / expected["path"])
    if _observed_schema(table) != expected_schema:
        raise ValueError(f"Allie embedding {name} row-index schema mismatch")
    if actual.get("count") != table.num_rows:
        raise ValueError(f"Allie embedding {name} row-index count mismatch")
    embedding_rows = table.column("embedding_row").to_pylist()
    if embedding_rows != list(range(table.num_rows)):
        raise ValueError(f"Allie embedding {name} embedding_row must be zero-based and contiguous")
    unique_key_count = table.group_by(list(expected["join_key"])).aggregate([]).num_rows
    if unique_key_count != table.num_rows:
        raise ValueError(f"Allie embedding {name} row-index has a duplicate join key")
    if name == "synth":
        thinking_times = np.asarray(table.column("move_thinking_time").to_pylist(), dtype=float)
        if not np.isfinite(thinking_times).all():
            raise ValueError("Allie embedding synth row-index has non-finite thinking times")
    if source_semantic_sha256 is not None:
        semantic_columns = tuple(
            column
            for column, _type in expected["schema"]
            if column != "embedding_row"
        )
        semantic_sha256 = _semantic_row_index_sha256(table, semantic_columns)
        if (
            actual.get("semantic_sha256") != semantic_sha256
            or source_semantic_sha256 != semantic_sha256
        ):
            raise ValueError(
                f"Allie embedding {name} row-index disagrees with source provenance"
            )
    return table.num_rows


def _validate_source_payload_bindings(
    source_provenance: dict[str, Any], files: dict[str, Any]
) -> None:
    for filename in SYNTH_ARRAY_FILENAMES:
        if (
            source_provenance["synth_arrays"][filename]["sha256"]
            != files[f"synth/{filename}"]["sha256"]
        ):
            raise ValueError(
                f"Allie embedding source provenance disagrees with copied payload synth/{filename}"
            )
    tournament_path = f"tournament/{TOURNAMENT_ARCHIVE_FILENAME}"
    if source_provenance["tournament_archive"]["sha256"] != files[tournament_path]["sha256"]:
        raise ValueError(
            f"Allie embedding source provenance disagrees with copied payload {tournament_path}"
        )


def validate_allie_embeddings(root: Path | str) -> dict[str, Any]:
    """Fail closed unless every Allie embedding payload and alignment contract agrees."""

    artifact_root = Path(root)
    manifest = _load_artifact_manifest(artifact_root)
    if manifest.get("artifact_contract_version") != ARTIFACT_CONTRACT_VERSION:
        raise ValueError("unsupported Allie embedding artifact contract version")
    if manifest.get("dataset_repository") != DATASET_REPOSITORY:
        raise ValueError("Allie embedding manifest has an unexpected dataset repository")
    _validate_reproducibility_metadata(manifest)
    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != _ALLIE_EMBEDDING_OUTPUT_FILES:
        raise ValueError("Allie embedding artifact inventory does not match the allowlist")
    actual_files = {
        str(path.relative_to(artifact_root))
        for path in artifact_root.rglob("*")
        if path.is_file() and path != artifact_root / "manifest.json"
    }
    if actual_files != _ALLIE_EMBEDDING_OUTPUT_FILES:
        raise ValueError("Allie embedding artifact tree contains missing or unexpected files")
    for relative_path in sorted(_ALLIE_EMBEDDING_OUTPUT_FILES):
        path = artifact_root / relative_path
        entry = files[relative_path]
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"Allie embedding manifest has no checksum for {relative_path}")
        if entry.get("bytes") != path.stat().st_size or _sha256(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {relative_path}")

    source_provenance = manifest["source_provenance"]
    _validate_source_payload_bindings(source_provenance, files)

    row_indexes = manifest.get("row_indexes")
    if not isinstance(row_indexes, dict) or set(row_indexes) != set(_ROW_INDEX_CONTRACTS):
        raise ValueError("Allie embedding manifest has an invalid row-index contract")
    row_counts = {
        name: _validate_row_index(
            artifact_root,
            name,
            row_indexes[name],
            source_provenance[f"{name}_manifest_csv"]["normalized_rows_sha256"],
        )
        for name in _ROW_INDEX_CONTRACTS
    }
    for filename in SYNTH_ARRAY_FILENAMES:
        relative_path = f"synth/{filename}"
        metadata = _array_metadata(artifact_root / relative_path, row_counts["synth"])
        if files[relative_path].get("numpy") != metadata:
            raise ValueError(f"Allie embedding manifest NumPy metadata mismatch for {relative_path}")
    tournament_path = f"tournament/{TOURNAMENT_ARCHIVE_FILENAME}"
    tournament_metadata = _npz_metadata(
        artifact_root / tournament_path, row_counts["tournament"]
    )
    if files[tournament_path].get("numpy") != tournament_metadata:
        raise ValueError(f"Allie embedding manifest NumPy metadata mismatch for {tournament_path}")
    return manifest


def _prepare_allie_embeddings_unlocked(
    *,
    synth_manifest_csv: Path,
    synth_arrays_dir: Path,
    tournament_manifest_csv: Path,
    tournament_archive: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create the validated Allie embedding tree with byte-identical NumPy payloads."""

    synth_manifest = Path(synth_manifest_csv)
    synth_arrays = Path(synth_arrays_dir)
    tournament_manifest = Path(tournament_manifest_csv)
    archive = Path(tournament_archive)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")

    synth_index = _load_synth_index(synth_manifest)
    tournament_index = _load_tournament_index(tournament_manifest)
    synth_metadata = _validate_synth_arrays(synth_arrays, synth_index.num_rows)
    if not archive.is_file():
        raise FileNotFoundError(archive)
    if archive.name != TOURNAMENT_ARCHIVE_FILENAME:
        raise ValueError(
            f"tournament archive filename must be {TOURNAMENT_ARCHIVE_FILENAME}: {archive.name}"
        )
    tournament_metadata = _npz_metadata(archive, tournament_index.num_rows)
    synth_semantic_sha256 = _semantic_row_index_sha256(
        synth_index, _SYNTH_COLUMNS
    )
    tournament_semantic_sha256 = _semantic_row_index_sha256(
        tournament_index, _TOURNAMENT_COLUMNS
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for filename in SYNTH_ARRAY_FILENAMES:
            source = synth_arrays / filename
            destination = temporary / "synth" / filename
            _copy_file(source, destination)
            if _sha256(source) != _sha256(destination):
                raise ValueError(f"checksum mismatch after copy for {filename}")
        archive_destination = temporary / "tournament" / TOURNAMENT_ARCHIVE_FILENAME
        _copy_file(archive, archive_destination)
        if _sha256(archive) != _sha256(archive_destination):
            raise ValueError("checksum mismatch after copy for tournament archive")

        synth_index_path = temporary / "synth" / "row_index.parquet"
        tournament_index_path = temporary / "tournament" / "row_index.parquet"
        pq.write_table(synth_index, synth_index_path)
        pq.write_table(tournament_index, tournament_index_path)

        files: dict[str, dict[str, object]] = {}
        for filename in SYNTH_ARRAY_FILENAMES:
            files[f"synth/{filename}"] = _file_record(
                temporary / "synth" / filename, synth_metadata[filename]
            )
        files["synth/row_index.parquet"] = _file_record(synth_index_path)
        files[f"tournament/{TOURNAMENT_ARCHIVE_FILENAME}"] = _file_record(
            archive_destination, tournament_metadata
        )
        files["tournament/row_index.parquet"] = _file_record(tournament_index_path)
        manifest: dict[str, object] = {
            "artifact_contract_version": ARTIFACT_CONTRACT_VERSION,
            "dataset_repository": DATASET_REPOSITORY,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_index_digest_format": _ROW_INDEX_DIGEST_FORMAT,
            "files": files,
            "row_indexes": {
                "synth": {
                    "path": "synth/row_index.parquet",
                    "count": synth_index.num_rows,
                    "schema": _schema_record(_SYNTH_INDEX_SCHEMA),
                    "join_key": ["player_id", "game_id", "half_move", "move_player"],
                    "semantic_sha256": synth_semantic_sha256,
                },
                "tournament": {
                    "path": "tournament/row_index.parquet",
                    "count": tournament_index.num_rows,
                    "schema": _schema_record(_TOURNAMENT_INDEX_SCHEMA),
                    "join_key": ["game_id", "player_id", "half_move", "move_player"],
                    "semantic_sha256": tournament_semantic_sha256,
                },
            },
            "source_provenance": {
                "synth_manifest_csv": {
                    "filename": synth_manifest.name,
                    "sha256": _sha256(synth_manifest),
                    "normalized_rows_sha256": synth_semantic_sha256,
                },
                "synth_arrays": {
                    filename: {
                        "filename": filename,
                        "sha256": files[f"synth/{filename}"]["sha256"],
                    }
                    for filename in SYNTH_ARRAY_FILENAMES
                },
                "tournament_manifest_csv": {
                    "filename": tournament_manifest.name,
                    "sha256": _sha256(tournament_manifest),
                    "normalized_rows_sha256": tournament_semantic_sha256,
                },
                "tournament_archive": {
                    "filename": archive.name,
                    "sha256": files[f"tournament/{TOURNAMENT_ARCHIVE_FILENAME}"]["sha256"],
                },
            },
            "preparation_command": _PREPARATION_COMMAND,
        }
        with (temporary / "manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        validate_allie_embeddings(temporary)
        if output.exists():
            raise FileExistsError(f"output directory already exists: {output}")
        temporary.replace(output)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_allie_embeddings(
    *,
    synth_manifest_csv: Path,
    synth_arrays_dir: Path,
    tournament_manifest_csv: Path,
    tournament_archive: Path,
    output_dir: Path,
) -> dict[str, object]:
    """Create the validated Allie embedding tree with byte-identical NumPy payloads."""

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        return _prepare_allie_embeddings_unlocked(
            synth_manifest_csv=synth_manifest_csv,
            synth_arrays_dir=synth_arrays_dir,
            tournament_manifest_csv=tournament_manifest_csv,
            tournament_archive=tournament_archive,
            output_dir=output,
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the public Allie embedding artifact bundle.")
    parser.add_argument("--synth-manifest-csv", type=Path, required=True)
    parser.add_argument("--synth-arrays-dir", type=Path, required=True)
    parser.add_argument("--tournament-manifest-csv", type=Path, required=True)
    parser.add_argument("--tournament-archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    prepare_allie_embeddings(
        synth_manifest_csv=args.synth_manifest_csv,
        synth_arrays_dir=args.synth_arrays_dir,
        tournament_manifest_csv=args.tournament_manifest_csv,
        tournament_archive=args.tournament_archive,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
