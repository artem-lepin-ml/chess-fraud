"""Resolve and validate the immutable Allie embedding artifact bundle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import zipfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import snapshot_download


DATASET_ID = "artemlepin/chess-fraud"
PUBLIC_DATASET_REVISION = "bd2804f268bf07c306217929db9b8dda5803392b"
ARTIFACT_SUBTREE = Path("artifacts/allie_embeddings")
ARTIFACT_CONTRACT_VERSION = "allie-embeddings-v1"
ALLOW_PATTERNS = ["artifacts/allie_embeddings/**"]
_FULL_SHA_PATTERN = re.compile(r"^[0-9a-fA-F]{40}$")
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
_SYNTH_ARRAYS = (
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
_EXPECTED_FILES = frozenset(
    {
        *(f"synth/{filename}" for filename in _SYNTH_ARRAYS),
        "synth/row_index.parquet",
        "tournament/embs_allie_2500.npz",
        "tournament/row_index.parquet",
    }
)
_ROW_INDEX_CONTRACTS = {
    "synth": {
        "path": "synth/row_index.parquet",
        "schema": (
            ("embedding_row", "int64"),
            ("player_id", "string"),
            ("game_id", "string"),
            ("half_move", "int64"),
            ("move_player", "string"),
            ("move_thinking_time", "float64"),
        ),
        "join_key": ("player_id", "game_id", "half_move", "move_player"),
        "finite_columns": ("move_thinking_time",),
    },
    "tournament": {
        "path": "tournament/row_index.parquet",
        "schema": (
            ("embedding_row", "int64"),
            ("game_id", "string"),
            ("player_id", "int64"),
            ("half_move", "int64"),
            ("move_player", "string"),
        ),
        "join_key": ("game_id", "player_id", "half_move", "move_player"),
    },
}


def resolve_allie_embedding_root(
    *, revision: str, environ: Mapping[str, str] | None = None
) -> Path:
    """Return a validated Allie embedding root from an override or pinned snapshot.

    A caller must supply a full immutable Hugging Face commit SHA.  A local
    override is deliberately validated before it is returned, preventing a
    half-copied or stale artifact directory from silently entering a tutorial.
    """
    if not _FULL_SHA_PATTERN.fullmatch(revision):
        raise ValueError(
            "Allie embedding artifact revision must be a full 40-character hexadecimal SHA."
        )

    environment = os.environ if environ is None else environ
    override = environment.get("CHESSFRAUD_ALLIE_EMBEDDING_DIR")
    if override:
        return _validate_allie_embedding_root(Path(override))

    print(
        "Allie embedding artifacts require approximately 9.56 GB plus cache overhead. "
        "Set CHESSFRAUD_ALLIE_EMBEDDING_DIR to use an existing validated copy or "
        "CHESSFRAUD_ALLIE_CACHE_DIR to choose the download cache."
    )
    download_kwargs: dict[str, Any] = {
        "repo_id": DATASET_ID,
        "repo_type": "dataset",
        "revision": revision,
        "allow_patterns": ALLOW_PATTERNS,
    }
    cache_dir = environment.get("CHESSFRAUD_ALLIE_CACHE_DIR")
    if cache_dir:
        download_kwargs["cache_dir"] = cache_dir
    try:
        snapshot_root = Path(snapshot_download(**download_kwargs))
    except Exception as error:
        raise RuntimeError(
            "Unable to download the pinned Allie embedding artifacts from "
            f"{DATASET_ID} at {revision}. Check network/authentication or set "
            "CHESSFRAUD_ALLIE_EMBEDDING_DIR to a validated local snapshot."
        ) from error

    try:
        return _validate_allie_embedding_root(snapshot_root)
    except ValueError as error:
        raise RuntimeError(
            "Downloaded Allie embedding artifacts failed validation; remove the bad cache "
            "or set CHESSFRAUD_ALLIE_EMBEDDING_DIR to a validated local snapshot. "
            f"Details: {error}"
        ) from error



def _validate_allie_embedding_root(snapshot_root: Path) -> Path:
    artifact_root = snapshot_root / ARTIFACT_SUBTREE
    manifest_path = artifact_root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Allie embedding artifact contract is missing {manifest_path}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"Allie embedding artifact manifest is unreadable at {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict):
        raise ValueError("Allie embedding artifact manifest must be a JSON object.")
    if manifest.get("artifact_contract_version") != ARTIFACT_CONTRACT_VERSION:
        raise ValueError(
            "Allie embedding artifact manifest has an unsupported artifact contract version."
        )
    if manifest.get("dataset_repository") != DATASET_ID:
        raise ValueError("Allie embedding artifact manifest names an unexpected dataset repository.")
    _validate_reproducibility_metadata(manifest)

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != _EXPECTED_FILES:
        raise ValueError("Allie embedding artifact manifest file allowlist does not match allie-embeddings-v1.")
    actual_files = {
        path.relative_to(artifact_root).as_posix()
        for path in artifact_root.rglob("*")
        if path.is_file() and path != manifest_path
    }
    if actual_files != _EXPECTED_FILES:
        raise ValueError("Allie embedding artifact directory has missing or unexpected payload files.")
    for relative_path in sorted(_EXPECTED_FILES):
        _validate_file(artifact_root, relative_path, files[relative_path])

    source_provenance = manifest["source_provenance"]
    _validate_source_payload_bindings(source_provenance, files)
    row_index_counts = _validate_row_indexes(
        artifact_root, manifest.get("row_indexes"), source_provenance
    )
    _validate_array_metadata(artifact_root, files, row_index_counts)
    return artifact_root


def _validate_file(artifact_root: Path, relative_path: str, metadata: object) -> None:
    if not isinstance(metadata, dict):
        raise ValueError(f"Allie embedding artifact manifest metadata is invalid for {relative_path}.")
    expected_size = metadata.get("bytes")
    expected_checksum = metadata.get("sha256")
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 0
        or not isinstance(expected_checksum, str)
        or _SHA256_PATTERN.fullmatch(expected_checksum) is None
    ):
        raise ValueError(f"Allie embedding artifact manifest checksum metadata is invalid for {relative_path}.")
    payload_path = artifact_root / relative_path
    if payload_path.stat().st_size != expected_size:
        raise ValueError(f"Allie embedding artifact byte size mismatch for {relative_path}.")
    if _sha256(payload_path) != expected_checksum:
        raise ValueError(f"Allie embedding artifact checksum mismatch for {relative_path}.")


def _validate_reproducibility_metadata(manifest: Mapping[str, object]) -> None:
    generated_at = manifest.get("generated_at_utc")
    if not isinstance(generated_at, str):
        raise ValueError("Allie embedding artifact manifest generated_at_utc is required.")
    try:
        generated_at_value = datetime.fromisoformat(generated_at.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("Allie embedding artifact manifest generated_at_utc must be ISO-8601 UTC.") from error
    if generated_at_value.tzinfo is None or generated_at_value.utcoffset() != timezone.utc.utcoffset(None):
        raise ValueError("Allie embedding artifact manifest generated_at_utc must use UTC.")
    if manifest.get("preparation_command") != _PREPARATION_COMMAND:
        raise ValueError("Allie embedding artifact manifest preparation command does not match allie-embeddings-v1.")
    if manifest.get("row_index_digest_format") != _ROW_INDEX_DIGEST_FORMAT:
        raise ValueError("Allie embedding artifact manifest row-index digest format does not match allie-embeddings-v1.")
    _validate_source_provenance(manifest.get("source_provenance"))


def _validate_source_provenance(value: object) -> None:
    if not isinstance(value, dict) or set(value) != {
        "synth_manifest_csv",
        "synth_arrays",
        "tournament_manifest_csv",
        "tournament_archive",
    }:
        raise ValueError("Allie embedding artifact manifest source provenance is incomplete.")
    _validate_source_file_record(
        value["synth_manifest_csv"], "synth manifest", normalized_rows=True
    )
    _validate_source_file_record(
        value["tournament_manifest_csv"], "tournament manifest", normalized_rows=True
    )
    _validate_source_file_record(
        value["tournament_archive"],
        "tournament archive",
        expected_filename="embs_allie_2500.npz",
    )
    synth_arrays = value["synth_arrays"]
    if not isinstance(synth_arrays, dict) or set(synth_arrays) != set(_SYNTH_ARRAYS):
        raise ValueError("Allie embedding artifact manifest source provenance is incomplete for Synth arrays.")
    for filename in _SYNTH_ARRAYS:
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
        raise ValueError(f"Allie embedding artifact source provenance is invalid for {name}.")
    filename = value.get("filename")
    checksum = value.get("sha256")
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or (expected_filename is not None and filename != expected_filename)
        or not isinstance(checksum, str)
        or _SHA256_PATTERN.fullmatch(checksum) is None
    ):
        raise ValueError(f"Allie embedding artifact source provenance is invalid for {name}.")
    if normalized_rows:
        normalized_checksum = value.get("normalized_rows_sha256")
        if (
            not isinstance(normalized_checksum, str)
            or _SHA256_PATTERN.fullmatch(normalized_checksum) is None
        ):
            raise ValueError(f"Allie embedding artifact source provenance is invalid for {name}.")


def _validate_source_payload_bindings(
    source_provenance: Mapping[str, object], files: Mapping[str, object]
) -> None:
    synth_arrays = source_provenance["synth_arrays"]
    if not isinstance(synth_arrays, dict):
        raise ValueError("Allie embedding artifact source provenance is incomplete for Synth arrays.")
    for filename in _SYNTH_ARRAYS:
        provenance = synth_arrays[filename]
        payload = files[f"synth/{filename}"]
        if (
            not isinstance(provenance, dict)
            or not isinstance(payload, dict)
            or provenance.get("sha256") != payload.get("sha256")
        ):
            raise ValueError(
                f"Allie embedding artifact source provenance disagrees with payload synth/{filename}."
            )
    archive_provenance = source_provenance["tournament_archive"]
    archive_payload = files["tournament/embs_allie_2500.npz"]
    if (
        not isinstance(archive_provenance, dict)
        or not isinstance(archive_payload, dict)
        or archive_provenance.get("sha256") != archive_payload.get("sha256")
    ):
        raise ValueError(
            "Allie embedding artifact source provenance disagrees with payload "
            "tournament/embs_allie_2500.npz."
        )


def _validate_row_indexes(
    artifact_root: Path,
    row_indexes: object,
    source_provenance: Mapping[str, object],
) -> dict[str, int]:
    if not isinstance(row_indexes, dict) or set(row_indexes) != set(_ROW_INDEX_CONTRACTS):
        raise ValueError("Allie embedding artifact manifest has invalid row-index definitions.")
    counts: dict[str, int] = {}
    for name, expected in _ROW_INDEX_CONTRACTS.items():
        metadata = row_indexes[name]
        if not isinstance(metadata, dict):
            raise ValueError(f"Allie embedding {name} row-index metadata must be an object.")
        if metadata.get("path") != expected["path"]:
            raise ValueError(f"Allie embedding {name} row-index path does not match allie-embeddings-v1.")
        if tuple(metadata.get("join_key", ())) != expected["join_key"]:
            raise ValueError(f"Allie embedding {name} row-index join key does not match allie-embeddings-v1.")
        manifest_schema = _schema_from_manifest(metadata.get("schema"), name)
        if manifest_schema != expected["schema"]:
            raise ValueError(f"Allie embedding {name} row-index manifest schema does not match allie-embeddings-v1.")
        parquet_path = artifact_root / expected["path"]
        parquet_file = _open_parquet(parquet_path, expected["path"])
        parquet_schema = tuple(
            (field.name, _arrow_type_label(str(field.type)))
            for field in parquet_file.schema_arrow
        )
        if parquet_schema != expected["schema"]:
            raise ValueError(f"Allie embedding {name} row-index Parquet schema does not match allie-embeddings-v1.")
        count = metadata.get("count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"Allie embedding {name} row-index count is invalid.")
        if parquet_file.metadata.num_rows != count:
            raise ValueError(f"Allie embedding {name} row-index row count does not match the manifest.")
        semantic_sha256 = _validate_row_index_contents(parquet_file, name, expected, count)
        source_record = source_provenance[f"{name}_manifest_csv"]
        source_semantic_sha256 = (
            source_record.get("normalized_rows_sha256")
            if isinstance(source_record, dict)
            else None
        )
        if (
            metadata.get("semantic_sha256") != semantic_sha256
            or source_semantic_sha256 != semantic_sha256
        ):
            raise ValueError(
                f"Allie embedding {name} row-index disagrees with source provenance."
            )
        counts[name] = count
    return counts


def _open_parquet(parquet_path: Path, relative_path: str) -> pq.ParquetFile:
    try:
        return pq.ParquetFile(parquet_path)
    except (OSError, pa.ArrowException) as error:
        raise ValueError(
            f"Allie embedding row-index Parquet file {relative_path} is unreadable: {error}"
        ) from error


def _validate_row_index_contents(
    parquet_file: pq.ParquetFile,
    name: str,
    expected: Mapping[str, object],
    count: int,
) -> str:
    join_key = expected["join_key"]
    if not isinstance(join_key, tuple):
        raise RuntimeError("Allie embedding row-index contract is invalid.")
    seen_embedding_rows = bytearray(count)
    seen_join_keys: set[tuple[object, ...]] = set()
    observed_rows = 0
    semantic_columns = tuple(
        column for column, _type_name in expected["schema"] if column != "embedding_row"
    )
    columns = ("embedding_row", *semantic_columns)
    digest = hashlib.sha256()
    digest.update(f"{_ROW_INDEX_DIGEST_FORMAT}\n".encode("ascii"))
    schema_by_name = {field.name: field for field in parquet_file.schema_arrow}
    digest.update(
        json.dumps(
            [
                (column, str(schema_by_name[column].type))
                for column in semantic_columns
            ],
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        + b"\n"
    )
    try:
        for batch in parquet_file.iter_batches(columns=columns, batch_size=65536):
            embedding_rows = batch.column("embedding_row").to_pylist()
            semantic_values = {
                column: batch.column(column).to_pylist() for column in semantic_columns
            }
            for index, embedding_row in enumerate(embedding_rows):
                if (
                    isinstance(embedding_row, bool)
                    or not isinstance(embedding_row, int)
                    or embedding_row < 0
                    or embedding_row >= count
                    or seen_embedding_rows[embedding_row]
                ):
                    raise ValueError(
                        f"Allie embedding {name} row-index embedding_row values must be unique and contiguous starting at zero."
                    )
                seen_embedding_rows[embedding_row] = 1
                key = tuple(semantic_values[column][index] for column in join_key)
                if any(value is None for value in key):
                    raise ValueError(f"Allie embedding {name} row-index join keys must not contain nulls.")
                if key in seen_join_keys:
                    raise ValueError(f"Allie embedding {name} row-index join keys must be unique.")
                seen_join_keys.add(key)
                for column in expected.get("finite_columns", ()):
                    value = semantic_values[column][index]
                    if value is None or not np.isfinite(value):
                        raise ValueError(
                            f"Allie embedding {name} row-index {column} values must be finite."
                        )
                row: list[object] = []
                for column in semantic_columns:
                    value = semantic_values[column][index]
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
                observed_rows += 1
    except (OSError, pa.ArrowException) as error:
        relative_path = expected["path"]
        raise ValueError(
            f"Allie embedding row-index Parquet file {relative_path} is unreadable: {error}"
        ) from error
    if observed_rows != count or not all(seen_embedding_rows):
        raise ValueError(
            f"Allie embedding {name} row-index embedding_row values must be unique and contiguous starting at zero."
        )
    return digest.hexdigest()


def _validate_array_metadata(
    artifact_root: Path, files: Mapping[str, object], row_index_counts: Mapping[str, int]
) -> None:
    synth_count = row_index_counts["synth"]
    for filename in _SYNTH_ARRAYS:
        relative_path = f"synth/{filename}"
        _validate_npy_metadata(
            artifact_root / relative_path,
            files[relative_path],
            relative_path,
            expected_rows=synth_count,
        )
    tournament_path = "tournament/embs_allie_2500.npz"
    _validate_npz_metadata(
        artifact_root / tournament_path,
        files[tournament_path],
        tournament_path,
        expected_rows=row_index_counts["tournament"],
    )


def _validate_npy_metadata(
    path: Path, file_metadata: object, relative_path: str, *, expected_rows: int
) -> None:
    metadata = file_metadata.get("numpy") if isinstance(file_metadata, dict) else None
    expected = _manifest_array_metadata(metadata, relative_path)
    try:
        array = np.load(path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Allie embedding NumPy metadata is unreadable for {relative_path}: {error}") from error
    actual = {"dtype": str(array.dtype), "shape": list(array.shape)}
    if actual != expected:
        raise ValueError(f"Allie embedding NumPy metadata does not match {relative_path}.")
    _validate_array_row_count(actual, relative_path, expected_rows)


def _validate_npz_metadata(
    path: Path, file_metadata: object, relative_path: str, *, expected_rows: int
) -> None:
    metadata = file_metadata.get("numpy") if isinstance(file_metadata, dict) else None
    if not isinstance(metadata, dict) or set(metadata) != {"members"}:
        raise ValueError(f"Allie embedding NumPy metadata is invalid for {relative_path}.")
    members = metadata["members"]
    if not isinstance(members, dict) or not members:
        raise ValueError(f"Allie embedding NumPy metadata is invalid for {relative_path}.")
    expected = {
        name: _manifest_array_metadata(member, f"{relative_path}:{name}")
        for name, member in members.items()
        if isinstance(name, str)
    }
    if len(expected) != len(members):
        raise ValueError(f"Allie embedding NumPy metadata is invalid for {relative_path}.")
    try:
        with zipfile.ZipFile(path) as archive:
            member_names = archive.namelist()
            if any(not member_name.endswith(".npy") for member_name in member_names):
                raise ValueError("NPZ archive contains a non-array member")
            actual = {
                member_name[:-4]: _npy_header_metadata(archive.open(member_name))
                for member_name in member_names
            }
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        raise ValueError(f"Allie embedding NumPy metadata is unreadable for {relative_path}: {error}") from error
    if actual != expected:
        raise ValueError(f"Allie embedding NumPy metadata does not match {relative_path}.")
    for member_name, member_metadata in actual.items():
        _validate_array_row_count(member_metadata, f"{relative_path}:{member_name}", expected_rows)


def _validate_array_row_count(
    metadata: Mapping[str, object], name: str, expected_rows: int
) -> None:
    shape = metadata["shape"]
    if not isinstance(shape, list) or not shape or shape[0] != expected_rows:
        raise ValueError(f"Allie embedding NumPy metadata row count does not match {name}.")


def _manifest_array_metadata(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != {"dtype", "shape"}:
        raise ValueError(f"Allie embedding NumPy metadata is invalid for {name}.")
    dtype = value.get("dtype")
    shape = value.get("shape")
    if (
        not isinstance(dtype, str)
        or not isinstance(shape, list)
        or any(isinstance(dimension, bool) or not isinstance(dimension, int) or dimension < 0 for dimension in shape)
    ):
        raise ValueError(f"Allie embedding NumPy metadata is invalid for {name}.")
    return {"dtype": dtype, "shape": shape}


def _npy_header_metadata(stream: BinaryIO) -> dict[str, object]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, _fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"unsupported NumPy array version {version}")
    return {"dtype": str(dtype), "shape": list(shape)}


def _schema_from_manifest(value: object, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(value, list):
        raise ValueError(f"Allie embedding {name} row-index manifest schema must be a list.")
    result: list[tuple[str, str]] = []
    for field in value:
        if not isinstance(field, dict):
            raise ValueError(f"Allie embedding {name} row-index schema contains an invalid field.")
        field_name = field.get("name")
        field_type = field.get("type")
        if not isinstance(field_name, str) or not isinstance(field_type, str):
            raise ValueError(f"Allie embedding {name} row-index schema contains an invalid field.")
        result.append((field_name, field_type))
    return tuple(result)


def _arrow_type_label(value: str) -> str:
    return "float64" if value == "double" else value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()
