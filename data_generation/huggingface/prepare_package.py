"""Assemble validated ChessFraud releases into a Hugging Face repository layout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

from data_generation.huggingface.prepare_allie_embeddings import (
    _output_lock,
    validate_allie_embeddings,
)


_COMPONENT_FILES = {
    "chess_fraud": {"chess_fraud.parquet": "data/chess_fraud/full.parquet"},
    "chess_fraud_synth": {
        "train.parquet": "data/chess_fraud_synth/train.parquet",
        "test.parquet": "data/chess_fraud_synth/test.parquet",
    },
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(release_directory: Path) -> dict[str, Any]:
    manifest_path = release_directory / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"manifest must contain a JSON object: {manifest_path}")
    return manifest


def _validate_component(
    release_directory: Path, manifest: dict[str, Any], filenames: Sequence[str]
) -> None:
    files = manifest.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"manifest has no file inventory: {release_directory}")
    for filename in filenames:
        path = release_directory / filename
        entry = files.get(filename)
        if not path.is_file():
            raise FileNotFoundError(path)
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError(f"manifest has no checksum for {filename}")
        if _sha256(path) != entry["sha256"]:
            raise ValueError(f"checksum mismatch for {filename}")


def _validate_allie_embeddings(root: Path) -> dict[str, Any]:
    return validate_allie_embeddings(root)


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _file_record(path: Path) -> dict[str, object]:
    return {"bytes": path.stat().st_size, "sha256": _sha256(path)}


def _prepare_package_unlocked(
    *,
    tournament_release_dir: Path | str,
    synth_release_dir: Path | str,
    card_path: Path | str,
    allie_embeddings_dir: Path | str,
    output_dir: Path | str,
    dataset_version: str,
) -> dict[str, Any]:
    """Create the unchanged core package plus the validated Allie embedding subtree."""

    tournament_directory = Path(tournament_release_dir)
    synth_directory = Path(synth_release_dir)
    card = Path(card_path)
    embedding_root = Path(allie_embeddings_dir)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(f"output directory already exists: {output}")
    if not dataset_version.strip():
        raise ValueError("dataset_version must not be empty")
    if not card.is_file():
        raise FileNotFoundError(card)

    component_directories = {"chess_fraud": tournament_directory, "chess_fraud_synth": synth_directory}
    component_manifests = {name: _load_manifest(path) for name, path in component_directories.items()}
    for name, mappings in _COMPONENT_FILES.items():
        _validate_component(component_directories[name], component_manifests[name], tuple(mappings))
    allie_manifest = _validate_allie_embeddings(embedding_root)

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        _copy_file(card, temporary / "README.md")
        for name, mappings in _COMPONENT_FILES.items():
            source_directory = component_directories[name]
            for source_name, destination_name in mappings.items():
                _copy_file(source_directory / source_name, temporary / destination_name)
            _copy_file(source_directory / "manifest.json", temporary / "manifests" / f"{name}.json")
        for source in sorted(embedding_root.rglob("*")):
            if source.is_file():
                _copy_file(source, temporary / "artifacts" / "allie_embeddings" / source.relative_to(embedding_root))

        package_files = {
            str(path.relative_to(temporary)): _file_record(path)
            for path in sorted(temporary.rglob("*"))
            if path.is_file()
        }
        package_manifest: dict[str, Any] = {
            "dataset_name": "ChessFraud",
            "dataset_version": dataset_version,
            "component_versions": {
                name: manifest.get("dataset_version") for name, manifest in component_manifests.items()
            },
            "configurations": {
                "chess_fraud": {"full": "data/chess_fraud/full.parquet"},
                "chess_fraud_synth": {
                    "train": "data/chess_fraud_synth/train.parquet",
                    "test": "data/chess_fraud_synth/test.parquet",
                },
            },
            "allie_embedding_contract_version": allie_manifest["artifact_contract_version"],
            "files": package_files,
        }
        with (temporary / "release_manifest.json").open("w", encoding="utf-8") as stream:
            json.dump(package_manifest, stream, indent=2, sort_keys=True)
            stream.write("\n")
        copied_allie_manifest = _validate_allie_embeddings(
            temporary / "artifacts" / "allie_embeddings"
        )
        if copied_allie_manifest != allie_manifest:
            raise ValueError("Allie embedding artifacts changed during package assembly")
        if output.exists():
            raise FileExistsError(f"output directory already exists: {output}")
        temporary.replace(output)
        return package_manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def prepare_package(
    *,
    tournament_release_dir: Path | str,
    synth_release_dir: Path | str,
    card_path: Path | str,
    allie_embeddings_dir: Path | str,
    output_dir: Path | str,
    dataset_version: str,
) -> dict[str, Any]:
    """Create the unchanged core package plus the validated Allie embedding subtree."""

    output = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        return _prepare_package_unlocked(
            tournament_release_dir=tournament_release_dir,
            synth_release_dir=synth_release_dir,
            card_path=card_path,
            allie_embeddings_dir=allie_embeddings_dir,
            output_dir=output,
            dataset_version=dataset_version,
        )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Assemble the combined ChessFraud package.")
    parser.add_argument("--tournament-release-dir", type=Path, required=True)
    parser.add_argument("--synth-release-dir", type=Path, required=True)
    parser.add_argument("--card", type=Path, required=True)
    parser.add_argument("--allie-embeddings-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dataset-version", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    prepare_package(
        tournament_release_dir=args.tournament_release_dir,
        synth_release_dir=args.synth_release_dir,
        card_path=args.card,
        allie_embeddings_dir=args.allie_embeddings_dir,
        output_dir=args.output_dir,
        dataset_version=args.dataset_version,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
