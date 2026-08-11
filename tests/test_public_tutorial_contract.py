from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from experiments.move_level.allie_embeddings import PUBLIC_DATASET_REVISION


REPO_ROOT = Path(__file__).resolve().parents[1]
MOVE_LEVEL_NOTEBOOK = (
    REPO_ROOT
    / "experiments/move_level/tutorial_reproduce_move_level_experiments.ipynb"
)
BASELINE_NOTEBOOK = (
    REPO_ROOT
    / "experiments/analisys/tutorial_transfer_synth_to_tournament.ipynb"
)
OLD_BASELINE_PATH = (
    REPO_ROOT
    / "experiments/move_level/tutorial_transfer_synth_to_tournament.ipynb"
)
REQUIREMENTS = REPO_ROOT / "requirements/tutorials.txt"
README = REPO_ROOT / "README.md"
HF_CARD = REPO_ROOT / "huggingface/README.md"

FINAL_REVISION = PUBLIC_DATASET_REVISION
DOI = "https://doi.org/10.1145/3770855.3817587"
MOVE_LEVEL_STABLE_OUTPUTS_SHA256 = (
    "88001dca74173652bb35df0b946347aaf53554d56c33604255016c88267a0ef0"
)
BASELINE_OUTPUTS_SHA256 = (
    "68486a37128ce3858a7b44912d4db2efbbf429ca458fa51e93ce17773ee49634"
)


def _notebook(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sources(notebook: dict) -> str:
    return "\n".join("".join(cell.get("source", [])) for cell in notebook["cells"])


def _outputs_digest(notebook: dict, *, first_cell: int = 0) -> str:
    payload = [
        cell.get("outputs", [])
        for cell in notebook["cells"][first_cell:]
        if cell["cell_type"] == "code"
    ]
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _execution_counts(notebook: dict) -> list[int | None]:
    return [
        cell.get("execution_count")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    ]


def test_baseline_tutorial_is_moved_without_losing_owner_version() -> None:
    assert BASELINE_NOTEBOOK.is_file()
    assert not OLD_BASELINE_PATH.exists()

    notebook = _notebook(BASELINE_NOTEBOOK)
    assert len(notebook["cells"]) == 15
    assert _execution_counts(notebook) == list(range(1, 9))
    assert _outputs_digest(notebook) == BASELINE_OUTPUTS_SHA256


def test_move_level_tutorial_uses_public_allie_bundle_and_row_indexes() -> None:
    notebook = _notebook(MOVE_LEVEL_NOTEBOOK)
    source = _sources(notebook)

    assert "DATASET_REVISION = PUBLIC_DATASET_REVISION" in source
    assert "resolve_allie_embedding_root" in source
    assert "experiments.move_level.detection" in source
    assert "synth/row_index.parquet" in source
    assert "tournament/row_index.parquet" in source
    assert source.count('validate="one_to_one"') >= 2

    assert _execution_counts(notebook) == list(range(1, 11))
    assert _outputs_digest(notebook, first_cell=4) == MOVE_LEVEL_STABLE_OUTPUTS_SHA256


def test_baseline_tutorial_pins_public_data_for_move_and_game_level_reproduction() -> None:
    source = _sources(_notebook(BASELINE_NOTEBOOK))

    assert FINAL_REVISION in source
    assert "Table 4" in source
    assert "Table 5" in source
    assert "move-level" in source
    assert "player-game" in source
    assert "requirements/tutorials.txt" in source


def test_notebooks_are_sanitized() -> None:
    forbidden_path_fragments = ("/Users/", "/home/", "/private/tmp/", "/tmp/")
    secret_pattern = re.compile(
        r"(?i)(?:hf_[a-z0-9]{20,}|api[_-]?key\s*=|access[_-]?token\s*=|password\s*=)"
    )

    for path in (MOVE_LEVEL_NOTEBOOK, BASELINE_NOTEBOOK):
        notebook = _notebook(path)
        serialized = json.dumps(notebook, ensure_ascii=False)
        assert not any(fragment in serialized for fragment in forbidden_path_fragments)
        assert not secret_pattern.search(serialized)
        assert all(
            output.get("output_type") != "error"
            for cell in notebook["cells"]
            for output in cell.get("outputs", [])
        )


def test_shared_tutorial_requirements_cover_both_notebooks() -> None:
    declared = {
        re.split(r"[<>=!~;\[]", line, maxsplit=1)[0].strip().lower()
        for line in REQUIREMENTS.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert {
        "datasets",
        "huggingface_hub",
        "ipykernel",
        "jupyter",
        "matplotlib",
        "nbclient",
        "nbformat",
        "numpy",
        "pandas",
        "pyarrow",
        "torch",
    } <= declared


def test_readme_links_immutable_public_tutorials_without_metric_duplication() -> None:
    readme = README.read_text(encoding="utf-8")

    assert DOI in readme
    assert "experiments/move_level/tutorial_reproduce_move_level_experiments.ipynb" in readme
    assert "experiments/analisys/tutorial_transfer_synth_to_tournament.ipynb" in readme
    assert "requirements/tutorials.txt" in readme
    assert FINAL_REVISION in readme
    assert f"https://huggingface.co/datasets/artemlepin/chess-fraud/tree/{FINAL_REVISION}" in readme
    assert "approximately 9.56 GB" in readme
    assert "resolve_allie_embedding_root" in readme

    tutorials = readme.split("## 🧪 Tutorials", maxsplit=1)[1].split("\n## ", maxsplit=1)[0]
    assert not re.search(r"(?i)(specificity|recall|macro[- ]?f1)\s*[:=]?\s*0\.\d+", tutorials)


def test_hugging_face_card_documents_the_semantic_embedding_contract() -> None:
    card = HF_CARD.read_text(encoding="utf-8")

    assert "`artifacts/allie_embeddings/`" in card
    assert "`allie-embeddings-v1`" in card
    assert "download_allie_embeddings" in card
    assert "CHESSFRAUD_ALLIE_EMBEDDING_DIR" in card
    assert "CHESSFRAUD_ALLIE_CACHE_DIR" in card
