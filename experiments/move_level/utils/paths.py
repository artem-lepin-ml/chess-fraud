from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


class _NpyDirSource:
    """Directory of per-column ``.npy`` files, accessed by ``src[col]`` which
    loads ``{col}.npy``. Mimics the subset of ``np.lib.npyio.NpzFile`` used by
    the dataset loaders (``__getitem__`` / ``__contains__``)."""

    def __init__(self, dir_path: Path):
        self._dir = Path(dir_path)

    def __contains__(self, key: str) -> bool:
        return (self._dir / f"{key}.npy").exists()

    def __getitem__(self, key: str) -> np.ndarray:
        p = self._dir / f"{key}.npy"
        if not p.exists():
            raise KeyError(f"Embedding file not found: {p}")
        return np.load(p)

    def __repr__(self) -> str:
        return f"_NpyDirSource({self._dir})"


def open_embedding_source(path: str | Path):
    """Open an embedding source. Accepts:

    * a path to an ``.npz`` file (archive of arrays);
    * a path to a directory of ``{col}.npy`` files (per-column layout).

    Returns a dict-like with ``[key]`` / ``in`` support.
    """
    p = Path(path)
    if p.is_dir():
        return _NpyDirSource(p)
    if not p.exists():
        raise FileNotFoundError(f"Embedding source not found: {p}")
    return np.load(str(p))


def get_repo_root() -> Path:
    """Find repo root via ``git rev-parse --show-toplevel``."""
    p = subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True
    ).strip()
    return Path(p)


def ensure_out_dir(out_dir: Path) -> Path:
    """Create standard output sub-directories and return *out_dir*."""
    for sub in (
        "models",
        "figures",
        "tables",
        "tables/classification_report",
        "tables/plots",
    ):
        (out_dir / sub).mkdir(parents=True, exist_ok=True)
    return out_dir


# ---------------------------------------------------------------------------
# Known dataset paths (relative to repo root)
# ---------------------------------------------------------------------------

def synth_csv_path(repo_root: Path | None = None) -> Path:
    root = repo_root or get_repo_root()
    return root / "data/processed/chess_fraud_synth.csv"


def tournament_csv_path(repo_root: Path | None = None) -> Path:
    root = repo_root or get_repo_root()
    return root / "data/processed/chess_fraud_tournament.csv"


def synth_emb_allie_path(repo_root: Path | None = None) -> Path:
    """Per-column directory of Allie embeddings for synth (one ``.npy`` per move column)."""
    root = repo_root or get_repo_root()
    return root / "data/processed/synth/04_embs_allie_2500"


def synth_emb_maia2_path(repo_root: Path | None = None) -> Path:
    root = repo_root or get_repo_root()
    return root / "data/processed/synth/embs_maia2_2050.npz"


def tournament_emb_allie_path(repo_root: Path | None = None) -> Path:
    root = repo_root or get_repo_root()
    return root / "data/processed/tournament/embs_allie_2500.npz"


def tournament_emb_maia2_path(repo_root: Path | None = None) -> Path:
    root = repo_root or get_repo_root()
    return root / "data/processed/tournament/embs_maia2_2050.npz"
