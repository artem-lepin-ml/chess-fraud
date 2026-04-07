from __future__ import annotations

import subprocess
from pathlib import Path


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
    root = repo_root or get_repo_root()
    return root / "data/processed/synth/embs_allie_2500.npz"


def synth_emb_maia2_path(repo_root: Path | None = None) -> Path:
    root = repo_root or get_repo_root()
    return root / "data/processed/synth/embs_maia2_2050.npz"


def tournament_emb_allie_path(repo_root: Path | None = None) -> Path:
    root = repo_root or get_repo_root()
    return root / "data/processed/tournament/embs_allie_2500.npz"


def tournament_emb_maia2_path(repo_root: Path | None = None) -> Path:
    root = repo_root or get_repo_root()
    return root / "data/processed/tournament/embs_maia2_2050.npz"
