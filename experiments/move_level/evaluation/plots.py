"""Heatmap plotting for metric tables."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def plot_metric_heatmap(
    tbl: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    vmin: float = 0.0,
    vmax: float = 1.0,
    figsize: tuple[int, int] = (10, 6),
    show_values: bool = True,
) -> None:
    """Plot and save a heatmap of *tbl* values."""
    mat = tbl.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(mat, vmin=vmin, vmax=vmax, cmap="Blues", aspect="auto")

    ax.set_xticks(range(tbl.shape[1]))
    ax.set_yticks(range(tbl.shape[0]))
    ax.set_xticklabels([str(c) for c in tbl.columns], rotation=45, ha="right")
    ax.set_yticklabels([str(r) for r in tbl.index])
    ax.set_title(title)

    if show_values:
        for i in range(tbl.shape[0]):
            for j in range(tbl.shape[1]):
                v = mat[i, j]
                s = "" if np.isnan(v) else f"{v:.2f}"
                ax.text(j, i, s, ha="center", va="center", fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_all_heatmaps(
    tables: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    """Plot heatmaps for all metric tables."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for metric_name, tbl in tables.items():
        plot_metric_heatmap(
            tbl,
            title=metric_name,
            out_path=out_dir / f"heatmap__{metric_name}.png",
        )
