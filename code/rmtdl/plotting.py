"""Shared 'watercolor' matplotlib style so every figure in the series looks like
one family (matches the blog's Aegean / clay palette)."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

# Default to the non-interactive Agg backend so the examples always run headless
# (CI, servers, broken Qt/Wayland plugins) and just write PNGs. Set MPLBACKEND
# yourself if you want an interactive window.
if os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

PALETTE = {
    "paper": "#FBF4E6",
    "ink": "#21303F",
    "clay": "#E25A3B",
    "blue": "#5B86A8",
    "lblue": "#9FC3E0",
    "sea": "#11A79C",
    "muted": "#6E7A86",
    "grid": "#E3DFD3",
}

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "outputs"


def use_watercolor_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["paper"],
            "axes.facecolor": PALETTE["paper"],
            "savefig.facecolor": PALETTE["paper"],
            "axes.edgecolor": "#D9D2C2",
            "axes.labelcolor": PALETTE["ink"],
            "text.color": PALETTE["ink"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "font.size": 11,
            "axes.grid": True,
            "grid.color": PALETTE["grid"],
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "figure.dpi": 120,
        }
    )


def save(fig, name: str) -> Path:
    """Save `fig` into the shared outputs/ dir and return the path."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    fig.savefig(path, bbox_inches="tight")
    print(f"  saved -> {path}")
    return path
