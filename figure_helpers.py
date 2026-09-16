#!/usr/bin/env python3
"""Shared style and export helpers for the five-wave publication figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "publication_outputs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

COL = {
    "navy": "#183B56",
    "blue": "#3B82A0",
    "sky": "#7CB7D3",
    "teal": "#2A9D8F",
    "orange": "#D97706",
    "vermillion": "#C94C2C",
    "gray": "#697386",
    "lightgray": "#D9DEE7",
    "pale": "#F3F7FA",
    "death": "#9AA3AE",
    "ink": "#17212B",
}


def setup_style() -> None:
    plt.rcParams.update({
        "font.family": "Arial",
        "font.size": 8.0,
        "axes.titlesize": 10.0,
        "axes.titleweight": "bold",
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "axes.edgecolor": "#7A8696",
        "axes.linewidth": 0.75,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save(fig, stem: str) -> None:
    fig.savefig(OUT / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(
        OUT / f"{stem}.tif",
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.06,
        pil_kwargs={"compression": "tiff_lzw"},
    )
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def arrow(ax, start, end, color=None, connectionstyle="arc3", lw=1.0):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=9,
        linewidth=lw,
        color=color or COL["gray"],
        connectionstyle=connectionstyle,
        shrinkA=2,
        shrinkB=2,
    )
    ax.add_patch(patch)
    return patch
