from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt

COLORS = {
    "navy": "#0C2A40",
    "blue": "#2B6F97",
    "teal": "#19A7A0",
    "mint": "#6ED4CD",
    "coral": "#FF7A59",
    "gold": "#E6B84A",
    "cream": "#F7F2EA",
    "slate": "#5B6B78",
    "red": "#C94C4C",
    "green": "#2A8C6B",
}


def set_finance_theme() -> None:
    mpl.rcParams.update(
        {
            "figure.figsize": (9.0, 4.8),
            "figure.dpi": 110,
            "savefig.dpi": 140,
            "axes.facecolor": COLORS["cream"],
            "figure.facecolor": "white",
            "axes.edgecolor": "#B8C2C8",
            "axes.labelcolor": COLORS["navy"],
            "axes.titlecolor": COLORS["navy"],
            "axes.titleweight": "bold",
            "axes.grid": True,
            "grid.color": "#D8DEDF",
            "grid.alpha": 0.75,
            "grid.linewidth": 0.7,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "xtick.color": COLORS["slate"],
            "ytick.color": COLORS["slate"],
            "legend.frameon": False,
            "lines.linewidth": 2,
        }
    )


def finish(ax: plt.Axes, source: str = "Synthetic teaching data") -> plt.Axes:
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(
        0,
        -0.18,
        f"Source: {source}",
        transform=ax.transAxes,
        fontsize=8,
        color=COLORS["slate"],
    )
    return ax

