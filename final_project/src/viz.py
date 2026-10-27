"""Shared figure style so every chart in the report and presentation looks the same."""

import matplotlib as mpl

PALETTE = {"primary": "#1f5f8b", "accent": "#d1495b", "muted": "#c9d3dd", "neutral": "#5b6770", "positive": "#2a9d8f"}


def style():
    mpl.rcParams.update({
        "font.family": "sans-serif", "font.size": 9, "axes.titlesize": 10, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "0.92", "grid.linewidth": 0.6,
        "axes.axisbelow": True, "figure.dpi": 110,
    })
