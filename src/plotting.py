"""Shared Matplotlib conventions and small plot primitives."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

MODEL_COLORS = {"LightGBM": "#4C78A8", "SchNet": "#F58518", "Ensemble": "#54A24B"}


def configure_matplotlib() -> None:
    """Apply the presentation style used throughout the notebooks."""
    plt.style.use("default")
    plt.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 180, "font.size": 10,
        "axes.titlesize": 12, "axes.labelsize": 10, "legend.fontsize": 9,
        "axes.grid": True, "grid.alpha": .20, "grid.linewidth": .6,
    })


def save_figure(
    name: str,
    directory: Path,
    *,
    figure=None,
    dpi: int = 180,
    show: bool = True,
) -> Path:
    """Save the current or supplied figure using the shared conventions."""
    directory.mkdir(parents=True, exist_ok=True)
    figure = figure or plt.gcf()
    figure.tight_layout()
    path = directory / f"{name}.png"
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    if show:
        plt.show()
    return path


def add_identity_line(ax, true, predicted, **kwargs):
    """Add a y=x line covering both supplied arrays."""
    limits = [min(np.min(true), np.min(predicted)), max(np.max(true), np.max(predicted))]
    ax.plot(limits, limits, "--", color="black", linewidth=1, **kwargs)
    return limits

