"""Convergence plot: best score over iterations."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd2.search.result import ExperimentResult

logger = logging.getLogger(__name__)


def plot_convergence(
    result: ExperimentResult,
    ax: Axes,
) -> list[str]:
    """Plot best score vs iteration from recorder data.

    Args:
        result: Experiment result containing a VizRecorder.
        ax: Matplotlib Axes to draw on.

    Returns:
        List of warnings (empty if successful).
    """
    warnings: list[str] = []
    scores = result.recorder.get("_best_score")

    if not scores:
        warnings.append("No _best_score data in recorder; skipping convergence plot")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best Score")
        ax.set_title("Convergence")
        ax.text(
            0.5,
            0.5,
            "No data",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return warnings

    iterations = list(range(len(scores)))
    ax.plot(iterations, scores, marker=".", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best Score")
    ax.set_title("Convergence")

    return warnings
