"""Coefficient bar chart: visualize fitted coefficients with optional ground truth.

Tier 1 plot (draws on provided Axes).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from torch import Tensor

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd2.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_NO_DATA_TEXT = "No data"
_BAR_ALPHA = 0.7
_GT_BAR_ALPHA = 0.4


def plot_coefficient_bar(
    result: ExperimentResult,
    ax: Axes,
    *,
    ground_truth: Tensor | None = None,
) -> list[str]:
    """Plot coefficient bar chart with optional ground truth overlay.

    Args:
        result: Experiment result containing final_eval.coefficients and .terms.
        ax: Matplotlib Axes to draw on.
        ground_truth: Optional ground truth coefficients for comparison.

    Returns:
        List of warnings (empty if successful).
    """
    warnings: list[str] = []

    terms = result.final_eval.terms
    coefficients = result.final_eval.coefficients

    # Handle missing data
    if terms is None or coefficients is None:
        warnings.append("No coefficient data available")
        ax.set_title("Coefficients")
        ax.text(
            0.5,
            0.5,
            _NO_DATA_TEXT,
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return warnings

    # Get coefficients as numpy
    coeff_np = np.array(coefficients.detach().cpu().numpy(), dtype=np.float64)

    # Filter to selected terms if available
    selected = result.final_eval.selected_indices
    if selected is not None:
        display_terms = [terms[i] for i in selected]
        display_coeffs = coeff_np[selected]
    else:
        display_terms = list(terms)
        display_coeffs = coeff_np

    if len(display_terms) == 0:
        warnings.append("No terms to display")
        ax.set_title("Coefficients")
        ax.text(
            0.5,
            0.5,
            _NO_DATA_TEXT,
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        return warnings

    # Filter NaN/Inf coefficients
    finite_mask = np.isfinite(display_coeffs)
    n_nonfinite = int(np.sum(~finite_mask))
    if n_nonfinite > 0:
        warnings.append(
            f"Filtered {n_nonfinite}/{len(display_coeffs)} non-finite coefficients"
        )
        display_coeffs = np.where(finite_mask, display_coeffs, 0.0)

    # Try to get pretty labels
    labels = _make_labels(display_terms)

    # Bar positions
    x = np.arange(len(labels))
    if ground_truth is not None:
        bar_width = 0.35
    else:
        # Adaptive: narrow for few bars (avoid "fence post" look),
        # wider for many bars (avoid hairlines). Bounded to [0.35, 0.55].
        bar_width = max(0.35, min(0.55, 2.0 / len(labels)))

    # Discovered coefficients (blue)
    ax.bar(
        x,
        display_coeffs,
        bar_width,
        label="Discovered",
        color="steelblue",
        alpha=_BAR_ALPHA,
    )

    # Ground truth overlay (red, semi-transparent)
    if ground_truth is not None:
        gt_np = np.array(ground_truth.detach().cpu().numpy(), dtype=np.float64)
        # Match ground truth to displayed terms
        if selected is not None and len(selected) > 0:
            gt_display = gt_np[selected] if len(gt_np) > max(selected) else gt_np
        else:
            gt_display = gt_np

        if len(gt_display) == len(display_coeffs):
            gt_display = np.where(np.isfinite(gt_display), gt_display, 0.0)
            ax.bar(
                x + bar_width,
                gt_display,
                bar_width,
                label="Ground Truth",
                color="firebrick",
                alpha=_GT_BAR_ALPHA,
            )
        else:
            warnings.append(
                f"Ground truth length {len(gt_display)} != "
                f"displayed terms {len(display_coeffs)}"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylabel("Coefficient")
    ax.set_title("Coefficients")
    ax.legend()

    return warnings


def _make_labels(terms: list[str]) -> list[str]:
    """Create display labels for terms, using LaTeX if available."""
    labels: list[str] = []
    for term in terms:
        try:
            from kd2.core.expr.sympy_bridge import to_latex

            labels.append(f"${to_latex(term)}$")
        except Exception:
            labels.append(term)
    return labels
