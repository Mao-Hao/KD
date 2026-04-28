"""Time-slice comparison plot: true vs predicted field at selected time steps.

Tier 2 plot (creates its own Figure).
"""

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from kd2.viz.plots._dim_utils import _pick_time_steps, _slice_nd_to_2d
from kd2.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd2.core.integrator import IntegrationResult
    from kd2.data.schema import PDEDataset
    from kd2.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150
_WARNING_FONTSIZE = 9
_WARNING_WRAP_WIDTH = 38


def plot_time_slices(
    result: ExperimentResult,
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    style: dict[str, Any] | None = None,
    n_slices: int = 3,
) -> tuple[Figure, list[str]]:
    """Plot true vs predicted field at selected time slices.

    Args:
        result: Experiment result (for metadata).
        dataset: PDEDataset with field data and coordinates.
        integration_result: Integration result with predicted field.
        style: Optional matplotlib style overrides.
        n_slices: Number of time slices to display.

    Returns:
        Tuple of (Figure, list of warnings).
    """
    warnings: list[str] = []

    # Extract field data
    field_name = dataset.lhs_field
    true_field = np.array(
        dataset.get_field(field_name).detach().cpu().numpy(), dtype=np.float64
    )

    # Guard: axis_order must be available for spatial slicing
    if dataset.axis_order is None:
        warnings.append("axis_order is None; skipping time slices")
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=_DEFAULT_DPI)
            ax.text(
                0.5,
                0.5,
                "No axis_order available",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=_WARNING_FONTSIZE,
                color="red",
            )
            ax.set_title("Time Slices")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings

    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes
    n_spatial = len(spatial_axes)

    # Guard: no spatial dimensions
    if n_spatial == 0:
        warnings.append("No spatial dimensions; skipping time slices")
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=_DEFAULT_DPI)
            ax.text(
                0.5,
                0.5,
                "No spatial dimensions",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=_WARNING_FONTSIZE,
                color="red",
            )
            ax.set_title("Time Slices")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings

    time_dim = dataset.axis_order.index(time_axis)
    n_t = true_field.shape[time_dim]

    # Pick time indices
    time_indices = _pick_time_steps(n_t, n_slices)
    n_cols = len(time_indices)

    # Get time coordinates
    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()

    # Extract predicted field
    pred_field, diverged = _extract_pred(integration_result, true_field.shape, warnings)

    with style_context(style):
        if n_spatial <= 1:
            fig = _render_1d_slices(
                true_field,
                pred_field,
                dataset,
                time_axis,
                spatial_axes,
                time_dim,
                time_indices,
                t_coords,
                n_cols,
                warnings,
                diverged=diverged,
                integration_result=integration_result,
            )
        else:
            fig = _render_2d_slices(
                true_field,
                pred_field,
                time_dim,
                time_indices,
                t_coords,
                n_cols,
                warnings,
                diverged=diverged,
                integration_result=integration_result,
            )

    return fig, warnings


def _extract_pred(
    integration_result: IntegrationResult,
    true_shape: tuple[int, ...],
    warnings: list[str],
) -> tuple[np.ndarray | None, bool]:
    """Extract predicted field as numpy array.

    Returns:
        Tuple of (predicted field or None, diverged flag).
        When success=False but predicted_field exists, returns the field
        with diverged=True.
    """
    if integration_result.predicted_field is None:
        msg = integration_result.warning or "Integration failed"
        warnings.append(msg)
        return None, False

    pred = np.array(
        integration_result.predicted_field.detach().cpu().numpy(), dtype=np.float64
    )

    if pred.shape != true_shape:
        warnings.append(f"Shape mismatch: true {true_shape} vs predicted {pred.shape}")
        return None, False

    diverged = not integration_result.success
    if diverged:
        warnings.append(integration_result.warning or "Integration did not succeed")

    return pred, diverged


def _render_1d_slices(
    true_field: np.ndarray,
    pred_field: np.ndarray | None,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    time_dim: int,
    time_indices: list[int],
    t_coords: np.ndarray,
    n_cols: int,
    warnings: list[str],
    *,
    diverged: bool = False,
    integration_result: IntegrationResult | None = None,
) -> Figure:
    """Render 1D spatial time slices: N columns of line plots."""
    fig, axes_arr = plt.subplots(
        1,
        n_cols,
        figsize=(5 * n_cols, 4),
        dpi=_DEFAULT_DPI,
        squeeze=False,
    )

    # Get spatial coordinates
    s_name = spatial_axes[0]
    s_coords = dataset.get_coords(s_name).detach().cpu().numpy()

    div_tag = _diverged_tag(diverged, integration_result)

    for col, t_idx in enumerate(time_indices):
        ax: Axes = axes_arr[0, col]
        t_val = float(t_coords[t_idx])

        # Extract slice at this time step
        true_slice = np.take(true_field, t_idx, axis=time_dim)
        true_display = np.where(np.isfinite(true_slice), true_slice, np.nan)
        ax.plot(s_coords, true_display, "b-", label="True", linewidth=1.5)

        if pred_field is not None:
            pred_slice = np.take(pred_field, t_idx, axis=time_dim)
            pred_display = np.where(np.isfinite(pred_slice), pred_slice, np.nan)
            pred_label = f"Predicted{div_tag}"
            ax.plot(s_coords, pred_display, "r--", label=pred_label, linewidth=1.5)
        else:
            _add_warning_text(ax, "No prediction")

        title = f"t = {t_val:.3g}{div_tag}" if div_tag else f"t = {t_val:.3g}"
        ax.set_title(title)
        ax.set_xlabel(s_name)
        if col == 0:
            ax.set_ylabel(dataset.lhs_field)
            ax.legend()

    fig.tight_layout()
    return fig


def _render_2d_slices(
    true_field: np.ndarray,
    pred_field: np.ndarray | None,
    time_dim: int,
    time_indices: list[int],
    t_coords: np.ndarray,
    n_cols: int,
    warnings: list[str],
    *,
    diverged: bool = False,
    integration_result: IntegrationResult | None = None,
) -> Figure:
    """Render 2D spatial time slices: 2xN grid of heatmaps."""
    n_rows = 2
    fig, axes_arr = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5 * n_cols, 4 * n_rows),
        dpi=_DEFAULT_DPI,
        squeeze=False,
    )

    div_tag = _diverged_tag(diverged, integration_result)

    for col, t_idx in enumerate(time_indices):
        t_val = float(t_coords[t_idx])

        true_slice = np.take(true_field, t_idx, axis=time_dim)
        if true_slice.ndim > 2:
            true_slice = _slice_nd_to_2d(true_slice, (0, 1))
        true_display = np.where(np.isfinite(true_slice), true_slice, np.nan)
        axes_arr[0, col].imshow(
            true_display, aspect="auto", origin="lower", rasterized=True
        )
        axes_arr[0, col].set_title(f"True (t={t_val:.3g})")

        if pred_field is not None:
            pred_slice = np.take(pred_field, t_idx, axis=time_dim)
            if pred_slice.ndim > 2:
                pred_slice = _slice_nd_to_2d(pred_slice, (0, 1))
            pred_display = np.where(np.isfinite(pred_slice), pred_slice, np.nan)
            axes_arr[1, col].imshow(
                pred_display, aspect="auto", origin="lower", rasterized=True
            )
            axes_arr[1, col].set_title(f"Predicted{div_tag} (t={t_val:.3g})")
        else:
            _warning_panel(axes_arr[1, col])

    fig.tight_layout()
    return fig


def _diverged_tag(
    diverged: bool,
    integration_result: IntegrationResult | None,
) -> str:
    """Build DIVERGED tag string for titles."""
    if not diverged:
        return ""
    if integration_result is not None and integration_result.diverged_at_t is not None:
        return f" (DIVERGED at t={integration_result.diverged_at_t:.3g})"
    return " (DIVERGED)"


def _add_warning_text(ax: Axes, msg: str) -> None:
    """Add warning annotation to an axes with data."""
    ax.text(
        0.5,
        0.95,
        msg,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=_WARNING_FONTSIZE,
        color="red",
    )


def _warning_panel(ax: Axes, msg: str = "No prediction available") -> None:
    """Show warning text in an empty panel.

    Wrap explicitly with textwrap so long messages don't spill across panels
    (matplotlib's ``wrap=True`` is unreliable in the SVG backend).
    """
    wrapped = textwrap.fill(msg, width=_WARNING_WRAP_WIDTH)
    ax.text(
        0.5,
        0.5,
        wrapped,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=_WARNING_FONTSIZE,
        color="red",
    )
    ax.set_xticks([])
    ax.set_yticks([])
