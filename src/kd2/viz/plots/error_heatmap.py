"""Error heatmap: spatial distribution of prediction error over time.

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
    from kd2.core.integrator import IntegrationResult
    from kd2.data.schema import PDEDataset
    from kd2.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_DEFAULT_DPI = 150
_WARNING_FONTSIZE = 11
_WARNING_WRAP_WIDTH = 60
_RESIDUAL_PERCENTILE = 99.0


def plot_error_heatmap(
    result: ExperimentResult,
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    """Plot spatial error heatmap (true - predicted) over time.

    Args:
        result: Experiment result (for metadata).
        dataset: PDEDataset with field data and coordinates.
        integration_result: Integration result with predicted field.
        style: Optional matplotlib style overrides.

    Returns:
        Tuple of (Figure, list of warnings).
    """
    warnings: list[str] = []

    # Check integration success
    if integration_result.predicted_field is None:
        msg = integration_result.warning or "Integration failed"
        warnings.append(msg)
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=_DEFAULT_DPI)
            wrapped = textwrap.fill(
                f"Cannot compute error: {msg}", width=_WARNING_WRAP_WIDTH
            )
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
            ax.set_title("Error Heatmap")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings

    # Track divergence for title annotation
    diverged = not integration_result.success
    if diverged:
        warnings.append(integration_result.warning or "Integration did not succeed")

    # Extract fields
    field_name = dataset.lhs_field
    true_field = np.array(
        dataset.get_field(field_name).detach().cpu().numpy(), dtype=np.float64
    )
    pred_field = np.array(
        integration_result.predicted_field.detach().cpu().numpy(), dtype=np.float64
    )

    if true_field.shape != pred_field.shape:
        warnings.append(
            f"Shape mismatch: true {true_field.shape} vs predicted {pred_field.shape}"
        )
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=(8, 5), dpi=_DEFAULT_DPI)
            ax.text(
                0.5,
                0.5,
                "Shape mismatch",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=_WARNING_FONTSIZE,
                color="red",
            )
            ax.set_title("Error Heatmap")
        return fig, warnings

    # Compute error
    error = true_field - pred_field
    error = np.where(np.isfinite(error), error, np.nan)

    # Guard: axis_order must be available for spatial rendering
    if dataset.axis_order is None:
        warnings.append("axis_order is None; skipping error heatmap")
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
            ax.set_title("Error Heatmap")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings

    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes
    n_spatial = len(spatial_axes)

    # Guard: no spatial dimensions
    if n_spatial == 0:
        warnings.append("No spatial dimensions; skipping error heatmap")
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
            ax.set_title("Error Heatmap")
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings

    time_dim = dataset.axis_order.index(time_axis)

    div_tag = _diverged_tag(diverged, integration_result)

    with style_context(style):
        if n_spatial <= 1:
            fig = _render_1d_error(
                error,
                dataset,
                time_axis,
                spatial_axes,
                time_dim,
                div_tag=div_tag,
            )
        else:
            fig = _render_2d_error(
                error,
                dataset,
                time_axis,
                time_dim,
                div_tag=div_tag,
            )

    return fig, warnings


def _diverged_tag(
    diverged: bool,
    integration_result: IntegrationResult,
) -> str:
    """Build DIVERGED tag string for titles."""
    if not diverged:
        return ""
    if integration_result.diverged_at_t is not None:
        return f" (DIVERGED at t={integration_result.diverged_at_t:.3g})"
    return " (DIVERGED)"


def _render_1d_error(
    error: np.ndarray,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    time_dim: int,
    *,
    div_tag: str = "",
) -> Figure:
    """Render 1D spatial error as imshow with axis coordinates."""
    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()
    s_name = spatial_axes[0]
    s_coords = dataset.get_coords(s_name).detach().cpu().numpy()

    # Ensure field is (spatial, time) for display
    error_2d = error.T if time_dim == 0 else error

    extent = (
        float(t_coords[0]),
        float(t_coords[-1]),
        float(s_coords[0]),
        float(s_coords[-1]),
    )

    # Symmetric color range centered at 0 (robust to outliers via p99)
    vmax = _robust_abs_max(error_2d)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5), dpi=_DEFAULT_DPI)
    im = ax.imshow(
        error_2d,
        aspect="auto",
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    fig.colorbar(im, ax=ax, label="Error (True - Predicted)")
    ax.set_xlabel(time_axis)
    ax.set_ylabel(s_name)
    ax.set_title(f"Error Heatmap{div_tag}")

    return fig


def _render_2d_error(
    error: np.ndarray,
    dataset: PDEDataset,
    time_axis: str,
    time_dim: int,
    *,
    div_tag: str = "",
) -> Figure:
    """Render 2D spatial error as heatmap at mid time step."""
    n_t = error.shape[time_dim]
    mid_indices = _pick_time_steps(n_t, 1)
    mid_idx = mid_indices[0]

    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()
    t_val = float(t_coords[mid_idx])

    error_slice = np.take(error, mid_idx, axis=time_dim)
    if error_slice.ndim > 2:
        error_slice = _slice_nd_to_2d(error_slice, (0, 1))

    vmax = _robust_abs_max(error_slice)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6), dpi=_DEFAULT_DPI)
    im = ax.imshow(
        error_slice,
        aspect="auto",
        origin="lower",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        rasterized=True,
    )
    fig.colorbar(im, ax=ax, label="Error (True - Predicted)")
    ax.set_title(f"Error Heatmap{div_tag} (t={t_val:.3g})")

    return fig


def _robust_abs_max(data: np.ndarray) -> float:
    """Robust |data| upper bound using p99; fall back to 1.0 if degenerate."""
    abs_data = np.abs(data[np.isfinite(data)])
    if abs_data.size == 0:
        return 1.0
    vmax = float(np.percentile(abs_data, _RESIDUAL_PERCENTILE))
    if not np.isfinite(vmax) or vmax == 0:
        vmax = float(abs_data.max()) if abs_data.size else 1.0
    if not np.isfinite(vmax) or vmax == 0:
        vmax = 1.0
    return vmax
