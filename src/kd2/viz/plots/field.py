"""Field comparison plot: true u-field vs integrated prediction vs residual.

Tier 2 plot that creates its own Figure. Layout adapts to spatial dimensionality:
- 1D spatial: pcolormesh(t, x) in 1x3 layout
- 2D spatial: 2xN heatmap at selected time steps
"""

from __future__ import annotations

import logging
import textwrap
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from kd2.viz.plots._dim_utils import _pick_time_steps, _slice_nd_to_2d
from kd2.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd2.core.integrator import IntegrationResult
    from kd2.data.schema import PDEDataset
    from kd2.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_FIELD_FIGSIZE_1D = (15, 4)
_N_TIME_SNAPSHOTS = 3
_DEFAULT_DPI = 150
_WARNING_FONTSIZE = 9
_WARNING_WRAP_WIDTH = 38
_RESIDUAL_PERCENTILE = 99.0


def plot_field_comparison(
    result: ExperimentResult,
    dataset: PDEDataset,
    integration_result: IntegrationResult,
    *,
    style: dict[str, Any] | None = None,
) -> tuple[Figure, list[str]]:
    """Plot True u-field vs Predicted (integrated) u-field vs Residual.

    Tier 2 plot: creates its own Figure (layout depends on dimensionality).

    Panels:
    - True: dataset.fields["u"].values (original physical field)
    - Predicted: integration_result.predicted_field (integrated from discovered eq.)
    - Residual: True - Predicted

    1D spatial: pcolormesh(t, x) in 1x3 layout
    2D spatial: 2xN heatmap (3 selected time steps)

    When integration_result.success is False, Predicted panel shows warning text.

    Args:
        result: Experiment result (for metadata / expression).
        dataset: PDEDataset with field data and coordinates.
        integration_result: Integration result with predicted field.
        style: Optional matplotlib style overrides.

    Returns:
        Tuple of (Figure, list of warnings).
    """
    warnings: list[str] = []

    # Extract true field
    field_name = dataset.lhs_field
    true_field = dataset.get_field(field_name).detach().cpu().numpy().astype(np.float64)

    # Guard: axis_order required for spatial layout
    if dataset.axis_order is None:
        warnings.append("axis_order is None; skipping field comparison")
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=_FIELD_FIGSIZE_1D, dpi=_DEFAULT_DPI)
            ax.text(
                0.5,
                0.5,
                "axis_order unavailable",
                transform=ax.transAxes,
                fontsize=14,
                color="red",
                ha="center",
                va="center",
            )
            ax.set_xticks([])
            ax.set_yticks([])
        return fig, warnings

    # Determine spatial dimensions
    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes
    n_spatial = len(spatial_axes)

    # Guard: no spatial dimensions
    if n_spatial == 0:
        warnings.append("No spatial dimensions; skipping field comparison")
        with style_context(style):
            fig, ax = plt.subplots(1, 1, figsize=_FIELD_FIGSIZE_1D, dpi=_DEFAULT_DPI)
            ax.text(
                0.5,
                0.5,
                "No spatial dimensions",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
        return fig, warnings

    # Get predicted field (may be None) and diverged flag
    pred_field, diverged = _extract_predicted(
        integration_result, true_field.shape, warnings
    )

    with style_context(style):
        if n_spatial <= 1:
            fig = _render_1d_spatial(
                true_field,
                pred_field,
                dataset,
                time_axis,
                spatial_axes,
                integration_result,
                warnings,
                diverged=diverged,
            )
        else:
            fig = _render_2d_spatial(
                true_field,
                pred_field,
                dataset,
                time_axis,
                spatial_axes,
                integration_result,
                warnings,
                diverged=diverged,
            )

    return fig, warnings


def _extract_predicted(
    integration_result: IntegrationResult,
    true_shape: tuple[int, ...],
    warnings: list[str],
) -> tuple[NDArray[np.floating] | None, bool]:
    """Extract predicted field from integration result.

    Returns:
        Tuple of (predicted field or None, diverged flag).
        When success=False but predicted_field exists, returns the field
        with diverged=True so the caller can mark it visually.
    """
    if integration_result.predicted_field is None:
        msg = integration_result.warning or "Integration failed"
        warnings.append(msg)
        return None, False

    pred = np.array(
        integration_result.predicted_field.detach().cpu().numpy(),
        dtype=np.float64,
    )

    if pred.shape != true_shape:
        warnings.append(f"Shape mismatch: true {true_shape} vs predicted {pred.shape}")
        return None, False

    diverged = not integration_result.success
    if diverged:
        warnings.append(integration_result.warning or "Integration did not succeed")

    return pred, diverged


def _render_1d_spatial(
    true_field: NDArray[np.floating],
    pred_field: NDArray[np.floating] | None,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    integration_result: IntegrationResult,
    warnings: list[str],
    *,
    diverged: bool = False,
) -> Figure:
    """Render 1D spatial field comparison as pcolormesh (1x3 layout)."""
    fig, axes_arr = plt.subplots(1, 3, figsize=_FIELD_FIGSIZE_1D, dpi=_DEFAULT_DPI)
    axes: list[Axes] = list(axes_arr.flat)

    # Get coordinate arrays
    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()

    assert (
        dataset.axis_order is not None
    ) # narrowing; guaranteed by outer guard # noqa: S101
    time_dim = dataset.axis_order.index(time_axis)

    # n_spatial >= 1 guaranteed by the n_spatial == 0 guard in plot_field_comparison
    s_name = spatial_axes[0]
    s_coords = dataset.get_coords(s_name).detach().cpu().numpy()

    # Ensure field is (spatial, time) for pcolormesh
    if time_dim == 0:
        true_2d = true_field.T
        pred_2d = pred_field.T if pred_field is not None else None
    else:
        true_2d = true_field
        pred_2d = pred_field

    _pcolormesh_panel(axes[0], t_coords, s_coords, true_2d, "True")

    if pred_2d is not None:
        pred_title = _predicted_title(diverged, integration_result)
        _pcolormesh_panel(axes[1], t_coords, s_coords, pred_2d, pred_title)
        residual = true_2d - pred_2d
        _pcolormesh_panel(
            axes[2], t_coords, s_coords, residual, "Residual", residual=True
        )
    else:
        _warning_panel(axes[1], integration_result)
        _warning_panel(axes[2], integration_result, label="Residual")

    fig.tight_layout()
    return fig


def _render_2d_spatial(
    true_field: NDArray[np.floating],
    pred_field: NDArray[np.floating] | None,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
    integration_result: IntegrationResult,
    warnings: list[str],
    *,
    diverged: bool = False,
) -> Figure:
    """Render 2D spatial field comparison as heatmaps at selected time steps."""
    assert (
        dataset.axis_order is not None
    ) # narrowing; guaranteed by outer guard # noqa: S101
    time_dim = dataset.axis_order.index(time_axis)
    n_t = true_field.shape[time_dim]
    time_indices = _pick_time_steps(n_t, _N_TIME_SNAPSHOTS)
    n_snaps = len(time_indices)

    # Always 3 rows (True, Predicted/Warning, Residual/Warning) for consistency
    n_rows = 3
    fig, axes_arr = plt.subplots(
        n_rows,
        n_snaps,
        figsize=(5 * n_snaps, 4 * n_rows),
        dpi=_DEFAULT_DPI,
        squeeze=False,
    )

    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()

    for col, t_idx in enumerate(time_indices):
        # Slice at this time step
        true_slice = np.take(true_field, t_idx, axis=time_dim)
        t_val = float(t_coords[t_idx])

        # If > 2D spatial, slice to 2D
        if true_slice.ndim > 2:
            true_slice = _slice_nd_to_2d(true_slice, (0, 1))

        _heatmap_panel(axes_arr[0, col], true_slice, f"True (t={t_val:.3g})")

        if pred_field is not None:
            pred_slice = np.take(pred_field, t_idx, axis=time_dim)
            if pred_slice.ndim > 2:
                pred_slice = _slice_nd_to_2d(pred_slice, (0, 1))

            pred_title = _predicted_title(diverged, integration_result, t_val=t_val)
            _heatmap_panel(axes_arr[1, col], pred_slice, pred_title)

            residual_slice = true_slice - pred_slice
            _heatmap_panel(
                axes_arr[2, col],
                residual_slice,
                f"Residual (t={t_val:.3g})",
                residual=True,
            )
        else:
            _warning_panel(axes_arr[1, col], integration_result)
            _warning_panel(axes_arr[2, col], integration_result, label="Residual")

    fig.tight_layout()
    return fig


def _predicted_title(
    diverged: bool,
    integration_result: IntegrationResult,
    *,
    t_val: float | None = None,
) -> str:
    """Build the Predicted panel title, adding DIVERGED marker if needed."""
    if not diverged:
        if t_val is not None:
            return f"Predicted (t={t_val:.3g})"
        return "Predicted"

    div_t = integration_result.diverged_at_t
    tag = f"DIVERGED at t={div_t:.3g}" if div_t is not None else "DIVERGED"

    if t_val is not None:
        return f"Predicted ({tag}, t={t_val:.3g})"
    return f"Predicted ({tag})"


def _pcolormesh_panel(
    ax: Axes,
    t_coords: NDArray[np.floating],
    s_coords: NDArray[np.floating],
    data: NDArray[np.floating],
    title: str,
    *,
    residual: bool = False,
) -> None:
    """Draw a single pcolormesh panel.

    When ``residual=True`` use a diverging colormap (RdBu_r) centered at zero
    with robust ±p99 limits and attach a colorbar so the magnitude is visible.
    """
    display = np.where(np.isfinite(data), data, np.nan)
    if residual:
        vmax = _robust_abs_max(display)
        mesh = ax.pcolormesh(
            t_coords,
            s_coords,
            display,
            shading="auto",
            rasterized=True,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.pcolormesh(t_coords, s_coords, display, shading="auto", rasterized=True)
    ax.set_xlabel("t")
    ax.set_ylabel("x")
    ax.set_title(title)


def _heatmap_panel(
    ax: Axes,
    data: NDArray[np.floating],
    title: str,
    *,
    residual: bool = False,
) -> None:
    """Draw a single heatmap (imshow) panel.

    When ``residual=True`` use a diverging colormap (RdBu_r) centered at zero
    with robust ±p99 limits and attach a colorbar.
    """
    display = np.where(np.isfinite(data), data, np.nan)
    if residual:
        vmax = _robust_abs_max(display)
        im = ax.imshow(
            display,
            aspect="auto",
            origin="lower",
            rasterized=True,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
        )
        ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.imshow(display, aspect="auto", origin="lower", rasterized=True)
    ax.set_title(title)


def _robust_abs_max(data: NDArray[np.floating]) -> float:
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


def _warning_panel(
    ax: Axes,
    integration_result: IntegrationResult,
    label: str = "Predicted",
) -> None:
    """Show warning text when integration failed.

    Wraps long messages explicitly with textwrap so the text stays inside
    the panel — matplotlib's ``wrap=True`` is unreliable in the SVG backend
    and lets long strings spill across neighbouring panels.
    """
    msg = integration_result.warning or "Integration failed"
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
    ax.set_title(label)
    ax.set_xticks([])
    ax.set_yticks([])
