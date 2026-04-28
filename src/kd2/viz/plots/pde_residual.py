"""PDE residual field plot: actual vs predicted spatial distribution.

Extracted from the original field.py — this is the u_t-level comparison
(ExperimentResult.actual vs .predicted reshaped to spatial grid).

When dataset is provided with axis info, renders with proper physical
coordinates:
- 1D spatial: pcolormesh(t, x) for true/predicted/residual
- 2D spatial: heatmap at mid time step
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

from kd2.viz.style import style_context

if TYPE_CHECKING:
    from matplotlib.axes import Axes

    from kd2.data.schema import PDEDataset
    from kd2.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_RESIDUAL_FIGSIZE = (15, 4)
_DEFAULT_DPI = 150
_WARNING_FONTSIZE = 12
_RESIDUAL_PERCENTILE = 99.0


def plot_pde_residual_field(
    result: ExperimentResult,
    *,
    style: dict[str, Any] | None = None,
    field_shape: tuple[int, ...] | None = None,
    dataset: PDEDataset | None = None,
) -> tuple[Figure, list[str]]:
    """Plot PDE residual field: result.actual vs result.predicted spatial distribution.

    Creates a Tier 2 figure with True / Predicted / Residual panels.

    When dataset is provided with axis information:
    - 1D spatial: uses pcolormesh(t, x) with physical coordinates
    - 2D spatial: heatmap at mid time step

    When dataset is None: falls back to generic reshape + imshow or 1D line plot.

    Args:
        result: Experiment result with actual/predicted tensors (1D flattened).
        style: Optional matplotlib style overrides.
        field_shape: Shape to reshape 1D data into 2D grid for display.
            If None, attempts to infer or produces a warning.
        dataset: Optional PDEDataset for axis-aware rendering.

    Returns:
        Tuple of (Figure, list of warnings).
    """
    warnings: list[str] = []

    actual = np.array(result.actual.detach().cpu().numpy(), dtype=np.float64)
    predicted = np.array(result.predicted.detach().cpu().numpy(), dtype=np.float64)

    # Handle NaN/Inf
    if not np.all(np.isfinite(actual)):
        warnings.append("Actual data contains NaN/Inf values")
    if not np.all(np.isfinite(predicted)):
        warnings.append("Predicted data contains NaN/Inf values")

    # Try axis-aware rendering if dataset is available
    if dataset is not None and _has_axis_info(dataset):
        with style_context(style):
            fig = _render_axis_aware(actual, predicted, dataset, field_shape, warnings)
        return fig, warnings

    # Generic fallback (no dataset or insufficient axis info)
    if dataset is None:
        logger.warning(
            "No dataset provided; falling back to 1D layout for PDE residual"
        )
    shape = _resolve_shape(actual, field_shape, warnings)

    with style_context(style):
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=_DEFAULT_DPI,
        )
        axes: list[Axes] = list(axes_arr.flat)

        if shape is not None:
            actual_2d = _safe_reshape(actual, shape, warnings, "actual")
            pred_2d = _safe_reshape(predicted, shape, warnings, "predicted")
        else:
            actual_2d = None
            pred_2d = None

        if actual_2d is not None and pred_2d is not None:
            _render_panel(axes[0], actual_2d, "True (u_t actual)")
            _render_panel(axes[1], pred_2d, "Predicted (u_t predicted)")
            residual = actual_2d - pred_2d
            _render_panel(axes[2], residual, "Residual", residual=True)
        else:
            # Fallback: 1D line plot
            _line_fallback(axes[0], actual, "True (u_t actual)")
            _line_fallback(axes[1], predicted, "Predicted (u_t predicted)")
            residual_1d = actual - predicted
            _line_fallback(axes[2], residual_1d, "Residual")

        fig.tight_layout()

    return fig, warnings


def _has_axis_info(dataset: PDEDataset) -> bool:
    """Check if dataset has sufficient axis info for axis-aware rendering."""
    return (
        dataset.axis_order is not None
        and dataset.axes is not None
        and len(dataset.axis_order) >= 2
    )


def _shape_matches_dataset(
    shape: tuple[int, ...],
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
) -> bool:
    """Check if reshaped data dimensions match dataset coordinate arrays."""
    assert dataset.axis_order is not None # noqa: S101
    assert dataset.axes is not None # noqa: S101

    # Build expected shape from dataset
    expected_sizes = []
    for axis_name in dataset.axis_order:
        if axis_name in dataset.axes:
            expected_sizes.append(dataset.axes[axis_name].values.numel())

    expected_shape = tuple(expected_sizes)
    return shape == expected_shape


def _render_axis_aware(
    actual: np.ndarray,
    predicted: np.ndarray,
    dataset: PDEDataset,
    field_shape: tuple[int, ...] | None,
    warnings: list[str],
) -> Figure:
    """Render with axis-aware coordinates from dataset."""
    import contextlib

    assert dataset.axis_order is not None # noqa: S101
    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes
    n_spatial = len(spatial_axes)

    # Resolve field shape from dataset if not provided
    if field_shape is None:
        with contextlib.suppress(ValueError, AttributeError):
            field_shape = dataset.get_shape()

    shape = _resolve_shape(actual, field_shape, warnings)
    if shape is None:
        # Fall back to generic
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=_DEFAULT_DPI,
        )
        axes: list[Axes] = list(axes_arr.flat)
        _line_fallback(axes[0], actual, "True (u_t actual)")
        _line_fallback(axes[1], predicted, "Predicted (u_t predicted)")
        _line_fallback(axes[2], actual - predicted, "Residual")
        fig.tight_layout()
        return fig

    actual_nd = _safe_reshape(actual, shape, warnings, "actual")
    pred_nd = _safe_reshape(predicted, shape, warnings, "predicted")

    if actual_nd is None or pred_nd is None:
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=_DEFAULT_DPI,
        )
        axes = list(axes_arr.flat)
        _line_fallback(axes[0], actual, "True (u_t actual)")
        _line_fallback(axes[1], predicted, "Predicted (u_t predicted)")
        _line_fallback(axes[2], actual - predicted, "Residual")
        fig.tight_layout()
        return fig

    residual_nd = actual_nd - pred_nd

    # Validate that reshaped dimensions match dataset coords
    # before attempting axis-aware rendering
    if not _shape_matches_dataset(shape, dataset, time_axis, spatial_axes):
        warnings.append("field_shape does not match dataset axes; using generic render")
        fig, axes_arr = plt.subplots(
            1,
            3,
            figsize=_RESIDUAL_FIGSIZE,
            dpi=_DEFAULT_DPI,
        )
        axes = list(axes_arr.flat)
        _render_panel(axes[0], actual_nd, "True (u_t actual)")
        _render_panel(axes[1], pred_nd, "Predicted (u_t predicted)")
        _render_panel(axes[2], residual_nd, "Residual", residual=True)
        fig.tight_layout()
        return fig

    if n_spatial <= 1 and len(shape) == 2:
        # 1D spatial: pcolormesh with (t, x) coordinates
        return _render_1d_axis_aware(
            actual_nd,
            pred_nd,
            residual_nd,
            dataset,
            time_axis,
            spatial_axes,
        )
    else:
        # 2D spatial: heatmap at mid time step
        time_dim = dataset.axis_order.index(time_axis)
        return _render_2d_axis_aware(
            actual_nd,
            pred_nd,
            residual_nd,
            dataset,
            time_axis,
            time_dim,
        )


def _render_1d_axis_aware(
    actual_2d: np.ndarray,
    pred_2d: np.ndarray,
    residual_2d: np.ndarray,
    dataset: PDEDataset,
    time_axis: str,
    spatial_axes: list[str],
) -> Figure:
    """Render 1D spatial with pcolormesh(t, x)."""
    assert dataset.axis_order is not None # noqa: S101

    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()
    s_name = spatial_axes[0]
    s_coords = dataset.get_coords(s_name).detach().cpu().numpy()

    time_dim = dataset.axis_order.index(time_axis)

    # Ensure data is (spatial, time) for pcolormesh
    if time_dim == 0:
        actual_display = actual_2d.T
        pred_display = pred_2d.T
        residual_display = residual_2d.T
    else:
        actual_display = actual_2d
        pred_display = pred_2d
        residual_display = residual_2d

    fig, axes_arr = plt.subplots(
        1,
        3,
        figsize=_RESIDUAL_FIGSIZE,
        dpi=_DEFAULT_DPI,
    )
    axes: list[Axes] = list(axes_arr.flat)

    _pcolormesh_panel(axes[0], t_coords, s_coords, actual_display, "True (u_t actual)")
    _pcolormesh_panel(
        axes[1], t_coords, s_coords, pred_display, "Predicted (u_t predicted)"
    )
    _pcolormesh_panel(
        axes[2], t_coords, s_coords, residual_display, "Residual", residual=True
    )

    fig.tight_layout()
    return fig


def _render_2d_axis_aware(
    actual_nd: np.ndarray,
    pred_nd: np.ndarray,
    residual_nd: np.ndarray,
    dataset: PDEDataset,
    time_axis: str,
    time_dim: int,
) -> Figure:
    """Render 2D spatial: heatmap at mid time step."""
    from kd2.viz.plots._dim_utils import _pick_time_steps, _slice_nd_to_2d

    n_t = actual_nd.shape[time_dim]
    mid_indices = _pick_time_steps(n_t, 1)
    mid_idx = mid_indices[0]

    t_coords = dataset.get_coords(time_axis).detach().cpu().numpy()
    t_val = float(t_coords[mid_idx])

    actual_slice = np.take(actual_nd, mid_idx, axis=time_dim)
    pred_slice = np.take(pred_nd, mid_idx, axis=time_dim)
    residual_slice = np.take(residual_nd, mid_idx, axis=time_dim)

    if actual_slice.ndim > 2:
        actual_slice = _slice_nd_to_2d(actual_slice, (0, 1))
        pred_slice = _slice_nd_to_2d(pred_slice, (0, 1))
        residual_slice = _slice_nd_to_2d(residual_slice, (0, 1))

    fig, axes_arr = plt.subplots(
        1,
        3,
        figsize=_RESIDUAL_FIGSIZE,
        dpi=_DEFAULT_DPI,
    )
    axes: list[Axes] = list(axes_arr.flat)

    _heatmap_panel(axes[0], actual_slice, f"True (t={t_val:.3g})")
    _heatmap_panel(axes[1], pred_slice, f"Predicted (t={t_val:.3g})")
    _heatmap_panel(axes[2], residual_slice, f"Residual (t={t_val:.3g})", residual=True)

    fig.tight_layout()
    return fig


def _pcolormesh_panel(
    ax: Axes,
    t_coords: np.ndarray,
    s_coords: np.ndarray,
    data: np.ndarray,
    title: str,
    *,
    residual: bool = False,
) -> None:
    """Draw a pcolormesh panel with physical coordinates.

    When ``residual=True`` use a diverging colormap (RdBu_r) centered at zero
    with robust ±p99 limits and attach a colorbar.
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
    data: np.ndarray,
    title: str,
    *,
    residual: bool = False,
) -> None:
    """Draw a heatmap panel.

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


def _resolve_shape(
    data: np.ndarray,
    field_shape: tuple[int, ...] | None,
    warnings: list[str],
) -> tuple[int, ...] | None:
    """Resolve the reshape target shape."""
    n = data.size

    if field_shape is not None:
        expected = 1
        for s in field_shape:
            expected *= s
        if expected != n:
            warnings.append(
                f"field_shape {field_shape} (size {expected}) "
                f"does not match data size {n}"
            )
            return None
        if len(field_shape) < 2:
            return None
        return field_shape

    # Try to infer square shape
    sqrt_n = int(math.isqrt(n))
    if sqrt_n * sqrt_n == n and sqrt_n > 1:
        return (sqrt_n, sqrt_n)

    warnings.append(f"No field_shape provided and cannot infer 2D shape from size {n}")
    return None


def _safe_reshape(
    data: np.ndarray,
    shape: tuple[int, ...],
    warnings: list[str],
    label: str,
) -> np.ndarray | None:
    """Reshape data, returning None on failure."""
    try:
        return data.reshape(shape)
    except ValueError:
        warnings.append(f"Cannot reshape {label} data to {shape}")
        return None


def _render_panel(
    ax: Axes,
    data: np.ndarray,
    title: str,
    *,
    residual: bool = False,
) -> None:
    """Render a 2D heatmap panel with NaN-safe colorbar.

    When ``residual=True`` use a diverging colormap (RdBu_r) centered at zero
    with robust ±p99 limits and attach a colorbar.
    """
    from kd2.viz.plots._dim_utils import _slice_nd_to_2d

    render_data = data
    if render_data.ndim > 2:
        render_data = _slice_nd_to_2d(render_data, (0, 1))
    # Replace non-finite for display
    display = np.where(np.isfinite(render_data), render_data, np.nan)
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


def _line_fallback(
    ax: Axes,
    data: np.ndarray,
    title: str,
) -> None:
    """Fallback 1D line plot when reshape is not possible."""
    ax.plot(data.ravel())
    ax.set_title(title)
    ax.set_xlabel("Index")
