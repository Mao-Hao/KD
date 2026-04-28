"""Finite difference derivative provider.

Implements DerivativeProvider using finite difference methods
for computing derivatives on regular grid data.
"""

from __future__ import annotations

import math

import numpy as np
import torch

from kd2.data.derivatives.base import DerivativeProvider
from kd2.data.schema import DataTopology, PDEDataset

# Minimum dx to avoid numerical overflow
_MIN_DX = 1e-15

# Maximum supported derivative order
_MAX_SUPPORTED_ORDER = 3

# Minimum points required for each order (4th-order accuracy)
_MIN_POINTS_FOR_ORDER = {
    1: 5, # 4th-order central diff needs 5 points
    2: 5, # 4th-order central diff needs 5 points
    3: 5, # f[i-2], f[i-1], f[i], f[i+1], f[i+2]
}

# Tolerance for uniform grid check (relative). rtol=1e-4 accepts
# torch.linspace(dtype=float32) drift (rel deviation up to ~3e-4 at
# n≈5000) while still catching real non-uniform grids (geometric / log
# spacing have rel deviation O(1)). Public so the FD provider, SGA delta
# map, and integrator share one source of truth for grid-uniformity rtol.
UNIFORM_GRID_RTOL = 1e-4

# Threshold for degenerate (effectively zero) spacing. Public so all FD
# entry points reject constant-coordinate axes with a clear diagnostic
# instead of silently dividing by ~0.
DX_ZERO_FLOOR = 1e-30


def central_diff(
    f: torch.Tensor,
    dx: float,
    axis: int,
    order: int,
    is_periodic: bool = False,
) -> torch.Tensor:
    """Compute central difference derivative.

    Accuracy:
    - 1st order: 4th-order accurate (O(dx^4))
    - 2nd order: 4th-order accurate (O(dx^4))
    - 3rd order: 2nd-order accurate (O(dx^2))

    Formulas:
    - 1st order: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)
    - 2nd order: (-f[i+2] + 16*f[i+1] - 30*f[i] + 16*f[i-1] - f[i-2]) / (12*dx^2)
    - 3rd order: (f[i+2] - 2*f[i+1] + 2*f[i-1] - f[i-2]) / (2*dx^3)

    Args:
        f: Input tensor.
        dx: Grid spacing.
        axis: Axis along which to differentiate (0, 1, 2, ...).
        order: Order of derivative (1, 2, or 3).
        is_periodic: If True, use wrap-around padding so boundary points
            get the same high-order interior stencil (no boundary formulas).
            Grid must NOT include duplicated endpoints (use [0, L) not [0, L]).

    Returns:
        Tensor of same shape as f containing derivative values.
        Boundary points use reduced-accuracy formulas (non-periodic)
        or interior-accuracy formulas (periodic).

    Raises:
        ValueError: If order not in {1, 2, 3} or axis invalid.
        ValueError: If dx is invalid (zero, negative, too small, or inf).
        ValueError: If input contains NaN or Inf.
        ValueError: If insufficient points for the requested order.
    """
    # Validate dx
    if math.isinf(dx):
        raise ValueError(f"dx must be finite, got {dx}")
    if dx <= 0:
        raise ValueError(f"dx must be positive, got {dx}")
    if dx < _MIN_DX:
        raise ValueError(f"dx is too small ({dx}), must be >= {_MIN_DX}")

    # Validate order
    if order < 1 or order > _MAX_SUPPORTED_ORDER:
        raise ValueError(f"order must be in [1, {_MAX_SUPPORTED_ORDER}], got {order}")

    # Validate axis
    if axis < 0 or axis >= f.dim():
        raise ValueError(f"axis {axis} out of range for {f.dim()}D tensor")

    # Validate input - check for NaN/Inf
    if torch.isnan(f).any():
        raise ValueError("Input tensor contains NaN values")
    if torch.isinf(f).any():
        raise ValueError("Input tensor contains Inf values")

    # Validate sufficient points
    n_points = f.shape[axis]
    min_points = _MIN_POINTS_FOR_ORDER[order]
    if n_points < min_points:
        raise ValueError(
            f"Order {order} derivative requires at least {min_points} points, "
            f"got {n_points}"
        )

    # Move axis to position 0 for easier slicing
    f_moved = f.movedim(axis, 0)

    if is_periodic:
        return _central_diff_periodic(f_moved, dx, order).movedim(0, axis)

    # Create output tensor with axis at position 0
    result_moved = torch.zeros_like(f_moved)

    if order == 1:
        # 4th-order 1st derivative: (-f[i+2] + 8*f[i+1] - 8*f[i-1] + f[i-2]) / (12*dx)
        # Interior points (need 2 points on each side)
        result_moved[2:-2] = (
            -f_moved[4:] + 8 * f_moved[3:-1] - 8 * f_moved[1:-3] + f_moved[:-4]
        ) / (12 * dx)

        # Near-boundary: use 2nd-order central difference
        # Points 1 and -2
        result_moved[1] = (f_moved[2] - f_moved[0]) / (2 * dx)
        result_moved[-2] = (f_moved[-1] - f_moved[-3]) / (2 * dx)

        # Boundary: forward/backward difference
        result_moved[0] = (f_moved[1] - f_moved[0]) / dx
        result_moved[-1] = (f_moved[-1] - f_moved[-2]) / dx

    elif order == 2:
        # 4th-order 2nd derivative formula (see docstring)
        # Interior points (need 2 points on each side)
        result_moved[2:-2] = (
            -f_moved[4:]
            + 16 * f_moved[3:-1]
            - 30 * f_moved[2:-2]
            + 16 * f_moved[1:-3]
            - f_moved[:-4]
        ) / (12 * dx**2)

        # Near-boundary: use 2nd-order central difference
        # Points 1 and -2
        result_moved[1] = (f_moved[2] - 2 * f_moved[1] + f_moved[0]) / (dx**2)
        result_moved[-2] = (f_moved[-1] - 2 * f_moved[-2] + f_moved[-3]) / (dx**2)

        # Boundary: forward/backward second derivative (2nd-order accurate)
        # Left boundary: f''(x0) = (2f0 - 5f1 + 4f2 - f3) / dx^2
        result_moved[0] = (
            2 * f_moved[0] - 5 * f_moved[1] + 4 * f_moved[2] - f_moved[3]
        ) / (dx**2)
        # Right boundary: f''(xn) = (2fn - 5f(n-1) + 4f(n-2) - f(n-3)) / dx^2
        result_moved[-1] = (
            2 * f_moved[-1] - 5 * f_moved[-2] + 4 * f_moved[-3] - f_moved[-4]
        ) / (dx**2)

    elif order == 3:
        # 3rd order: (f[i+2] - 2*f[i+1] + 2*f[i-1] - f[i-2]) / (2*dx^3)
        # This is 2nd-order accurate (standard formula)
        # Interior points (need at least 2 points on each side)
        result_moved[2:-2] = (
            f_moved[4:] - 2 * f_moved[3:-1] + 2 * f_moved[1:-3] - f_moved[:-4]
        ) / (2 * dx**3)

        # Boundary: use one-sided differences
        # Left boundary points (0, 1)
        # Forward 3rd derivative: (-f[i] + 3f[i+1] - 3f[i+2] + f[i+3]) / dx^3
        result_moved[0] = (
            -f_moved[0] + 3 * f_moved[1] - 3 * f_moved[2] + f_moved[3]
        ) / (dx**3)
        result_moved[1] = (
            -f_moved[1] + 3 * f_moved[2] - 3 * f_moved[3] + f_moved[4]
        ) / (dx**3)

        # Right boundary points (-2, -1)
        # Backward 3rd derivative: (f[i] - 3f[i-1] + 3f[i-2] - f[i-3]) / dx^3
        result_moved[-1] = (
            f_moved[-1] - 3 * f_moved[-2] + 3 * f_moved[-3] - f_moved[-4]
        ) / (dx**3)
        result_moved[-2] = (
            f_moved[-2] - 3 * f_moved[-3] + 3 * f_moved[-4] - f_moved[-5]
        ) / (dx**3)

    # Move axis back to original position
    return result_moved.movedim(0, axis)


def _central_diff_periodic(
    f_moved: torch.Tensor,
    dx: float,
    order: int,
) -> torch.Tensor:
    """Compute central diff on periodic data (axis already at position 0).

    Pads with wrap-around so all points use the interior stencil.
    Grid must be on [0, L) without duplicated endpoints.

    Args:
        f_moved: Input tensor with differentiation axis at position 0.
        dx: Grid spacing.
        order: Derivative order (1, 2, or 3).

    Returns:
        Derivative tensor, same shape as f_moved.
    """
    pad = 2 # all stencils need 2 ghost points on each side
    # Wrap-around padding: [f[-2:], f, f[:2]]
    f_padded = torch.cat([f_moved[-pad:], f_moved, f_moved[:pad]], dim=0)

    if order == 1:
        result = (
            -f_padded[4:] + 8 * f_padded[3:-1] - 8 * f_padded[1:-3] + f_padded[:-4]
        ) / (12 * dx)
    elif order == 2:
        result = (
            -f_padded[4:]
            + 16 * f_padded[3:-1]
            - 30 * f_padded[2:-2]
            + 16 * f_padded[1:-3]
            - f_padded[:-4]
        ) / (12 * dx**2)
    else: # order == 3
        result = (
            f_padded[4:] - 2 * f_padded[3:-1] + 2 * f_padded[1:-3] - f_padded[:-4]
        ) / (2 * dx**3)

    return result


def is_uniform_grid(
    coords: torch.Tensor | np.ndarray,
    rtol: float = UNIFORM_GRID_RTOL,
) -> bool:
    """Return True iff `coords` are uniformly spaced within `rtol` of |dx0|.

    Predicate: ``max(abs(diffs - dx0)) <= rtol * abs(dx0)``.
    Equivalent to ``np.allclose(diffs, dx0, rtol=rtol, atol=0)``.

    The integrator's ``np.allclose`` form is the canonical predicate
    (per-element, more permissive than max-min span — matches the
    integrator's pre-fix behavior) and both FD and integrator paths
    route through this single helper.

    Args:
        coords: 1D coordinate tensor (or numpy array) along a single axis.
        rtol: Relative tolerance applied to ``|dx0|``.

    Returns:
        True when the coordinates are uniformly spaced. False otherwise
        (including grids with fewer than two points or degenerate
        spacing). The helper itself never raises — error reporting is
        the call site's responsibility.
    """
    arr = (
        coords.detach().cpu().numpy()
        if isinstance(coords, torch.Tensor)
        else np.asarray(coords)
    )
    if arr.ndim != 1 or arr.size < 2:
        return False
    diffs = np.diff(arr.astype(np.float64))
    dx0 = float(diffs[0])
    # Reject inf/NaN dx0 (overflow on near-MAX_DOUBLE coords, e.g.
    # `[-1.7e308, 1.7e308]` produces `[inf]`, and `np.allclose([inf], inf)`
    # is True — falsely accepting a degenerate grid).
    if not np.isfinite(dx0):
        return False
    # Reject non-positive dx0: descending grids ([0.4, 0.3, ..., 0.0]) and
    # constant axes both have non-positive dx0. FD stencils silently flip
    # signs when dx<0; PDEDataset BYOD safeguards expect monotonic
    # increasing axes, so the caller is responsible for flipping.
    if dx0 <= 0:
        return False
    if dx0 < DX_ZERO_FLOOR:
        return False
    return bool(np.allclose(diffs, dx0, rtol=rtol, atol=0.0))


def _check_uniform_grid(values: torch.Tensor, axis_name: str) -> float:
    """Check if grid values are uniformly spaced and return spacing.

    Args:
        values: 1D tensor of coordinate values.
        axis_name: Name of axis for error message.

    Returns:
        Grid spacing dx.

    Raises:
        ValueError: If the grid has fewer than two points, degenerate
            spacing (``|dx| < DX_ZERO_FLOOR``), or non-uniform spacing
            beyond ``UNIFORM_GRID_RTOL``.
    """
    if values.numel() < 2:
        raise ValueError(f"Axis '{axis_name}' must have at least 2 points")

    diffs = values[1:] - values[:-1]
    dx = diffs[0].item()

    # Reject inf/NaN spacing before all other checks — `dx=inf` (overflow
    # on extreme coords like `[-1.7e308, 1.7e308]`) silently divides FD
    # output by inf and yields zeros instead of raising.
    if not math.isfinite(dx):
        raise ValueError(
            f"Axis '{axis_name}' has non-finite spacing dx={dx}; "
            f"finite-difference stencils require finite dx."
        )

    # Reject negative dx (descending coords) — FD stencils silently flip
    # the sign of derivatives when dx<0. PDEDataset BYOD safeguards
    # expect monotonic increasing axes; flip the data before calling.
    if dx < 0:
        raise ValueError(
            f"Axis '{axis_name}' has decreasing spacing dx={dx:.6g}; "
            f"finite-difference stencils require monotonic increasing "
            f"coordinates (flip the array before fitting)."
        )

    # Reject degenerate (effectively zero) spacing before the uniform
    # check — a constant-coordinate axis would otherwise divide by ~0
    # downstream. Mirrors integrator._check_spatial_uniformity.
    if abs(dx) < DX_ZERO_FLOOR:
        raise ValueError(
            f"Axis '{axis_name}' has degenerate spacing dx={dx:.6g}; "
            f"finite-difference stencils require nonzero dx."
        )

    # Delegate to the canonical predicate so FD + integrator stay in lockstep.
    if not is_uniform_grid(values, rtol=UNIFORM_GRID_RTOL):
        # Report the per-element drift (matches the predicate, which is
        # `max|d - dx0| <= rtol*|dx0|`). The previous min/max range was
        # misleading after we switched to the np.allclose-style check.
        max_dev = float((diffs - dx).abs().max().item())
        raise ValueError(
            f"Axis '{axis_name}' has non-uniform spacing "
            f"(dx[0]={dx:.6g}, max deviation={max_dev:.6g})"
        )

    return dx


class FiniteDiffProvider(DerivativeProvider):
    """Finite difference derivative provider.

    Computes derivatives using finite difference methods on regular grid data.
    All derivatives are precomputed during initialization for efficiency.

    Attributes:
        dataset: The underlying PDE dataset (must have Grid topology).
        max_order: Maximum derivative order to precompute.
        method: Finite difference method ("central", "forward", "backward").
        accuracy: Accuracy order of the finite difference scheme.

    Example:
        >>> provider = FiniteDiffProvider(dataset, max_order=3)
        >>> u_x = provider.get_derivative("u", "x", order=1)
        >>> u_xx = provider.get_derivative("u", "x", order=2)
    """

    def __init__(
        self,
        dataset: PDEDataset,
        max_order: int = 3,
        method: str = "central",
        accuracy: int = 4,
    ) -> None:
        """Initialize finite difference provider.

        Args:
            dataset: PDE dataset (must have Grid topology).
            max_order: Maximum derivative order to precompute (1-3).
            method: Finite difference method ("central").
            accuracy: Accuracy order (only 4 is supported currently).

        Raises:
            ValueError: If dataset topology is not Grid.
            ValueError: If max_order is not in valid range.
            ValueError: If method is not supported.
            ValueError: If accuracy is not 4.
            ValueError: If grid is not uniform.
        """
        # Validate accuracy (only 4 is supported currently)
        if accuracy != 4:
            raise ValueError(
                f"Only accuracy=4 is supported currently, got {accuracy}. "
                f"Other accuracy levels will be added currently."
            )

        # Validate topology
        if dataset.topology != DataTopology.GRID:
            raise ValueError(
                f"FiniteDiffProvider requires Grid topology, "
                f"got {dataset.topology.value}"
            )

        # Validate max_order
        if max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {max_order}")
        if max_order > _MAX_SUPPORTED_ORDER:
            raise ValueError(
                f"max_order must be <= {_MAX_SUPPORTED_ORDER}, got {max_order}"
            )

        # Validate method (only central supported currently)
        if method != "central":
            raise ValueError(f"Method '{method}' not supported, use 'central'")

        self._dataset = dataset
        self._max_order = max_order
        self._method = method
        self._accuracy = accuracy

        # Validate uniform grids and compute spacing
        self._dx: dict[str, float] = {}
        self._axis_indices: dict[str, int] = {}
        self._is_periodic: dict[str, bool] = {}

        if dataset.axes is None or dataset.axis_order is None:
            raise ValueError("Dataset must have axes and axis_order defined")

        for i, axis_name in enumerate(dataset.axis_order):
            axis_info = dataset.axes[axis_name]
            dx = _check_uniform_grid(axis_info.values, axis_name)
            self._dx[axis_name] = dx
            self._axis_indices[axis_name] = i
            self._is_periodic[axis_name] = axis_info.is_periodic

        # Precompute all derivatives
        self._cache: dict[tuple[str, str, int], torch.Tensor] = {}
        self._precompute_derivatives()

    @property
    def coords(self) -> dict[str, torch.Tensor]:
        """Coordinate tensors keyed by axis name."""
        if self._dataset.axes is None:
            return {}
        return {name: axis.values for name, axis in self._dataset.axes.items()}

    def _precompute_derivatives(self) -> None:
        """Precompute all derivatives up to max_order."""
        if self._dataset.fields is None:
            return

        for field_name, field_data in self._dataset.fields.items():
            for axis_name in self._axis_indices:
                axis_idx = self._axis_indices[axis_name]
                dx = self._dx[axis_name]

                for order in range(1, self._max_order + 1):
                    periodic = self._is_periodic[axis_name]
                    deriv = central_diff(
                        field_data.values,
                        dx,
                        axis_idx,
                        order,
                        is_periodic=periodic,
                    )
                    self._cache[(field_name, axis_name, order)] = deriv

    def get_derivative(
        self,
        field: str,
        axis: str,
        order: int,
    ) -> torch.Tensor:
        """Get precomputed derivative.

        Args:
            field: Field name (e.g., "u").
            axis: Axis name (e.g., "x").
            order: Derivative order (1, 2, 3).

        Returns:
            Tensor of derivative values.

        Raises:
            KeyError: If field or axis not found.
            ValueError: If order exceeds max_order or is invalid.
        """
        # Validate order
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if order > self._max_order:
            raise ValueError(f"order {order} exceeds max_order {self._max_order}")

        # Validate field exists
        if self._dataset.fields is None or field not in self._dataset.fields:
            raise KeyError(f"Field '{field}' not found in dataset")

        # Validate axis exists
        if axis not in self._axis_indices:
            raise KeyError(f"Axis '{axis}' not found in dataset")

        # Get from cache and return a clone to prevent external modification
        key = (field, axis, order)
        return self._cache[key].clone()

    def diff(
        self,
        expression: torch.Tensor,
        axis: str,
        order: int,
    ) -> torch.Tensor:
        """Compute open-form derivative of an arbitrary tensor expression.

        Applies central finite differences to a runtime tensor expression
        along the specified axis. Reuses the same `central_diff()` function
        and grid metadata (dx, axis index, periodicity) as `get_derivative()`.

        Unlike ``get_derivative()``, this method is not constrained by
        ``max_order`` — it computes derivatives on-the-fly rather than
        looking up precomputed cache entries. The order limit is only
        what ``central_diff()`` supports (currently 1-3).

        Args:
            expression: Tensor expression to differentiate. Must have
                the same shape as the dataset grid along the
                differentiation axis.
            axis: Axis name (e.g., "x", "t").
            order: Derivative order (1, 2, or 3).

        Returns:
            Tensor of derivative values, same shape as expression.

        Raises:
            KeyError: If axis not found in dataset.
            ValueError: If order is invalid, expression shape doesn't
                match grid, or expression has issues (delegated to
                central_diff).
        """
        if axis not in self._axis_indices:
            raise KeyError(f"Axis '{axis}' not found in dataset")

        axis_idx = self._axis_indices[axis]

        # Validate expression shape matches grid along differentiation axis
        if self._dataset.axes is not None and axis in self._dataset.axes:
            expected_size = len(self._dataset.axes[axis].values)
            actual_size = (
                expression.shape[axis_idx] if axis_idx < expression.dim() else -1
            )
            if actual_size != expected_size:
                raise ValueError(
                    f"Expression shape[{axis_idx}]={actual_size} doesn't match "
                    f"grid axis '{axis}' size={expected_size}"
                )

        dx = self._dx[axis]
        is_periodic = self._is_periodic[axis]

        return central_diff(expression, dx, axis_idx, order, is_periodic=is_periodic)

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        """Return list of available precomputed derivatives.

        Returns:
            List of (field, axis, order) tuples.
        """
        return list(self._cache.keys())
