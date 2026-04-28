"""Dimension-adaptive slice utilities for ND field visualization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def _slice_nd_to_2d(
    field: NDArray[np.floating],
    axes: tuple[int, int],
) -> NDArray[np.floating]:
    """Take middle slices of ND array to produce 2D result.

    For axes not in the ``axes`` pair, slice at their midpoint index.

    Args:
        field: N-dimensional array.
        axes: The two axis indices to keep.

    Returns:
        2D array with only the two selected axes remaining.

    Raises:
        ValueError: If axes are out of range or duplicated.
    """
    ndim = field.ndim
    ax0, ax1 = axes

    # Validate
    if ax0 == ax1:
        raise ValueError(f"axes must be distinct, got ({ax0}, {ax1})")
    for ax in (ax0, ax1):
        if ax < 0 or ax >= ndim:
            raise ValueError(f"Axis {ax} out of range for {ndim}-D array")

    # Already 2D: return as-is
    if ndim == 2:
        return field

    # Build index: slice(None) for kept axes, midpoint int for others
    keep = {ax0, ax1}
    idx: list[int | slice] = []
    for i in range(ndim):
        if i in keep:
            idx.append(slice(None))
        else:
            idx.append(field.shape[i] // 2)

    return field[tuple(idx)]


def _pick_time_steps(n_t: int, n: int = 3) -> list[int]:
    """Select evenly-spaced time step indices including endpoints.

    Args:
        n_t: Total number of time steps.
        n: Desired number of indices (default 3 = first/mid/last).

    Returns:
        List of indices (length ``min(n, n_t)``).

    Raises:
        ValueError: If n_t < 1.
    """
    if n_t < 1:
        raise ValueError(f"n_t must be >= 1, got {n_t}")

    actual_n = min(n, n_t)

    if actual_n == 1:
        return [0]

    # Evenly spaced from 0 to n_t-1 inclusive
    indices = np.linspace(0, n_t - 1, actual_n)
    result = sorted(set(int(round(x)) for x in indices))
    return result
