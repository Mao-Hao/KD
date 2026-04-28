"""Shared utilities for synthetic data generators."""

from __future__ import annotations

import torch

# Axis names for spatial dimensions (up to 3D)
SPATIAL_AXIS_NAMES = ("x", "y", "z")

# Axis name for time
AXIS_T = "t"

# Field name
FIELD_U = "u"


def broadcast_grids(coords: list[torch.Tensor]) -> list[torch.Tensor]:
    """Broadcast 1D coordinate arrays to N-D grids.

    Each coordinate is reshaped so that it varies only along its
    own axis and is broadcast-compatible with all other coordinates.

    Args:
        coords: List of 1D coordinate tensors.

    Returns:
        List of N-D tensors, one per coordinate, all broadcast to the
        same shape.
    """
    ndim = len(coords)
    full_shape = tuple(c.numel() for c in coords)
    grids = []
    for i, c in enumerate(coords):
        shape = [1] * ndim
        shape[i] = c.numel()
        grids.append(c.reshape(shape).expand(full_shape))
    return grids
