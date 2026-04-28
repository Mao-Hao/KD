"""Execution context for expression evaluation.

This module defines:
- ExecutionContext: Encapsulates all data needed for expression evaluation
  and derives PDE spatial axes from dataset metadata

Design principles:
- Device-aware tensor operations
- Immutable context during execution
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from kd2.data import PDEDataset
    from kd2.data.derivatives import DerivativeProvider


@dataclass
class ExecutionContext:
    """Execution context encapsulating all data needed for expression evaluation.

    The ExecutionContext provides access to:
    - Field values (e.g., u, v)
    - Coordinate values (e.g., x, t)
    - Precomputed derivatives (e.g., u_x, u_xx)
    - Named constants (e.g., pi, nu)
    - Spatial axes derived from dataset.axis_order and dataset.lhs_axis

    Attributes:
        dataset: The underlying PDE dataset.
        derivative_provider: Provider for precomputed derivatives.
        constants: Dictionary of named constants.
        device: Target device for tensor operations.

    Examples:
        >>> context = ExecutionContext(
        ... dataset=dataset,
        ... derivative_provider=provider,
        ... constants={"pi": 3.14159, "nu": 0.1},
        ... )
        >>> u = context.get_variable("u")
        >>> u_x = context.get_derivative("u", "x", 1)
    """

    dataset: PDEDataset
    derivative_provider: DerivativeProvider
    constants: dict[str, float] = field(default_factory=dict)
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))

    def get_variable(self, name: str) -> torch.Tensor:
        """Get variable value by name.

        Looks up the variable in the following order:
        1. Field values (e.g., "u", "v")
        2. Coordinate values (e.g., "x", "t")

        Args:
            name: Variable name to look up.

        Returns:
            Tensor of variable values.

        Raises:
            KeyError: If variable not found in dataset.
        """
        # Check if it's a field
        if self.dataset.fields is not None and name in self.dataset.fields:
            return self.dataset.fields[name].values.to(self.device)

        # Check if it's a coordinate axis
        if self.dataset.axes is not None and name in self.dataset.axes:
            coord = self.dataset.axes[name].values.to(self.device)
            # Broadcast coordinate to field shape
            return self._broadcast_coord(name, coord)

        raise KeyError(f"Variable '{name}' not found in dataset")

    def _broadcast_coord(self, axis_name: str, coord: torch.Tensor) -> torch.Tensor:
        """Broadcast 1D coordinate to field shape.

        Args:
            axis_name: Name of the axis.
            coord: 1D tensor of coordinate values.

        Returns:
            Tensor broadcast to field shape.
        """
        if self.dataset.axis_order is None or self.dataset.axes is None:
            raise ValueError("Dataset must have axis_order and axes defined")

        # Get field shape
        field_shape = self.dataset.get_shape()

        # Find which dimension this axis corresponds to
        axis_idx = self.dataset.axis_order.index(axis_name)

        # Create shape for broadcasting: (1, 1, ..., n, ..., 1, 1)
        # where n is at axis_idx position
        broadcast_shape = [1] * len(field_shape)
        broadcast_shape[axis_idx] = coord.shape[0]

        # Reshape and expand to field shape
        coord_reshaped = coord.view(*broadcast_shape)
        return coord_reshaped.expand(*field_shape)

    def get_derivative(self, field_name: str, axis: str, order: int) -> torch.Tensor:
        """Get precomputed derivative value.

        Args:
            field_name: Name of the field (e.g., "u").
            axis: Name of the differentiation axis (e.g., "x", "t").
            order: Order of the derivative (1, 2, 3, ...).

        Returns:
            Tensor of derivative values.

        Raises:
            KeyError: If field or axis not found.
            ValueError: If order is invalid.
        """
        return self.derivative_provider.get_derivative(field_name, axis, order).to(
            self.device
        )

    def get_constant(self, name: str) -> float:
        """Get named constant value.

        Args:
            name: Constant name to look up.

        Returns:
            Constant value as float.

        Raises:
            KeyError: If constant not found.
        """
        if name not in self.constants:
            raise KeyError(f"Constant '{name}' not found in context")
        return self.constants[name]

    def diff(self, expression: torch.Tensor, axis: str, order: int = 1) -> torch.Tensor:
        """Compute derivative of an expression via the derivative provider.

        Convenience method that delegates to derivative_provider.diff().

        Args:
            expression: Tensor expression to differentiate.
            axis: Axis name to differentiate along (e.g., "x", "t").
            order: Order of the derivative (default 1).

        Returns:
            Tensor of derivative values.

        Raises:
            NotImplementedError: If provider does not support open-form diff.
            ValueError: If expression is not connected to computation graph.
        """
        return self.derivative_provider.diff(expression, axis, order)

    @property
    def spatial_axes(self) -> list[str]:
        """Spatial axes derived from dataset metadata.

        The single source of truth is ``dataset.spatial_axes``. This property
        delegates without caching the result, which avoids dataset/context
        divergence.

        Returns an empty list when ``axis_order`` is missing or ``lhs_axis``
        is empty. It intentionally does not inspect ``dataset.axes`` or
        ``dataset.fields`` because SCATTERED datasets may still expose
        ``axis_order`` and ``lhs_axis`` without grid data.
        """
        return self.dataset.spatial_axes
