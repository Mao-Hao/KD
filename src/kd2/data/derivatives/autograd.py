"""Autograd-based derivative provider.

Implements DerivativeProvider using PyTorch automatic differentiation
for computing derivatives through neural network models.

This is useful when the field is represented by a neural network
surrogate model, and derivatives are computed by backpropagation
through the computation graph.

Adapted from upstream surrogate-modelling work on PINN derivatives.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from torch import Tensor

from kd2.data.derivatives.base import DerivativeProvider
from kd2.data.schema import PDEDataset

logger = logging.getLogger(__name__)

# Maximum derivative order reported by available_derivatives
_DEFAULT_MAX_ORDER = 3


class AutogradProvider(DerivativeProvider):
    """Autograd-based derivative provider.

    Computes derivatives using PyTorch's autograd through a neural
    network model. Supports arbitrary-order derivatives via repeated
    application of torch.autograd.grad.

    Attributes:
        model: Neural network model mapping coords -> field values.
        coords: Dict mapping axis names to coordinate tensors
            (must have requires_grad=True).
        dataset: PDE dataset providing metadata (field names, axes).
    """

    def __init__(
        self,
        model: nn.Module,
        coords: dict[str, Tensor],
        dataset: PDEDataset,
        max_order: int = _DEFAULT_MAX_ORDER,
    ) -> None:
        """Initialize autograd provider.

        Args:
            model: Neural network model. Should accept coordinate tensors
                as keyword arguments or positional arguments matching
                axis_order, and return either a single Tensor (mapped to
                lhs_field) or a dict[str, Tensor] for multi-field output.
            coords: Dict mapping axis names to coordinate tensors.
                All tensors must have requires_grad=True.
            dataset: PDE dataset providing metadata.
            max_order: Maximum derivative order for available_derivatives.

        Raises:
            ValueError: If any coord tensor lacks requires_grad=True.
            ValueError: If coord keys don't match dataset axis_order.
            ValueError: If max_order < 1.
        """
        # Validate max_order
        if max_order < 1:
            raise ValueError(f"max_order must be >= 1, got {max_order}")

        self._validate_coords(coords, dataset)

        self.model = model
        self._coords = coords
        self.dataset = dataset
        self._max_order = max_order

    @property
    def coords(self) -> dict[str, Tensor]:
        """Coordinate tensors keyed by axis name."""
        return self._coords

    @staticmethod
    def _validate_coords(
        coords: dict[str, Tensor],
        dataset: PDEDataset,
    ) -> None:
        """Validate coordinate tensors.

        Args:
            coords: Coordinate tensors to validate.
            dataset: Dataset providing expected axis names.

        Raises:
            ValueError: If requires_grad is missing or keys mismatch.
        """
        # Check requires_grad
        for name, tensor in coords.items():
            if not tensor.requires_grad:
                raise ValueError(
                    f"Coordinate '{name}' must have requires_grad=True. "
                    f"Use tensor.requires_grad_(True) or pass "
                    f"requires_grad=True at creation."
                )

        # Check coords contain all dataset axes
        if dataset.axis_order is not None:
            expected = set(dataset.axis_order)
            actual = set(coords.keys())
            missing = expected - actual
            if missing:
                raise ValueError(
                    f"Coords missing required axes from dataset: {missing}. "
                    f"Coords have: {actual}, dataset needs: {expected}"
                )

    def _forward_model(self) -> dict[str, Tensor]:
        """Run the model forward pass and return field outputs as a dict.

        Uses enable_grad() so that model outputs have grad_fn even when
        called from a no_grad context (e.g., Evaluator).

        Returns:
            Dict mapping field names to output tensors.
        """
        # enable_grad: model output must have grad_fn for autograd.grad
        with torch.enable_grad():
            output = self.model(**self._coords)

        # Normalize output to dict
        if isinstance(output, dict):
            return output
        # Single tensor output -> map to lhs_field
        return {self.dataset.lhs_field: output}

    def get_field(self, name: str) -> Tensor:
        """Get model output for a named field.

        Args:
            name: Field name (e.g., "u", "v").

        Returns:
            Tensor of field values from the model forward pass.

        Raises:
            KeyError: If the field name is not produced by the model.
        """
        fields = self._forward_model()
        if name not in fields:
            raise KeyError(
                f"Field '{name}' not found. Available fields: {list(fields.keys())}"
            )
        return fields[name]

    def diff(
        self,
        expression: Tensor,
        axis: str,
        order: int,
    ) -> Tensor:
        """Compute derivative of an expression using autograd.

        Uses torch.autograd.grad with create_graph=True to support
        higher-order derivatives.

        Args:
            expression: Tensor expression to differentiate. Must be
                connected to coords through the computation graph.
            axis: Axis name to differentiate along (e.g., "x", "t").
            order: Order of the derivative (must be >= 1).

        Returns:
            Tensor of derivative values, same shape as expression.

        Raises:
            KeyError: If axis not found in coords.
            ValueError: If order < 1 or order > max_order.
            ValueError: If expression is not connected to the coordinate
                in the computation graph.
            TypeError: If order is not an integer.
        """
        # Validate order type
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")

        # Validate order value Enforce upper bound to match
        # FiniteDiffProvider's contract; each autograd.grad call with
        # create_graph=True compounds the computation graph, so an
        # unbounded order would cause exponential memory blow-up.
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")
        if order > self._max_order:
            raise ValueError(
                f"order {order} exceeds max_order {self._max_order}; "
                f"pass max_order=... at AutogradProvider construction to raise"
            )

        # Validate axis (KeyError for unknown axis)
        if axis not in self._coords:
            raise KeyError(
                f"Axis '{axis}' not found in coords. "
                f"Available axes: {list(self._coords.keys())}"
            )

        coord = self._coords[axis]
        result = expression

        for _step in range(order):
            # enable_grad: autograd.grad needs grad tracking even in no_grad
            with torch.enable_grad():
                try:
                    (result,) = torch.autograd.grad(
                        outputs=result,
                        inputs=coord,
                        grad_outputs=torch.ones_like(result),
                        create_graph=True,
                        retain_graph=True,
                    )
                except RuntimeError as e:
                    msg = str(e)
                    if (
                        "does not require grad" in msg
                        or "One of the differentiated Tensors" in msg
                    ):
                        raise ValueError(
                            f"Cannot compute derivative: expression is "
                            f"not connected to coordinate '{axis}' in "
                            f"the computation graph. Ensure the expression "
                            f"depends on coord '{axis}' via the model."
                        ) from e
                    raise

        return result

    def get_derivative(
        self,
        field: str,
        axis: str,
        order: int,
    ) -> Tensor:
        """Get derivative of a field with respect to an axis.

        Convenience method that combines get_field + diff.
        Uses torch.enable_grad() internally so that autograd works
        even when called from a no_grad context (e.g., Evaluator).

        Args:
            field: Field name (e.g., "u").
            axis: Axis name (e.g., "x", "t").
            order: Derivative order (must be >= 1).

        Returns:
            Tensor of derivative values.

        Raises:
            KeyError: If field or axis not found.
            ValueError: If order < 1.
        """
        # Validate order early for clear error messages
        if not isinstance(order, int):
            raise TypeError(f"order must be an integer, got {type(order).__name__}")
        if order < 1:
            raise ValueError(f"order must be >= 1, got {order}")

        # Validate axis early
        if axis not in self._coords:
            raise KeyError(
                f"Axis '{axis}' not found in coords. "
                f"Available axes: {list(self._coords.keys())}"
            )

        # enable_grad() lets autograd work inside Evaluator's no_grad context
        with torch.enable_grad():
            field_tensor = self.get_field(field)
            return self.diff(field_tensor, axis, order)

    def available_derivatives(self) -> list[tuple[str, str, int]]:
        """Return list of available derivatives.

        For AutogradProvider, derivatives are computed on-demand rather
        than precomputed. This returns all possible (field, axis, order)
        combinations up to max_order.

        Returns:
            List of (field, axis, order) tuples.
        """
        result: list[tuple[str, str, int]] = []

        # Get field names from dataset
        field_names: list[str] = []
        if self.dataset.fields is not None:
            field_names = list(self.dataset.fields.keys())
        elif self.dataset.lhs_field:
            field_names = [self.dataset.lhs_field]

        axis_names = list(self._coords.keys())

        for field_name in field_names:
            for axis_name in axis_names:
                for order in range(1, self._max_order + 1):
                    result.append((field_name, axis_name, order))

        return result
