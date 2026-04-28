"""Abstract base class for derivative providers.

This module defines the DerivativeProvider protocol that all
derivative computation implementations must follow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class DerivativeProvider(ABC):
    """Abstract base class for derivative providers.

    A derivative provider computes derivatives of field data with respect
    to coordinate axes. Implementations may use different methods:
    - Finite differences (FiniteDiffProvider)
    - Automatic differentiation (AutogradProvider)
    - Neural network approximation

    Attributes:
        dataset: The underlying PDE dataset.
    """

    @property
    def coords(self) -> dict[str, torch.Tensor]:
        """Coordinate tensors keyed by axis name.

        Subclasses can override this to expose runtime coordinate tensors.
        The default implementation returns an empty mapping so existing
        lightweight test doubles do not need to implement it.
        """
        return {}

    @abstractmethod
    def get_derivative(
        self,
        field: str,
        axis: str,
        order: int,
    ) -> torch.Tensor:
        """Get precomputed derivative of a field.

        Args:
            field: Name of the field (e.g., "u", "v").
            axis: Name of the axis to differentiate along (e.g., "x", "t").
            order: Order of the derivative (1, 2, 3, ...).

        Returns:
            Tensor of derivative values with same shape as field data.

        Raises:
            KeyError: If field or axis not found.
            ValueError: If order is invalid.
        """
        raise NotImplementedError

    @abstractmethod
    def diff(
        self,
        expression: torch.Tensor,
        axis: str,
        order: int,
    ) -> torch.Tensor:
        """Compute derivative of an expression (open-form differentiation).

        This method computes derivatives of arbitrary tensor expressions,
        not just precomputed field derivatives.

        Args:
            expression: Tensor expression to differentiate.
            axis: Name of the axis to differentiate along.
            order: Order of the derivative.

        Returns:
            Tensor of derivative values.

        Raises:
            NotImplementedError: If open-form diff is not supported.
        """
        raise NotImplementedError

    @abstractmethod
    def available_derivatives(self) -> list[tuple[str, str, int]]:
        """Return list of available precomputed derivatives.

        Returns:
            List of (field, axis, order) tuples representing available
            precomputed derivatives.

        Example:
            >>> provider.available_derivatives()
            [("u", "x", 1), ("u", "x", 2), ("u", "t", 1), ...]
        """
        raise NotImplementedError
