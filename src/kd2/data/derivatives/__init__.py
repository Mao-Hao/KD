"""Derivative providers for PDE discovery.

This module provides interfaces and implementations for computing
derivatives of field data:

- DerivativeProvider: Abstract base class for derivative computation
- FiniteDiffProvider: Finite difference implementation (grid data)
- AutogradProvider: Autograd-based implementation (neural network models)

Example:
    >>> from kd2.data.derivatives import FiniteDiffProvider
    >>> provider = FiniteDiffProvider(dataset, max_order=3)
    >>> u_x = provider.get_derivative("u", "x", order=1)
"""

from kd2.data.derivatives.autograd import AutogradProvider
from kd2.data.derivatives.base import DerivativeProvider
from kd2.data.derivatives.finite_diff import FiniteDiffProvider

__all__ = [
    "AutogradProvider",
    "DerivativeProvider",
    "FiniteDiffProvider",
]
