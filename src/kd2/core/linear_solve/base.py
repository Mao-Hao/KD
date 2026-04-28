"""Base classes for linear solvers.

This module defines the abstract interface and data structures for
sparse linear solvers used in coefficient optimization.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch


@dataclass
class SolveResult:
    """Result from a sparse solver.

    Attributes:
        coefficients: Solution vector (n_terms,)
        residual: Squared residual ||y - theta @ xi||^2
        r2: R-squared (coefficient of determination)
        condition_number: Condition number of theta matrix (for diagnostics)
        selected_indices: Indices of non-zero coefficients (for sparse solvers)
    """

    coefficients: torch.Tensor
    residual: float
    r2: float
    condition_number: float
    selected_indices: list[int] | None = None


class SparseSolver(ABC):
    """Abstract base class for sparse solvers.

    Solves the linear system y = theta @ xi for xi, where:
    - theta: Feature matrix (n_samples, n_terms)
    - y: Target vector (n_samples,) or (n_samples, 1)
    - xi: Coefficient vector (n_terms,)
    """

    @abstractmethod
    def solve(
        self,
        theta: torch.Tensor,
        y: torch.Tensor,
    ) -> SolveResult:
        """Solve y = theta @ xi for xi.

        Args:
            theta: Feature matrix (n_samples, n_terms)
            y: Target vector (n_samples,) or (n_samples, 1)

        Returns:
            SolveResult containing coefficients and diagnostics
        """
        raise NotImplementedError
