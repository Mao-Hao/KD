"""Least squares solver implementation.

This module provides a standard least squares solver using torch.linalg.lstsq.
"""

from __future__ import annotations

import torch

from kd2.core.linear_solve.base import SolveResult, SparseSolver

# Epsilon for numerical comparisons (e.g., R2 denominator check)
_R2_EPS = 1e-15


class LeastSquaresSolver(SparseSolver):
    """Least squares solver using torch.linalg.lstsq.

    Solves y = theta @ xi by minimizing ||y - theta @ xi||^2.

    For overdetermined systems (n > m), finds the least squares solution.
    For underdetermined systems (n < m), finds the minimum norm solution.

    Attributes:
        rcond: Cutoff for small singular values. If None, uses machine precision.
    """

    def __init__(self, rcond: float | None = None) -> None:
        """Initialize solver.

        Args:
            rcond: Cutoff ratio for small singular values.
                   Singular values s[i] <= rcond * max(s) are treated as zero.
                   If None, uses machine precision based default.
        """
        self.rcond = rcond

    def solve(
        self,
        theta: torch.Tensor,
        y: torch.Tensor,
    ) -> SolveResult:
        """Solve y = theta @ xi using least squares.

        Args:
            theta: Feature matrix (n_samples, n_terms)
            y: Target vector (n_samples,) or (n_samples, 1)

        Returns:
            SolveResult with coefficients, residual, r2, and condition number

        Raises:
            ValueError: If inputs contain NaN, Inf, are empty, or have
                        mismatched dimensions
        """
        # Validate inputs
        self._validate_inputs(theta, y)

        # Normalize y to 1D (use squeeze(-1) to avoid 0D tensor for shape (1,1))
        y_1d = y.squeeze(-1) if y.dim() == 2 else y

        # Compute condition number
        condition_number = self._compute_condition_number(theta)

        # Solve using lstsq
        # torch.linalg.lstsq signature: lstsq(A, B, rcond=None, *, driver=None)
        # Returns: solution, residuals, rank, singular_values
        result = torch.linalg.lstsq(theta, y_1d.unsqueeze(1), rcond=self.rcond)
        coefficients = result.solution.squeeze()

        # Ensure coefficients is 1D (n_terms,)
        if coefficients.dim() == 0:
            coefficients = coefficients.unsqueeze(0)

        # Compute residual: ||y - theta @ xi||^2
        y_pred = theta @ coefficients
        residual = ((y_1d - y_pred) ** 2).sum().item()

        # Compute R^2: 1 - SS_res / SS_tot
        r2 = self._compute_r2(y_1d, y_pred)

        return SolveResult(
            coefficients=coefficients,
            residual=residual,
            r2=r2,
            condition_number=condition_number,
            selected_indices=None,
        )

    def _validate_inputs(self, theta: torch.Tensor, y: torch.Tensor) -> None:
        """Validate input tensors.

        Args:
            theta: Feature matrix
            y: Target vector

        Raises:
            ValueError: If inputs are invalid
        """
        # Check theta is 2D
        if theta.dim() != 2:
            raise ValueError(f"theta must be 2D, got {theta.dim()}D")

        # Check y is at least 1D (reject 0D tensors)
        if y.dim() < 1:
            raise ValueError(f"y must be at least 1D, got {y.dim()}D (0D tensor)")

        # Check for empty tensors
        if theta.shape[0] == 0 or theta.shape[1] == 0:
            raise ValueError("theta is empty (0 rows or 0 columns)")

        # Check dtype consistency (avoid silent precision loss)
        if theta.dtype != y.dtype:
            raise ValueError(
                f"dtype mismatch: theta is {theta.dtype}, y is {y.dtype}. "
                "Cast to same dtype before solving to avoid precision loss."
            )

        # Check for NaN
        if torch.isnan(theta).any():
            raise ValueError("theta contains NaN values")
        if torch.isnan(y).any():
            raise ValueError("y contains NaN values")

        # Check for Inf
        if torch.isinf(theta).any():
            raise ValueError("theta contains Inf values")
        if torch.isinf(y).any():
            raise ValueError("y contains Inf values")

        # Normalize y for dimension check (use squeeze(-1) to avoid 0D for shape (1,1))
        y_1d = y.squeeze(-1) if y.dim() == 2 else y

        # Check y shape compatibility
        if y.dim() == 2 and y.shape[1] != 1:
            raise ValueError(f"y must be 1D or (n, 1), got shape {y.shape}")

        # Check dimension match
        n_samples = theta.shape[0]
        if y_1d.shape[0] != n_samples:
            raise ValueError(
                f"dimension mismatch: theta has {n_samples} rows, "
                f"y has {y_1d.shape[0]} elements"
            )

    def _compute_condition_number(self, theta: torch.Tensor) -> float:
        """Compute condition number of theta matrix.

        Args:
            theta: Feature matrix

        Returns:
            Condition number (inf for singular/all-zero matrices)
        """
        # Check for all-zero matrix
        if (theta == 0).all():
            return float("inf")

        try:
            cond: float = float(torch.linalg.cond(theta).item())
            return cond
        except RuntimeError:
            # Handle singular or near-singular matrices
            return float("inf")

    def _compute_r2(self, y: torch.Tensor, y_pred: torch.Tensor) -> float:
        """Compute R-squared (coefficient of determination).

        R^2 = 1 - SS_res / SS_tot

        When SS_tot is near zero (constant or near-constant y), returns 0.0
        as per design decision to avoid numerical instability.

        Args:
            y: True target values
            y_pred: Predicted values

        Returns:
            R-squared value, clamped to [-1.0, 1.0] for numerical stability
        """
        ss_res = ((y - y_pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()

        # Handle constant or near-constant y case (SS_tot near 0)
        # Using epsilon comparison instead of exact == 0.0 to handle
        # floating point precision issues with near-constant y
        if ss_tot < _R2_EPS:
            return 0.0

        r2 = 1.0 - ss_res / ss_tot

        # Clamp R2 to valid range for numerical stability
        # R2 can technically be negative (worse than mean prediction)
        # but extreme values indicate numerical issues
        return max(-1.0, min(1.0, r2))
