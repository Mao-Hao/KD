"""Sequential Threshold Ridge (STRidge) regression solver.

Ported from a predecessor implementation.
This is a PyTorch implementation that preserves exact algorithmic equivalence
with the numpy reference.

Algorithm: iteratively threshold small coefficients and re-solve on the
remaining support, producing sparse solutions to y = theta @ xi.
"""

from __future__ import annotations

import logging

import torch

from kd2.core.linear_solve.base import SolveResult, SparseSolver

logger = logging.getLogger(__name__)

# Default parameters matching the predecessor
_DEFAULT_TOL = 0.1
_DEFAULT_LAM = 0.0
_DEFAULT_MAX_ITER = 10
_DEFAULT_NORMALIZE = 2

# Numerical safety
_ZERO_COL_EPS = 1e-14
_R2_EPS = 1e-15
_COND_WARN_THRESHOLD = 1e10


class STRidgeSolver(SparseSolver):
    """Sequential Threshold Ridge regression solver.

    Iteratively thresholds small coefficients and re-solves the linear system
    on the remaining support. Produces sparse solutions.

    Attributes:
        tol: Threshold below which coefficients are set to zero.
        lam: Ridge regularization parameter. 0 uses OLS.
        max_iter: Maximum number of thresholding iterations.
        normalize: Norm order for column normalization (0 to skip).
    """

    def __init__(
        self,
        tol: float = _DEFAULT_TOL,
        lam: float = _DEFAULT_LAM,
        max_iter: int = _DEFAULT_MAX_ITER,
        normalize: int = _DEFAULT_NORMALIZE,
    ) -> None:
        self.tol = tol
        self.lam = lam
        self.max_iter = max_iter
        self.normalize = normalize

    def solve(
        self,
        theta: torch.Tensor,
        y: torch.Tensor,
    ) -> SolveResult:
        """Solve y = theta @ xi using STRidge with self.tol.

        Args:
            theta: Feature matrix (n_samples, n_terms)
            y: Target vector (n_samples,) or (n_samples, 1)

        Returns:
            SolveResult with coefficients, residual, r2, condition_number,
            and selected_indices.

        Raises:
            ValueError: If inputs contain NaN, Inf, are empty, or have
                        mismatched dimensions.
        """
        return self.solve_with_tol(theta, y, tol=self.tol)

    def solve_with_tol(
        self,
        theta: torch.Tensor,
        y: torch.Tensor,
        tol: float,
    ) -> SolveResult:
        """Solve y = theta @ xi using STRidge with a specified tolerance.

        Args:
            theta: Feature matrix (n_samples, n_terms)
            y: Target vector (n_samples,) or (n_samples, 1)
            tol: Threshold for zeroing small coefficients.

        Returns:
            SolveResult with coefficients, residual, r2, condition_number,
            and selected_indices.

        Raises:
            ValueError: If inputs contain NaN, Inf, are empty, or have
                        mismatched dimensions.
        """
        self._validate_inputs(theta, y)

        with torch.no_grad():
            y_1d = y.squeeze(-1) if y.dim() == 2 else y
            _n, d = theta.shape

            # Condition number on original matrix
            condition_number = _compute_condition_number(theta)
            if condition_number > _COND_WARN_THRESHOLD:
                logger.warning(
                    "High condition number %.2e detected in theta matrix",
                    condition_number,
                )

            # Step 1: Detect and remove zero columns
            zero_mask = _detect_zero_columns(theta)
            nonzero_cols = (~zero_mask).nonzero(as_tuple=True)[0]

            if nonzero_cols.numel() == 0:
                # All columns are zero
                return _build_result(
                    torch.zeros(d, dtype=theta.dtype, device=theta.device),
                    theta,
                    y_1d,
                    condition_number,
                )

            x0 = theta[:, nonzero_cols]
            d_reduced = x0.shape[1]

            # Step 2: Column normalization
            x_norm, mreg = _normalize_columns(x0, self.normalize)

            # Step 3: Initial solve
            w = _initial_solve(x_norm, y_1d, self.lam, d_reduced)

            # Step 4: Iterative thresholding
            w, biginds = _iterative_threshold(
                x_norm,
                y_1d,
                w,
                tol,
                self.lam,
                self.max_iter,
                d_reduced,
            )

            # Step 5: Debiased OLS on final support
            if len(biginds) > 0:
                w[biginds] = _lstsq(x_norm[:, biginds], y_1d)

            # Step 6: Un-normalize
            if self.normalize != 0:
                w = mreg * w

            # Step 7: Map back to original column indices
            full_w = torch.zeros(d, dtype=theta.dtype, device=theta.device)
            full_w[nonzero_cols] = w.squeeze(-1)

            return _build_result(full_w, theta, y_1d, condition_number)

    def _validate_inputs(self, theta: torch.Tensor, y: torch.Tensor) -> None:
        """Validate input tensors.

        Raises:
            ValueError: If inputs are invalid.
        """
        if theta.dim() != 2:
            raise ValueError(f"theta must be 2D, got {theta.dim()}D")

        if y.dim() < 1:
            raise ValueError(f"y must be at least 1D, got {y.dim()}D")

        if theta.shape[0] == 0 or theta.shape[1] == 0:
            raise ValueError("theta is empty (0 rows or 0 columns)")

        if theta.dtype != y.dtype:
            raise ValueError(
                f"dtype mismatch: theta is {theta.dtype}, y is {y.dtype}. "
                "Cast to same dtype before solving to avoid precision loss."
            )

        if torch.isnan(theta).any():
            raise ValueError("theta contains NaN values")
        if torch.isnan(y).any():
            raise ValueError("y contains NaN values")

        if torch.isinf(theta).any():
            raise ValueError("theta contains Inf values")
        if torch.isinf(y).any():
            raise ValueError("y contains Inf values")

        y_1d = y.squeeze(-1) if y.dim() == 2 else y
        if y.dim() == 2 and y.shape[1] != 1:
            raise ValueError(f"y must be 1D or (n, 1), got shape {y.shape}")

        if y_1d.shape[0] != theta.shape[0]:
            raise ValueError(
                f"dimension mismatch: theta has {theta.shape[0]} rows, "
                f"y has {y_1d.shape[0]} elements"
            )


# ============================================================
# Private helper functions
# ============================================================


def _detect_zero_columns(theta: torch.Tensor) -> torch.Tensor:
    """Return boolean mask of columns with norm < epsilon."""
    col_norms = torch.linalg.norm(theta, dim=0)
    mask: torch.Tensor = col_norms < _ZERO_COL_EPS
    return mask


def _normalize_columns(
    x0: torch.Tensor, normalize: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize columns of x0 by their p-norm.

    Args:
        x0: Matrix to normalize (n, d).
        normalize: Norm order (0 to skip normalization).

    Returns:
        (x_norm, mreg) where x_norm is normalized, mreg is (d, 1) scale factors.
    """
    d = x0.shape[1]
    if normalize == 0:
        mreg = torch.ones(d, 1, dtype=x0.dtype, device=x0.device)
        return x0, mreg

    mreg = torch.zeros(d, 1, dtype=x0.dtype, device=x0.device)
    x_norm = torch.zeros_like(x0)
    for i in range(d):
        col_norm = torch.linalg.norm(x0[:, i], ord=normalize).item()
        mreg[i, 0] = 1.0 / col_norm
        x_norm[:, i] = mreg[i, 0] * x0[:, i]
    return x_norm, mreg


def _lstsq(mat: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Solve mat @ x = b via least squares. Returns x as (k, 1)."""
    b_2d = b.unsqueeze(1) if b.dim() == 1 else b
    sol: torch.Tensor = torch.linalg.lstsq(mat, b_2d).solution
    return sol


def _ridge_solve(x: torch.Tensor, y: torch.Tensor, lam: float, d: int) -> torch.Tensor:
    """Solve (X^T X + lam I) w = X^T y. Returns w as (d, 1)."""
    y_2d = y.unsqueeze(1) if y.dim() == 1 else y
    xtx = x.T @ x + lam * torch.eye(d, dtype=x.dtype, device=x.device)
    xty = x.T @ y_2d
    sol: torch.Tensor = torch.linalg.lstsq(xtx, xty).solution
    return sol


def _initial_solve(
    x: torch.Tensor, y: torch.Tensor, lam: float, d: int
) -> torch.Tensor:
    """Compute initial solution: ridge if lam != 0, OLS otherwise."""
    if lam != 0:
        return _ridge_solve(x, y, lam, d)
    return _lstsq(x, y)


def _iterative_threshold(
    x: torch.Tensor,
    y: torch.Tensor,
    w: torch.Tensor,
    tol: float,
    lam: float,
    max_iter: int,
    d: int,
) -> tuple[torch.Tensor, list[int]]:
    """Run iterative thresholding loop.

    Returns:
        (w, biginds) -- updated coefficients and final support indices.
    """
    num_relevant = d
    biginds: list[int] = list(range(d))

    for j in range(max_iter):
        # Identify small coefficients
        smallinds = (w.abs() < tol).squeeze(-1).nonzero(as_tuple=True)[0]
        smallinds_set = set(smallinds.tolist())

        # Complement definition (NaN-safe)
        new_biginds = [i for i in range(d) if i not in smallinds_set]

        # Convergence: support size unchanged
        if num_relevant == len(new_biginds):
            break
        num_relevant = len(new_biginds)

        # Empty support check
        if len(new_biginds) == 0:
            if j == 0:
                # Return initial w (before any thresholding)
                return w, biginds
            else:
                break

        biginds = new_biginds

        # Zero out small coefficients
        w[smallinds] = 0

        # Re-solve on support
        if lam != 0:
            w[biginds] = _ridge_solve(x[:, biginds], y, lam, len(biginds))
        else:
            w[biginds] = _lstsq(x[:, biginds], y)

    return w, biginds


def _compute_condition_number(theta: torch.Tensor) -> float:
    """Compute condition number of theta matrix."""
    if (theta == 0).all():
        return float("inf")
    try:
        return float(torch.linalg.cond(theta).item())
    except RuntimeError:
        return float("inf")


def _compute_r2(y: torch.Tensor, y_pred: torch.Tensor) -> float:
    """Compute R-squared (coefficient of determination)."""
    ss_res = ((y - y_pred) ** 2).sum().item()
    ss_tot = ((y - y.mean()) ** 2).sum().item()
    if ss_tot < _R2_EPS:
        return 0.0
    r2 = 1.0 - ss_res / ss_tot
    return max(-1.0, min(1.0, r2))


def _build_result(
    full_w: torch.Tensor,
    theta: torch.Tensor,
    y_1d: torch.Tensor,
    condition_number: float,
) -> SolveResult:
    """Build SolveResult from full coefficient vector."""
    y_pred = theta @ full_w
    residual = ((y_1d - y_pred) ** 2).sum().item()
    r2 = _compute_r2(y_1d, y_pred)

    # Selected indices: positions with non-zero coefficients
    nonzero_mask = full_w.abs() > _ZERO_COL_EPS
    selected_indices = nonzero_mask.nonzero(as_tuple=True)[0].tolist()

    return SolveResult(
        coefficients=full_w,
        residual=residual,
        r2=r2,
        condition_number=condition_number,
        selected_indices=selected_indices,
    )
