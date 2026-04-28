"""SGA tolerance sweep + AIC scoring + evaluate_candidate pipeline.

Implements the predecessor Train() algorithm: adaptive tolerance sweep over STRidge
to find the sparsity level that minimizes AIC (Akaike Information Criterion).

Algorithm:
  1. OLS baseline (tol=0) to establish initial AIC
  2. Sweep tolerance upward, tracking best AIC
  3. On improvement: advance tolerance
  4. On overshoot: back off and shrink step size

The sweep calls STRidge at each tolerance level. The returned coefficients
include the ridge regularization effect (lam), which influences both the
AIC score and the final solution.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import Tensor

from kd2.core.metrics import aic_no_n
from kd2.search.sga.config import SGAConfig
from kd2.search.sga.evaluate import DiffContext, build_theta, prune_invalid_terms
from kd2.search.sga.pde import PDE

_LOGGER = logging.getLogger(__name__)

# Backward-compatible re-export (implementation moved to kd2.core.metrics)
compute_aic = aic_no_n

# Numerical safety constants
_ZERO_COL_EPS = 1e-14
"""Threshold for counting a coefficient as non-zero."""


@dataclass
class TrainResult:
    """Result from a single SGA training sweep.

    Attributes:
        coefficients: Solution vector (n_terms,).
        selected_indices: Indices of active (non-zero) terms.
        aic_score: AIC score for model selection.
        mse: Mean squared error (SSR / n_samples).
        best_tol: Tolerance value that achieved the best AIC.
    """

    coefficients: Tensor
    selected_indices: list[int]
    aic_score: float
    mse: float
    best_tol: float


@dataclass
class CandidateResult:
    """Result from evaluate_candidate with genotype-aligned information.

    Bundles the TrainResult with the pruned PDE genotype so that
    coefficients, selected_indices, and PDE terms are all aligned
    to the same surviving-terms basis.

    Attributes:
        train_result: The TrainResult from train_sweep.
        pruned_pde: PDE with only the surviving (valid) terms.
        valid_term_indices: Original indices of surviving terms in the
            input PDE (maps pruned_pde.terms[i] back to original PDE).
    """

    train_result: TrainResult
    pruned_pde: PDE
    valid_term_indices: list[int]

    # -- Delegate TrainResult fields for backward compatibility ----------------

    @property
    def coefficients(self) -> Tensor:
        """Solution vector from the train sweep."""
        return self.train_result.coefficients

    @property
    def selected_indices(self) -> list[int]:
        """Indices of active (non-zero) terms in the theta matrix."""
        return self.train_result.selected_indices

    @property
    def aic_score(self) -> float:
        """AIC score for model selection."""
        return self.train_result.aic_score

    @property
    def mse(self) -> float:
        """Mean squared error."""
        return self.train_result.mse

    @property
    def best_tol(self) -> float:
        """Tolerance value that achieved the best AIC."""
        return self.train_result.best_tol


def _invalid_result(n_terms: int, device: torch.device) -> TrainResult:
    """Create a TrainResult representing an invalid/empty solution."""
    return TrainResult(
        coefficients=torch.zeros(n_terms, device=device),
        selected_indices=[],
        aic_score=float("inf"),
        mse=float("inf"),
        best_tol=0.0,
    )


def _active_mask(w: Tensor) -> Tensor:
    """Boolean mask of active (non-zero) coefficients: ``|w| > eps``.

    Single source of truth for the coefficient-activity threshold so
    ``_count_active`` and ``_selected_indices`` cannot drift apart if
    the criterion is tightened
    """
    return w.abs() > _ZERO_COL_EPS


def _count_active(w: Tensor) -> int:
    """Count non-zero coefficients in a weight vector."""
    return int(_active_mask(w).sum().item())


def _selected_indices(w: Tensor) -> list[int]:
    """Get indices of non-zero coefficients."""
    return _active_mask(w).nonzero(as_tuple=True)[0].tolist()


def _compute_mse(theta: Tensor, y_1d: Tensor, w: Tensor, n_samples: int) -> float:
    """Compute mean squared error: ||y - theta @ w||^2 / n."""
    residual = ((y_1d - theta @ w) ** 2).sum().item()
    return residual / n_samples


def _validate_inputs(theta: Tensor, y: Tensor) -> None:
    """Validate theta and y for train_sweep.

    Raises:
        ValueError: If inputs contain NaN, Inf, are empty, or have
            mismatched dimensions.
    """
    if theta.dim() != 2:
        raise ValueError(f"theta must be 2D, got {theta.dim()}D")

    if y.dim() < 1:
        raise ValueError(f"y must be at least 1D, got {y.dim()}D")

    y_1d = y.squeeze(-1) if y.dim() == 2 else y
    if y_1d.shape[0] != theta.shape[0]:
        raise ValueError(
            f"dimension mismatch: theta has {theta.shape[0]} rows, "
            f"y has {y_1d.shape[0]} elements"
        )

    if torch.isnan(theta).any():
        raise ValueError("theta contains NaN values")
    if torch.isnan(y).any():
        raise ValueError("y contains NaN values")

    if torch.isinf(theta).any():
        raise ValueError("theta contains Inf values")
    if torch.isinf(y).any():
        raise ValueError("y contains Inf values")


def _stridge_no_debias(
    theta: Tensor,
    y_1d: Tensor,
    lam: float,
    max_iter: int,
    tol: float,
    normalize: int,
) -> Tensor:
    """Run STRidge without the final debiasing step.

    Returns full-size coefficient vector (n_terms,) with ridge-regularized
    coefficients (not debiased). This preserves the lam effect in the
    returned coefficients, which is important for AIC-based model selection.

    Args:
        theta: Feature matrix (n_samples, n_terms).
        y_1d: Target vector (n_samples,).
        lam: Ridge regularization parameter (0 = OLS).
        max_iter: Maximum thresholding iterations.
        tol: Coefficient threshold.
        normalize: Column norm order (0 to skip normalization).

    Returns:
        Coefficient vector (n_terms,).
    """
    n, d = theta.shape

    if d == 0:
        return torch.zeros(0, dtype=theta.dtype, device=theta.device)

    # Step 1: Detect and remove zero columns
    col_norms = torch.linalg.norm(theta, dim=0)
    nonzero_mask = col_norms > _ZERO_COL_EPS
    nonzero_cols = nonzero_mask.nonzero(as_tuple=True)[0]

    if nonzero_cols.numel() == 0:
        return torch.zeros(d, dtype=theta.dtype, device=theta.device)

    x0 = theta[:, nonzero_cols]
    d_reduced = x0.shape[1]

    # Step 2: Column normalization
    if normalize != 0:
        mreg = torch.zeros(d_reduced, 1, dtype=theta.dtype, device=theta.device)
        x_norm = torch.zeros_like(x0)
        for i in range(d_reduced):
            cn = torch.linalg.norm(x0[:, i], ord=normalize).item()
            mreg[i, 0] = 1.0 / cn
            x_norm[:, i] = mreg[i, 0] * x0[:, i]
    else:
        x_norm = x0
        mreg = torch.ones(d_reduced, 1, dtype=theta.dtype, device=theta.device)

    # Step 3: Initial solve
    w = _solve(x_norm, y_1d, lam, d_reduced)

    # Step 4: Iterative thresholding
    num_relevant = d_reduced
    biginds: list[int] = list(range(d_reduced))

    for j in range(max_iter):
        smallinds = (w.abs() < tol).squeeze(-1).nonzero(as_tuple=True)[0]
        smallinds_set = set(smallinds.tolist())
        new_biginds = [i for i in range(d_reduced) if i not in smallinds_set]

        if num_relevant == len(new_biginds):
            break
        num_relevant = len(new_biginds)

        if len(new_biginds) == 0:
            if j == 0:
                # Return initial w (before thresholding)
                w_out = mreg * w if normalize != 0 else w
                full_w = torch.zeros(d, dtype=theta.dtype, device=theta.device)
                full_w[nonzero_cols] = w_out.squeeze(-1)
                return full_w
            else:
                break

        biginds = new_biginds
        w[smallinds] = 0
        w[biginds] = _solve(x_norm[:, biginds], y_1d, lam, len(biginds))

    # Step 5: NO debiasing (unlike STRidgeSolver.solve_with_tol)
    # The lam effect is preserved in the coefficients.

    # Step 6: Un-normalize
    if normalize != 0:
        w = mreg * w

    # Step 7: Map back to original column indices
    full_w = torch.zeros(d, dtype=theta.dtype, device=theta.device)
    full_w[nonzero_cols] = w.squeeze(-1)
    return full_w


def _solve(x: Tensor, y: Tensor, lam: float, d: int) -> Tensor:
    """Solve X @ w = y with optional ridge regularization.

    Returns w as (d, 1) tensor.
    """
    y_2d = y.unsqueeze(1) if y.dim() == 1 else y
    if lam != 0:
        xtx = x.T @ x + lam * torch.eye(d, dtype=x.dtype, device=x.device)
        xty = x.T @ y_2d
        sol: Tensor = torch.linalg.lstsq(xtx, xty).solution
        return sol
    sol_ols: Tensor = torch.linalg.lstsq(x, y_2d).solution
    return sol_ols


def train_sweep(
    theta: Tensor,
    y: Tensor,
    config: SGAConfig,
) -> TrainResult:
    """Adaptive tolerance sweep (the predecessor Train() algorithm).

    Sweeps STRidge tolerance from 0 upward, evaluating AIC at each step.
    Uses backoff with shrinking step size when AIC worsens.

    The sweep uses ridge-regularized coefficients (without debiasing) for
    AIC evaluation. This ensures that different lam values lead to different
    model selection outcomes. The returned coefficients reflect the solution
    at the best tolerance found.

    Args:
        theta: Feature matrix (n_samples, n_terms).
        y: Target vector (n_samples,) or (n_samples, 1).
        config: SGA configuration with sweep parameters.

    Returns:
        TrainResult with the best AIC solution found.
    """
    _validate_inputs(theta, y)

    n_terms = theta.shape[1]
    device = theta.device

    if n_terms == 0:
        return _invalid_result(0, device)

    n_samples = theta.shape[0]
    y_1d = y.squeeze(-1) if y.dim() == 2 else y

    with torch.no_grad():
        return _train_sweep_impl(theta, y_1d, n_samples, n_terms, device, config)


def _train_sweep_impl(
    theta: Tensor,
    y_1d: Tensor,
    n_samples: int,
    n_terms: int,
    device: torch.device,
    config: SGAConfig,
) -> TrainResult:
    """Inner implementation of train_sweep (runs under no_grad)."""
    # Step 1: OLS baseline (tol=0, lam=0, the predecessor convention)
    w_baseline = _stridge_no_debias(
        theta,
        y_1d,
        lam=0.0,
        max_iter=config.str_iters,
        tol=0.0,
        normalize=config.normalize,
    )
    mse = _compute_mse(theta, y_1d, w_baseline, n_samples)
    k = _count_active(w_baseline)
    best_aic = aic_no_n(mse, k, config.aic_ratio)
    best = TrainResult(
        coefficients=w_baseline.clone(),
        selected_indices=_selected_indices(w_baseline),
        aic_score=best_aic,
        mse=mse,
        best_tol=0.0,
    )

    # Step 2: Adaptive tolerance sweep
    tol = config.d_tol
    d_tol = config.d_tol

    for iter_idx in range(config.maxit):
        w = _stridge_no_debias(
            theta,
            y_1d,
            lam=config.lam,
            max_iter=config.str_iters,
            tol=tol,
            normalize=config.normalize,
        )
        mse = _compute_mse(theta, y_1d, w, n_samples)
        k = _count_active(w)
        aic = aic_no_n(mse, k, config.aic_ratio)

        if aic <= best_aic:
            # Improvement: record and advance
            best_aic = aic
            best = TrainResult(
                coefficients=w.clone(),
                selected_indices=_selected_indices(w),
                aic_score=aic,
                mse=mse,
                best_tol=tol,
            )
            tol += d_tol
        else:
            # Overshoot: backoff and shrink step
            tol = max(0.0, tol - 2.0 * d_tol)
            remaining = config.maxit - iter_idx
            if remaining > 0:
                d_tol = 2.0 * d_tol / remaining
            tol += d_tol

    return best


def evaluate_candidate(
    pde: PDE,
    data_dict: dict[str, Tensor],
    default_terms: Tensor | None,
    y: Tensor,
    config: SGAConfig,
    diff_ctx: DiffContext | None = None,
) -> CandidateResult:
    """Full pipeline: prune_invalid_terms -> build_theta -> train_sweep.

    Executes all PDE terms against data, prunes invalid terms from the
    genotype, builds the feature matrix (with optional default terms
    prepended), then runs the tolerance sweep to find the best sparse
    solution.

    The returned CandidateResult bundles the TrainResult with the pruned
    PDE, ensuring coefficients and selected_indices are aligned with the
    surviving terms. The input PDE is NOT modified.

    Args:
        pde: PDE candidate with term Trees.
        data_dict: Mapping from variable names to data tensors.
        default_terms: Optional pre-computed default features to prepend.
        y: Target vector (n_samples,) or (n_samples, 1).
        config: SGA configuration.
        diff_ctx: Optional finite-difference context for derivative execution
            and lhs-axis filtering.

    Returns:
        CandidateResult with the best AIC solution and pruned genotype.
    """
    # Step 1: Execute + prune PDE terms (genotype sync)
    pruned_pde, valid_terms, valid_term_indices = prune_invalid_terms(
        pde,
        data_dict,
        diff_ctx=diff_ctx,
    )

    # Step 2: Build theta matrix (prepend defaults)
    theta = build_theta(valid_terms, default_terms)

    # Step 3: If no columns, return invalid
    if theta.shape[1] == 0:
        return CandidateResult(
            train_result=_invalid_result(0, theta.device),
            pruned_pde=pruned_pde,
            valid_term_indices=valid_term_indices,
        )

    # Step 4: Train sweep
    train_result = train_sweep(theta, y, config)
    return CandidateResult(
        train_result=train_result,
        pruned_pde=pruned_pde,
        valid_term_indices=valid_term_indices,
    )
