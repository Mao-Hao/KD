"""Evaluator for LINEAR mode expression evaluation.

This module provides the Evaluator class which encapsulates the complete
evaluation pipeline for symbolic regression:

1. Execute terms to build feature matrix Θ
2. Solve y = Θξ via least squares
3. Compute evaluation metrics (MSE, NMSE, R², AIC)

The Evaluator is the main interface for coefficient recovery.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from kd2.core.metrics import ScorerFn, make_aic_scorer
from kd2.core.metrics import nmse as _metrics_nmse

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from kd2.core.executor import ExecutionContext
    from kd2.core.expr import PythonExecutor
    from kd2.core.linear_solve import SparseSolver


@dataclass
class EvaluationResult:
    """Result from evaluating an expression or term list.

    Attributes:
        mse: Mean squared error between prediction and target.
        nmse: Normalized MSE (MSE / variance of target).
        r2: R-squared (coefficient of determination).
        aic: Akaike Information Criterion (if computed).
        complexity: Number of active terms for AIC.
            Selected count for sparse solvers, total for dense.
        coefficients: Fitted coefficients (one per term).
        is_valid: Whether evaluation succeeded.
        error_message: Error description if is_valid is False.
        selected_indices: Indices of terms selected by sparse solver
            (None for dense solvers). Relative to ``terms``.
        residuals: Detached residual tensor (predicted - actual),
            shape ``(n_samples,)``. Sign convention: positive means
            over-prediction.
        terms: Term strings used for evaluation (defensive copy).
        expression: Original expression string, set only by
            ``evaluate_expression()``.
    """

    mse: float
    nmse: float
    r2: float
    aic: float | None = None
    complexity: int = 0
    coefficients: Tensor | None = None
    is_valid: bool = True
    error_message: str = ""
    selected_indices: list[int] | None = None
    residuals: Tensor | None = None
    terms: list[str] | None = None
    expression: str = ""


class Evaluator:
    """Expression evaluator for LINEAR mode coefficient fitting.

    The Evaluator orchestrates the evaluation pipeline:
    1. Execute each term to compute Θ matrix columns
    2. Flatten and solve y = Θξ via least squares
    3. Compute quality metrics

    This is the main entry point for evaluating candidate expressions
    in symbolic regression.

    Examples:
        >>> evaluator = Evaluator(executor, solver, context, lhs=u_t)
        >>> result = evaluator.evaluate_terms(["mul(u, u_x)", "u_xx"])
        >>> print(f"Coefficients: {result.coefficients}")
        >>> print(f"R²: {result.r2}")

        >>> result = evaluator.evaluate_expression("add(mul(u, u_x), u_xx)")
        >>> # Automatically splits and evaluates
    """

    def __init__(
        self,
        executor: PythonExecutor,
        solver: SparseSolver,
        context: ExecutionContext,
        lhs: Tensor,
        penalty_value: float = 1e10,
        scorer: ScorerFn | None = None,
    ) -> None:
        """Initialize the Evaluator.

        Args:
            executor: PythonExecutor for term evaluation.
            solver: SparseSolver (e.g., LeastSquaresSolver) for coefficient fitting.
            context: ExecutionContext with variables and derivatives.
            lhs: Target tensor (e.g., u_t for PDE discovery).
            penalty_value: Value to return for invalid expressions.
            scorer: Optional custom scorer ``(mse, k) -> score``.
                Defaults to ``make_aic_scorer(n_samples)``.
        """
        self._executor = executor
        self._solver = solver
        self._context = context
        self._lhs = lhs
        self._penalty_value = penalty_value

        # Precompute flattened LHS for solving
        self._lhs_flat = lhs.flatten()
        self._n_samples = self._lhs_flat.shape[0]

        # Precompute LHS variance for NMSE
        self._lhs_var = self._lhs_flat.var().item()

        # Scorer for model selection (default: standard AIC)
        self._scorer = scorer or make_aic_scorer(self._n_samples)

    @property
    def lhs_target(self) -> Tensor:
        """Return the flattened regression target."""
        return self._lhs_flat.detach()

    @property
    def executor(self) -> PythonExecutor:
        """Expose the injected term executor (read-only)."""
        return self._executor

    @property
    def solver(self) -> SparseSolver:
        """Expose the injected linear solver (read-only)."""
        return self._solver

    @property
    def context(self) -> ExecutionContext:
        """Expose the injected execution context (read-only)."""
        return self._context

    def build_theta_matrix(
        self,
        terms: list[str],
        *,
        skip_invalid: bool = False,
    ) -> tuple[Tensor, list[str]]:
        """Execute terms and return the Theta matrix used for fitting."""
        if not terms:
            raise ValueError("Empty term list")
        with torch.no_grad():
            theta, valid_terms = self._build_theta(
                terms,
                skip_invalid=skip_invalid,
            )
        return theta.detach(), list(valid_terms)

    def evaluate_terms(
        self, terms: list[str], *, skip_invalid: bool = False
    ) -> EvaluationResult:
        """Evaluate a list of terms (core API).

        This is the primary evaluation method. Given a list of term expressions:
        1. Execute each term to get tensor values
        2. Build feature matrix Θ (n_samples × n_terms)
        3. Solve y = Θξ for coefficients ξ
        4. Compute evaluation metrics

        Args:
            terms: List of term expression strings.
            skip_invalid: If True, skip terms that fail execution,
                contain NaN/Inf, or are all zeros. Default False
                preserves existing raise-on-failure behavior.

        Returns:
            EvaluationResult with coefficients and metrics.
        """
        # Handle empty terms
        if not terms:
            return self._make_invalid_result("Empty term list")

        # Use no_grad to prevent autograd graph accumulation
        # (important when input tensors have requires_grad=True).
        # catch ``torch.OutOfMemoryError`` so deep diff_* candidates
        # that blow up GPU memory inside ``torch.autograd.grad`` degrade to
        # an invalid result instead of crashing the whole search run.
        try:
            with torch.no_grad():
                return self._evaluate_terms_impl(terms, skip_invalid=skip_invalid)
        except torch.cuda.OutOfMemoryError as exc:
            logger.warning(
                "Autograd OOM during evaluate_terms; returning invalid "
                "result. terms=%s error=%s",
                terms[:3],
                exc,
            )
            _release_cuda_memory()
            return self._make_invalid_result("autograd OOM")

    def _build_theta(
        self, terms: list[str], *, skip_invalid: bool = False
    ) -> tuple[Tensor, list[str]]:
        """Execute terms and build the feature matrix Theta.

        Args:
            terms: List of term expression strings.
            skip_invalid: If True, skip bad terms instead of raising.

        Returns:
            Tuple of (theta matrix, valid_terms list).

        Raises:
            ValueError: If execution fails (when skip_invalid=False),
                theta contains NaN/Inf, or all terms are filtered out.
        """
        theta_columns: list[Tensor] = []
        valid_terms: list[str] = []

        for term in terms:
            # Layer 1: Execution exception
            try:
                result = self._executor.execute(term, self._context)
                col = result.value.flatten()
            except torch.cuda.OutOfMemoryError:
                # bubble to evaluate_terms' OOM handler so the
                # caching allocator can be flushed and the candidate
                # reported as "autograd OOM" rather than relabelled as
                # a generic Execution error.
                raise
            except Exception as e:
                if not skip_invalid:
                    raise ValueError(f"Execution error for '{term}': {e}") from e
                logger.debug("skip_invalid: skipping '%s' (execution error)", term)
                continue

            # Layer 2: NaN/Inf check (only when skip_invalid)
            if skip_invalid and not torch.isfinite(col).all():
                logger.debug("skip_invalid: skipping '%s' (NaN/Inf)", term)
                continue

            # Layer 3: All-zero check (only when skip_invalid)
            if skip_invalid and (col == 0).all():
                logger.debug("skip_invalid: skipping '%s' (all zeros)", term)
                continue

            theta_columns.append(col)
            valid_terms.append(term)

        # All terms filtered
        if not theta_columns:
            if skip_invalid:
                raise ValueError("All terms filtered by skip_invalid")
            raise ValueError("No valid terms")

        try:
            theta = torch.stack(theta_columns, dim=1)
        except torch.cuda.OutOfMemoryError:
            # defer to evaluate_terms' OOM handler.
            raise
        except Exception as e:
            raise ValueError(f"Failed to build theta matrix: {e}") from e

        # When skip_invalid=False, still check the whole theta for NaN/Inf
        if not skip_invalid and (torch.isnan(theta).any() or torch.isinf(theta).any()):
            raise ValueError("Theta contains NaN or Inf")

        return theta, valid_terms

    def _get_complexity(self, selected_indices: list[int] | None, n_terms: int) -> int:
        """Compute model complexity from selected indices or term count.

        Args:
            selected_indices: Indices selected by sparse solver, or None.
            n_terms: Total number of terms.

        Returns:
            Complexity count for AIC computation.
        """
        if selected_indices is not None:
            return len(selected_indices)
        return n_terms

    def _evaluate_terms_impl(
        self, terms: list[str], *, skip_invalid: bool = False
    ) -> EvaluationResult:
        """Internal implementation of evaluate_terms (called within no_grad)."""
        try:
            theta, valid_terms = self._build_theta(terms, skip_invalid=skip_invalid)
        except ValueError as e:
            return self._make_invalid_result(str(e))

        try:
            solve_result = self._solver.solve(theta, self._lhs_flat)
        except torch.cuda.OutOfMemoryError:
            # defer to evaluate_terms' OOM handler.
            raise
        except Exception as e:
            return self._make_invalid_result(f"Solver error: {e}")

        coefficients = solve_result.coefficients
        y_pred = theta @ coefficients
        mse = ((self._lhs_flat - y_pred) ** 2).mean().item()

        if not math.isfinite(mse):
            return self._make_invalid_result("MSE is NaN or Inf")

        nmse_val = _metrics_nmse(mse, self._lhs_var)
        r2 = solve_result.r2
        complexity = self._get_complexity(
            solve_result.selected_indices, len(valid_terms)
        )
        aic_val = self._scorer(mse, complexity)

        return EvaluationResult(
            mse=mse,
            nmse=nmse_val,
            r2=r2,
            aic=aic_val,
            complexity=complexity,
            coefficients=coefficients,
            is_valid=True,
            error_message="",
            selected_indices=solve_result.selected_indices,
            residuals=(y_pred - self._lhs_flat).detach(),
            terms=list(valid_terms),
        )

    def evaluate_expression(self, expr: str) -> EvaluationResult:
        """Evaluate a single expression (convenience API).

        Automatically splits the expression using split_terms() and
        then calls evaluate_terms().

        Args:
            expr: Expression string, e.g., "add(mul(u, u_x), u_xx)"

        Returns:
            EvaluationResult with coefficients and metrics.
        """
        from kd2.core.expr.terms import split_terms

        try:
            terms = split_terms(expr, self._executor.registry)
        except Exception as e:
            result = self._make_invalid_result(f"split_terms error: {e}")
            result.expression = expr
            return result

        result = self.evaluate_terms(terms)
        result.expression = expr
        return result

    def _make_invalid_result(self, error_message: str) -> EvaluationResult:
        """Create an invalid EvaluationResult with penalty values.

        Args:
            error_message: Description of the error.

        Returns:
            EvaluationResult with is_valid=False and penalty metrics.
        """
        return EvaluationResult(
            mse=self._penalty_value,
            nmse=self._penalty_value,
            r2=-float("inf"),
            aic=float("inf"),
            complexity=0,
            coefficients=None,
            is_valid=False,
            error_message=error_message,
            selected_indices=None,
            residuals=None,
            terms=None,
            expression="",
        )


def _release_cuda_memory() -> None:
    """Best-effort reclaim of fragmented CUDA memory after an OOM.

    ``torch.autograd.grad`` can leave the caching allocator holding
    reserved-but-unallocated blocks after an OutOfMemoryError; calling
    ``empty_cache`` returns those to CUDA so the next candidate has a
    real chance of succeeding rather than OOMing on the residue alone.
    No-op on CPU-only builds / when CUDA is not initialised.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
