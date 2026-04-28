"""Model selection metrics -- standalone pure functions + scorer protocol.

Provides unified AIC/BIC/NMSE metrics used by both the platform Evaluator
and SGA plugin. Each metric is a pure function; extra parameters (n, ratio)
are bound via factory functions that return a ScorerFn.

Design:.
"""

from __future__ import annotations

import math
from collections.abc import Callable

# Scorer type alias

ScorerFn = Callable[[float, int], float]
"""Unified scorer signature: (mse, k) -> score.

``mse`` is mean squared error; ``k`` is model complexity (active term count).
Lower scores indicate better models.
"""

# Named constants

_NMSE_EPS: float = 1e-15
"""Default epsilon for NMSE target-variance guard."""

_MSE_FLOOR: float = 1e-15
"""MSE values at or below this threshold are treated as perfect fit."""


# Pure metric functions


def aic(mse: float, k: int, n: int) -> float:
    """Standard AIC = n * log(MSE) + 2k.

    Args:
        mse: Mean squared error.
        k: Number of active terms (model complexity).
        n: Number of data samples.

    Returns:
        AIC value. Returns ``-inf`` for perfect fit (mse <= 1e-15).
        Returns ``+inf`` for invalid input (NaN, Inf, mse < 0, or n <= 0).
    """
    if n <= 0:
        return float("inf")
    if not math.isfinite(mse) or mse < 0.0:
        return float("inf")
    if mse <= _MSE_FLOOR:
        return -float("inf")
    return n * math.log(mse) + 2.0 * k


def aic_no_n(mse: float, k: int, ratio: float = 1.0) -> float:
    """Ranking-only AIC (same-dataset): 2k * ratio + 2 * log(MSE).

    Omits *n* because within a single dataset all candidates share the
    same sample size, so *n* does not affect relative ranking. Used
    internally by SGA.

    Args:
        mse: Mean squared error. Must be positive for a valid score.
        k: Number of active terms.
        ratio: AIC penalty ratio (scales the complexity term).

    Returns:
        AIC score. Returns ``+inf`` if mse <= 0, NaN, or Inf.
    """
    if not math.isfinite(mse) or mse <= 0.0:
        return float("inf")
    return 2.0 * k * ratio + 2.0 * math.log(mse)


def aicc(mse: float, k: int, n: int) -> float:
    """Corrected AIC: AIC + 2k(k+1) / (n - k - 1).

    Small-sample correction that penalises complexity more heavily when
    the sample-to-parameter ratio is low.

    Args:
        mse: Mean squared error.
        k: Number of active terms.
        n: Number of data samples.

    Returns:
        AICc value. Returns ``+inf`` when n <= k + 1 (correction
        undefined) or when input is invalid. Returns ``-inf`` for
        perfect fit (mse == 0) when n > k + 1.
    """
    if n <= k + 1:
        return float("inf")
    base = aic(mse, k, n)
    if not math.isfinite(base):
        return base
    correction = 2.0 * k * (k + 1) / (n - k - 1)
    return base + correction


def bic(mse: float, k: int, n: int) -> float:
    """BIC = n * log(MSE) + k * log(n).

    Bayesian Information Criterion -- favours parsimony more than AIC
    for moderate-to-large sample sizes.

    Args:
        mse: Mean squared error.
        k: Number of active terms.
        n: Number of data samples.

    Returns:
        BIC value. Returns ``-inf`` for perfect fit (mse == 0).
        Returns ``+inf`` for invalid input.
    """
    if not math.isfinite(mse) or mse < 0.0:
        return float("inf")
    if mse <= _MSE_FLOOR:
        return -float("inf")
    if n <= 0:
        return float("inf")
    return n * math.log(mse) + k * math.log(n)


def nmse(mse: float, target_var: float, eps: float = _NMSE_EPS) -> float:
    """Normalized MSE: MSE / Var(target).

    Falls back to raw MSE when the target variance is near zero
    (constant target), preventing division by zero.

    Args:
        mse: Mean squared error.
        target_var: Variance of the regression target.
        eps: Minimum variance threshold.

    Returns:
        Normalized MSE value. Non-finite *mse* is passed through unchanged.
    """
    if not math.isfinite(mse):
        return mse
    if target_var > eps:
        return mse / target_var
    return mse


# Factory functions (bind extra params, return ScorerFn)


def make_aic_scorer(n: int) -> ScorerFn:
    """Create an AIC scorer with *n* pre-bound.

    Args:
        n: Number of data samples.

    Returns:
        ``ScorerFn`` that computes ``aic(mse, k, n)``.
    """

    def _scorer(mse: float, k: int) -> float:
        return aic(mse, k, n)

    return _scorer


def make_sga_scorer(ratio: float = 1.0) -> ScorerFn:
    """Create an SGA ranking scorer with *ratio* pre-bound.

    Args:
        ratio: AIC penalty ratio.

    Returns:
        ``ScorerFn`` that computes ``aic_no_n(mse, k, ratio)``.
    """

    def _scorer(mse: float, k: int) -> float:
        return aic_no_n(mse, k, ratio)

    return _scorer


def make_bic_scorer(n: int) -> ScorerFn:
    """Create a BIC scorer with *n* pre-bound.

    Args:
        n: Number of data samples.

    Returns:
        ``ScorerFn`` that computes ``bic(mse, k, n)``.
    """

    def _scorer(mse: float, k: int) -> float:
        return bic(mse, k, n)

    return _scorer
