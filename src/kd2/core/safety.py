"""Numerical safety utilities.

Protected operations that guarantee finite outputs from finite inputs.
All numerical code in kd2 must use these instead of raw arithmetic.

Note: These functions protect against common numerical issues (division by zero,
overflow) but do NOT sanitize Inf/NaN inputs - those pass through unchanged.
If input validation is needed, check inputs explicitly before calling.
"""

import torch
from torch import Tensor


def safe_div(a: Tensor, b: Tensor, eps: float = 1e-10) -> Tensor:
    """Protected division: ``a / b`` that avoids division-by-zero.

    When ``b`` is near zero, adds ``eps`` with the sign of ``b``
    to avoid division by zero while preserving sign semantics.

    Note:
        This function guarantees finite output only for FINITE inputs.
        Inf and NaN inputs pass through unchanged (Inf/1 = Inf, NaN/1 = NaN).
        If input validation is needed, check ``torch.isfinite()`` before calling.

    Args:
        a: Numerator tensor (should be finite for guaranteed finite output).
        b: Denominator tensor (should be finite for guaranteed finite output).
        eps: Small constant added to denominator near zero.

    Returns:
        Result tensor. Finite if both inputs are finite.
    """
    sign_b = torch.sign(b)
    sign_b = torch.where(sign_b == 0, torch.ones_like(sign_b), sign_b)
    return a / (b + eps * sign_b)


def safe_exp(x: Tensor, min_val: float = -50.0, max_val: float = 50.0) -> Tensor:
    """Protected exponential: clamps input to prevent overflow and underflow.

    Clamps input to [min_val, max_val] before computing exp(), preventing:
    - Overflow: exp(large positive) → Inf
    - Underflow: exp(large negative) → 0.0 (which breaks 1/exp(x))

    The default range [-50, 50] ensures:
    - exp(-50) ≈ 1.9e-22 (non-zero, safe for 1/exp)
    - exp(50) ≈ 5.2e+21 (large but finite)

    Note:
        Clamping causes gradient to vanish outside the clamped range.
        For gradient-sensitive applications, consider alternatives.

    Args:
        x: Input tensor.
        min_val: Minimum value before clamping (default: -50.0).
        max_val: Maximum value before clamping (default: 50.0).

    Returns:
        ``exp(clamp(x, min=min_val, max=max_val))``, always positive and finite.
    """
    return torch.exp(torch.clamp(x, min=min_val, max=max_val))


def safe_log(x: Tensor, eps: float = 1e-10) -> Tensor:
    """Protected logarithm: takes log of ``|x|`` clamped above eps.

    Handles zero, negative, and near-zero inputs safely by:
    1. Taking absolute value (handles negative inputs)
    2. Clamping to minimum eps (handles zero and near-zero)

    Note:
        Inf inputs pass through: log(|Inf|) = Inf.
        NaN inputs pass through: log(|NaN|) = NaN.
        If input validation is needed, check ``torch.isfinite()`` before calling.

    Args:
        x: Input tensor (any sign, should be finite for guaranteed finite output).
        eps: Minimum absolute value before taking log.

    Returns:
        ``log(clamp(|x|, min=eps))``. Finite if input is finite.
    """
    return torch.log(torch.clamp(x.abs(), min=eps))
