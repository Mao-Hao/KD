"""Conservative expression transforms for kd2 IR."""

from __future__ import annotations

import sympy # type: ignore[import-untyped]

from kd2.core.expr.sympy_bridge import _sympy_to_ir, to_sympy


def expand_linear_diffs(code: str, *, strict: bool = True) -> str:
    """Expand only the linear open-form derivative subset in kd2 IR.

    This is not a full algebraic simplifier. It preserves nested finite
    difference semantics by keeping compound names such as ``u_x_x`` instead
    of folding them into direct-order names such as ``u_xx``. Nonlinear
    derivatives like ``diff_x(mul(u, u_x))`` remain open-form diff calls.
    """
    if not code.strip():
        return code
    try:
        return _sympy_to_ir(sympy.expand(to_sympy(code, strict=True)))
    except Exception:
        if strict:
            raise
        return code


__all__ = ["expand_linear_diffs"]
