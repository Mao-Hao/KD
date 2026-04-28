"""Expression module for Python AST-based IR.

This module provides:
- FunctionRegistry: Registry of operators and terminals
- Validator: AST validation
- Canonicalizer: Expression canonicalization
- PythonExecutor: Expression execution
- split_terms: Term splitting for LINEAR mode
"""

from kd2.core.expr.canonicalizer import (
    canonical_hash,
    canonicalize,
    canonicalize_code,
)
from kd2.core.expr.executor import ExecutorResult, PythonExecutor, has_open_form_diff
from kd2.core.expr.registry import FunctionRegistry
from kd2.core.expr.simplify import expand_linear_diffs
from kd2.core.expr.terms import split_terms
from kd2.core.expr.validator import (
    ALLOWED_NODES,
    get_function_calls,
    validate_expr,
)

__all__ = [
    # Registry
    "FunctionRegistry",
    # Validator
    "ALLOWED_NODES",
    "validate_expr",
    "get_function_calls",
    # Canonicalizer
    "canonicalize",
    "canonicalize_code",
    "canonical_hash",
    # Executor
    "PythonExecutor",
    "ExecutorResult",
    "has_open_form_diff",
    "expand_linear_diffs",
    # Terms
    "split_terms",
]
