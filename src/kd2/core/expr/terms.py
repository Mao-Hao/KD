"""Term splitting for LINEAR mode evaluation.

This module provides split_terms() which splits expressions along top-level
add/sub operators into individual terms for coefficient fitting.

Examples:
    "add(a, b)" -> ["a", "b"]
    "add(a, add(b, c))" -> ["a", "b", "c"]
    "sub(a, b)" -> ["a", "neg(b)"]
    "mul(u, u_x)" -> ["mul(u, u_x)"] # single term, not split
"""

from __future__ import annotations

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kd2.core.expr.registry import FunctionRegistry


def _reject_infix_nodes(tree: ast.AST) -> None:
    """Reject infix / unary / boolean operators in the AST.

    kd2 IR only allows ast.Call, ast.Name, ast.Constant nodes.
    Infix expressions like ``u + v`` or ``-u`` violate the IR contract.

    Raises:
        ValueError: If any forbidden node type is found.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.BinOp):
            raise ValueError(
                "Infix operators (e.g., 'a + b') are not allowed. "
                "Use function-call IR: add(a, b), mul(a, b), etc."
            )
        if isinstance(node, ast.UnaryOp):
            # Allow negative numeric literals: -1, -1.5, -2e3
            # Python 3.11 parses these as UnaryOp(USub, Constant)
            if isinstance(node.op, (ast.USub, ast.UAdd)) and isinstance(
                node.operand, ast.Constant
            ):
                continue
            raise ValueError(
                "Unary operators (e.g., '-x') are not allowed. "
                "Use function-call IR: neg(x)."
            )
        if isinstance(node, ast.BoolOp):
            raise ValueError("Boolean operators are not allowed in kd2 IR.")


def split_terms(expr: str, registry: FunctionRegistry) -> list[str]:
    """Split expression along top-level add/sub into term list.

    For LINEAR mode evaluation, we need to decompose an expression into
    individual terms that can be executed separately and combined via
    linear coefficients: y = c1*term1 + c2*term2 + ...

    The function recursively flattens:
    - add(a, b) -> [a, b]
    - sub(a, b) -> [a, neg(b)]
    - Nested add/sub are flattened
    - Other operators (mul, sin, etc.) remain as single terms

    Args:
        expr: Python expression string, e.g., "add(mul(u, u_x), u_xx)"
        registry: FunctionRegistry for querying operator properties.

    Returns:
        List of term strings, each representing an independent term.

    Raises:
        ValueError: If expr is empty, whitespace-only, or has syntax errors.

    Examples:
        >>> reg = FunctionRegistry.create_default()
        >>> split_terms("add(a, b)", reg)
        ['a', 'b']
        >>> split_terms("sub(a, b)", reg)
        ['a', 'neg(b)']
        >>> split_terms("add(mul(u, u_x), u_xx)", reg)
        ['mul(u, u_x)', 'u_xx']
    """
    # Validate input
    if not expr or not expr.strip():
        raise ValueError("Expression cannot be empty")

    # Parse expression
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in expression: {e}") from e

    # Reject infix/unary/boolean operators — IR only allows Call/Name/Constant
    _reject_infix_nodes(tree)

    # Collect terms with sign tracking
    # Returns list of (node, is_negated) pairs
    terms = _collect_terms(tree.body, negated=False)

    # Convert back to strings
    result = []
    for node, is_negated in terms:
        term_str = ast.unparse(node)
        if is_negated:
            term_str = f"neg({term_str})"
        result.append(term_str)

    return result


def _collect_terms(
    node: ast.expr,
    negated: bool,
) -> list[tuple[ast.expr, bool]]:
    """Recursively collect terms from AST node.

    Args:
        node: AST expression node.
        negated: Whether this subtree is negated (from sub()).

    Returns:
        List of (node, is_negated) tuples.
    """
    # Check if this is an add or sub call
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        func_name = node.func.id

        if func_name == "add":
            # Validate arity
            if len(node.args) != 2:
                raise ValueError(
                    f"add() requires exactly 2 arguments, got {len(node.args)}"
                )

            # add(a, b) with negated=False -> collect(a, False) + collect(b, False)
            # add(a, b) with negated=True -> collect(a, True) + collect(b, True)
            left_terms = _collect_terms(node.args[0], negated)
            right_terms = _collect_terms(node.args[1], negated)
            return left_terms + right_terms

        elif func_name == "sub":
            # Validate arity
            if len(node.args) != 2:
                raise ValueError(
                    f"sub() requires exactly 2 arguments, got {len(node.args)}"
                )

            # sub(a, b) = a - b
            # negated=False: collect(a, False) + collect(b, True)
            # negated=True: -(a-b) = -a+b -> collect(a, True) + collect(b, False)
            left_terms = _collect_terms(node.args[0], negated)
            right_terms = _collect_terms(node.args[1], not negated)
            return left_terms + right_terms

    # Not add/sub at top level - this is a single term
    return [(node, negated)]
