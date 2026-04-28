"""Prefix notation bridging functions.

This module provides bidirectional conversion between Python expressions
and prefix token sequences, enabling compatibility with algorithms
that require prefix notation (e.g., DISCOVER).

Functions:
    python_to_prefix: Convert Python expression to prefix tokens.
    prefix_to_python: Convert prefix tokens back to Python expression.
"""

import ast
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kd2.core.expr.registry import FunctionRegistry


def python_to_prefix(code: str) -> list[str]:
    """Convert Python expression to prefix token sequence.

    Traverses the AST in pre-order to produce prefix notation.

    Args:
        code: Valid Python expression string (e.g., "add(mul(x, y), z)").

    Returns:
        List of tokens in prefix order.

    Raises:
        SyntaxError: If code is not valid Python syntax.
        ValueError: If code contains unsupported AST nodes or is empty.

    Examples:
        >>> python_to_prefix("add(mul(u, u_x), mul(C, u_xx))")
        ['add', 'mul', 'u', 'u_x', 'mul', 'C', 'u_xx']

        >>> python_to_prefix("x")
        ['x']

        >>> python_to_prefix("3.14")
        ['3.14']
    """
    # Handle empty/whitespace input
    if not code or not code.strip():
        raise ValueError("Empty expression")

    # Parse expression - raises SyntaxError if invalid
    tree = ast.parse(code, mode="eval")
    return _traverse(tree.body)


def _traverse(node: ast.expr) -> list[str]:
    """Recursively traverse AST node and produce prefix tokens.

    Args:
        node: AST expression node.

    Returns:
        List of tokens in prefix order.

    Raises:
        ValueError: If node type is not supported.
    """
    if isinstance(node, ast.Call):
        # Function call: func_name followed by args
        if not isinstance(node.func, ast.Name):
            raise ValueError(
                f"Only simple function calls supported, got {type(node.func).__name__}"
            )
        tokens = [node.func.id]
        for arg in node.args:
            tokens.extend(_traverse(arg))
        return tokens

    elif isinstance(node, ast.Name):
        # Variable reference
        return [node.id]

    elif isinstance(node, ast.Constant):
        # Numeric or other constant
        return [_format_constant(node.value)]

    elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        # Special case: negative constant -5 -> ["-5"]
        if isinstance(node.operand, ast.Constant):
            value = node.operand.value
            if isinstance(value, (int, float)):
                return [_format_constant(-value)]
            raise ValueError(f"Unary minus not supported on {type(value).__name__}")
        # Other unary minus: not supported (should use neg() function)
        raise ValueError("Unary minus on non-constant not supported; use neg()")

    elif isinstance(node, ast.BinOp):
        raise ValueError(
            "Binary operators (+, -, *, /) not supported; use function calls"
        )

    elif isinstance(node, ast.Subscript):
        raise ValueError("Subscript operations not supported")

    elif isinstance(node, ast.Attribute):
        raise ValueError("Attribute access not supported")

    elif isinstance(node, ast.List):
        raise ValueError("List literals not supported")

    elif isinstance(node, ast.Dict):
        raise ValueError("Dict literals not supported")

    else:
        raise ValueError(f"Unsupported AST node type: {type(node).__name__}")


def _format_constant(value: object) -> str:
    """Format a constant value as a string token.

    Args:
        value: Constant value (int, float, etc.)

    Returns:
        String representation.
    """
    if isinstance(value, float):
        # Preserve scientific notation and avoid unnecessary precision
        formatted = repr(value)
        # Python repr() may produce "1e-10" or "0.0001" depending on value
        return formatted
    return str(value)


def prefix_to_python(tokens: list[str], registry: "FunctionRegistry") -> str:
    """Convert prefix token sequence to Python expression.

    Uses the registry to determine function arities for proper reconstruction.

    Args:
        tokens: List of tokens in prefix order.
        registry: FunctionRegistry to look up function arities.

    Returns:
        Python expression string.

    Raises:
        ValueError: If tokens form an invalid prefix expression.
        ValueError: If stack is not properly balanced.

    Examples:
        >>> registry = FunctionRegistry.create_default()
        >>> prefix_to_python(['add', 'mul', 'u', 'u_x', 'mul', 'C', 'u_xx'], registry)
        'add(mul(u, u_x), mul(C, u_xx))'

        >>> prefix_to_python(['x'], registry)
        'x'
    """
    if not tokens:
        raise ValueError("Empty token list")

    stack: list[str] = []

    # Process tokens in reverse order
    for token in reversed(tokens):
        if registry.has(token):
            arity = registry.get_arity(token)
            if arity == 0:
                # Zero-arity function (terminal): push as-is
                stack.append(token)
            else:
                # Function with arguments
                if len(stack) < arity:
                    raise ValueError(
                        f"Not enough arguments for '{token}': "
                        f"need {arity}, have {len(stack)}"
                    )
                args = [stack.pop() for _ in range(arity)]
                expr = f"{token}({', '.join(args)})"
                stack.append(expr)
        else:
            # Unknown token: treat as variable/constant (arity=0)
            stack.append(token)

    # After processing all tokens, stack should have exactly one element
    if len(stack) != 1:
        raise ValueError(
            f"Unbalanced expression: {len(stack)} elements remain on stack"
        )

    return stack[0]
