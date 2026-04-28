"""Expression validator for Python AST subset.

The validator ensures expressions belong to the allowed AST subset:
- Only ast.Expression, ast.Call, ast.Name, ast.Constant nodes
- Function calls must be bare (no method calls)
- Function names must be in the allowed set

This provides safe expression validation before eval().
"""

import ast

# Allowed AST node types for kd2 expressions
ALLOWED_NODES: set[type] = {
    ast.Expression, # Top-level wrapper
    ast.Call, # Function calls: add(a, b), sin(x)
    ast.Name, # Variables: u, u_x, x, t
    ast.Constant, # Numbers: 1.0, 0.1, -2.5
    ast.Load, # Load context for Name nodes (required by ast.walk)
}


def _is_negative_constant(node: ast.AST) -> bool:
    """Check if node is a negative constant literal like -2.5.

    In Python AST, -2.5 parses as UnaryOp(USub, Constant(2.5)),
    not Constant(-2.5). This helper identifies such patterns.

    Args:
        node: AST node to check.

    Returns:
        True if node is UnaryOp(USub, Constant).
    """
    if not isinstance(node, ast.UnaryOp):
        return False
    if not isinstance(node.op, ast.USub):
        return False
    return isinstance(node.operand, ast.Constant)


def validate_expr(code: str, allowed_funcs: set[str]) -> bool:
    """Validate that expression belongs to the allowed AST subset.

    Validates that:
    1. The code can be parsed as a Python expression
    2. All AST nodes are in ALLOWED_NODES (with special case for negative constants)
    3. All function calls are bare (no method calls like x.sin())
    4. All function names are in allowed_funcs

    Args:
        code: Expression string to validate (e.g., "add(u, sin(x))").
        allowed_funcs: Set of allowed function names (e.g., {"add", "sin", "mul"}).

    Returns:
        True if the expression is valid, False otherwise.

    Examples:
        >>> validate_expr("add(u, v)", {"add", "mul"})
        True
        >>> validate_expr("u * v", {"add", "mul"}) # BinOp not allowed
        False
        >>> validate_expr("x.sin()", {"sin"}) # Method call not allowed
        False
        >>> validate_expr("unknown(x)", {"add"}) # Unknown function
        False
    """
    # Strip whitespace to avoid indentation errors
    code = code.strip()

    # Try to parse as expression
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return False

    # Collect nodes that are part of negative constant patterns
    # These should be skipped during validation
    negative_const_nodes: set[int] = set()
    for node in ast.walk(tree):
        if _is_negative_constant(node):
            # Mark the UnaryOp, USub, and Constant as allowed
            negative_const_nodes.add(id(node))
            negative_const_nodes.add(id(node.op)) # type: ignore
            negative_const_nodes.add(id(node.operand)) # type: ignore

    # Walk all nodes and validate
    for node in ast.walk(tree):
        # Skip nodes that are part of negative constant patterns
        if id(node) in negative_const_nodes:
            continue

        # Check node type is allowed
        if type(node) not in ALLOWED_NODES:
            return False

        # For Call nodes, validate function reference
        if isinstance(node, ast.Call):
            # Function must be a bare name (not attribute/method call)
            if not isinstance(node.func, ast.Name):
                return False
            # Function name must be in allowed set
            if node.func.id not in allowed_funcs:
                return False

    return True


def get_function_calls(code: str) -> set[str]:
    """Extract all function names called in the expression.

    Useful for checking what functions are used before validation.

    Args:
        code: Expression string to analyze.

    Returns:
        Set of function names found in the expression.
        Returns empty set if parsing fails.

    Examples:
        >>> get_function_calls("add(sin(x), mul(y, z))")
        {'add', 'sin', 'mul'}
        >>> get_function_calls("u")
        set()
    """
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError:
        return set()

    func_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_names.add(node.func.id)

    return func_names
