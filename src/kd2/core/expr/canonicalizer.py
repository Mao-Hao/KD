"""Expression canonicalizer for AST normalization.

The canonicalizer transforms expressions into a canonical form where:
- Arguments of commutative functions are sorted deterministically
- Equivalent expressions produce identical representations

This enables effective caching and deduplication.
"""

import ast
import hashlib

from kd2.core.expr.registry import FunctionRegistry

# Default maximum recursion depth for canonicalization
DEFAULT_MAX_DEPTH: int = 1000


def canonicalize(
    node: ast.expr,
    registry: FunctionRegistry,
    _depth: int = 0,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> ast.expr:
    """Canonicalize AST by sorting commutative function arguments.

    Transforms an AST into canonical form:
    - For commutative functions (e.g., add, mul), arguments are sorted
      by their AST dump representation
    - Sorting is recursive: children are canonicalized before sorting
    - Non-commutative functions and terminals are unchanged

    Args:
        node: AST expression node to canonicalize.
        registry: FunctionRegistry to check commutativity.
        _depth: Internal recursion depth counter (do not set manually).
        max_depth: Maximum recursion depth (default 1000).

    Returns:
        Canonicalized AST node (may share structure with input).

    Raises:
        RecursionError: If max_depth is exceeded.

    Examples:
        >>> import ast
        >>> tree = ast.parse("add(b, a)", mode="eval")
        >>> canonical = canonicalize(tree.body, registry)
        >>> ast.unparse(canonical)
        'add(a, b)'

        >>> tree = ast.parse("sub(b, a)", mode="eval") # non-commutative
        >>> canonical = canonicalize(tree.body, registry)
        >>> ast.unparse(canonical)
        'sub(b, a)'
    """
    if _depth > max_depth:
        raise RecursionError(
            f"Expression depth {_depth} exceeds max_depth {max_depth}"
        )

    if isinstance(node, ast.Call):
        # Recursively canonicalize children first
        new_args = [
            canonicalize(arg, registry, _depth + 1, max_depth)
            for arg in node.args
        ]

        # Get function name (must be ast.Name for our subset)
        if isinstance(node.func, ast.Name):
            func_name = node.func.id

            # Sort arguments for commutative functions
            # Only sort if function is registered and commutative
            if registry.has(func_name) and registry.is_commutative(func_name):
                new_args = sorted(new_args, key=lambda a: ast.dump(a))

        # Return new Call node with canonicalized arguments
        return ast.Call(func=node.func, args=new_args, keywords=[])

    # Name and Constant nodes are unchanged
    return node


def canonical_hash(
    code: str,
    registry: FunctionRegistry,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> str:
    """Compute hash of the canonical form of an expression.

    Two semantically equivalent expressions (differing only by
    commutative argument order) will have the same canonical hash.

    Args:
        code: Expression string to hash.
        registry: FunctionRegistry to check commutativity.
        max_depth: Maximum recursion depth for canonicalization (default 1000).

    Returns:
        16-character hexadecimal hash string.

    Raises:
        SyntaxError: If code cannot be parsed as an expression.
        RecursionError: If expression depth exceeds max_depth.

    Examples:
        >>> # Same hash for equivalent expressions
        >>> h1 = canonical_hash("add(a, b)", registry)
        >>> h2 = canonical_hash("add(b, a)", registry)
        >>> h1 == h2
        True

        >>> # Different hash for non-equivalent expressions
        >>> h1 = canonical_hash("sub(a, b)", registry)
        >>> h2 = canonical_hash("sub(b, a)", registry)
        >>> h1 != h2
        True
    """
    # Parse the code (raises SyntaxError if invalid)
    tree = ast.parse(code, mode="eval")

    # Canonicalize the AST
    canonical = canonicalize(tree.body, registry, max_depth=max_depth)

    # Unparse to get canonical string representation
    canonical_code = ast.unparse(canonical)

    # Compute SHA-256 hash and return first 16 hex chars
    return hashlib.sha256(canonical_code.encode()).hexdigest()[:16]


def canonicalize_code(
    code: str,
    registry: FunctionRegistry,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> str:
    """Canonicalize an expression string.

    Convenience function that parses, canonicalizes, and unparses.

    Args:
        code: Expression string to canonicalize.
        registry: FunctionRegistry to check commutativity.
        max_depth: Maximum recursion depth for canonicalization (default 1000).

    Returns:
        Canonicalized expression string.

    Raises:
        SyntaxError: If code cannot be parsed as an expression.
        RecursionError: If expression depth exceeds max_depth.

    Examples:
        >>> canonicalize_code("add(b, a)", registry)
        'add(a, b)'
    """
    # Parse the code (raises SyntaxError if invalid)
    tree = ast.parse(code, mode="eval")

    # Canonicalize the AST
    canonical = canonicalize(tree.body, registry, max_depth=max_depth)

    # Unparse to string
    return ast.unparse(canonical)
