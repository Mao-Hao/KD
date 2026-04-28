"""Tests for expression canonicalizer.

Test coverage:
- smoke: Functions exist and are callable
- unit: Commutative reordering, non-commutative unchanged, hash equivalence
- numerical: N/A (canonicalizer is not numerical)

TDD Status: RED - Tests written before implementation.
"""

import ast

import pytest

from kd2.core.expr.canonicalizer import (
    DEFAULT_MAX_DEPTH,
    canonical_hash,
    canonicalize,
    canonicalize_code,
)
from kd2.core.expr.registry import FunctionRegistry

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_registry() -> FunctionRegistry:
    """Create a mock registry with commutative/non-commutative functions.

    This fixture creates a minimal registry for testing canonicalization
    without requiring a full implementation.
    """
    # Since FunctionRegistry is not yet implemented, we need to handle this
    # For now, we'll test with the expectation that create_default works
    # Tests will fail at RED stage, which is expected
    return FunctionRegistry.create_default()


# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.smoke
class TestCanonicalizerSmoke:
    """Smoke tests: function existence and basic callability."""

    def test_canonicalize_exists(self) -> None:
        """canonicalize function exists and is callable."""
        assert callable(canonicalize)

    def test_canonical_hash_exists(self) -> None:
        """canonical_hash function exists and is callable."""
        assert callable(canonical_hash)

    def test_canonicalize_code_exists(self) -> None:
        """canonicalize_code function exists and is callable."""
        assert callable(canonicalize_code)


# =============================================================================
# Unit Tests - Commutative Reordering
# =============================================================================


@pytest.mark.unit
class TestCommutativeReordering:
    """Unit tests for commutative function argument sorting."""

    def test_add_arguments_sorted(self, mock_registry: FunctionRegistry) -> None:
        """add(b, a) becomes add(a, b) after canonicalization."""
        result = canonicalize_code("add(b, a)", mock_registry)
        assert result == "add(a, b)"

    def test_add_already_sorted(self, mock_registry: FunctionRegistry) -> None:
        """add(a, b) stays add(a, b)."""
        result = canonicalize_code("add(a, b)", mock_registry)
        assert result == "add(a, b)"

    def test_mul_arguments_sorted(self, mock_registry: FunctionRegistry) -> None:
        """mul(y, x) becomes mul(x, y) after canonicalization."""
        result = canonicalize_code("mul(y, x)", mock_registry)
        assert result == "mul(x, y)"

    def test_sorting_is_lexicographic(self, mock_registry: FunctionRegistry) -> None:
        """Arguments are sorted by ast.dump (lexicographic on structure)."""
        # 'a' < 'b' < 'c' lexicographically
        result = canonicalize_code("add(c, a)", mock_registry)
        assert result == "add(a, c)"

    def test_nested_commutative_outer(self, mock_registry: FunctionRegistry) -> None:
        """Outer commutative function arguments are sorted."""
        # add(mul(x, y), add(a, b)) - outer add should sort its children
        result = canonicalize_code("add(mul(x, y), add(a, b))", mock_registry)
        # Both children are calls; sorting by ast.dump
        # Need to determine which comes first in sorted order
        # The point is consistent ordering, not specific order
        tree1 = ast.parse(result, mode="eval")
        assert isinstance(tree1.body, ast.Call)
        assert tree1.body.func.id == "add" # type: ignore

    def test_nested_commutative_inner(self, mock_registry: FunctionRegistry) -> None:
        """Inner commutative function arguments are also sorted."""
        result = canonicalize_code("add(mul(b, a), x)", mock_registry)
        # mul(b, a) should become mul(a, b)
        assert "mul(a, b)" in result

    def test_deeply_nested_sorting(self, mock_registry: FunctionRegistry) -> None:
        """Deeply nested commutative functions are all sorted."""
        # add(mul(b, a), add(d, c))
        result = canonicalize_code("add(mul(b, a), add(d, c))", mock_registry)
        # Inner mul: mul(a, b)
        # Inner add: add(c, d)
        # Outer add: sorted by ast.dump of children
        assert "mul(a, b)" in result
        assert "add(c, d)" in result


# =============================================================================
# Unit Tests - Non-Commutative Functions
# =============================================================================


@pytest.mark.unit
class TestNonCommutative:
    """Unit tests for non-commutative functions (unchanged order)."""

    def test_sub_unchanged(self, mock_registry: FunctionRegistry) -> None:
        """sub(a, b) stays sub(a, b) - order matters."""
        result = canonicalize_code("sub(a, b)", mock_registry)
        assert result == "sub(a, b)"

    def test_sub_reversed_unchanged(self, mock_registry: FunctionRegistry) -> None:
        """sub(b, a) stays sub(b, a) - not reordered."""
        result = canonicalize_code("sub(b, a)", mock_registry)
        assert result == "sub(b, a)"

    def test_div_unchanged(self, mock_registry: FunctionRegistry) -> None:
        """div(a, b) stays div(a, b)."""
        result = canonicalize_code("div(a, b)", mock_registry)
        assert result == "div(a, b)"

    def test_mixed_commutative_noncommutative(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Mixed expression: only commutative parts are reordered."""
        # sub(mul(b, a), x) - mul is commutative, sub is not
        result = canonicalize_code("sub(mul(b, a), x)", mock_registry)
        # mul becomes mul(a, b), sub keeps order
        assert result == "sub(mul(a, b), x)"


# =============================================================================
# Unit Tests - Unary Functions
# =============================================================================


@pytest.mark.unit
class TestUnaryFunctions:
    """Unit tests for unary function handling."""

    def test_unary_unchanged(self, mock_registry: FunctionRegistry) -> None:
        """Unary functions are unchanged (no arguments to sort)."""
        result = canonicalize_code("sin(x)", mock_registry)
        assert result == "sin(x)"

    def test_unary_nested_sorted(self, mock_registry: FunctionRegistry) -> None:
        """Unary function with commutative child: child is sorted."""
        result = canonicalize_code("sin(add(b, a))", mock_registry)
        assert result == "sin(add(a, b))"

    def test_chain_of_unary(self, mock_registry: FunctionRegistry) -> None:
        """Chain of unary functions is unchanged."""
        result = canonicalize_code("sin(cos(exp(x)))", mock_registry)
        assert result == "sin(cos(exp(x)))"


# =============================================================================
# Unit Tests - Terminals (Variables and Constants)
# =============================================================================


@pytest.mark.unit
class TestTerminals:
    """Unit tests for terminal handling."""

    def test_variable_unchanged(self, mock_registry: FunctionRegistry) -> None:
        """Bare variable is unchanged."""
        result = canonicalize_code("x", mock_registry)
        assert result == "x"

    def test_constant_unchanged(self, mock_registry: FunctionRegistry) -> None:
        """Constant is unchanged."""
        result = canonicalize_code("1.0", mock_registry)
        assert result == "1.0"

    def test_negative_constant_unchanged(self, mock_registry: FunctionRegistry) -> None:
        """Negative constant is unchanged (if represented as Constant)."""
        # Note: parsing behavior may vary; test actual behavior
        tree = ast.parse("-1.0", mode="eval")
        # Could be UnaryOp or Constant depending on Python version
        # We just verify canonicalize handles it
        canonical = canonicalize(tree.body, mock_registry)
        assert canonical is not None


# =============================================================================
# Unit Tests - Hash Equivalence
# =============================================================================


@pytest.mark.unit
class TestHashEquivalence:
    """Unit tests for canonical_hash equivalence."""

    def test_same_expression_same_hash(self, mock_registry: FunctionRegistry) -> None:
        """Identical expressions have identical hash."""
        h1 = canonical_hash("add(a, b)", mock_registry)
        h2 = canonical_hash("add(a, b)", mock_registry)
        assert h1 == h2

    def test_commutative_reorder_same_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Commutatively equivalent expressions have same hash."""
        h1 = canonical_hash("add(a, b)", mock_registry)
        h2 = canonical_hash("add(b, a)", mock_registry)
        assert h1 == h2

    def test_mul_commutative_same_hash(self, mock_registry: FunctionRegistry) -> None:
        """mul(x, y) and mul(y, x) have same hash."""
        h1 = canonical_hash("mul(x, y)", mock_registry)
        h2 = canonical_hash("mul(y, x)", mock_registry)
        assert h1 == h2

    def test_nested_commutative_same_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Nested commutative equivalents have same hash."""
        h1 = canonical_hash("add(mul(a, b), mul(c, d))", mock_registry)
        h2 = canonical_hash("add(mul(d, c), mul(b, a))", mock_registry)
        assert h1 == h2

    def test_deeply_nested_same_hash(self, mock_registry: FunctionRegistry) -> None:
        """Deeply nested equivalents have same hash."""
        h1 = canonical_hash("add(add(a, b), add(c, d))", mock_registry)
        h2 = canonical_hash("add(add(d, c), add(b, a))", mock_registry)
        assert h1 == h2


# =============================================================================
# Unit Tests - Hash Difference
# =============================================================================


@pytest.mark.unit
class TestHashDifference:
    """Unit tests for non-equivalent expressions having different hash."""

    def test_different_expressions_different_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Different expressions have different hash."""
        h1 = canonical_hash("add(a, b)", mock_registry)
        h2 = canonical_hash("mul(a, b)", mock_registry)
        assert h1 != h2

    def test_sub_not_commutative_different_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """sub(a, b) and sub(b, a) have different hash (non-commutative)."""
        h1 = canonical_hash("sub(a, b)", mock_registry)
        h2 = canonical_hash("sub(b, a)", mock_registry)
        assert h1 != h2

    def test_div_not_commutative_different_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """div(a, b) and div(b, a) have different hash (non-commutative)."""
        h1 = canonical_hash("div(a, b)", mock_registry)
        h2 = canonical_hash("div(b, a)", mock_registry)
        assert h1 != h2

    def test_different_variables_different_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Expressions with different variables have different hash."""
        h1 = canonical_hash("add(x, y)", mock_registry)
        h2 = canonical_hash("add(a, b)", mock_registry)
        assert h1 != h2

    def test_different_constants_different_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Expressions with different constants have different hash."""
        h1 = canonical_hash("add(x, 1.0)", mock_registry)
        h2 = canonical_hash("add(x, 2.0)", mock_registry)
        assert h1 != h2


# =============================================================================
# Unit Tests - Hash Format
# =============================================================================


@pytest.mark.unit
class TestHashFormat:
    """Unit tests for hash format requirements."""

    def test_hash_is_string(self, mock_registry: FunctionRegistry) -> None:
        """canonical_hash returns a string."""
        h = canonical_hash("add(a, b)", mock_registry)
        assert isinstance(h, str)

    def test_hash_is_hexadecimal(self, mock_registry: FunctionRegistry) -> None:
        """canonical_hash returns hexadecimal string."""
        h = canonical_hash("add(a, b)", mock_registry)
        # Should only contain hex characters
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_length_is_16(self, mock_registry: FunctionRegistry) -> None:
        """canonical_hash returns 16-character string."""
        h = canonical_hash("add(a, b)", mock_registry)
        assert len(h) == 16

    def test_hash_deterministic(self, mock_registry: FunctionRegistry) -> None:
        """Same input always produces same hash."""
        results = [canonical_hash("sin(add(x, y))", mock_registry) for _ in range(10)]
        assert len(set(results)) == 1 # All same


# =============================================================================
# Unit Tests - Error Handling
# =============================================================================


@pytest.mark.unit
class TestErrorHandling:
    """Unit tests for error handling."""

    def test_canonical_hash_syntax_error(self, mock_registry: FunctionRegistry) -> None:
        """canonical_hash raises SyntaxError for invalid code."""
        with pytest.raises(SyntaxError):
            canonical_hash("add(a, b", mock_registry)

    def test_canonicalize_code_syntax_error(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """canonicalize_code raises SyntaxError for invalid code."""
        with pytest.raises(SyntaxError):
            canonicalize_code("add(a, b", mock_registry)


# =============================================================================
# Unit Tests - Recursion Depth Limit
# =============================================================================


@pytest.mark.unit
class TestRecursionDepthLimit:
    """Unit tests for recursion depth limit"""

    def test_default_max_depth_constant_exists(self) -> None:
        """DEFAULT_MAX_DEPTH constant is defined and reasonable."""
        assert DEFAULT_MAX_DEPTH == 1000

    def test_shallow_expression_succeeds(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Shallow expressions work with small max_depth."""
        result = canonicalize_code("add(a, b)", mock_registry, max_depth=10)
        assert result == "add(a, b)"

    def test_exceeds_max_depth_raises_recursion_error(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Expression exceeding max_depth raises RecursionError."""
        # Build a nested expression: sin(sin(sin(x))) with depth 3
        expr = "sin(sin(sin(x)))"
        # Depth 0 -> sin, Depth 1 -> sin, Depth 2 -> sin, Depth 3 -> x
        # With max_depth=2, depth 3 exceeds limit
        with pytest.raises(RecursionError) as exc_info:
            canonicalize_code(expr, mock_registry, max_depth=2)
        assert "exceeds max_depth" in str(exc_info.value)

    def test_exact_max_depth_succeeds(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Expression at exactly max_depth succeeds."""
        # sin(x) has depth 1 (sin at depth 0, x at depth 1)
        expr = "sin(x)"
        result = canonicalize_code(expr, mock_registry, max_depth=1)
        assert result == "sin(x)"

    def test_canonical_hash_respects_max_depth(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """canonical_hash also respects max_depth parameter."""
        expr = "sin(sin(sin(x)))"
        with pytest.raises(RecursionError):
            canonical_hash(expr, mock_registry, max_depth=2)

    def test_canonicalize_direct_respects_max_depth(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Direct canonicalize() call respects max_depth."""
        tree = ast.parse("sin(sin(sin(x)))", mode="eval")
        with pytest.raises(RecursionError):
            canonicalize(tree.body, mock_registry, max_depth=2)

    def test_deeply_nested_within_limit_succeeds(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Deep nesting within limit works correctly."""
        # Build expression with depth 5: add(add(add(add(add(x, y), z), w), v), u)
        expr = "add(add(add(add(add(x, y), z), w), v), u)"
        # This should succeed with default max_depth
        result = canonicalize_code(expr, mock_registry)
        assert "add" in result

    def test_error_message_includes_depth_info(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """RecursionError message includes depth information."""
        expr = "sin(sin(x))"
        with pytest.raises(RecursionError) as exc_info:
            canonicalize_code(expr, mock_registry, max_depth=1)
        error_msg = str(exc_info.value)
        assert "2" in error_msg # depth that exceeded
        assert "1" in error_msg # max_depth


# =============================================================================
# Unit Tests - Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Unit tests for edge cases."""

    def test_single_variable(self, mock_registry: FunctionRegistry) -> None:
        """Single variable canonicalizes to itself."""
        result = canonicalize_code("x", mock_registry)
        assert result == "x"

    def test_single_constant(self, mock_registry: FunctionRegistry) -> None:
        """Single constant canonicalizes to itself."""
        result = canonicalize_code("42", mock_registry)
        assert result == "42"

    def test_whitespace_preserved_or_normalized(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Expression with extra whitespace is normalized."""
        result = canonicalize_code("add( a, b )", mock_registry)
        # ast.unparse produces canonical formatting
        assert result == "add(a, b)"

    def test_many_commutative_arguments(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Function with many arguments (if supported) is sorted."""
        # Standard binary functions have arity 2, but test the concept
        # add only takes 2 args, so this tests nested adds
        result = canonicalize_code("add(add(c, b), a)", mock_registry)
        # Inner add sorted: add(b, c)
        # Then outer: add(a, add(b, c)) or add(add(b, c), a) depending on sort
        assert "add(b, c)" in result

    def test_complex_real_world_expression(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Complex expression typical in PDE discovery."""
        # u * u_x + C * u_xx in functional form
        expr = "add(mul(u, u_x), mul(C, u_xx))"
        result = canonicalize_code(expr, mock_registry)

        # Verify it's a valid expression
        tree = ast.parse(result, mode="eval")
        assert isinstance(tree.body, ast.Call)

    def test_equivalent_complex_expressions_same_hash(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """Equivalent complex expressions have same hash."""
        # Original: add(mul(u, u_x), mul(C, u_xx))
        # Reordered: add(mul(u_xx, C), mul(u_x, u))
        h1 = canonical_hash("add(mul(u, u_x), mul(C, u_xx))", mock_registry)
        h2 = canonical_hash("add(mul(u_xx, C), mul(u_x, u))", mock_registry)
        assert h1 == h2


# =============================================================================
# Unit Tests - AST Node Handling
# =============================================================================


@pytest.mark.unit
class TestASTNodeHandling:
    """Unit tests for direct AST node manipulation."""

    def test_canonicalize_name_node(self, mock_registry: FunctionRegistry) -> None:
        """canonicalize handles ast.Name node."""
        tree = ast.parse("x", mode="eval")
        result = canonicalize(tree.body, mock_registry)
        assert isinstance(result, ast.Name)
        assert result.id == "x"

    def test_canonicalize_constant_node(self, mock_registry: FunctionRegistry) -> None:
        """canonicalize handles ast.Constant node."""
        tree = ast.parse("1.0", mode="eval")
        result = canonicalize(tree.body, mock_registry)
        assert isinstance(result, ast.Constant)
        assert result.value == 1.0

    def test_canonicalize_call_node(self, mock_registry: FunctionRegistry) -> None:
        """canonicalize handles ast.Call node."""
        tree = ast.parse("sin(x)", mode="eval")
        result = canonicalize(tree.body, mock_registry)
        assert isinstance(result, ast.Call)

    def test_canonicalize_returns_valid_ast(
        self, mock_registry: FunctionRegistry
    ) -> None:
        """canonicalize returns valid AST that can be unparsed."""
        tree = ast.parse("add(mul(b, a), x)", mode="eval")
        result = canonicalize(tree.body, mock_registry)

        # Should be unparseable
        code = ast.unparse(result)
        assert isinstance(code, str)
        assert len(code) > 0

        # Should be re-parseable
        reparsed = ast.parse(code, mode="eval")
        assert reparsed is not None
