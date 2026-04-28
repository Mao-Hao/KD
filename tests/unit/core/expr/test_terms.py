"""Tests for split_terms function.

Test coverage:
- smoke: Basic interface and simple cases
- unit: Core splitting logic for add/sub/nested expressions
- edge cases: Empty, single term, deeply nested

Note: Tests written in TDD TDD red phase - implementation in progress.
"""

from __future__ import annotations

import pytest

from kd2.core.expr import FunctionRegistry, split_terms

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def registry() -> FunctionRegistry:
    """Default FunctionRegistry with standard operators."""
    return FunctionRegistry.create_default()


# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.smoke
class TestSplitTermsSmoke:
    """Smoke tests: basic interface existence."""

    def test_split_terms_exists(self) -> None:
        """split_terms function is importable."""
        from kd2.core.expr import split_terms

        assert callable(split_terms)

    def test_split_terms_returns_list(self, registry: FunctionRegistry) -> None:
        """split_terms returns a list of strings."""
        result = split_terms("u", registry)
        assert isinstance(result, list)
        assert all(isinstance(item, str) for item in result)

    def test_single_variable(self, registry: FunctionRegistry) -> None:
        """Single variable returns as single term."""
        result = split_terms("u", registry)
        assert result == ["u"]


# =============================================================================
# Unit Tests - Single Term (No Splitting)
# =============================================================================


@pytest.mark.unit
class TestSingleTerm:
    """Tests for expressions that should NOT be split."""

    def test_single_variable(self, registry: FunctionRegistry) -> None:
        """Variable name stays as single term."""
        assert split_terms("u", registry) == ["u"]
        assert split_terms("v", registry) == ["v"]
        assert split_terms("u_x", registry) == ["u_x"]
        assert split_terms("u_xx", registry) == ["u_xx"]

    def test_mul_expression(self, registry: FunctionRegistry) -> None:
        """mul() is not splittable."""
        result = split_terms("mul(u, v)", registry)
        assert result == ["mul(u, v)"]

    def test_mul_with_derivative(self, registry: FunctionRegistry) -> None:
        """mul with derivative stays as single term."""
        result = split_terms("mul(u, u_x)", registry)
        assert result == ["mul(u, u_x)"]

    def test_div_expression(self, registry: FunctionRegistry) -> None:
        """div() is not splittable."""
        result = split_terms("div(u, v)", registry)
        assert result == ["div(u, v)"]

    def test_unary_function(self, registry: FunctionRegistry) -> None:
        """Unary functions stay as single term."""
        assert split_terms("sin(u)", registry) == ["sin(u)"]
        assert split_terms("cos(u)", registry) == ["cos(u)"]
        assert split_terms("neg(u)", registry) == ["neg(u)"]
        assert split_terms("n2(u)", registry) == ["n2(u)"]

    def test_nested_non_add_sub(self, registry: FunctionRegistry) -> None:
        """Nested non-add/sub expressions stay as single term."""
        result = split_terms("mul(sin(u), cos(v))", registry)
        assert result == ["mul(sin(u), cos(v))"]

    def test_numeric_constant(self, registry: FunctionRegistry) -> None:
        """Numeric constants stay as single term."""
        assert split_terms("1.5", registry) == ["1.5"]
        assert split_terms("0", registry) == ["0"]


# =============================================================================
# Unit Tests - add() Splitting
# =============================================================================


@pytest.mark.unit
class TestAddSplitting:
    """Tests for splitting add() expressions."""

    def test_simple_add(self, registry: FunctionRegistry) -> None:
        """add(a, b) splits into [a, b]."""
        result = split_terms("add(a, b)", registry)
        assert result == ["a", "b"]

    def test_add_variables(self, registry: FunctionRegistry) -> None:
        """add() with variables splits correctly."""
        result = split_terms("add(u, v)", registry)
        assert result == ["u", "v"]

    def test_add_with_derivatives(self, registry: FunctionRegistry) -> None:
        """add() with derivatives splits correctly."""
        result = split_terms("add(u_x, u_xx)", registry)
        assert result == ["u_x", "u_xx"]

    def test_nested_add_right(self, registry: FunctionRegistry) -> None:
        """add(a, add(b, c)) flattens to [a, b, c]."""
        result = split_terms("add(a, add(b, c))", registry)
        assert result == ["a", "b", "c"]

    def test_nested_add_left(self, registry: FunctionRegistry) -> None:
        """add(add(a, b), c) flattens to [a, b, c]."""
        result = split_terms("add(add(a, b), c)", registry)
        assert result == ["a", "b", "c"]

    def test_deeply_nested_add(self, registry: FunctionRegistry) -> None:
        """Deeply nested add flattens completely."""
        # add(add(a, add(b, c)), add(d, e))
        result = split_terms("add(add(a, add(b, c)), add(d, e))", registry)
        assert result == ["a", "b", "c", "d", "e"]

    def test_add_with_complex_terms(self, registry: FunctionRegistry) -> None:
        """add() with complex sub-expressions."""
        # add(mul(u, u_x), u_xx)
        result = split_terms("add(mul(u, u_x), u_xx)", registry)
        assert result == ["mul(u, u_x)", "u_xx"]

    def test_add_preserves_inner_add_in_non_top(
        self, registry: FunctionRegistry
    ) -> None:
        """add inside mul is not split (not top-level)."""
        # mul(add(a, b), c) - the inner add is NOT at top level
        result = split_terms("mul(add(a, b), c)", registry)
        assert result == ["mul(add(a, b), c)"]


# =============================================================================
# Unit Tests - sub() Handling
# =============================================================================


@pytest.mark.unit
class TestSubHandling:
    """Tests for sub() expressions - should wrap second arg in neg()."""

    def test_simple_sub(self, registry: FunctionRegistry) -> None:
        """sub(a, b) becomes [a, neg(b)]."""
        result = split_terms("sub(a, b)", registry)
        assert result == ["a", "neg(b)"]

    def test_sub_with_variables(self, registry: FunctionRegistry) -> None:
        """sub() with variables."""
        result = split_terms("sub(u, v)", registry)
        assert result == ["u", "neg(v)"]

    def test_sub_with_complex_second_arg(self, registry: FunctionRegistry) -> None:
        """sub() wraps complex second arg in neg()."""
        # sub(a, mul(b, c)) -> [a, neg(mul(b, c))]
        result = split_terms("sub(a, mul(b, c))", registry)
        assert result == ["a", "neg(mul(b, c))"]

    def test_sub_first_arg_is_add(self, registry: FunctionRegistry) -> None:
        """sub with add as first arg flattens the add part."""
        # sub(add(a, b), c) -> [a, b, neg(c)]
        result = split_terms("sub(add(a, b), c)", registry)
        assert result == ["a", "b", "neg(c)"]

    def test_sub_second_arg_is_add(self, registry: FunctionRegistry) -> None:
        """sub with add as second arg - does NOT flatten the add.

        sub(a, add(b, c)) = a - (b + c) = a - b - c
        This should become [a, neg(b), neg(c)], not [a, neg(add(b, c))]

        Design decision: We need to distribute the negation.
        """
        result = split_terms("sub(a, add(b, c))", registry)
        # Distributing: a - (b + c) = a + (-b) + (-c)
        assert result == ["a", "neg(b)", "neg(c)"]

    def test_sub_second_arg_is_sub(self, registry: FunctionRegistry) -> None:
        """sub with sub as second arg.

        sub(a, sub(b, c)) = a - (b - c) = a - b + c
        This should become [a, neg(b), c]
        """
        result = split_terms("sub(a, sub(b, c))", registry)
        # a - (b - c) = a + (-b) + c
        assert result == ["a", "neg(b)", "c"]


# =============================================================================
# Unit Tests - Mixed add/sub
# =============================================================================


@pytest.mark.unit
class TestMixedAddSub:
    """Tests for mixed add/sub expressions."""

    def test_add_then_sub(self, registry: FunctionRegistry) -> None:
        """add(a, sub(b, c)) flattens with neg."""
        # add(a, sub(b, c)) = a + (b - c) = a + b - c
        # = [a, b, neg(c)]
        result = split_terms("add(a, sub(b, c))", registry)
        assert result == ["a", "b", "neg(c)"]

    def test_sub_then_add(self, registry: FunctionRegistry) -> None:
        """sub(a, add(b, c)) = a - b - c."""
        result = split_terms("sub(a, add(b, c))", registry)
        assert result == ["a", "neg(b)", "neg(c)"]

    def test_complex_mixed(self, registry: FunctionRegistry) -> None:
        """Complex mix of add/sub."""
        # add(sub(a, b), sub(c, d))
        # = (a - b) + (c - d)
        # = a - b + c - d
        # = [a, neg(b), c, neg(d)]
        result = split_terms("add(sub(a, b), sub(c, d))", registry)
        assert result == ["a", "neg(b)", "c", "neg(d)"]


# =============================================================================
# Unit Tests - Burgers Equation Examples
# =============================================================================


@pytest.mark.unit
class TestBurgersExamples:
    """Tests using Burgers equation typical expressions."""

    def test_burgers_rhs_split(self, registry: FunctionRegistry) -> None:
        """Burgers RHS: add(mul(u, u_x), u_xx) -> [mul(u, u_x), u_xx]."""
        result = split_terms("add(mul(u, u_x), u_xx)", registry)
        assert result == ["mul(u, u_x)", "u_xx"]

    def test_burgers_full_equation_split(self, registry: FunctionRegistry) -> None:
        """Full Burgers: sub(mul(u, u_x), mul(nu, u_xx)).

        u_t = -u*u_x + nu*u_xx
        RHS = sub(neg(mul(u, u_x)), neg(mul(nu, u_xx)))
        Actually let's test: add(neg(mul(u, u_x)), mul(nu, u_xx))
        """
        # Testing a realistic Burgers term structure
        result = split_terms("add(neg(mul(u, u_x)), mul(nu, u_xx))", registry)
        assert result == ["neg(mul(u, u_x))", "mul(nu, u_xx)"]


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge case tests."""

    def test_empty_string_raises(self, registry: FunctionRegistry) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            split_terms("", registry)

    def test_whitespace_only_raises(self, registry: FunctionRegistry) -> None:
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError):
            split_terms(" ", registry)

    def test_invalid_syntax_raises(self, registry: FunctionRegistry) -> None:
        """Invalid syntax raises ValueError."""
        with pytest.raises(ValueError):
            split_terms("add(a, )", registry) # Trailing comma

        with pytest.raises(ValueError):
            split_terms("add(a", registry) # Missing closing paren

    def test_triple_add_argument_not_supported(
        self, registry: FunctionRegistry
    ) -> None:
        """add() with 3 arguments - Python syntax valid, but not our DSL.

        add(a, b, c) is valid Python but our add is binary.
        The behavior depends on ast.parse - it will parse but shouldn't work.
        """
        # This should either raise or handle gracefully
        # Since add is binary, the expression is semantically invalid
        # But ast.parse will accept it - split_terms should validate or handle
        with pytest.raises((ValueError, TypeError)):
            split_terms("add(a, b, c)", registry)

    def test_unknown_function_preserves(self, registry: FunctionRegistry) -> None:
        """Unknown function names are preserved (not split).

        The function is unknown to registry, but split_terms only cares
        about add/sub at top level. Unknown functions should be preserved.
        """
        result = split_terms("unknown_func(a, b)", registry)
        assert result == ["unknown_func(a, b)"]

    def test_very_deep_nesting(self, registry: FunctionRegistry) -> None:
        """Handle deeply nested expressions."""
        # add(add(add(add(a, b), c), d), e)
        expr = "add(add(add(add(a, b), c), d), e)"
        result = split_terms(expr, registry)
        assert result == ["a", "b", "c", "d", "e"]


# =============================================================================
# Unit Tests - neg() Simplification (Optional Enhancement)
# =============================================================================


@pytest.mark.unit
class TestNegSimplification:
    """Tests for neg() simplification.

    Design decision: Do we simplify neg(neg(x)) -> x?
    For now, we don't simplify - preserve the structure.
    """

    def test_double_neg_preserved(self, registry: FunctionRegistry) -> None:
        """neg(neg(x)) is preserved, not simplified.

        sub(a, neg(b)) = a - (-b) = a + b
        But we represent as [a, neg(neg(b))]

        If we want simplification, we'd need [a, b].
        Current design: preserve structure, no simplification.
        """
        # sub(a, neg(b)) -> [a, neg(neg(b))]
        result = split_terms("sub(a, neg(b))", registry)
        # The neg(b) gets wrapped in another neg due to sub rule
        assert result == ["a", "neg(neg(b))"]

    def test_existing_neg_in_expression(self, registry: FunctionRegistry) -> None:
        """neg() inside expression is preserved."""
        result = split_terms("add(neg(a), b)", registry)
        assert result == ["neg(a)", "b"]


# =============================================================================
#: Infix Rejection Tests
# =============================================================================


@pytest.mark.unit
class TestInfixRejection:
    """Tests for infix operator rejection.

     D2: split_terms() must detect ast.BinOp (infix operators
    like +, -, *) and raise ValueError. The IR contract requires funcall
    format -- infix input is a plugin bug that should be caught early.
    """

    def test_infix_add_rejected(self, registry: FunctionRegistry) -> None:
        """Infix 'u + u_xx' must be rejected with ValueError.

        This is the primary bug: infix expressions are accepted
        by Python eval but not split by split_terms, causing silent
        single-term fitting with wrong coefficients.
        """
        with pytest.raises(ValueError, match="(?i)infix|BinOp|operator"):
            split_terms("u + u_xx", registry)

    def test_infix_sub_rejected(self, registry: FunctionRegistry) -> None:
        """Infix 'u - u_xx' must be rejected."""
        with pytest.raises(ValueError, match="(?i)infix|BinOp|operator"):
            split_terms("u - u_xx", registry)

    def test_infix_mul_rejected(self, registry: FunctionRegistry) -> None:
        """Infix 'u * u_xx' must be rejected."""
        with pytest.raises(ValueError, match="(?i)infix|BinOp|operator"):
            split_terms("u * u_xx", registry)

    def test_funcall_add_still_works(self, registry: FunctionRegistry) -> None:
        """Funcall 'add(u, u_xx)' must still work (regression guard).

        This confirms that the infix rejection does not accidentally
        break the valid funcall path.
        """
        result = split_terms("add(u, u_xx)", registry)
        assert result == ["u", "u_xx"]

    def test_infix_nested_in_funcall_rejected(self, registry: FunctionRegistry) -> None:
        """Infix inside funcall: 'add(u + v, u_xx)' must be rejected.

        Even if the top-level node is a funcall, infix operators
        anywhere in the AST should trigger rejection -- they indicate
        the expression is not in proper IR format.
        """
        with pytest.raises(ValueError, match="(?i)infix|BinOp|operator"):
            split_terms("add(u + v, u_xx)", registry)
