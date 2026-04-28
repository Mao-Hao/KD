"""Tests for prefix notation bridging functions.

Tests cover:
- Smoke: Basic conversion works
- Happy path: Various valid expressions
- Edge cases: Empty, single variable, pure constant
- Round-trip: python -> prefix -> python
- Error handling: Invalid syntax, unknown functions
- Numerical constants: int, float, negative
"""

import pytest

from kd2.core.compat.prefix import prefix_to_python, python_to_prefix
from kd2.core.expr.registry import FunctionRegistry


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def default_registry() -> FunctionRegistry:
    """Create default registry with standard operators."""
    return FunctionRegistry.create_default()


@pytest.fixture
def extended_registry() -> FunctionRegistry:
    """Registry with additional test functions."""
    reg = FunctionRegistry.create_default()
    # Add some custom functions for testing
    reg.register("sqrt", lambda x: x**0.5, arity=1)
    reg.register("pow", lambda x, y: x**y, arity=2)
    return reg


# =============================================================================
# Smoke Tests - Basic functionality
# =============================================================================


class TestPythonToPrefixSmoke:
    """Smoke tests for python_to_prefix."""

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        """python_to_prefix should be callable."""
        assert callable(python_to_prefix)

    @pytest.mark.smoke
    def test_simple_binary_expression(self) -> None:
        """Basic binary operation should convert correctly."""
        result = python_to_prefix("add(x, y)")
        assert result == ["add", "x", "y"]

    @pytest.mark.smoke
    def test_simple_unary_expression(self) -> None:
        """Basic unary operation should convert correctly."""
        result = python_to_prefix("sin(x)")
        assert result == ["sin", "x"]


class TestPrefixToPythonSmoke:
    """Smoke tests for prefix_to_python."""

    @pytest.mark.smoke
    def test_function_exists_and_callable(self) -> None:
        """prefix_to_python should be callable."""
        assert callable(prefix_to_python)

    @pytest.mark.smoke
    def test_simple_binary_expression(self, default_registry: FunctionRegistry) -> None:
        """Basic binary operation should convert correctly."""
        result = prefix_to_python(["add", "x", "y"], default_registry)
        assert result == "add(x, y)"

    @pytest.mark.smoke
    def test_simple_unary_expression(self, default_registry: FunctionRegistry) -> None:
        """Basic unary operation should convert correctly."""
        result = prefix_to_python(["sin", "x"], default_registry)
        assert result == "sin(x)"


# =============================================================================
# Happy Path Tests - python_to_prefix
# =============================================================================


class TestPythonToPrefixHappyPath:
    """Happy path tests for python_to_prefix."""

    @pytest.mark.unit
    def test_nested_binary_operations(self) -> None:
        """Nested binary operations should preserve structure."""
        result = python_to_prefix("add(mul(a, b), sub(c, d))")
        assert result == ["add", "mul", "a", "b", "sub", "c", "d"]

    @pytest.mark.unit
    def test_deeply_nested_expression(self) -> None:
        """Deeply nested expressions should work."""
        # add(mul(u, u_x), mul(C, u_xx)) - typical PDE term
        result = python_to_prefix("add(mul(u, u_x), mul(C, u_xx))")
        assert result == ["add", "mul", "u", "u_x", "mul", "C", "u_xx"]

    @pytest.mark.unit
    def test_mixed_unary_binary(self) -> None:
        """Mixed unary and binary operations."""
        result = python_to_prefix("add(sin(x), cos(y))")
        assert result == ["add", "sin", "x", "cos", "y"]

    @pytest.mark.unit
    def test_chained_unary(self) -> None:
        """Chained unary operations."""
        result = python_to_prefix("sin(cos(exp(x)))")
        assert result == ["sin", "cos", "exp", "x"]

    @pytest.mark.unit
    def test_three_level_nesting(self) -> None:
        """Three levels of nesting."""
        result = python_to_prefix("add(mul(sin(x), y), z)")
        assert result == ["add", "mul", "sin", "x", "y", "z"]

    @pytest.mark.unit
    def test_multiple_variables(self) -> None:
        """Expression with many different variables."""
        result = python_to_prefix("add(add(a, b), add(c, d))")
        assert result == ["add", "add", "a", "b", "add", "c", "d"]

    @pytest.mark.unit
    def test_derivative_style_names(self) -> None:
        """Derivative-style variable names (u_x, u_xx, etc.)."""
        result = python_to_prefix("mul(u_x, u_xx)")
        assert result == ["mul", "u_x", "u_xx"]


# =============================================================================
# Happy Path Tests - prefix_to_python
# =============================================================================


class TestPrefixToPythonHappyPath:
    """Happy path tests for prefix_to_python."""

    @pytest.mark.unit
    def test_nested_binary_operations(self, default_registry: FunctionRegistry) -> None:
        """Nested binary operations should restore correctly."""
        tokens = ["add", "mul", "a", "b", "sub", "c", "d"]
        result = prefix_to_python(tokens, default_registry)
        assert result == "add(mul(a, b), sub(c, d))"

    @pytest.mark.unit
    def test_deeply_nested_expression(self, default_registry: FunctionRegistry) -> None:
        """Deeply nested expressions should restore correctly."""
        tokens = ["add", "mul", "u", "u_x", "mul", "C", "u_xx"]
        result = prefix_to_python(tokens, default_registry)
        assert result == "add(mul(u, u_x), mul(C, u_xx))"

    @pytest.mark.unit
    def test_mixed_unary_binary(self, default_registry: FunctionRegistry) -> None:
        """Mixed unary and binary operations."""
        tokens = ["add", "sin", "x", "cos", "y"]
        result = prefix_to_python(tokens, default_registry)
        assert result == "add(sin(x), cos(y))"

    @pytest.mark.unit
    def test_chained_unary(self, default_registry: FunctionRegistry) -> None:
        """Chained unary operations."""
        tokens = ["sin", "cos", "exp", "x"]
        result = prefix_to_python(tokens, default_registry)
        assert result == "sin(cos(exp(x)))"

    @pytest.mark.unit
    def test_derivative_style_names(self, default_registry: FunctionRegistry) -> None:
        """Derivative-style variable names."""
        tokens = ["mul", "u_x", "u_xx"]
        result = prefix_to_python(tokens, default_registry)
        assert result == "mul(u_x, u_xx)"


# =============================================================================
# Edge Cases - Single elements
# =============================================================================


class TestSingleElement:
    """Edge cases for single-element expressions."""

    @pytest.mark.unit
    def test_python_to_prefix_single_variable(self) -> None:
        """Single variable should return one-element list."""
        result = python_to_prefix("x")
        assert result == ["x"]

    @pytest.mark.unit
    def test_python_to_prefix_single_variable_with_underscore(self) -> None:
        """Variable with underscore."""
        result = python_to_prefix("u_x")
        assert result == ["u_x"]

    @pytest.mark.unit
    def test_prefix_to_python_single_variable(
        self, default_registry: FunctionRegistry
    ) -> None:
        """Single variable token should return just the variable."""
        result = prefix_to_python(["x"], default_registry)
        assert result == "x"

    @pytest.mark.unit
    def test_prefix_to_python_single_variable_with_underscore(
        self, default_registry: FunctionRegistry
    ) -> None:
        """Variable with underscore."""
        result = prefix_to_python(["u_x"], default_registry)
        assert result == "u_x"


# =============================================================================
# Edge Cases - Constants
# =============================================================================


class TestConstants:
    """Tests for numeric constants."""

    @pytest.mark.unit
    def test_python_to_prefix_integer(self) -> None:
        """Integer constant."""
        result = python_to_prefix("42")
        assert result == ["42"]

    @pytest.mark.unit
    def test_python_to_prefix_float(self) -> None:
        """Float constant."""
        result = python_to_prefix("3.14")
        assert result == ["3.14"]

    @pytest.mark.unit
    def test_python_to_prefix_negative_integer(self) -> None:
        """Negative integer (unary minus in AST).

        Design decision: Negative constants are output as literal strings
        (e.g., ["-5"]) rather than unary operations (e.g., ["neg", "5"]).
        This is more concise and matches the design doc's `str(node.value)`.
        """
        result = python_to_prefix("-5")
        assert result == ["-5"]

    @pytest.mark.unit
    def test_python_to_prefix_negative_float(self) -> None:
        """Negative float.

        Design decision: Negative constants are output as literal strings.
        """
        result = python_to_prefix("-3.14")
        assert result == ["-3.14"]

    @pytest.mark.unit
    def test_python_to_prefix_scientific_notation(self) -> None:
        """Scientific notation."""
        result = python_to_prefix("1e-10")
        assert result == ["1e-10"]

    @pytest.mark.unit
    def test_python_to_prefix_constant_in_expression(self) -> None:
        """Constant mixed with variables."""
        result = python_to_prefix("mul(2.5, x)")
        assert result == ["mul", "2.5", "x"]

    @pytest.mark.unit
    def test_prefix_to_python_integer(self, default_registry: FunctionRegistry) -> None:
        """Integer constant in prefix."""
        result = prefix_to_python(["42"], default_registry)
        assert result == "42"

    @pytest.mark.unit
    def test_prefix_to_python_float(self, default_registry: FunctionRegistry) -> None:
        """Float constant in prefix."""
        result = prefix_to_python(["3.14"], default_registry)
        assert result == "3.14"

    @pytest.mark.unit
    def test_prefix_to_python_negative(self, default_registry: FunctionRegistry) -> None:
        """Negative constant in prefix."""
        result = prefix_to_python(["-5"], default_registry)
        assert result == "-5"

    @pytest.mark.unit
    def test_prefix_to_python_constant_in_expression(
        self, default_registry: FunctionRegistry
    ) -> None:
        """Constant mixed with variables."""
        result = prefix_to_python(["mul", "2.5", "x"], default_registry)
        assert result == "mul(2.5, x)"


# =============================================================================
# Edge Cases - Empty and minimal
# =============================================================================


class TestEmptyAndMinimal:
    """Edge cases for empty and minimal inputs."""

    @pytest.mark.unit
    def test_python_to_prefix_empty_raises(self) -> None:
        """Empty string should raise an error."""
        with pytest.raises((SyntaxError, ValueError)):
            python_to_prefix("")

    @pytest.mark.unit
    def test_python_to_prefix_whitespace_only_raises(self) -> None:
        """Whitespace-only string should raise an error."""
        with pytest.raises((SyntaxError, ValueError)):
            python_to_prefix(" ")

    @pytest.mark.unit
    def test_prefix_to_python_empty_raises(
        self, default_registry: FunctionRegistry
    ) -> None:
        """Empty token list should raise an error."""
        with pytest.raises(ValueError):
            prefix_to_python([], default_registry)


# =============================================================================
# Round-trip Tests
# =============================================================================


class TestRoundTrip:
    """Round-trip conversion tests: python -> prefix -> python."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "expr",
        [
            "x",
            "add(x, y)",
            "mul(a, b)",
            "sin(x)",
            "cos(y)",
            "exp(z)",
            "add(mul(x, y), z)",
            "sin(cos(x))",
            "add(sin(x), cos(y))",
            "mul(u_x, u_xx)",
            "add(mul(u, u_x), mul(C, u_xx))",
            "div(sub(a, b), add(c, d))",
            "neg(x)",
            "n2(x)",
            "n3(y)",
        ],
    )
    def test_round_trip_preserves_expression(
        self, default_registry: FunctionRegistry, expr: str
    ) -> None:
        """Converting to prefix and back should preserve the expression."""
        prefix = python_to_prefix(expr)
        restored = prefix_to_python(prefix, default_registry)
        assert restored == expr

    @pytest.mark.unit
    def test_round_trip_with_constants(
        self, default_registry: FunctionRegistry
    ) -> None:
        """Round-trip with numeric constants."""
        expr = "add(mul(2, x), 3.14)"
        prefix = python_to_prefix(expr)
        restored = prefix_to_python(prefix, default_registry)
        assert restored == expr

    @pytest.mark.unit
    def test_round_trip_complex_pde_term(
        self, default_registry: FunctionRegistry
    ) -> None:
        """Round-trip with complex PDE-like expression."""
        # Burgers equation term: u * u_x + nu * u_xx
        expr = "add(mul(u, u_x), mul(nu, u_xx))"
        prefix = python_to_prefix(expr)
        restored = prefix_to_python(prefix, default_registry)
        assert restored == expr

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "expr,expected_prefix",
        [
            # Standalone negative constants
            ("-5", ["-5"]),
            ("-3.14", ["-3.14"]),
            ("-1e-10", ["-1e-10"]),
            # Negative constants in expressions
            ("add(x, -5)", ["add", "x", "-5"]),
            ("mul(-2.5, y)", ["mul", "-2.5", "y"]),
            ("add(-1, -2)", ["add", "-1", "-2"]),
        ],
    )
    def test_round_trip_negative_constants(
        self, default_registry: FunctionRegistry, expr: str, expected_prefix: list[str]
    ) -> None:
        """Round-trip for negative constants.

        Negative constants should be preserved as literal strings (e.g., "-5")
        not converted to unary operations (e.g., "neg(5)").
        """
        # Verify python -> prefix produces expected format
        prefix = python_to_prefix(expr)
        assert prefix == expected_prefix

        # Verify round-trip: prefix -> python -> prefix
        restored = prefix_to_python(prefix, default_registry)
        prefix_again = python_to_prefix(restored)
        assert prefix_again == expected_prefix


# =============================================================================
# Error Handling - python_to_prefix
# =============================================================================


class TestPythonToPrefixErrors:
    """Error handling tests for python_to_prefix."""

    @pytest.mark.unit
    def test_invalid_syntax_raises(self) -> None:
        """Invalid Python syntax should raise SyntaxError."""
        with pytest.raises(SyntaxError):
            python_to_prefix("add x y") # Missing parentheses - actual syntax error

    @pytest.mark.unit
    def test_incomplete_expression_raises(self) -> None:
        """Incomplete expression should raise."""
        with pytest.raises(SyntaxError):
            python_to_prefix("add(x") # Missing closing paren

    @pytest.mark.unit
    def test_statement_not_expression_raises(self) -> None:
        """Statement (not expression) should raise."""
        with pytest.raises(SyntaxError):
            python_to_prefix("x = 5") # Assignment is statement

    @pytest.mark.unit
    def test_multiple_expressions_raises(self) -> None:
        """Multiple expressions should raise."""
        with pytest.raises(SyntaxError):
            python_to_prefix("x; y") # Multiple statements

    @pytest.mark.unit
    def test_method_call_raises(self) -> None:
        """Method calls (obj.method()) should raise ValueError."""
        # We only support simple function calls, not methods
        with pytest.raises((ValueError, AttributeError)):
            python_to_prefix("x.sin()")

    @pytest.mark.unit
    def test_subscript_raises(self) -> None:
        """Subscript operations should raise ValueError."""
        with pytest.raises(ValueError):
            python_to_prefix("x[0]")

    @pytest.mark.unit
    def test_binary_operator_raises(self) -> None:
        """Binary operator syntax (+, *, etc.) should raise ValueError."""
        # We only support function call syntax, not operators
        with pytest.raises(ValueError):
            python_to_prefix("x + y")

    @pytest.mark.unit
    def test_list_literal_raises(self) -> None:
        """List literal should raise ValueError."""
        with pytest.raises(ValueError):
            python_to_prefix("[1, 2, 3]")


# =============================================================================
# Error Handling - prefix_to_python
# =============================================================================


class TestPrefixToPythonErrors:
    """Error handling tests for prefix_to_python."""

    @pytest.mark.unit
    def test_unbalanced_too_few_args(self, default_registry: FunctionRegistry) -> None:
        """Too few arguments for function should raise."""
        # add requires 2 args, but only x is available
        with pytest.raises(ValueError):
            prefix_to_python(["add", "x"], default_registry)

    @pytest.mark.unit
    def test_unbalanced_extra_tokens(self, default_registry: FunctionRegistry) -> None:
        """Extra tokens after valid expression should raise."""
        # sin(x) consumes 2 tokens, y is extra
        with pytest.raises(ValueError):
            prefix_to_python(["sin", "x", "y"], default_registry)

    @pytest.mark.unit
    def test_unknown_function_used_as_variable(
        self, default_registry: FunctionRegistry
    ) -> None:
        """Unknown name is treated as variable (arity 0)."""
        # unknown_func is not in registry, so treated as variable
        result = prefix_to_python(["add", "unknown_var", "x"], default_registry)
        assert result == "add(unknown_var, x)"

    @pytest.mark.unit
    def test_deeply_unbalanced(self, default_registry: FunctionRegistry) -> None:
        """Deeply unbalanced expression should raise."""
        # add(add(x, y), ???) - missing second arg
        with pytest.raises(ValueError):
            prefix_to_python(["add", "add", "x", "y"], default_registry)


# =============================================================================
# Extended Registry Tests
# =============================================================================


class TestExtendedRegistry:
    """Tests with extended registry (custom functions)."""

    @pytest.mark.unit
    def test_custom_unary_function(self, extended_registry: FunctionRegistry) -> None:
        """Custom unary function should work."""
        result = prefix_to_python(["sqrt", "x"], extended_registry)
        assert result == "sqrt(x)"

    @pytest.mark.unit
    def test_custom_binary_function(self, extended_registry: FunctionRegistry) -> None:
        """Custom binary function should work."""
        result = prefix_to_python(["pow", "x", "2"], extended_registry)
        assert result == "pow(x, 2)"

    @pytest.mark.unit
    def test_mixed_default_and_custom(
        self, extended_registry: FunctionRegistry
    ) -> None:
        """Mix of default and custom functions."""
        tokens = ["add", "sqrt", "x", "pow", "y", "2"]
        result = prefix_to_python(tokens, extended_registry)
        assert result == "add(sqrt(x), pow(y, 2))"


# =============================================================================
# Specific Algorithm Compatibility (DISCOVER)
# =============================================================================


class TestDISCOVERCompatibility:
    """Tests for DISCOVER algorithm compatibility."""

    @pytest.mark.unit
    def test_discover_style_expression(self) -> None:
        """DISCOVER-style expression should convert correctly."""
        # Typical DISCOVER output format
        code = "add(mul(u, u_x), mul(C, u_xx))"
        prefix = python_to_prefix(code)
        expected = ["add", "mul", "u", "u_x", "mul", "C", "u_xx"]
        assert prefix == expected

    @pytest.mark.unit
    def test_discover_reconstruction(
        self, default_registry: FunctionRegistry
    ) -> None:
        """DISCOVER prefix should reconstruct correctly."""
        # Input from DISCOVER algorithm
        prefix_from_discover = ["add", "mul", "u", "u_x", "mul", "C", "u_xx"]
        result = prefix_to_python(prefix_from_discover, default_registry)
        assert result == "add(mul(u, u_x), mul(C, u_xx))"

    @pytest.mark.unit
    def test_burgers_equation_term(self) -> None:
        """Burgers equation term: u*u_x + nu*u_xx."""
        code = "add(mul(u, u_x), mul(nu, u_xx))"
        prefix = python_to_prefix(code)
        assert prefix == ["add", "mul", "u", "u_x", "mul", "nu", "u_xx"]

    @pytest.mark.unit
    def test_heat_equation_term(self) -> None:
        """Heat equation term: alpha*u_xx."""
        code = "mul(alpha, u_xx)"
        prefix = python_to_prefix(code)
        assert prefix == ["mul", "alpha", "u_xx"]

    @pytest.mark.unit
    def test_wave_equation_term(self) -> None:
        """Wave equation term: c^2*u_xx."""
        code = "mul(n2(c), u_xx)"
        prefix = python_to_prefix(code)
        assert prefix == ["mul", "n2", "c", "u_xx"]
