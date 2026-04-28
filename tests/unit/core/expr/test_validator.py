"""Tests for expression validator.

Test coverage:
- smoke: Function exists and is callable
- unit: Valid expressions, invalid expressions, edge cases
- numerical: N/A (validator is not numerical)

TDD Status: RED - Tests written before implementation.
"""

import pytest

from kd2.core.expr.validator import (
    ALLOWED_NODES,
    get_function_calls,
    validate_expr,
)

# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.smoke
class TestValidatorSmoke:
    """Smoke tests: function existence and basic callability."""

    def test_validate_expr_exists(self) -> None:
        """validate_expr function exists and is callable."""
        assert callable(validate_expr)

    def test_get_function_calls_exists(self) -> None:
        """get_function_calls function exists and is callable."""
        assert callable(get_function_calls)

    def test_allowed_nodes_is_set(self) -> None:
        """ALLOWED_NODES is a set of types."""
        assert isinstance(ALLOWED_NODES, set)
        assert len(ALLOWED_NODES) > 0
        for node_type in ALLOWED_NODES:
            assert isinstance(node_type, type)


# =============================================================================
# Unit Tests - Valid Expressions
# =============================================================================


@pytest.mark.unit
class TestValidExpressions:
    """Unit tests for expressions that should be valid."""

    @pytest.fixture
    def allowed_funcs(self) -> set[str]:
        """Standard set of allowed functions."""
        return {"add", "mul", "sub", "div", "sin", "cos", "exp", "neg", "n2", "n3"}

    def test_simple_function_call(self, allowed_funcs: set[str]) -> None:
        """Simple function call is valid."""
        assert validate_expr("add(u, v)", allowed_funcs) is True

    def test_unary_function_call(self, allowed_funcs: set[str]) -> None:
        """Unary function call is valid."""
        assert validate_expr("sin(x)", allowed_funcs) is True

    def test_constant_expression(self, allowed_funcs: set[str]) -> None:
        """Bare constant is valid."""
        assert validate_expr("1.0", allowed_funcs) is True
        assert validate_expr("42", allowed_funcs) is True
        assert validate_expr("-2.5", allowed_funcs) is True

    def test_variable_expression(self, allowed_funcs: set[str]) -> None:
        """Bare variable (Name) is valid."""
        assert validate_expr("u", allowed_funcs) is True
        assert validate_expr("u_x", allowed_funcs) is True
        assert validate_expr("x", allowed_funcs) is True

    def test_nested_function_calls(self, allowed_funcs: set[str]) -> None:
        """Nested function calls are valid."""
        assert validate_expr("add(sin(x), cos(y))", allowed_funcs) is True
        assert validate_expr("mul(add(u, v), sub(x, y))", allowed_funcs) is True

    def test_deeply_nested_expression(self, allowed_funcs: set[str]) -> None:
        """Deeply nested expressions are valid."""
        expr = "add(mul(sin(u), cos(v)), neg(exp(x)))"
        assert validate_expr(expr, allowed_funcs) is True

    def test_function_with_constant_arg(self, allowed_funcs: set[str]) -> None:
        """Function call with constant argument is valid."""
        assert validate_expr("mul(u, 2.0)", allowed_funcs) is True
        assert validate_expr("add(1.0, 2.0)", allowed_funcs) is True

    def test_negative_constant(self, allowed_funcs: set[str]) -> None:
        """Negative constant is valid (ast.Constant with negative value)."""
        # Note: -2.5 parses as ast.UnaryOp(-) + ast.Constant(2.5) in older Python
        # But ast.Constant(-2.5) in newer Python when literal is negative
        # The expr "neg(2.5)" is the kd2 way to represent negation
        assert validate_expr("neg(2.5)", allowed_funcs) is True

    def test_zero_arity_terminal(self, allowed_funcs: set[str]) -> None:
        """Variable names (terminals) are valid."""
        assert validate_expr("u_x", allowed_funcs) is True
        assert validate_expr("u_xx", allowed_funcs) is True


# =============================================================================
# Unit Tests - Invalid Expressions
# =============================================================================


@pytest.mark.unit
class TestInvalidExpressions:
    """Unit tests for expressions that should be invalid."""

    @pytest.fixture
    def allowed_funcs(self) -> set[str]:
        """Standard set of allowed functions."""
        return {"add", "mul", "sub", "div", "sin", "cos"}

    def test_infix_operator_binop(self, allowed_funcs: set[str]) -> None:
        """Infix binary operator (BinOp) is not allowed."""
        assert validate_expr("u * v", allowed_funcs) is False
        assert validate_expr("u + v", allowed_funcs) is False
        assert validate_expr("u - v", allowed_funcs) is False
        assert validate_expr("u / v", allowed_funcs) is False
        assert validate_expr("u ** 2", allowed_funcs) is False

    def test_unary_minus_operator(self, allowed_funcs: set[str]) -> None:
        """Unary minus operator (UnaryOp) is not allowed."""
        # -x parses as UnaryOp(op=USub(), operand=Name('x'))
        assert validate_expr("-x", allowed_funcs) is False

    def test_method_call(self, allowed_funcs: set[str]) -> None:
        """Method call (Attribute) is not allowed."""
        assert validate_expr("x.sin()", allowed_funcs) is False
        assert validate_expr("tensor.mean()", allowed_funcs) is False
        assert validate_expr("obj.method(a, b)", allowed_funcs) is False

    def test_subscript(self, allowed_funcs: set[str]) -> None:
        """Subscript is not allowed."""
        assert validate_expr("x[0]", allowed_funcs) is False
        assert validate_expr("data[i]", allowed_funcs) is False

    def test_list_literal(self, allowed_funcs: set[str]) -> None:
        """List literal is not allowed."""
        assert validate_expr("[1, 2, 3]", allowed_funcs) is False

    def test_dict_literal(self, allowed_funcs: set[str]) -> None:
        """Dict literal is not allowed."""
        assert validate_expr("{'a': 1}", allowed_funcs) is False

    def test_comprehension(self, allowed_funcs: set[str]) -> None:
        """List comprehension is not allowed."""
        assert validate_expr("[x for x in items]", allowed_funcs) is False

    def test_lambda(self, allowed_funcs: set[str]) -> None:
        """Lambda expression is not allowed."""
        assert validate_expr("lambda x: x + 1", allowed_funcs) is False

    def test_comparison(self, allowed_funcs: set[str]) -> None:
        """Comparison operators are not allowed."""
        assert validate_expr("x > 0", allowed_funcs) is False
        assert validate_expr("a == b", allowed_funcs) is False

    def test_boolean_operator(self, allowed_funcs: set[str]) -> None:
        """Boolean operators are not allowed."""
        assert validate_expr("a and b", allowed_funcs) is False
        assert validate_expr("a or b", allowed_funcs) is False
        assert validate_expr("not a", allowed_funcs) is False

    def test_conditional_expression(self, allowed_funcs: set[str]) -> None:
        """Conditional (ternary) expression is not allowed."""
        assert validate_expr("a if cond else b", allowed_funcs) is False


# =============================================================================
# Unit Tests - Unknown Functions
# =============================================================================


@pytest.mark.unit
class TestUnknownFunctions:
    """Unit tests for function name validation."""

    def test_unknown_function_rejected(self) -> None:
        """Function not in allowed_funcs is rejected."""
        allowed = {"add", "mul"}
        assert validate_expr("unknown_func(x)", allowed) is False

    def test_typo_function_rejected(self) -> None:
        """Typo in function name is rejected."""
        allowed = {"sin", "cos"}
        assert validate_expr("sine(x)", allowed) is False # typo: sine vs sin

    def test_case_sensitive_rejection(self) -> None:
        """Function names are case-sensitive."""
        allowed = {"sin", "cos"}
        assert validate_expr("Sin(x)", allowed) is False
        assert validate_expr("SIN(x)", allowed) is False

    def test_nested_unknown_function(self) -> None:
        """Unknown function nested inside valid expression is rejected."""
        allowed = {"add", "mul"}
        assert validate_expr("add(unknown(x), y)", allowed) is False

    def test_all_functions_must_be_allowed(self) -> None:
        """All functions in expression must be in allowed set."""
        allowed = {"add"}
        # mul is not allowed
        assert validate_expr("add(mul(a, b), c)", allowed) is False


# =============================================================================
# Unit Tests - Syntax Errors
# =============================================================================


@pytest.mark.unit
class TestSyntaxErrors:
    """Unit tests for syntax error handling."""

    @pytest.fixture
    def allowed_funcs(self) -> set[str]:
        """Standard set of allowed functions."""
        return {"add", "mul", "sin"}

    def test_syntax_error_unbalanced_parens(self, allowed_funcs: set[str]) -> None:
        """Unbalanced parentheses returns False, not raises."""
        assert validate_expr("add(a, b", allowed_funcs) is False
        assert validate_expr("add(a, b))", allowed_funcs) is False

    def test_syntax_error_missing_comma(self, allowed_funcs: set[str]) -> None:
        """Missing comma returns False."""
        assert validate_expr("add(a b)", allowed_funcs) is False

    def test_syntax_error_invalid_token(self, allowed_funcs: set[str]) -> None:
        """Invalid token returns False."""
        assert validate_expr("add(a, @b)", allowed_funcs) is False

    def test_syntax_error_statement(self, allowed_funcs: set[str]) -> None:
        """Statement (not expression) returns False."""
        assert validate_expr("x = 1", allowed_funcs) is False
        assert validate_expr("import math", allowed_funcs) is False


# =============================================================================
# Unit Tests - Edge Cases
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Unit tests for edge cases."""

    @pytest.fixture
    def allowed_funcs(self) -> set[str]:
        """Standard set of allowed functions."""
        return {"add", "mul", "sin"}

    def test_empty_string(self, allowed_funcs: set[str]) -> None:
        """Empty string returns False."""
        assert validate_expr("", allowed_funcs) is False

    def test_whitespace_only(self, allowed_funcs: set[str]) -> None:
        """Whitespace-only string returns False."""
        assert validate_expr(" ", allowed_funcs) is False
        assert validate_expr("\n\t", allowed_funcs) is False

    def test_empty_allowed_funcs(self) -> None:
        """Empty allowed_funcs rejects all function calls."""
        assert validate_expr("add(a, b)", set()) is False
        # But bare variable should still be valid
        assert validate_expr("x", set()) is True
        assert validate_expr("1.0", set()) is True

    def test_whitespace_around_expression(self, allowed_funcs: set[str]) -> None:
        """Whitespace around expression is handled."""
        assert validate_expr(" add(a, b) ", allowed_funcs) is True
        assert validate_expr("\nadd(a, b)\n", allowed_funcs) is True

    def test_expression_with_float_scientific_notation(
        self, allowed_funcs: set[str]
    ) -> None:
        """Scientific notation constants are valid."""
        assert validate_expr("1e-10", allowed_funcs) is True
        assert validate_expr("add(x, 1e5)", allowed_funcs) is True

    def test_expression_with_boolean_constant(self, allowed_funcs: set[str]) -> None:
        """Boolean constants are technically valid (ast.Constant)."""
        # True/False are ast.Constant in Python 3.8+
        # Whether to allow them is a design choice; currently allowed
        # as they are ast.Constant nodes
        result = validate_expr("True", allowed_funcs)
        # Could be True or False depending on design decision
        assert isinstance(result, bool)

    def test_expression_with_none_constant(self, allowed_funcs: set[str]) -> None:
        """None constant is valid (ast.Constant)."""
        result = validate_expr("None", allowed_funcs)
        assert isinstance(result, bool)

    def test_very_long_expression(self, allowed_funcs: set[str]) -> None:
        """Very long expression can be validated."""
        # Build a deeply nested expression
        expr = "x"
        for _ in range(100):
            expr = f"sin({expr})"
        assert validate_expr(expr, allowed_funcs) is True

    def test_many_arguments_rejected(self, allowed_funcs: set[str]) -> None:
        """Function call with too many arguments is still syntactically valid.

        Note: arity checking is done by FunctionRegistry, not validator.
        Validator only checks AST structure.
        """
        # This is syntactically valid but semantically wrong
        # Validator allows it; executor will fail on arity mismatch
        assert validate_expr("add(a, b, c)", allowed_funcs) is True


# =============================================================================
# Unit Tests - get_function_calls()
# =============================================================================


@pytest.mark.unit
class TestGetFunctionCalls:
    """Unit tests for get_function_calls helper."""

    def test_single_function(self) -> None:
        """Extract single function name."""
        result = get_function_calls("sin(x)")
        assert result == {"sin"}

    def test_multiple_functions(self) -> None:
        """Extract multiple function names."""
        result = get_function_calls("add(sin(x), mul(y, z))")
        assert result == {"add", "sin", "mul"}

    def test_no_functions(self) -> None:
        """Expression with no functions returns empty set."""
        result = get_function_calls("x")
        assert result == set()

    def test_constant_no_functions(self) -> None:
        """Constant expression returns empty set."""
        result = get_function_calls("1.0")
        assert result == set()

    def test_nested_same_function(self) -> None:
        """Repeated function appears once in result."""
        result = get_function_calls("add(add(a, b), add(c, d))")
        assert result == {"add"}

    def test_syntax_error_returns_empty(self) -> None:
        """Syntax error returns empty set, not raises."""
        result = get_function_calls("add(a, b")
        assert result == set()

    def test_empty_string_returns_empty(self) -> None:
        """Empty string returns empty set."""
        result = get_function_calls("")
        assert result == set()


# =============================================================================
# Unit Tests - Security Considerations
# =============================================================================


@pytest.mark.unit
class TestSecurityConsiderations:
    """Unit tests for security-relevant validation."""

    @pytest.fixture
    def allowed_funcs(self) -> set[str]:
        """Standard set of allowed functions."""
        return {"add", "mul", "sin"}

    def test_dunder_attribute_rejected(self, allowed_funcs: set[str]) -> None:
        """Expressions accessing dunder attributes are rejected."""
        # These would be method calls (Attribute node), should be rejected
        assert validate_expr("x.__class__", allowed_funcs) is False
        assert validate_expr("x.__dict__", allowed_funcs) is False

    def test_builtin_function_not_in_allowed(self, allowed_funcs: set[str]) -> None:
        """Built-in functions not in allowed_funcs are rejected."""
        assert validate_expr("eval(code)", allowed_funcs) is False
        assert validate_expr("exec(code)", allowed_funcs) is False
        assert validate_expr("__import__('os')", allowed_funcs) is False

    def test_call_on_call_result_rejected(self, allowed_funcs: set[str]) -> None:
        """Calling result of another call is rejected (not bare function)."""
        # get_func(name)(x) would be Call(func=Call(...))
        # func must be ast.Name, not ast.Call
        assert validate_expr("get_func('sin')(x)", allowed_funcs) is False
