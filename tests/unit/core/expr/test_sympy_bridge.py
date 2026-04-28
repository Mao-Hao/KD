"""Tests for SymPy bridge: IR funcall string <-> SymPy expression conversion.

TDD red phase -- tests written against the interface spec
Module under test: src/kd2/core/expr/sympy_bridge.py

Functions:
  to_sympy(code, *, strict=True) -> sympy.Expr
  to_latex(code, *, strict=True) -> str
  format_pde(terms, coefficients, *, lhs="u_t") -> FormattedEquation
  are_equivalent(expr_a, expr_b) -> bool
  symbolic_diff(code, var) -> str
"""

from __future__ import annotations

import pytest
import sympy
import torch

from kd2.core.expr.sympy_bridge import (
    FormattedEquation,
    are_equivalent,
    format_pde,
    symbolic_diff,
    to_latex,
    to_sympy,
)

# ===========================================================================
# to_sympy: basic operators
# ===========================================================================


class TestToSympyBasic:
    """to_sympy converts IR funcall strings to SymPy expressions."""

    @pytest.mark.smoke
    def test_single_variable(self) -> None:
        result = to_sympy("u")
        assert result == sympy.Symbol("u")

    def test_add(self) -> None:
        result = to_sympy("add(u, u_x)")
        assert result == sympy.Symbol("u") + sympy.Symbol("u_x")

    def test_mul(self) -> None:
        result = to_sympy("mul(u, u_x)")
        assert result == sympy.Symbol("u") * sympy.Symbol("u_x")

    def test_sub(self) -> None:
        result = to_sympy("sub(u, x)")
        assert result == sympy.Symbol("u") - sympy.Symbol("x")

    def test_div(self) -> None:
        result = to_sympy("div(u, x)")
        assert result == sympy.Symbol("u") / sympy.Symbol("x")

    def test_n2(self) -> None:
        """n2(u) -> u**2."""
        result = to_sympy("n2(u)")
        assert result == sympy.Symbol("u") ** 2

    def test_n3(self) -> None:
        """n3(u) -> u**3."""
        result = to_sympy("n3(u)")
        assert result == sympy.Symbol("u") ** 3


# ===========================================================================
# to_sympy: derivative operators
# ===========================================================================


class TestToSympyDerivatives:
    """to_sympy handles diff_x/diff2_x as terminal symbol mappings."""

    def test_diff_x_simple(self) -> None:
        """diff_x(u) -> Symbol("u_x") in display mode."""
        result = to_sympy("diff_x(u)")
        assert result == sympy.Symbol("u_x")

    def test_diff2_x_simple(self) -> None:
        """diff2_x(u) -> Symbol("u_xx")."""
        result = to_sympy("diff2_x(u)")
        assert result == sympy.Symbol("u_xx")

    def test_diff_t(self) -> None:
        """diff_t(u) -> Symbol("u_t")."""
        result = to_sympy("diff_t(u)")
        assert result == sympy.Symbol("u_t")

    def test_diff2_t(self) -> None:
        """diff2_t(u) -> Symbol("u_tt")."""
        result = to_sympy("diff2_t(u)")
        assert result == sympy.Symbol("u_tt")

    def test_nested_diff_produces_compound_derivative_symbol(self) -> None:
        """diff_y(u_x) stays in terminal naming form as u_x_y."""
        result = to_sympy("diff_y(u_x)")
        assert result == sympy.Symbol("u_x_y")

    def test_nested_higher_diff_produces_compound_symbol(self) -> None:
        """diff2_y(u_xx) becomes u_xx_yy for downstream integrator parsing."""
        result = to_sympy("diff2_y(u_xx)")
        assert result == sympy.Symbol("u_xx_yy")

    def test_nested_same_axis_diff_preserves_fd_sequence(self) -> None:
        """diff_x(diff_x(u)) stays as sequential one-step derivatives."""
        result = to_sympy("diff_x(diff_x(u))")
        assert result == sympy.Symbol("u_x_x")

    def test_linear_diff_distributes_over_add(self) -> None:
        """diff_x(u_x + diff_x(u)) expands without lossy d1_x fallback."""
        result = to_sympy("diff_x(add(u_x, diff_x(u)))")
        assert result == 2 * sympy.Symbol("u_x_x")

    def test_linear_diff_extracts_numeric_scalar(self) -> None:
        """diff_x(2*u_x) extracts the numeric scalar but not product-rules."""
        result = to_sympy("diff_x(mul(2, u_x))")
        assert result == 2 * sympy.Symbol("u_x_x")

    def test_nonlinear_product_diff_remains_open_form(self) -> None:
        """diff_x(u*u_x) must not be product-rule expanded."""
        u = sympy.Symbol("u")
        u_x = sympy.Symbol("u_x")
        diff_x = sympy.Function("diff_x")

        result = to_sympy("diff_x(mul(u, u_x))")

        assert result == diff_x(u * u_x)

    def test_function_diff_remains_round_trippable_open_form(self) -> None:
        """diff_x(sin(u)) must not serialize as invalid Symbol('sin(u)_x')."""
        u = sympy.Symbol("u")
        diff_x = sympy.Function("diff_x")

        result = to_sympy("diff_x(sin(u))")

        assert result == diff_x(sympy.sin(u))

    def test_diff_of_integer_literal_returns_zero(self) -> None:
        """diff_x(0) must coerce the Python int via sympify, not crash."""
        assert to_sympy("diff_x(0)") == sympy.Integer(0)

    def test_diff_of_float_literal_returns_zero(self) -> None:
        """diff_x(2.5) must coerce the Python float via sympify, not crash."""
        assert to_sympy("diff_x(2.5)") == sympy.Integer(0)


# ===========================================================================
# to_sympy: special operators
# ===========================================================================


class TestToSympySpecialOperators:
    """to_sympy preserves special-operator structure."""

    def test_lap_simple(self) -> None:
        """lap(u) remains an uninterpreted SymPy Function application."""
        u = sympy.Symbol("u")
        lap = sympy.Function("lap")

        result = to_sympy("lap(u)")

        assert result == lap(u)

    def test_lap_nested_expression(self) -> None:
        """lap can wrap regular kd2 IR expressions."""
        u = sympy.Symbol("u")
        lap = sympy.Function("lap")

        result = to_sympy("lap(n2(u))")

        assert result == lap(u**2)


# ===========================================================================
# to_sympy: nested expressions
# ===========================================================================


class TestToSympyNested:
    """to_sympy handles nested funcall expressions."""

    def test_add_mul(self) -> None:
        """add(mul(u, u_x), u_xx) -> u*u_x + u_xx."""
        result = to_sympy("add(mul(u, u_x), u_xx)")
        u = sympy.Symbol("u")
        u_x = sympy.Symbol("u_x")
        u_xx = sympy.Symbol("u_xx")
        expected = u * u_x + u_xx
        assert sympy.expand(result - expected) == 0

    def test_mul_with_nested_add(self) -> None:
        """mul(u, add(x, t)) -> u*(x + t)."""
        result = to_sympy("mul(u, add(x, t))")
        u, x, t = sympy.symbols("u x t")
        expected = u * (x + t)
        assert sympy.expand(result - expected) == 0

    def test_deeply_nested(self) -> None:
        """div(n3(u), add(x, t)) -> u**3 / (x + t)."""
        result = to_sympy("div(n3(u), add(x, t))")
        u, x, t = sympy.symbols("u x t")
        expected = u**3 / (x + t)
        # Use simplify for rational expressions
        assert sympy.simplify(result - expected) == 0


# ===========================================================================
# to_sympy: numeric literals
# ===========================================================================


class TestToSympyNumericLiterals:
    """to_sympy handles numeric coefficient literals."""

    def test_integer_coefficient(self) -> None:
        """mul(2, u) -> 2*u."""
        result = to_sympy("mul(2, u)")
        assert result == 2 * sympy.Symbol("u")

    def test_float_coefficient(self) -> None:
        """mul(0.1, u_xx) -> 0.1*u_xx."""
        result = to_sympy("mul(0.1, u_xx)")
        u_xx = sympy.Symbol("u_xx")
        expected = sympy.Float(0.1) * u_xx
        assert abs(float(sympy.simplify(result - expected))) < 1e-15

    def test_negative_coefficient(self) -> None:
        """mul(-1.0, u) -> -u."""
        result = to_sympy("mul(-1.0, u)")
        u = sympy.Symbol("u")
        expected = sympy.Float(-1.0) * u
        assert abs(float(sympy.simplify(result - expected))) < 1e-15


# ===========================================================================
# to_sympy: error handling
# ===========================================================================


class TestToSympyErrors:
    """Error handling: strict vs. non-strict mode."""

    def test_strict_invalid_raises(self) -> None:
        """Invalid code with strict=True should raise ValueError."""
        with pytest.raises(ValueError):
            to_sympy("@@@invalid!!!", strict=True)

    def test_nonstrict_invalid_returns_symbol_fallback(self) -> None:
        """Invalid code with strict=False should return Symbol(code) fallback."""
        code = "@@@invalid!!!"
        result = to_sympy(code, strict=False)
        assert isinstance(result, sympy.Basic)

    def test_strict_is_default(self) -> None:
        """strict=True is the default."""
        with pytest.raises(ValueError):
            to_sympy("@@@invalid!!!")

    def test_empty_string_raises_strict(self) -> None:
        """Empty string in strict mode should raise."""
        with pytest.raises(ValueError):
            to_sympy("", strict=True)

    def test_empty_string_nonstrict_fallback(self) -> None:
        """Empty string in non-strict mode should not crash."""
        result = to_sympy("", strict=False)
        assert isinstance(result, sympy.Basic)


# ===========================================================================
# to_latex
# ===========================================================================


class TestToLatex:
    """to_latex converts IR funcall strings to LaTeX."""

    @pytest.mark.smoke
    def test_returns_string(self) -> None:
        result = to_latex("add(u, u_x)")
        assert isinstance(result, str)

    def test_subscript_rendering(self) -> None:
        """Variables with underscores should get subscript rendering.
        u_x -> u_{x}, u_xx -> u_{xx}."""
        result = to_latex("u_x")
        assert "u_{x}" in result or "u_x" in result

    def test_add_latex(self) -> None:
        """add(u, u_x) should produce LaTeX with +."""
        result = to_latex("add(u, u_x)")
        assert "+" in result

    def test_numeric_coefficient_preserved(self) -> None:
        """Numeric coefficients should appear in LaTeX output."""
        result = to_latex("mul(0.1, u_xx)")
        assert "0.1" in result

    def test_strict_invalid_raises(self) -> None:
        """strict=True should propagate ValueError from to_sympy."""
        with pytest.raises(ValueError):
            to_latex("@@@invalid!!!", strict=True)

    def test_nonstrict_invalid_returns_string(self) -> None:
        """strict=False should return some string, even for invalid input."""
        result = to_latex("@@@invalid!!!", strict=False)
        assert isinstance(result, str)


# ===========================================================================
# format_pde
# ===========================================================================


class TestFormatPde:
    """format_pde builds a complete PDE equation from terms + coefficients."""

    @pytest.mark.smoke
    def test_returns_formatted_equation(self) -> None:
        result = format_pde(
            ["mul(u, u_x)", "u_xx"],
            torch.tensor([-1.0, 0.1]),
        )
        assert isinstance(result, FormattedEquation)

    def test_has_required_fields(self) -> None:
        """FormattedEquation should have .latex, .sympy_expr, .lhs, .rhs."""
        result = format_pde(
            ["mul(u, u_x)", "u_xx"],
            torch.tensor([-1.0, 0.1]),
        )
        assert hasattr(result, "latex")
        assert hasattr(result, "sympy_expr")
        assert hasattr(result, "lhs")
        assert hasattr(result, "rhs")

    def test_lhs_default_is_u_t(self) -> None:
        """Default lhs is 'u_t'."""
        result = format_pde(
            ["u"],
            torch.tensor([1.0]),
        )
        assert result.lhs == "u_t"

    def test_custom_lhs(self) -> None:
        """Custom lhs label should be used."""
        result = format_pde(
            ["u"],
            torch.tensor([1.0]),
            lhs="v_t",
        )
        assert result.lhs == "v_t"

    def test_latex_contains_lhs(self) -> None:
        """LaTeX output should contain the lhs label."""
        result = format_pde(
            ["u"],
            torch.tensor([1.0]),
            lhs="u_t",
        )
        # LaTeX should mention u_t somewhere (possibly with subscript)
        assert "u" in result.latex

    def test_burgers_equation(self) -> None:
        """Burgers: u_t = -u*u_x + 0.1*u_xx.
        The LaTeX should contain the equation, not raw IR."""
        result = format_pde(
            ["mul(u, u_x)", "u_xx"],
            torch.tensor([-1.0, 0.1]),
        )
        latex = result.latex
        assert isinstance(latex, str)
        assert len(latex) > 0
        # Should not contain raw IR function names
        assert "mul(" not in latex
        assert "add(" not in latex

    def test_single_term(self) -> None:
        """Single term PDE should work."""
        result = format_pde(
            ["u_xx"],
            torch.tensor([1.0]),
        )
        assert isinstance(result.latex, str)
        assert result.rhs is not None

    def test_zero_coefficient_excluded(self) -> None:
        """Terms with zero coefficient should be excluded from the expression."""
        result = format_pde(
            ["u", "u_x", "u_xx"],
            torch.tensor([1.0, 0.0, 0.5]),
        )
        # The rhs should NOT contain u_x as a free symbol (zero coefficient)
        rhs_symbols = result.rhs.free_symbols
        assert sympy.Symbol("u_x") not in rhs_symbols
        # But u and u_xx should remain
        assert sympy.Symbol("u") in rhs_symbols
        assert sympy.Symbol("u_xx") in rhs_symbols

    def test_empty_terms_list(self) -> None:
        """Empty terms list should produce a valid result (likely 0 rhs)."""
        result = format_pde(
            [],
            torch.tensor([]),
        )
        assert isinstance(result, FormattedEquation)
        assert isinstance(result.latex, str)

    def test_lap_term_does_not_crash(self) -> None:
        """format_pde accepts lap(u) as an uninterpreted SymPy term."""
        result = format_pde(
            ["lap(u)"],
            torch.tensor([1.0]),
            lhs="u_t",
        )
        lap = sympy.Function("lap")
        u = sympy.Symbol("u")

        assert isinstance(result.latex, str)
        assert result.rhs == lap(u)


# ===========================================================================
# format_pde: properties (anti-reverse-engineering)
# ===========================================================================


class TestFormatPdeProperties:
    """Property-based checks for format_pde."""

    def test_sympy_expr_is_equation(self) -> None:
        """sympy_expr should be an Eq object (full equation, not just RHS)."""
        result = format_pde(
            ["u", "u_xx"],
            torch.tensor([2.0, -0.5]),
        )
        # sympy_expr is Eq(lhs, rhs) per 
        assert isinstance(result.sympy_expr, sympy.Eq)

    def test_rhs_matches_manual_construction(self) -> None:
        """The rhs field should equal manually constructed SymPy RHS."""
        result = format_pde(
            ["u", "u_xx"],
            torch.tensor([2.0, -0.5]),
        )
        u = sympy.Symbol("u")
        u_xx = sympy.Symbol("u_xx")
        expected_rhs = 2.0 * u + (-0.5) * u_xx
        diff = sympy.simplify(result.rhs - expected_rhs)
        assert abs(float(diff)) < 1e-12

    def test_coefficient_sign_preserved(self) -> None:
        """Negative coefficients should produce subtraction in the result."""
        result = format_pde(
            ["u"],
            torch.tensor([-3.0]),
        )
        # Use rhs (not sympy_expr which is an Eq)
        rhs_float = float(result.rhs.subs(sympy.Symbol("u"), 1.0))
        assert rhs_float < 0

    def test_selected_indices(self) -> None:
        """selected_indices filters which terms are included."""
        result = format_pde(
            ["u", "u_x", "u_xx"],
            torch.tensor([1.0, 2.0, 3.0]),
            selected_indices=[0, 2], # skip u_x
        )
        rhs_symbols = result.rhs.free_symbols
        assert sympy.Symbol("u_x") not in rhs_symbols
        assert sympy.Symbol("u") in rhs_symbols
        assert sympy.Symbol("u_xx") in rhs_symbols

    def test_selected_indices_empty_list(self) -> None:
        """selected_indices=[] means no terms selected (empty RHS).
        Must NOT be treated as None (which means all terms)."""
        result = format_pde(
            ["u", "u_x"],
            torch.tensor([1.0, 2.0]),
            selected_indices=[],
        )
        # Empty selection → rhs should be 0
        assert result.rhs == 0


# ===========================================================================
# are_equivalent
# ===========================================================================


class TestAreEquivalent:
    """are_equivalent checks algebraic equivalence of two IR strings."""

    @pytest.mark.smoke
    def test_identical(self) -> None:
        assert are_equivalent("add(u, x)", "add(u, x)") is True

    def test_commutativity_add(self) -> None:
        """add(u, x) is equivalent to add(x, u)."""
        assert are_equivalent("add(u, x)", "add(x, u)") is True

    def test_commutativity_mul(self) -> None:
        """mul(u, x) is equivalent to mul(x, u)."""
        assert are_equivalent("mul(u, x)", "mul(x, u)") is True

    def test_distributivity(self) -> None:
        """mul(u, add(x, t)) == add(mul(u, x), mul(u, t))."""
        assert (
            are_equivalent(
                "mul(u, add(x, t))",
                "add(mul(u, x), mul(u, t))",
            )
            is True
        )

    def test_non_equivalent(self) -> None:
        """Clearly different expressions should return False."""
        assert are_equivalent("add(u, x)", "mul(u, x)") is False

    def test_sub_not_commutative(self) -> None:
        """sub(u, x) is NOT equivalent to sub(x, u)."""
        assert are_equivalent("sub(u, x)", "sub(x, u)") is False

    def test_lap_identical_expressions(self) -> None:
        """Identical lap expressions are equivalent."""
        assert are_equivalent("lap(u)", "lap(u)") is True

    def test_lap_different_arguments_not_equivalent(self) -> None:
        """lap(u) and lap(v) preserve different arguments."""
        assert are_equivalent("lap(u)", "lap(v)") is False


# ===========================================================================
# symbolic_diff
# ===========================================================================


class TestSymbolicDiff:
    """symbolic_diff differentiates an IR expression w.r.t. a variable."""

    @pytest.mark.smoke
    def test_returns_string(self) -> None:
        result = symbolic_diff("n2(u)", "u")
        assert isinstance(result, str)

    def test_n2_derivative(self) -> None:
        """d/du(u^2) = 2*u. Result should represent 2*u."""
        result = symbolic_diff("n2(u)", "u")
        # Parse the result back to SymPy to verify
        result_sympy = to_sympy(result, strict=False)
        u = sympy.Symbol("u")
        expected = 2 * u
        assert sympy.expand(result_sympy - expected) == 0

    def test_n3_derivative(self) -> None:
        """d/du(u^3) = 3*u^2."""
        result = symbolic_diff("n3(u)", "u")
        result_sympy = to_sympy(result, strict=False)
        u = sympy.Symbol("u")
        expected = 3 * u**2
        assert sympy.expand(result_sympy - expected) == 0

    def test_diff_wrt_other_variable_is_zero(self) -> None:
        """d/dx(u) = 0 when u and x are independent symbols."""
        result = symbolic_diff("u", "x")
        result_sympy = to_sympy(result, strict=False)
        assert result_sympy == 0

    def test_mul_derivative(self) -> None:
        """d/du(mul(u, x)) = x (product rule, x independent of u)."""
        result = symbolic_diff("mul(u, x)", "u")
        result_sympy = to_sympy(result, strict=False)
        x = sympy.Symbol("x")
        assert sympy.expand(result_sympy - x) == 0


# ===========================================================================
# Negative tests (>= 20%)
# ===========================================================================


class TestSympyBridgeNegative:
    """Negative and failure-injection tests for sympy bridge."""

    def test_to_sympy_unbalanced_parens_strict_raises(self) -> None:
        """Unbalanced parentheses should raise in strict mode."""
        with pytest.raises(ValueError):
            to_sympy("add(u, x", strict=True)

    def test_to_sympy_unknown_function_strict_raises(self) -> None:
        """Unknown function name in strict mode should raise."""
        with pytest.raises(ValueError):
            to_sympy("unknown_func(u, x)", strict=True)

    def test_format_pde_mismatched_length_raises(self) -> None:
        """terms and coefficients length mismatch should raise."""
        with pytest.raises((ValueError, RuntimeError)):
            format_pde(
                ["u", "u_x", "u_xx"],
                torch.tensor([1.0, 2.0]), # 2 coeffs for 3 terms
            )

    def test_are_equivalent_invalid_input(self) -> None:
        """Invalid input to are_equivalent should raise or return False."""
        try:
            result = are_equivalent("@@@", "add(u, x)")
            assert result is False
        except (ValueError, SyntaxError):
            pass

    def test_symbolic_diff_empty_var_raises(self) -> None:
        """Empty variable name should raise or handle gracefully."""
        with pytest.raises((ValueError, TypeError)):
            symbolic_diff("n2(u)", "")

    def test_format_pde_nan_coefficient(self) -> None:
        """NaN coefficient should either raise or produce valid output."""
        try:
            result = format_pde(
                ["u"],
                torch.tensor([float("nan")]),
            )
            # If it returns, the LaTeX should still be a string
            assert isinstance(result.latex, str)
        except (ValueError, RuntimeError):
            pass

    def test_format_pde_inf_coefficient(self) -> None:
        """Inf coefficient should either raise or produce valid output."""
        try:
            result = format_pde(
                ["u"],
                torch.tensor([float("inf")]),
            )
            assert isinstance(result.latex, str)
        except (ValueError, RuntimeError):
            pass


# ===========================================================================
# M1/M2: _sympy_to_ir — trig/exp/log function support
# ===========================================================================


class TestSympyToIrTrigExpLog:
    """_sympy_to_ir must handle sin/cos/exp/log SymPy expressions.

    Currently _sympy_to_ir raises ValueError for these because it
    has no branches for sympy.sin, sympy.cos, sympy.exp, sympy.log.
    """

    def test_sympy_to_ir_sin(self) -> None:
        """_sympy_to_ir(sympy.sin(x)) -> 'sin(x)'."""
        from kd2.core.expr.sympy_bridge import _sympy_to_ir

        x = sympy.Symbol("x")
        result = _sympy_to_ir(sympy.sin(x))
        # Parse the result back through to_sympy to verify round-trip
        assert result == "sin(x)"

    def test_sympy_to_ir_cos(self) -> None:
        """_sympy_to_ir(sympy.cos(x)) -> 'cos(x)'."""
        from kd2.core.expr.sympy_bridge import _sympy_to_ir

        x = sympy.Symbol("x")
        result = _sympy_to_ir(sympy.cos(x))
        assert result == "cos(x)"

    def test_sympy_to_ir_exp(self) -> None:
        """_sympy_to_ir(sympy.exp(x)) -> 'exp(x)'."""
        from kd2.core.expr.sympy_bridge import _sympy_to_ir

        x = sympy.Symbol("x")
        result = _sympy_to_ir(sympy.exp(x))
        assert result == "exp(x)"

    def test_sympy_to_ir_log(self) -> None:
        """_sympy_to_ir(sympy.log(x)) -> 'log(x)'."""
        from kd2.core.expr.sympy_bridge import _sympy_to_ir

        x = sympy.Symbol("x")
        result = _sympy_to_ir(sympy.log(x))
        assert result == "log(x)"


class TestSympyToIrLap:
    """_sympy_to_ir must preserve uninterpreted lap(...) calls."""

    def test_sympy_to_ir_lap(self) -> None:
        """_sympy_to_ir(lap(u)) -> 'lap(u)'."""
        from kd2.core.expr.sympy_bridge import _sympy_to_ir

        lap_func = sympy.Function("lap")
        u = sympy.Symbol("u")

        result = _sympy_to_ir(lap_func(u))

        assert result == "lap(u)"


class TestSympyToIrOpenFormDiff:
    """_sympy_to_ir must preserve unsupported open-form diff calls."""

    def test_sympy_to_ir_diff_sin_round_trip(self) -> None:
        """diff_x(sin(u)) serializes to legal kd2 IR, not sin(u)_x."""
        from kd2.core.expr.sympy_bridge import _sympy_to_ir

        u = sympy.Symbol("u")
        diff_x = sympy.Function("diff_x")

        result = _sympy_to_ir(diff_x(sympy.sin(u)))

        assert result == "diff_x(sin(u))"
        assert to_sympy(result) == diff_x(sympy.sin(u))

    def test_sympy_to_ir_diff_lap_round_trip(self) -> None:
        """diff_x(lap(u)) remains parseable for /M2 round-trip."""
        from kd2.core.expr.sympy_bridge import _sympy_to_ir

        u = sympy.Symbol("u")
        diff_x = sympy.Function("diff_x")
        lap = sympy.Function("lap")

        result = _sympy_to_ir(diff_x(lap(u)))

        assert result == "diff_x(lap(u))"
        assert to_sympy(result) == diff_x(lap(u))


# ===========================================================================
# M1/M2: symbolic_diff with trig functions
# ===========================================================================


class TestSymbolicDiffWithTrig:
    """symbolic_diff must not crash on expressions containing sin/cos/exp.

    Currently symbolic_diff -> sympy.diff -> _sympy_to_ir, and the last
    step fails because _sympy_to_ir has no sin/cos/exp branches.
    """

    def test_symbolic_diff_with_sin(self) -> None:
        """d/du(sin(u)) = cos(u). Must not crash; result must round-trip."""
        result = symbolic_diff("sin(u)", "u")
        # The result should be a valid IR string representing cos(u)
        result_sympy = to_sympy(result, strict=False)
        u = sympy.Symbol("u")
        expected = sympy.cos(u)
        assert sympy.expand(result_sympy - expected) == 0

    def test_symbolic_diff_with_exp(self) -> None:
        """d/du(exp(u)) = exp(u). Self-derivative property."""
        result = symbolic_diff("exp(u)", "u")
        result_sympy = to_sympy(result, strict=False)
        u = sympy.Symbol("u")
        expected = sympy.exp(u)
        assert sympy.expand(result_sympy - expected) == 0

    def test_symbolic_diff_mul_sin(self) -> None:
        """d/du(mul(u, sin(u))) = sin(u) + u*cos(u). Product rule with trig."""
        result = symbolic_diff("mul(u, sin(u))", "u")
        result_sympy = to_sympy(result, strict=False)
        u = sympy.Symbol("u")
        expected = sympy.sin(u) + u * sympy.cos(u)
        assert sympy.expand(result_sympy - expected) == 0


class TestSymbolicDiffWithLap:
    """symbolic_diff must serialize derivatives that contain lap(...) terms."""

    def test_symbolic_diff_mul_v_lap_u_wrt_v(self) -> None:
        """d/dv(v * lap(u)) = lap(u)."""
        result = symbolic_diff("mul(v, lap(u))", "v")

        assert result == "lap(u)"


# ===========================================================================
# M1/M1: _derivative_symbol_name for non-Symbol inputs
# ===========================================================================


class TestDerivativeSymbolNameNonSymbol:
    """_derivative_symbol_name should preserve structure for Apply types.

    Currently, for non-Symbol inputs (like sin(u)), it falls back to
    the generic 'd{ord}_{axis}' format, losing the expression structure.
    After the fix, it should produce something that retains the function
    information, e.g., including 'sin' in the output.
    """

    def test_derivative_symbol_name_preserves_function(self) -> None:
        """_derivative_symbol_name on sin(u) should retain 'sin' info.

        The exact output format is an implementation choice, but it must
        NOT be the generic 'd1_x' fallback -- that loses all structure.
        """
        from kd2.core.expr.sympy_bridge import _derivative_symbol_name

        u = sympy.Symbol("u")
        expr = sympy.sin(u)
        result = _derivative_symbol_name(expr, "x", 1)

        # Must NOT be the lossy generic fallback
        assert result != "d1_x", (
            "_derivative_symbol_name fell back to generic 'd1_x' "
            "for sin(u), losing expression structure"
        )
        # The result should be a non-empty string
        assert isinstance(result, str)
        assert len(result) > 0
