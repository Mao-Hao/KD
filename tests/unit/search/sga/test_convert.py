"""Tests for SGA tree-to-kd2 expression converter.

TDD red phase -- tests written against the interface spec before implementation.
Covers tree_to_kd2_expr (single tree) and pde_to_kd2_expr (multi-term + coefficients).

Format: funcall syntax -- e.g. "mul(u, x)" instead of "mul u x".
d/d^2 operators are mapped to diff_axis/diff2_axis.
"""

from __future__ import annotations

import pytest

from kd2.search.sga.convert import pde_to_kd2_expr, tree_to_kd2_expr
from kd2.search.sga.pde import PDE
from kd2.search.sga.tree import Node, Tree

# Helpers


def _leaf(name: str) -> Node:
    return Node(name=name, arity=0, children=[])


def _unary(op: str, child: Node) -> Node:
    return Node(name=op, arity=1, children=[child])


def _binary(op: str, left: Node, right: Node) -> Node:
    return Node(name=op, arity=2, children=[left, right])


def _tree(root: Node) -> Tree:
    return Tree(root=root)


# ===========================================================================
# tree_to_kd2_expr: leaf nodes
# ===========================================================================


class TestTreeToKd2ExprLeaf:
    """Leaf nodes (variables) pass through without mapping."""

    @pytest.mark.smoke
    def test_single_variable(self) -> None:
        tree = _tree(_leaf("u"))
        assert tree_to_kd2_expr(tree) == "u"

    def test_various_variables(self) -> None:
        """Variables like u, x, t, u_x, u_t should pass through unchanged."""
        for var in ("u", "x", "t", "u_x", "u_t", "u_xx", "u_tt"):
            tree = _tree(_leaf(var))
            result = tree_to_kd2_expr(tree)
            assert result == var, f"Expected '{var}', got '{result}'"


# ===========================================================================
# tree_to_kd2_expr: D1 name mapping (funcall format)
# ===========================================================================


class TestTreeToKd2ExprNameMapping:
    """D1 name mapping: SGA internal names -> kd2 funcall syntax."""

    def test_binary_add(self) -> None:
        """+ -> add(u, x)."""
        tree = _tree(_binary("+", _leaf("u"), _leaf("x")))
        result = tree_to_kd2_expr(tree)
        assert result == "add(u, x)"

    def test_binary_sub(self) -> None:
        """- -> sub(u, x)."""
        tree = _tree(_binary("-", _leaf("u"), _leaf("x")))
        result = tree_to_kd2_expr(tree)
        assert result == "sub(u, x)"

    def test_binary_mul(self) -> None:
        """* -> mul(u, x)."""
        tree = _tree(_binary("*", _leaf("u"), _leaf("x")))
        result = tree_to_kd2_expr(tree)
        assert result == "mul(u, x)"

    def test_binary_div(self) -> None:
        """/ -> div(u, x)."""
        tree = _tree(_binary("/", _leaf("u"), _leaf("x")))
        result = tree_to_kd2_expr(tree)
        assert result == "div(u, x)"

    def test_unary_square(self) -> None:
        """^2 -> n2(u)."""
        tree = _tree(_unary("^2", _leaf("u")))
        result = tree_to_kd2_expr(tree)
        assert result == "n2(u)"

    def test_unary_cube(self) -> None:
        """^3 -> n3(u)."""
        tree = _tree(_unary("^3", _leaf("u")))
        result = tree_to_kd2_expr(tree)
        assert result == "n3(u)"


# ===========================================================================
# tree_to_kd2_expr: d / d^2 operators
# ===========================================================================


class TestTreeToKd2ExprDerivatives:
    """d and d^2 operators map to diff_axis and diff2_axis funcall syntax.

    d and d^2 have arity=2: children[0] is the expression, children[1]
    is a leaf node whose name is the axis variable (e.g. "x", "t").
    """

    def test_d_simple(self) -> None:
        """d(u, x) -> diff_x(u)."""
        tree = _tree(_binary("d", _leaf("u"), _leaf("x")))
        result = tree_to_kd2_expr(tree)
        assert result == "diff_x(u)"

    def test_d_axis_t(self) -> None:
        """d(u, t) -> diff_t(u)."""
        tree = _tree(_binary("d", _leaf("u"), _leaf("t")))
        result = tree_to_kd2_expr(tree)
        assert result == "diff_t(u)"

    def test_d2_simple(self) -> None:
        """d^2(u, x) -> diff2_x(u)."""
        tree = _tree(_binary("d^2", _leaf("u"), _leaf("x")))
        result = tree_to_kd2_expr(tree)
        assert result == "diff2_x(u)"

    def test_d2_axis_t(self) -> None:
        """d^2(u, t) -> diff2_t(u)."""
        tree = _tree(_binary("d^2", _leaf("u"), _leaf("t")))
        result = tree_to_kd2_expr(tree)
        assert result == "diff2_t(u)"

    def test_d_with_expression_arg(self) -> None:
        """d(*(u, u_x), x) -> diff_x(mul(u, u_x))."""
        expr = _binary("*", _leaf("u"), _leaf("u_x"))
        tree = _tree(_binary("d", expr, _leaf("x")))
        result = tree_to_kd2_expr(tree)
        assert result == "diff_x(mul(u, u_x))"

    def test_d2_with_expression_arg(self) -> None:
        """d^2(+(u, x), t) -> diff2_t(add(u, x))."""
        expr = _binary("+", _leaf("u"), _leaf("x"))
        tree = _tree(_binary("d^2", expr, _leaf("t")))
        result = tree_to_kd2_expr(tree)
        assert result == "diff2_t(add(u, x))"

    def test_nested_d_of_d2(self) -> None:
        """d(d^2(u, x), x) -> diff_x(diff2_x(u))."""
        inner = _binary("d^2", _leaf("u"), _leaf("x"))
        tree = _tree(_binary("d", inner, _leaf("x")))
        result = tree_to_kd2_expr(tree)
        assert result == "diff_x(diff2_x(u))"

    def test_d_in_larger_tree(self) -> None:
        """*(d(u, x), u) -> mul(diff_x(u), u)."""
        d_node = _binary("d", _leaf("u"), _leaf("x"))
        tree = _tree(_binary("*", d_node, _leaf("u")))
        result = tree_to_kd2_expr(tree)
        assert result == "mul(diff_x(u), u)"


# ===========================================================================
# tree_to_kd2_expr: nested trees (funcall format)
# ===========================================================================


class TestTreeToKd2ExprNested:
    """Nested expressions produce correct funcall syntax strings."""

    def test_binary_with_unary_child(self) -> None:
        """*(^2(u), x) -> mul(n2(u), x)."""
        inner = _unary("^2", _leaf("u"))
        tree = _tree(_binary("*", inner, _leaf("x")))
        result = tree_to_kd2_expr(tree)
        assert result == "mul(n2(u), x)"

    def test_deeply_nested(self) -> None:
        """/(^3(u), +(x, t)) -> div(n3(u), add(x, t))."""
        cube_u = _unary("^3", _leaf("u"))
        add_xt = _binary("+", _leaf("x"), _leaf("t"))
        tree = _tree(_binary("/", cube_u, add_xt))
        result = tree_to_kd2_expr(tree)
        assert result == "div(n3(u), add(x, t))"

    def test_triple_nesting(self) -> None:
        """^2(*(u, x)) -> n2(mul(u, x))."""
        inner = _binary("*", _leaf("u"), _leaf("x"))
        tree = _tree(_unary("^2", inner))
        result = tree_to_kd2_expr(tree)
        assert result == "n2(mul(u, x))"

    def test_complex_tree(self) -> None:
        """*(+(u, x), -(u_x, t)) -> mul(add(u, x), sub(u_x, t))."""
        left = _binary("+", _leaf("u"), _leaf("x"))
        right = _binary("-", _leaf("u_x"), _leaf("t"))
        tree = _tree(_binary("*", left, right))
        result = tree_to_kd2_expr(tree)
        assert result == "mul(add(u, x), sub(u_x, t))"


# ===========================================================================
# tree_to_kd2_expr: properties (anti-reverse-engineering)
# ===========================================================================


class TestTreeToKd2ExprProperties:
    """Property-based checks that are harder to reverse-engineer."""

    def test_output_is_valid_python_funcall(self) -> None:
        """The output must be a valid Python expression (parseable by compile).
        This replaces the old token-count property since funcall format
        no longer has a trivial token-to-node mapping."""
        tree = _tree(_binary("*", _leaf("u"), _leaf("x")))
        result = tree_to_kd2_expr(tree)
        # Should be parseable as Python
        compile(result, "<test>", "eval")

    def test_output_is_valid_python_funcall_nested(self) -> None:
        """Nested funcall output is valid Python syntax."""
        inner = _unary("^2", _leaf("u"))
        tree = _tree(_binary("/", inner, _leaf("x")))
        result = tree_to_kd2_expr(tree)
        compile(result, "<test>", "eval")

    def test_leaves_appear_in_output(self) -> None:
        """All variable names in the tree must appear in the output string."""
        tree = _tree(
            _binary(
                "*",
                _binary("+", _leaf("u"), _leaf("x")),
                _leaf("t"),
            )
        )
        result = tree_to_kd2_expr(tree)
        for var in ("u", "x", "t"):
            assert var in result

    def test_mapped_operator_not_in_output(self) -> None:
        """Raw SGA operators (+, -, *, /) should NOT appear as function names.
        They should be mapped to add, sub, mul, div."""
        tree = _tree(_binary("+", _binary("*", _leaf("u"), _leaf("x")), _leaf("t")))
        result = tree_to_kd2_expr(tree)
        # Check the funcall names, not individual characters
        # (parentheses, commas etc. contain these chars naturally)
        assert "add(" in result
        assert "mul(" in result

    def test_return_type_is_str(self) -> None:
        tree = _tree(_leaf("u"))
        result = tree_to_kd2_expr(tree)
        assert isinstance(result, str)

    def test_d_operator_not_raw_in_output(self) -> None:
        """Raw 'd' and 'd^2' should not appear as function names.
        They should be mapped to diff_axis and diff2_axis."""
        tree = _tree(_binary("d", _leaf("u"), _leaf("x")))
        result = tree_to_kd2_expr(tree)
        # Should NOT contain a bare "d(" call
        assert result.startswith("diff_")
        assert "d^2" not in result

    def test_d2_operator_maps_to_diff2(self) -> None:
        """d^2 maps to diff2_axis, not d^2 or d2."""
        tree = _tree(_binary("d^2", _leaf("u"), _leaf("x")))
        result = tree_to_kd2_expr(tree)
        assert result.startswith("diff2_")


# ===========================================================================
# pde_to_kd2_expr: without coefficients (funcall format)
# ===========================================================================


class TestPdeToKd2ExprNoCoefficents:
    """pde_to_kd2_expr without coefficients wraps terms in nested add()."""

    @pytest.mark.smoke
    def test_single_term(self) -> None:
        """A single-term PDE should return just the term expression."""
        pde = PDE(terms=[_tree(_binary("*", _leaf("u"), _leaf("x")))])
        result = pde_to_kd2_expr(pde)
        assert result == "mul(u, x)"

    def test_two_terms(self) -> None:
        """Two terms: add(term1, term2)."""
        pde = PDE(
            terms=[
                _tree(_leaf("u")),
                _tree(_leaf("x")),
            ]
        )
        result = pde_to_kd2_expr(pde)
        assert result == "add(u, x)"

    def test_three_terms(self) -> None:
        """Three terms: add(term1, add(term2, term3))."""
        pde = PDE(
            terms=[
                _tree(_leaf("u")),
                _tree(_leaf("x")),
                _tree(_leaf("t")),
            ]
        )
        result = pde_to_kd2_expr(pde)
        assert result == "add(u, add(x, t))"

    def test_four_terms_right_associative(self) -> None:
        """Four terms: add(t1, add(t2, add(t3, t4))) -- right-associative."""
        pde = PDE(
            terms=[
                _tree(_leaf("a")),
                _tree(_leaf("b")),
                _tree(_leaf("c")),
                _tree(_leaf("d")),
            ]
        )
        result = pde_to_kd2_expr(pde)
        assert result == "add(a, add(b, add(c, d)))"

    def test_pde_with_d_operator_term(self) -> None:
        """PDE terms containing d operators use funcall format."""
        pde = PDE(
            terms=[
                _tree(_binary("d", _leaf("u"), _leaf("x"))),
                _tree(_leaf("u")),
            ]
        )
        result = pde_to_kd2_expr(pde)
        assert result == "add(diff_x(u), u)"


# ===========================================================================
# pde_to_kd2_expr: with coefficients (funcall format)
# ===========================================================================


class TestPdeToKd2ExprWithCoefficients:
    """pde_to_kd2_expr with coefficients wraps each term in mul(coeff, term)."""

    def test_single_term_with_coefficient(self) -> None:
        """Single term + coefficient: mul(c, term)."""
        pde = PDE(terms=[_tree(_leaf("u"))])
        result = pde_to_kd2_expr(pde, coefficients=[2.5])
        # Should contain the coefficient value and the variable
        assert "2.5" in result
        assert "u" in result

    def test_two_terms_with_coefficients(self) -> None:
        """Two terms: add(mul(c1, term1), mul(c2, term2))."""
        pde = PDE(terms=[_tree(_leaf("u")), _tree(_leaf("x"))])
        result = pde_to_kd2_expr(pde, coefficients=[1.0, -0.5])
        # Both coefficients and variables must appear
        assert "1.0" in result or "1" in result
        assert "-0.5" in result or "0.5" in result
        assert "u" in result
        assert "x" in result

    def test_coefficient_wraps_each_term(self) -> None:
        """Each term should be wrapped: mul(coeff, term_expr)."""
        pde = PDE(terms=[_tree(_leaf("u"))])
        result = pde_to_kd2_expr(pde, coefficients=[3.0])
        # The result should start with mul( (coefficient wrapping)
        assert result.startswith("mul("), f"Expected 'mul(' prefix, got '{result}'"

    def test_coefficient_format_is_numeric_literal(self) -> None:
        """Coefficients should be numeric literals, not _const suffixed."""
        pde = PDE(terms=[_tree(_leaf("u"))])
        result = pde_to_kd2_expr(pde, coefficients=[0.1])
        # Must NOT contain _const suffix (old format)
        assert "_const" not in result
        # Should contain numeric literal in funcall: mul(0.1, u)
        assert result == "mul(0.1, u)"

    def test_funcall_format_with_two_terms(self) -> None:
        """Verify full funcall format for two terms with coefficients."""
        pde = PDE(terms=[_tree(_leaf("u")), _tree(_leaf("x"))])
        result = pde_to_kd2_expr(pde, coefficients=[1.0, -0.5])
        # Should be funcall: add(mul(1.0, u), mul(-0.5, x))
        assert result == "add(mul(1.0, u), mul(-0.5, x))"


# ===========================================================================
# pde_to_kd2_expr: edge cases
# ===========================================================================


class TestPdeToKd2ExprEdgeCases:
    """Edge cases and error handling for pde_to_kd2_expr."""

    def test_empty_pde_returns_empty_or_raises(self) -> None:
        """An empty PDE (no terms) should return empty string or raise ValueError."""
        pde = PDE(terms=[])
        try:
            result = pde_to_kd2_expr(pde)
            # If it returns, should be empty or some default
            assert isinstance(result, str)
        except (ValueError, IndexError):
            # Raising on empty is also acceptable
            pass

    def test_coefficient_count_mismatch_raises(self) -> None:
        """Mismatched coefficients count should raise an error."""
        pde = PDE(terms=[_tree(_leaf("u")), _tree(_leaf("x"))])
        with pytest.raises((ValueError, IndexError)):
            pde_to_kd2_expr(pde, coefficients=[1.0]) # 1 coeff for 2 terms

    def test_return_type_is_str(self) -> None:
        pde = PDE(terms=[_tree(_leaf("u"))])
        result = pde_to_kd2_expr(pde)
        assert isinstance(result, str)

    def test_complex_terms_with_coefficients(self) -> None:
        """Coefficients work with complex (non-leaf) terms."""
        term1 = _tree(_binary("*", _leaf("u"), _leaf("x")))
        term2 = _tree(_unary("^2", _leaf("t")))
        pde = PDE(terms=[term1, term2])
        result = pde_to_kd2_expr(pde, coefficients=[1.5, -2.0])
        # Result should be a valid non-empty string
        assert len(result) > 0
        # Should contain mul (from both coefficient wrapping and term1)
        assert "mul(" in result

    def test_output_is_valid_python_expression(self) -> None:
        """The pde_to_kd2_expr output must be parseable as Python."""
        pde = PDE(
            terms=[
                _tree(_binary("*", _leaf("u"), _leaf("u_x"))),
                _tree(_leaf("u_xx")),
            ]
        )
        result = pde_to_kd2_expr(pde, coefficients=[-1.0, 0.1])
        compile(result, "<test>", "eval")


# ===========================================================================
# Negative tests (>= 20%)
# ===========================================================================


class TestConverterNegative:
    """Negative and failure-injection tests for converters."""

    def test_tree_to_kd2_expr_preserves_unknown_ops(self) -> None:
        """Unknown operator names should either pass through or raise clearly.
        This tests forward compatibility: if new operators are added to SGA,
        the converter should not silently drop them."""
        tree = _tree(_unary("sin", _leaf("u")))
        try:
            result = tree_to_kd2_expr(tree)
            # If it succeeds, the unknown op should appear in output
            assert "sin" in result
        except (KeyError, ValueError):
            # Raising on unknown op is also acceptable
            pass

    def test_pde_to_kd2_expr_none_coefficients_treated_as_no_coefficients(
        self,
    ) -> None:
        """coefficients=None should behave the same as no coefficients."""
        pde = PDE(terms=[_tree(_leaf("u")), _tree(_leaf("x"))])
        result_none = pde_to_kd2_expr(pde, coefficients=None)
        result_default = pde_to_kd2_expr(pde)
        assert result_none == result_default

    def test_empty_coefficients_list_with_empty_pde(self) -> None:
        """Empty coefficients with empty PDE should not crash."""
        pde = PDE(terms=[])
        try:
            result = pde_to_kd2_expr(pde, coefficients=[])
            assert isinstance(result, str)
        except (ValueError, IndexError):
            pass

    def test_d_operator_with_non_leaf_axis_raises_or_handles(self) -> None:
        """d operator expects a leaf as the axis child.
        If a non-leaf axis is given, should either handle gracefully
        or raise a clear error."""
        # Construct d(u, +(x, t)) -- axis is an expression, not a leaf
        bad_axis = _binary("+", _leaf("x"), _leaf("t"))
        tree = _tree(_binary("d", _leaf("u"), bad_axis))
        try:
            result = tree_to_kd2_expr(tree)
            # If it succeeds, should still produce some string
            assert isinstance(result, str)
        except (ValueError, AttributeError, TypeError):
            # Raising on invalid axis is acceptable
            pass
