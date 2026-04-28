"""Tests for SGA term execution, filtering, and theta construction.

TDD red phase -- tests written against the interface spec before implementation.
Tests use the operator names from config.py (+, -, *, /, ^2, ^3) since
those are what Node.name stores.
"""

from __future__ import annotations

import pytest
import torch
from torch import Tensor

# Will fail until implementation exists
from kd2.search.sga.evaluate import DiffContext, build_theta, execute_pde, execute_tree
from kd2.search.sga.pde import PDE
from kd2.search.sga.tree import Node, Tree

# -- Constants ----------------------------------------------------------------

N_SAMPLES = 50
RTOL = 1e-5
ATOL = 1e-8


# -- Helpers ------------------------------------------------------------------


def _leaf(name: str) -> Node:
    """Shorthand for a leaf node."""
    return Node(name=name, arity=0, children=[])


def _unary(op: str, child: Node) -> Node:
    """Shorthand for a unary operator node."""
    return Node(name=op, arity=1, children=[child])


def _binary(op: str, left: Node, right: Node) -> Node:
    """Shorthand for a binary operator node."""
    return Node(name=op, arity=2, children=[left, right])


def _make_data(seed: int = 42) -> dict[str, Tensor]:
    """Build a reproducible data_dict with keys u, x, t."""
    gen = torch.Generator().manual_seed(seed)
    return {
        "u": torch.randn(N_SAMPLES, generator=gen),
        "x": torch.randn(N_SAMPLES, generator=gen),
        "t": torch.randn(N_SAMPLES, generator=gen),
    }


# ===========================================================================
# Smoke: function existence and basic callability
# ===========================================================================


class TestSmoke:
    """Verify that the public API is importable and callable."""

    @pytest.mark.smoke
    def test_execute_tree_callable(self) -> None:
        tree = Tree(root=_leaf("u"))
        data = {"u": torch.ones(5)}
        result = execute_tree(tree, data)
        assert isinstance(result, Tensor)

    @pytest.mark.smoke
    def test_execute_pde_callable(self) -> None:
        tree = Tree(root=_leaf("u"))
        pde = PDE(terms=[tree])
        data = {"u": torch.ones(5)}
        valid_terms, valid_indices = execute_pde(pde, data)
        assert isinstance(valid_terms, Tensor)
        assert isinstance(valid_indices, list)

    @pytest.mark.smoke
    def test_build_theta_callable(self) -> None:
        vt = torch.randn(10, 3)
        result = build_theta(vt)
        assert isinstance(result, Tensor)


# ===========================================================================
# execute_tree: leaf nodes
# ===========================================================================


class TestExecuteTreeLeaf:
    """Leaf nodes should look up the variable in data_dict and return it."""

    def test_single_leaf_returns_data(self) -> None:
        """execute_tree on a leaf 'u' returns the tensor for 'u'."""
        data = _make_data()
        tree = Tree(root=_leaf("u"))
        result = execute_tree(tree, data)
        torch.testing.assert_close(result, data["u"], rtol=RTOL, atol=ATOL)

    def test_different_variable(self) -> None:
        """Different leaf names select different data columns."""
        data = _make_data()
        tree_x = Tree(root=_leaf("x"))
        result = execute_tree(tree_x, data)
        torch.testing.assert_close(result, data["x"], rtol=RTOL, atol=ATOL)

    def test_missing_variable_raises(self) -> None:
        """Looking up a name not in data_dict should raise KeyError."""
        data = {"u": torch.ones(5)}
        tree = Tree(root=_leaf("nonexistent"))
        with pytest.raises(KeyError):
            execute_tree(tree, data)


# ===========================================================================
# execute_tree: binary operators
# ===========================================================================


class TestExecuteTreeBinaryOps:
    """Binary operators: +, -, *, /."""

    def test_add(self) -> None:
        """+(u, x) -> u + x."""
        data = _make_data()
        tree = Tree(root=_binary("+", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        expected = data["u"] + data["x"]
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_sub(self) -> None:
        """-(u, x) -> u - x."""
        data = _make_data()
        tree = Tree(root=_binary("-", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        expected = data["u"] - data["x"]
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_mul(self) -> None:
        """*(u, x) -> u * x."""
        data = _make_data()
        tree = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        expected = data["u"] * data["x"]
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_div_nonzero_denominator(self) -> None:
        """/(u, x) with nonzero x behaves like u / x (approximately)."""
        data = _make_data()
        # Ensure no exact zeros in denominator for this test
        data["x"] = data["x"].clamp(min=0.1)
        tree = Tree(root=_binary("/", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        expected = data["u"] / data["x"]
        # Slightly larger tolerance since safe_div adds epsilon
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)


# ===========================================================================
# execute_tree: unary operators
# ===========================================================================


class TestExecuteTreeUnaryOps:
    """Unary operators: ^2, ^3."""

    def test_square(self) -> None:
        """^2(u) -> u**2."""
        data = _make_data()
        tree = Tree(root=_unary("^2", _leaf("u")))
        result = execute_tree(tree, data)
        expected = data["u"] ** 2
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_cube(self) -> None:
        """^3(u) -> u**3."""
        data = _make_data()
        tree = Tree(root=_unary("^3", _leaf("u")))
        result = execute_tree(tree, data)
        expected = data["u"] ** 3
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)


# ===========================================================================
# execute_tree: nested / composite expressions
# ===========================================================================


class TestExecuteTreeNested:
    """Complex expression trees should compose correctly."""

    def test_add_of_products(self) -> None:
        """+(*(u, x), ^2(u)) -> u*x + u^2."""
        data = _make_data()
        mul_ux = _binary("*", _leaf("u"), _leaf("x"))
        sq_u = _unary("^2", _leaf("u"))
        root = _binary("+", mul_ux, sq_u)
        tree = Tree(root=root)

        result = execute_tree(tree, data)
        expected = data["u"] * data["x"] + data["u"] ** 2
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_deeply_nested(self) -> None:
        """*(^3(u), -(x, t)) -> u^3 * (x - t)."""
        data = _make_data()
        cube_u = _unary("^3", _leaf("u"))
        sub_xt = _binary("-", _leaf("x"), _leaf("t"))
        root = _binary("*", cube_u, sub_xt)
        tree = Tree(root=root)

        result = execute_tree(tree, data)
        expected = data["u"] ** 3 * (data["x"] - data["t"])
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_chained_unary(self) -> None:
        """^2(^3(u)) -> (u^3)^2 = u^6."""
        data = _make_data()
        # Use small values to avoid overflow
        data["u"] = torch.linspace(-1.0, 1.0, N_SAMPLES)
        cube_u = _unary("^3", _leaf("u"))
        sq_cube = _unary("^2", cube_u)
        tree = Tree(root=sq_cube)

        result = execute_tree(tree, data)
        expected = data["u"] ** 6
        torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)

    def test_div_in_nested_context(self) -> None:
        """+(/(u, x), t) -> safe_div(u, x) + t."""
        data = _make_data()
        data["x"] = data["x"].clamp(min=0.5) # avoid near-zero
        div_ux = _binary("/", _leaf("u"), _leaf("x"))
        root = _binary("+", div_ux, _leaf("t"))
        tree = Tree(root=root)

        result = execute_tree(tree, data)
        expected = data["u"] / data["x"] + data["t"]
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)


# ===========================================================================
# execute_tree: stateless guarantee
# ===========================================================================


class TestExecuteTreeStateless:
    """execute_tree must NOT modify the tree"""

    def test_tree_unchanged_after_execution(self) -> None:
        """Tree string representation is identical before and after execute."""
        data = _make_data()
        root = _binary("*", _unary("^2", _leaf("u")), _leaf("x"))
        tree = Tree(root=root)

        str_before = str(tree)
        _ = execute_tree(tree, data)
        str_after = str(tree)

        assert str_before == str_after

    def test_tree_structurally_equal_after_execution(self) -> None:
        """Tree structure (via __eq__) is unchanged after execution."""
        data = _make_data()
        root = _binary("+", _leaf("u"), _leaf("x"))
        tree = Tree(root=root)
        tree_copy = tree.copy()

        _ = execute_tree(tree, data)
        assert tree == tree_copy


# ===========================================================================
# execute_tree: output shape
# ===========================================================================


class TestExecuteTreeShape:
    """Output should be 1D with n_samples elements."""

    def test_output_is_1d(self) -> None:
        data = _make_data()
        tree = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        assert result.ndim == 1

    def test_output_length_matches_input(self) -> None:
        data = _make_data()
        tree = Tree(root=_leaf("u"))
        result = execute_tree(tree, data)
        assert result.shape[0] == N_SAMPLES


# ===========================================================================
# execute_pde: basic filtering
# ===========================================================================


class TestExecutePdeFiltering:
    """execute_pde should filter out NaN, Inf, and all-zero columns."""

    def test_all_valid_terms_kept(self) -> None:
        """When all terms are valid, all indices are returned."""
        data = _make_data()
        t1 = Tree(root=_leaf("u"))
        t2 = Tree(root=_leaf("x"))
        t3 = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        pde = PDE(terms=[t1, t2, t3])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 3
        assert valid_indices == [0, 1, 2]

    def test_nan_column_filtered(self) -> None:
        """A term that produces NaN should be excluded."""
        # Division by zero can produce NaN/Inf; alternatively craft data with NaN
        data = _make_data()
        data["z"] = torch.full((N_SAMPLES,), float("nan"))
        t_valid = Tree(root=_leaf("u"))
        t_nan = Tree(root=_leaf("z"))
        pde = PDE(terms=[t_valid, t_nan])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 1
        assert 0 in valid_indices
        assert 1 not in valid_indices

    def test_inf_column_filtered(self) -> None:
        """A term that produces Inf should be excluded."""
        data = _make_data()
        data["inf_var"] = torch.full((N_SAMPLES,), float("inf"))
        t_valid = Tree(root=_leaf("u"))
        t_inf = Tree(root=_leaf("inf_var"))
        pde = PDE(terms=[t_valid, t_inf])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 1
        assert 0 in valid_indices
        assert 1 not in valid_indices

    def test_zero_column_filtered(self) -> None:
        """A term that is all zeros should be excluded."""
        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        t_valid = Tree(root=_leaf("u"))
        t_zero = Tree(root=_leaf("zeros"))
        pde = PDE(terms=[t_valid, t_zero])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 1
        assert 0 in valid_indices
        assert 1 not in valid_indices

    def test_valid_indices_map_back_correctly(self) -> None:
        """valid_indices should be indices into the original pde.terms list."""
        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        # terms: [u, zeros, x, zeros, *(u,x)]
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
                Tree(root=_leaf("x")),
                Tree(root=_leaf("zeros")),
                Tree(root=_binary("*", _leaf("u"), _leaf("x"))),
            ]
        )

        valid_terms, valid_indices = execute_pde(pde, data)
        # Expect indices 0, 2, 4 to survive
        assert valid_indices == [0, 2, 4]
        assert valid_terms.shape[1] == 3


# ===========================================================================
# execute_pde: output shape and dtype
# ===========================================================================


class TestExecutePdeShape:
    """Verify shape contract of execute_pde output."""

    def test_output_shape(self) -> None:
        """valid_terms has shape (n_samples, n_valid_terms)."""
        data = _make_data()
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("x")),
            ]
        )
        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape == (N_SAMPLES, 2)

    def test_output_is_2d(self) -> None:
        data = _make_data()
        pde = PDE(terms=[Tree(root=_leaf("u"))])
        valid_terms, _ = execute_pde(pde, data)
        assert valid_terms.ndim == 2


# ===========================================================================
# execute_pde: edge cases
# ===========================================================================


class TestExecutePdeEdgeCases:
    """Edge cases: all filtered, single term, etc."""

    def test_all_terms_filtered_returns_empty(self) -> None:
        """When every term is invalid, return empty tensor + empty indices."""
        data = _make_data()
        data["nan_var"] = torch.full((N_SAMPLES,), float("nan"))
        data["inf_var"] = torch.full((N_SAMPLES,), float("inf"))
        pde = PDE(
            terms=[
                Tree(root=_leaf("nan_var")),
                Tree(root=_leaf("inf_var")),
            ]
        )

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 0
        assert valid_indices == []

    def test_single_valid_term(self) -> None:
        """A PDE with one valid term should return that term."""
        data = _make_data()
        pde = PDE(terms=[Tree(root=_leaf("u"))])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape == (N_SAMPLES, 1)
        assert valid_indices == [0]
        torch.testing.assert_close(valid_terms[:, 0], data["u"], rtol=RTOL, atol=ATOL)

    def test_empty_pde(self) -> None:
        """A PDE with no terms at all."""
        data = _make_data()
        pde = PDE(terms=[])

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_terms.shape[1] == 0
        assert valid_indices == []

    def test_pde_not_modified(self) -> None:
        """execute_pde must NOT modify the PDE object itself."""
        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
            ]
        )
        original_width = pde.width
        original_str = str(pde)

        _ = execute_pde(pde, data)

        assert pde.width == original_width
        assert str(pde) == original_str


# ===========================================================================
# execute_pde: division-induced filtering
# ===========================================================================


class TestExecutePdeDivisionGuard:
    """Division by zero should either be handled by safe_div or cause filtering."""

    def test_div_by_zero_does_not_crash(self) -> None:
        """/(u, zeros) must not raise an exception."""
        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        tree = Tree(root=_binary("/", _leaf("u"), _leaf("zeros")))
        pde = PDE(terms=[tree])

        # Should not raise. Result may be filtered or safe_div-protected.
        valid_terms, valid_indices = execute_pde(pde, data)
        # If safe_div is used: term survives with finite values
        # If guard-filtered: term is removed
        if len(valid_indices) > 0:
            # safe_div path: result should be finite
            assert torch.isfinite(valid_terms).all()

    def test_div_by_near_zero_result_is_finite(self) -> None:
        """Division by near-zero values should produce finite output via safe_div."""
        data = _make_data()
        data["small"] = torch.full((N_SAMPLES,), 1e-15)
        tree = Tree(root=_binary("/", _leaf("u"), _leaf("small")))

        result = execute_tree(tree, data)
        # safe_div should ensure the result is finite
        assert torch.isfinite(result).all()


# ===========================================================================
# build_theta: concatenation
# ===========================================================================


class TestBuildTheta:
    """build_theta concatenates default_terms (prepend) + valid_terms."""

    def test_no_defaults(self) -> None:
        """Without default_terms, return valid_terms as-is."""
        vt = torch.randn(N_SAMPLES, 3)
        result = build_theta(vt)
        torch.testing.assert_close(result, vt, rtol=RTOL, atol=ATOL)

    def test_no_defaults_explicit_none(self) -> None:
        """Passing None explicitly should behave identically."""
        vt = torch.randn(N_SAMPLES, 3)
        result = build_theta(vt, default_terms=None)
        torch.testing.assert_close(result, vt, rtol=RTOL, atol=ATOL)

    def test_with_defaults_prepended(self) -> None:
        """default_terms should appear as the leftmost columns."""
        gen = torch.Generator().manual_seed(99)
        defaults = torch.randn(N_SAMPLES, 2, generator=gen)
        valid = torch.randn(N_SAMPLES, 3, generator=gen)

        result = build_theta(valid, default_terms=defaults)

        assert result.shape == (N_SAMPLES, 5)
        # First 2 columns are from defaults
        torch.testing.assert_close(result[:, :2], defaults, rtol=RTOL, atol=ATOL)
        # Last 3 columns are from valid_terms
        torch.testing.assert_close(result[:, 2:], valid, rtol=RTOL, atol=ATOL)

    def test_output_shape_correct(self) -> None:
        """Shape should be (n_samples, n_defaults + n_valid)."""
        n_defaults = 4
        n_valid = 7
        defaults = torch.randn(N_SAMPLES, n_defaults)
        valid = torch.randn(N_SAMPLES, n_valid)

        result = build_theta(valid, default_terms=defaults)
        assert result.shape == (N_SAMPLES, n_defaults + n_valid)

    def test_with_empty_valid_terms(self) -> None:
        """If valid_terms has 0 columns, result is just defaults."""
        defaults = torch.randn(N_SAMPLES, 3)
        empty_valid = torch.empty(N_SAMPLES, 0)

        result = build_theta(empty_valid, default_terms=defaults)
        assert result.shape == (N_SAMPLES, 3)
        torch.testing.assert_close(result, defaults, rtol=RTOL, atol=ATOL)


# ===========================================================================
# execute_pde: runtime-error containment (3a2/H6)
# ===========================================================================


class TestExecutePdeRuntimeErrorContainment:
    """Torch RuntimeError from one term must NOT crash the whole evaluation.

    Regression for 3a2/H6: ``execute_pde`` previously caught only
    ``(KeyError, ValueError)``. A mutated tree that triggers a torch
    op-level RuntimeError (shape mismatch, dtype, OOM) would propagate
    out and abort the entire SGA generation.
    """

    def test_runtime_error_term_skipped_keeps_siblings(self) -> None:
        """A term that raises RuntimeError is dropped; valid siblings survive."""
        # Mismatched-shape data triggers torch.add RuntimeError when combined.
        data: dict[str, Tensor] = {
            "u": torch.randn(N_SAMPLES),
            "x": torch.randn(N_SAMPLES),
            # Intentionally different size to force torch.add shape mismatch.
            "v_short": torch.randn(N_SAMPLES // 2),
        }
        good = Tree(root=_leaf("u"))
        bad = Tree(root=_binary("+", _leaf("u"), _leaf("v_short")))
        also_good = Tree(root=_leaf("x"))
        pde = PDE(terms=[good, bad, also_good])

        valid_terms, valid_indices = execute_pde(pde, data)

        # Bad term skipped; the two valid siblings retained with original indices.
        assert valid_indices == [0, 2]
        assert valid_terms.shape == (N_SAMPLES, 2)

    def test_runtime_error_only_returns_empty(self) -> None:
        """If every term raises RuntimeError, return empty result without raising."""
        data: dict[str, Tensor] = {
            "u": torch.randn(N_SAMPLES),
            "v_short": torch.randn(N_SAMPLES // 2),
        }
        bad1 = Tree(root=_binary("+", _leaf("u"), _leaf("v_short")))
        bad2 = Tree(root=_binary("*", _leaf("u"), _leaf("v_short")))
        pde = PDE(terms=[bad1, bad2])

        valid_terms, valid_indices = execute_pde(pde, data)

        assert valid_indices == []
        assert valid_terms.shape[1] == 0


# ===========================================================================
# Numerical: NaN/Inf injection and extreme values
# ===========================================================================


class TestNumericalStability:
    """Numerical edge cases: NaN, Inf, extreme values."""

    @pytest.mark.numerical
    def test_nan_input_in_data_dict(self) -> None:
        """If data_dict contains NaN, execute_tree should not crash."""
        data = {"u": torch.tensor([float("nan"), 1.0, 2.0])}
        tree = Tree(root=_leaf("u"))
        result = execute_tree(tree, data)
        # NaN passes through leaf lookup
        assert result.shape[0] == 3

    @pytest.mark.numerical
    def test_inf_input_in_data_dict(self) -> None:
        """If data_dict contains Inf, execute_tree should not crash."""
        data = {"u": torch.tensor([float("inf"), 1.0, 2.0])}
        tree = Tree(root=_unary("^2", _leaf("u")))
        result = execute_tree(tree, data)
        assert result.shape[0] == 3

    @pytest.mark.numerical
    def test_large_values_no_crash(self) -> None:
        """Very large input values should not cause exceptions."""
        data = {"u": torch.full((10,), 1e30)}
        tree = Tree(root=_unary("^2", _leaf("u")))
        # ^2 of 1e30 = 1e60 which is finite in float64 but Inf in float32
        result = execute_tree(tree, data)
        assert result.shape[0] == 10

    @pytest.mark.numerical
    def test_execute_pde_mixed_nan_inf_valid(self) -> None:
        """PDE with a mix of NaN, Inf, and valid terms filters correctly."""
        n = 20
        data = {
            "good": torch.randn(n),
            "nan_data": torch.full((n,), float("nan")),
            "inf_data": torch.full((n,), float("inf")),
        }
        pde = PDE(
            terms=[
                Tree(root=_leaf("good")),
                Tree(root=_leaf("nan_data")),
                Tree(root=_leaf("inf_data")),
            ]
        )
        valid_terms, valid_indices = execute_pde(pde, data)
        # Only the good term should survive
        assert valid_indices == [0]
        assert valid_terms.shape == (n, 1)
        assert torch.isfinite(valid_terms).all()


# ===========================================================================
# Property-based: algebraic identities
# ===========================================================================


class TestAlgebraicProperties:
    """Algebraic properties that must hold regardless of input data."""

    def test_mul_commutativity_in_result(self) -> None:
        """*(u, x) and *(x, u) should produce the same numerical result."""
        data = _make_data()
        tree_ux = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        tree_xu = Tree(root=_binary("*", _leaf("x"), _leaf("u")))

        r_ux = execute_tree(tree_ux, data)
        r_xu = execute_tree(tree_xu, data)
        torch.testing.assert_close(r_ux, r_xu, rtol=RTOL, atol=ATOL)

    def test_add_commutativity_in_result(self) -> None:
        """+(u, x) and +(x, u) should produce the same numerical result."""
        data = _make_data()
        tree_ux = Tree(root=_binary("+", _leaf("u"), _leaf("x")))
        tree_xu = Tree(root=_binary("+", _leaf("x"), _leaf("u")))

        r_ux = execute_tree(tree_ux, data)
        r_xu = execute_tree(tree_xu, data)
        torch.testing.assert_close(r_ux, r_xu, rtol=RTOL, atol=ATOL)

    def test_sub_anticommutativity(self) -> None:
        """-(u, x) = -(-(x, u)) pointwise."""
        data = _make_data()
        tree_ux = Tree(root=_binary("-", _leaf("u"), _leaf("x")))
        tree_xu = Tree(root=_binary("-", _leaf("x"), _leaf("u")))

        r_ux = execute_tree(tree_ux, data)
        r_xu = execute_tree(tree_xu, data)
        torch.testing.assert_close(r_ux, -r_xu, rtol=RTOL, atol=ATOL)

    def test_square_is_nonnegative(self) -> None:
        """^2(u) should always be >= 0."""
        data = _make_data()
        tree = Tree(root=_unary("^2", _leaf("u")))
        result = execute_tree(tree, data)
        assert (result >= 0).all()

    def test_cube_preserves_sign(self) -> None:
        """^3(u) should have the same sign as u (where u != 0)."""
        data = _make_data()
        # Avoid exact zeros
        data["u"] = data["u"].clamp(min=0.01)
        tree = Tree(root=_unary("^3", _leaf("u")))
        result = execute_tree(tree, data)
        assert (result > 0).all() # all positive since input > 0

    def test_identity_via_div_self(self) -> None:
        """/(u, u) should be approximately 1 (via safe_div when u != 0)."""
        data = _make_data()
        # Ensure no zeros
        data["u"] = data["u"].abs().clamp(min=0.1)
        tree = Tree(root=_binary("/", _leaf("u"), _leaf("u")))
        result = execute_tree(tree, data)
        expected = torch.ones_like(result)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)


# ===========================================================================
# Operator coverage: all 6 dispatch entries
# ===========================================================================


class TestAllOperatorsCovered:
    """Ensure every operator in the dispatch table is exercised."""

    @pytest.mark.parametrize("op", ["+", "-", "*", "/"])
    def test_binary_op_executes(self, op: str) -> None:
        """Binary op '{op}' should be executable without error."""
        data = _make_data()
        data["u"] = data["u"].abs().clamp(min=0.1)
        data["x"] = data["x"].abs().clamp(min=0.1)
        tree = Tree(root=_binary(op, _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data)
        assert result.shape == (N_SAMPLES,)
        assert torch.isfinite(result).all()

    @pytest.mark.parametrize("op", ["^2", "^3"])
    def test_unary_op_executes(self, op: str) -> None:
        """Unary op '{op}' should be executable without error."""
        data = _make_data()
        tree = Tree(root=_unary(op, _leaf("u")))
        result = execute_tree(tree, data)
        assert result.shape == (N_SAMPLES,)
        assert torch.isfinite(result).all()


# ===========================================================================
# Negative tests: error handling
# ===========================================================================


class TestErrorHandling:
    """Invalid inputs should produce clear errors or safe behavior."""

    def test_unknown_operator_raises(self) -> None:
        """An operator not in the dispatch table should raise an error."""
        data = _make_data()
        tree = Tree(root=_unary("sin", _leaf("u")))
        with pytest.raises((KeyError, ValueError)):
            execute_tree(tree, data)

    def test_unknown_binary_operator_raises(self) -> None:
        """An unknown binary operator should raise an error."""
        data = _make_data()
        tree = Tree(root=_binary("mod", _leaf("u"), _leaf("x")))
        with pytest.raises((KeyError, ValueError)):
            execute_tree(tree, data)

    def test_arity_mismatch_unary_with_two_children(self) -> None:
        """A node marked arity=1 but having 2 children is structurally invalid.

        The executor may raise or may only look at children[0]; either way
        this tests that it does not silently produce wrong results.
        """
        data = _make_data()
        # Arity 1 but 2 children -- structurally inconsistent
        bad_node = Node(name="^2", arity=1, children=[_leaf("u"), _leaf("x")])
        tree = Tree(root=bad_node)
        # We allow any of: raises, or returns u^2 (ignoring extra child)
        # But result must not silently depend on the extra child
        try:
            result = execute_tree(tree, data)
            # If no error, verify it's just u^2 (ignoring extra child)
            expected = data["u"] ** 2
            torch.testing.assert_close(result, expected, rtol=RTOL, atol=ATOL)
        except (IndexError, ValueError, TypeError):
            pass # acceptable to raise

    def test_build_theta_mismatched_rows_raises(self) -> None:
        """default_terms and valid_terms with different n_samples should fail."""
        # First verify build_theta works at all with valid input
        ok_result = build_theta(torch.randn(10, 2), default_terms=torch.randn(10, 1))
        assert ok_result.shape == (10, 3)

        # Now test the mismatch case
        defaults = torch.randn(10, 2)
        valid = torch.randn(20, 3)
        with pytest.raises((RuntimeError, ValueError)):
            build_theta(valid, default_terms=defaults)


# ===========================================================================
# Near-zero column threshold
# ===========================================================================


class TestZeroColumnThreshold:
    """Near-zero columns (norm < eps) should be filtered, not just exact zeros."""

    def test_near_zero_column_filtered(self) -> None:
        """A column with very small but nonzero values should be filtered."""
        data = _make_data()
        data["tiny"] = torch.full((N_SAMPLES,), 1e-15)
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("tiny")),
            ]
        )

        valid_terms, valid_indices = execute_pde(pde, data)
        # The tiny column should be treated as effectively zero
        assert 1 not in valid_indices
        assert 0 in valid_indices

    def test_small_but_nonzero_column_kept(self) -> None:
        """A column with small but meaningful values should survive."""
        data = _make_data()
        data["small_meaningful"] = torch.full((N_SAMPLES,), 0.01)
        pde = PDE(
            terms=[
                Tree(root=_leaf("small_meaningful")),
            ]
        )

        valid_terms, valid_indices = execute_pde(pde, data)
        assert valid_indices == [0]


# ===========================================================================
# Derivative operator execution: d(expr, axis), d^2(expr, axis)
# ===========================================================================


def _make_grid_data(
    nx: int = 50,
    dx: float = 0.1,
) -> dict[str, Tensor]:
    """Build a 1D spatial grid data_dict with known derivatives.

    u = sin(x), so d(u, x) = cos(x), d^2(u, x) = -sin(x).
    Also provides the grid spacing via 'dx' metadata key.
    """
    x = torch.linspace(0.0, (nx - 1) * dx, nx)
    u = torch.sin(x)
    return {
        "u": u,
        "x": x,
    }


def _make_diff_ctx(
    nx: int = 50,
    dx: float = 0.1,
    lhs_axis: str | None = None,
) -> DiffContext:
    """Build a 1D DiffContext for derivative execution."""
    delta = {"x": dx}
    if lhs_axis is not None:
        delta[lhs_axis] = 0.02
    return DiffContext(
        field_shape=(nx,),
        axis_map={"x": 0},
        delta=delta,
        lhs_axis=lhs_axis,
    )


class TestDerivativeExecution:
    """execute_tree must support d(expr, axis) and d^2(expr, axis)."""

    def test_d_dispatch_exists(self) -> None:
        """'d' must be in DISPATCH or handled by execute_tree."""

        # d must be handled (either in DISPATCH or special-cased)
        # We test by constructing a simple d(u, x) tree and executing it
        data = _make_grid_data()
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0])
        d_node = _binary("d", _leaf("u"), _leaf("x"))
        tree = Tree(root=d_node)
        # Should not raise ValueError("Unknown operator 'd'")
        result = execute_tree(tree, data, diff_ctx=diff_ctx)
        assert isinstance(result, Tensor)

    def test_d2_dispatch_exists(self) -> None:
        """'d^2' must be handled by execute_tree."""
        data = _make_grid_data()
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0])
        d2_node = _binary("d^2", _leaf("u"), _leaf("x"))
        tree = Tree(root=d2_node)
        result = execute_tree(tree, data, diff_ctx=diff_ctx)
        assert isinstance(result, Tensor)

    def test_d_simple_leaf(self) -> None:
        """d(u, x) should compute du/dx via finite differences."""
        nx = 50
        dx = 0.1
        data = _make_grid_data(nx=nx, dx=dx)
        diff_ctx = _make_diff_ctx(nx=nx, dx=dx)

        tree = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data, diff_ctx=diff_ctx)

        # u = sin(x) -> du/dx = cos(x)
        # Finite differences have error O(dx^2) ~ 0.01
        x = data["x"]
        expected = torch.cos(x)
        # Use generous tolerance for finite difference approximation
        torch.testing.assert_close(result, expected, rtol=0.05, atol=0.02)

    def test_d2_simple_leaf(self) -> None:
        """d^2(u, x) should compute d^2u/dx^2 via finite differences."""
        nx = 50
        dx = 0.1
        data = _make_grid_data(nx=nx, dx=dx)
        diff_ctx = _make_diff_ctx(nx=nx, dx=dx)

        tree = Tree(root=_binary("d^2", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data, diff_ctx=diff_ctx)

        # u = sin(x) -> d^2u/dx^2 = -sin(x)
        x = data["x"]
        expected = -torch.sin(x)
        # Finite differences have larger error for second derivative
        torch.testing.assert_close(result, expected, rtol=0.1, atol=0.05)

    def test_d_composite_expr(self) -> None:
        """d(u*u, x) = d(u^2, x) should compute d(u^2)/dx via FD.

        For u = sin(x): d(sin^2(x))/dx = 2*sin(x)*cos(x) = sin(2x).
        """
        nx = 50
        dx = 0.1
        data = _make_grid_data(nx=nx, dx=dx)
        diff_ctx = _make_diff_ctx(nx=nx, dx=dx)

        # d(*(u, u), x)
        mul_node = _binary("*", _leaf("u"), _leaf("u"))
        d_node = _binary("d", mul_node, _leaf("x"))
        tree = Tree(root=d_node)
        result = execute_tree(tree, data, diff_ctx=diff_ctx)

        x = data["x"]
        expected = torch.sin(2.0 * x) # 2*sin(x)*cos(x) = sin(2x)
        torch.testing.assert_close(result, expected, rtol=0.1, atol=0.05)

    def test_d_nested_derivative(self) -> None:
        """d(d(u, x), x) = d^2u/dx^2. This tests nested derivative execution.

        For u = sin(x): d^2u/dx^2 = -sin(x).
        """
        nx = 100
        dx = 0.05
        data = _make_grid_data(nx=nx, dx=dx)
        diff_ctx = _make_diff_ctx(nx=nx, dx=dx)

        # d(d(u, x), x)
        inner_d = _binary("d", _leaf("u"), _leaf("x"))
        outer_d = _binary("d", inner_d, _leaf("x"))
        tree = Tree(root=outer_d)
        result = execute_tree(tree, data, diff_ctx=diff_ctx)

        x = data["x"]
        expected = -torch.sin(x)
        # Nested FD has more error; be generous
        torch.testing.assert_close(result, expected, rtol=0.2, atol=0.1)

    def test_d_output_shape(self) -> None:
        """d(u, x) output must be 1D with same length as input."""
        data = _make_grid_data()
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0])
        tree = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        result = execute_tree(tree, data, diff_ctx=diff_ctx)
        assert result.ndim == 1
        assert result.shape[0] == data["u"].shape[0]

    def test_d_stateless(self) -> None:
        """execute_tree with d/d^2 must not modify the tree."""
        data = _make_grid_data()
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0])
        tree = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        str_before = str(tree)
        _ = execute_tree(tree, data, diff_ctx=diff_ctx)
        assert str(tree) == str_before


class TestDerivativeLHSRejection:
    """Terms with d(expr, lhs_axis) must be rejected in execute_pde."""

    def test_d_along_lhs_axis_is_filtered(self) -> None:
        """d(u, t) where t is lhs_axis should be filtered out by execute_pde."""
        data = _make_grid_data()
        data["t"] = torch.linspace(0.0, 1.0, data["u"].shape[0])
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0], lhs_axis="t")

        tree_ok = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        tree_bad = Tree(root=_binary("d", _leaf("u"), _leaf("t")))
        pde = PDE(terms=[tree_ok, tree_bad])

        valid_terms, valid_indices = execute_pde(pde, data, diff_ctx=diff_ctx)
        # Only the d(u, x) term should survive
        assert 0 in valid_indices
        assert 1 not in valid_indices

    def test_d2_along_lhs_axis_is_filtered(self) -> None:
        """d^2(u, t) where t is lhs_axis should be filtered out."""
        data = _make_grid_data()
        data["t"] = torch.linspace(0.0, 1.0, data["u"].shape[0])
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0], lhs_axis="t")

        tree_ok = Tree(root=_leaf("u"))
        tree_bad = Tree(root=_binary("d^2", _leaf("u"), _leaf("t")))
        pde = PDE(terms=[tree_ok, tree_bad])

        valid_terms, valid_indices = execute_pde(pde, data, diff_ctx=diff_ctx)
        assert 0 in valid_indices
        assert 1 not in valid_indices


# ===========================================================================
# T4: Genotype pruning — prune_invalid_terms syncs PDE.terms
# ===========================================================================


class TestPruneInvalidTerms:
    """prune_invalid_terms should remove invalid terms from PDE.terms,
    keeping PDE genotype in sync with the valid theta columns."""

    def test_prune_removes_nan_terms(self) -> None:
        """After pruning, PDE.terms should not contain the NaN term."""
        from kd2.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["z"] = torch.full((N_SAMPLES,), float("nan"))
        t_valid = Tree(root=_leaf("u"))
        t_nan = Tree(root=_leaf("z"))
        pde = PDE(terms=[t_valid, t_nan])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 1
        assert str(pruned_pde.terms[0]) == str(t_valid)
        assert valid_terms.shape[1] == 1
        assert valid_indices == [0]

    def test_prune_removes_zero_terms(self) -> None:
        """After pruning, all-zero terms should be removed from PDE.terms."""
        from kd2.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        t_valid = Tree(root=_leaf("u"))
        t_zero = Tree(root=_leaf("zeros"))
        pde = PDE(terms=[t_valid, t_zero])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 1
        assert str(pruned_pde.terms[0]) == str(t_valid)

    def test_prune_removes_inf_terms(self) -> None:
        """After pruning, Inf terms should be removed from PDE.terms."""
        from kd2.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["inf_var"] = torch.full((N_SAMPLES,), float("inf"))
        t_valid = Tree(root=_leaf("u"))
        t_inf = Tree(root=_leaf("inf_var"))
        pde = PDE(terms=[t_valid, t_inf])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 1
        assert str(pruned_pde.terms[0]) == str(t_valid)

    def test_prune_preserves_valid_term_order(self) -> None:
        """Surviving terms should maintain their original relative order."""
        from kd2.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        # terms: [u, zeros, x, zeros, *(u,x)]
        t_u = Tree(root=_leaf("u"))
        t_z1 = Tree(root=_leaf("zeros"))
        t_x = Tree(root=_leaf("x"))
        t_z2 = Tree(root=_leaf("zeros"))
        t_ux = Tree(root=_binary("*", _leaf("u"), _leaf("x")))
        pde = PDE(terms=[t_u, t_z1, t_x, t_z2, t_ux])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 3
        assert valid_indices == [0, 2, 4]
        assert str(pruned_pde.terms[0]) == str(t_u)
        assert str(pruned_pde.terms[1]) == str(t_x)
        assert str(pruned_pde.terms[2]) == str(t_ux)

    def test_prune_returns_aligned_theta(self) -> None:
        """valid_terms columns should correspond 1:1 to pruned_pde.terms."""
        from kd2.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
                Tree(root=_leaf("x")),
            ]
        )

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        # Column 0 of valid_terms should match pruned_pde.terms[0] = "u"
        assert valid_terms.shape[1] == pruned_pde.width
        torch.testing.assert_close(valid_terms[:, 0], data["u"], rtol=RTOL, atol=ATOL)
        torch.testing.assert_close(valid_terms[:, 1], data["x"], rtol=RTOL, atol=ATOL)

    def test_prune_does_not_mutate_original(self) -> None:
        """prune_invalid_terms should return a new PDE, not modify the input."""
        from kd2.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["zeros"] = torch.zeros(N_SAMPLES)
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
            ]
        )
        original_width = pde.width
        original_str = str(pde)

        pruned_pde, _, _ = prune_invalid_terms(pde, data)

        # Original should be unchanged
        assert pde.width == original_width
        assert str(pde) == original_str
        # Pruned should be different
        assert pruned_pde.width == 1

    def test_prune_all_invalid_returns_empty(self) -> None:
        """When all terms are invalid, pruned PDE should be empty."""
        from kd2.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        data["nan_var"] = torch.full((N_SAMPLES,), float("nan"))
        data["inf_var"] = torch.full((N_SAMPLES,), float("inf"))
        pde = PDE(
            terms=[
                Tree(root=_leaf("nan_var")),
                Tree(root=_leaf("inf_var")),
            ]
        )

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 0
        assert valid_terms.shape[1] == 0
        assert valid_indices == []

    def test_prune_empty_pde(self) -> None:
        """Pruning an empty PDE should return an empty PDE."""
        from kd2.search.sga.evaluate import prune_invalid_terms

        data = _make_data()
        pde = PDE(terms=[])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(pde, data)
        assert pruned_pde.width == 0
        assert valid_indices == []

    def test_prune_with_lhs_axis_filtering(self) -> None:
        """lhs_axis-derivative terms should be pruned from the genotype too."""
        from kd2.search.sga.evaluate import prune_invalid_terms

        data = _make_grid_data()
        data["t"] = torch.linspace(0.0, 1.0, data["u"].shape[0])
        diff_ctx = _make_diff_ctx(nx=data["u"].shape[0], lhs_axis="t")

        tree_ok = Tree(root=_binary("d", _leaf("u"), _leaf("x")))
        tree_bad = Tree(root=_binary("d", _leaf("u"), _leaf("t")))
        pde = PDE(terms=[tree_ok, tree_bad])

        pruned_pde, valid_terms, valid_indices = prune_invalid_terms(
            pde, data, diff_ctx=diff_ctx
        )
        assert pruned_pde.width == 1
        assert str(pruned_pde.terms[0]) == str(tree_ok)
        assert valid_indices == [0]
