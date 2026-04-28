"""Tests for SGAConfig and operator pool constants.

TDD red phase -- tests written against the interface spec before implementation.
"""

from __future__ import annotations

import pytest

from kd2.search.sga.config import SGAConfig

# ===========================================================================
# SGAConfig defaults
# ===========================================================================


class TestSGAConfigDefaults:
    """Default values must match the D2 parameter specification."""

    @pytest.mark.smoke
    def test_config_is_instantiable(self) -> None:
        """SGAConfig() should be callable with no arguments."""
        config = SGAConfig()
        assert config is not None

    def test_ga_defaults(self) -> None:
        config = SGAConfig()
        assert config.num == 20
        assert config.p_var == 0.5
        assert config.p_mute == 0.3
        assert config.p_cro == 0.5
        assert config.p_rep == 1.0
        assert config.seed == 0

    def test_tree_structure_defaults(self) -> None:
        config = SGAConfig()
        assert config.depth == 4
        assert config.width == 5

    def test_evaluation_defaults(self) -> None:
        config = SGAConfig()
        assert config.aic_ratio == 1.0
        assert config.lam == 0.0
        assert config.d_tol == 1.0
        assert config.maxit == 10
        assert config.str_iters == 10
        assert config.normalize == 2


# ===========================================================================
# SGAConfig does NOT contain generations
# ===========================================================================


class TestSGAConfigNoGenerations:
    """generations is a Runner concern, not a config concern"""

    def test_no_generations_field(self) -> None:
        config = SGAConfig()
        assert not hasattr(config, "generations")

    def test_no_sga_run_field(self) -> None:
        """sga_run was the predecessor name for generations -- should not exist."""
        config = SGAConfig()
        assert not hasattr(config, "sga_run")


# ===========================================================================
# SGAConfig custom overrides
# ===========================================================================


class TestSGAConfigCustom:
    """SGAConfig fields can be overridden at construction."""

    def test_override_single_field(self) -> None:
        config = SGAConfig(num=50)
        assert config.num == 50

    def test_override_multiple_fields(self) -> None:
        config = SGAConfig(depth=6, width=8, p_mute=0.5)
        assert config.depth == 6
        assert config.width == 8
        assert config.p_mute == 0.5

    def test_other_defaults_preserved_after_override(self) -> None:
        """Overriding one field should not change others."""
        config = SGAConfig(num=100)
        assert config.p_var == 0.5
        assert config.depth == 4
        assert config.seed == 0


# ===========================================================================
# SGAConfig is a dataclass
# ===========================================================================


class TestSGAConfigDataclass:
    """SGAConfig should be a proper dataclass."""

    def test_has_dataclass_fields(self) -> None:
        """Should have __dataclass_fields__ attribute."""
        import dataclasses

        assert dataclasses.is_dataclass(SGAConfig)

    def test_equality(self) -> None:
        """Two configs with same values should be equal."""
        a = SGAConfig()
        b = SGAConfig()
        assert a == b

    def test_inequality(self) -> None:
        a = SGAConfig(num=10)
        b = SGAConfig(num=20)
        assert a != b


# ===========================================================================
# Operator pool constants
# ===========================================================================


class TestOperatorPools:
    """Operator pool constants define valid SGA symbols."""

    @pytest.mark.smoke
    def test_ops_pool_importable(self) -> None:
        from kd2.search.sga.config import OPS

        assert isinstance(OPS, (list, tuple))
        assert len(OPS) > 0

    def test_ops_contains_all_operators(self) -> None:
        """OPS should have: +, -, *, /, ^2, ^3, d, d^2."""
        from kd2.search.sga.config import OPS

        op_names = {name for name, _ in OPS}
        expected = {"+", "-", "*", "/", "^2", "^3", "d", "d^2"}
        assert expected == op_names

    def test_ops_arities_correct(self) -> None:
        """Binary ops have arity 2, unary ops have arity 1."""
        from kd2.search.sga.config import OPS

        op_dict = {name: arity for name, arity in OPS}
        assert op_dict["+"] == 2
        assert op_dict["-"] == 2
        assert op_dict["*"] == 2
        assert op_dict["/"] == 2
        assert op_dict["^2"] == 1
        assert op_dict["^3"] == 1
        assert op_dict["d"] == 2
        assert op_dict["d^2"] == 2

    def test_root_excludes_plus_minus(self) -> None:
        """ROOT pool should not contain + or - (tree root is never additive)."""
        from kd2.search.sga.config import ROOT

        root_names = {name for name, _ in ROOT}
        assert "+" not in root_names
        assert "-" not in root_names

    def test_root_contains_mul_div_powers_and_deriv(self) -> None:
        """ROOT should contain *, /, ^2, ^3, d, d^2."""
        from kd2.search.sga.config import ROOT

        root_names = {name for name, _ in ROOT}
        assert {"*", "/", "^2", "^3", "d", "d^2"}.issubset(root_names)

    def test_op1_only_unary(self) -> None:
        """OP1 should only contain unary operators (arity 1)."""
        from kd2.search.sga.config import OP1

        for name, arity in OP1:
            assert arity == 1, f"{name} has arity {arity}, expected 1"

    def test_op1_contents(self) -> None:
        """OP1 = ^2, ^3."""
        from kd2.search.sga.config import OP1

        op1_names = {name for name, _ in OP1}
        assert op1_names == {"^2", "^3"}

    def test_op2_only_binary(self) -> None:
        """OP2 should only contain binary operators (arity 2)."""
        from kd2.search.sga.config import OP2

        for name, arity in OP2:
            assert arity == 2, f"{name} has arity {arity}, expected 2"

    def test_op2_contents(self) -> None:
        """OP2 = +, -, *, /, d, d^2."""
        from kd2.search.sga.config import OP2

        op2_names = {name for name, _ in OP2}
        assert op2_names == {"+", "-", "*", "/", "d", "d^2"}


# ===========================================================================
# Operator pool structural invariants
# ===========================================================================


class TestOperatorPoolInvariants:
    """Cross-pool consistency checks."""

    def test_op1_and_op2_partition_ops_by_arity(self) -> None:
        """OP1 + OP2 should cover the same operators as OPS."""
        from kd2.search.sga.config import OP1, OP2, OPS

        ops_names = {name for name, _ in OPS}
        combined = {name for name, _ in OP1} | {name for name, _ in OP2}
        assert combined == ops_names

    def test_op1_and_op2_are_disjoint(self) -> None:
        """No operator should be in both OP1 and OP2."""
        from kd2.search.sga.config import OP1, OP2

        op1_names = {name for name, _ in OP1}
        op2_names = {name for name, _ in OP2}
        assert op1_names.isdisjoint(op2_names)

    def test_root_is_subset_of_ops(self) -> None:
        """Every ROOT operator should also be in OPS."""
        from kd2.search.sga.config import OPS, ROOT

        ops_names = {name for name, _ in OPS}
        root_names = {name for name, _ in ROOT}
        assert root_names.issubset(ops_names)

    def test_all_pool_entries_are_name_arity_tuples(self) -> None:
        """Every entry in every pool must be (str, int)."""
        from kd2.search.sga.config import OP1, OP2, OPS, ROOT

        for pool_name, pool in [
            ("OPS", OPS),
            ("ROOT", ROOT),
            ("OP1", OP1),
            ("OP2", OP2),
        ]:
            for entry in pool:
                assert isinstance(entry, tuple), (
                    f"{pool_name} entry is not a tuple: {entry}"
                )
                assert len(entry) == 2, f"{pool_name} entry has wrong length: {entry}"
                name, arity = entry
                assert isinstance(name, str), f"{pool_name} name is not str: {name}"
                assert isinstance(arity, int), f"{pool_name} arity is not int: {arity}"


# ===========================================================================
# Negative tests
# ===========================================================================


class TestSGAConfigNegative:
    """Edge cases and error handling for SGAConfig."""

    def test_probability_fields_accept_boundary_values(self) -> None:
        """Probabilities at 0.0 and 1.0 should be accepted without error."""
        config = SGAConfig(p_var=0.0, p_mute=0.0, p_cro=0.0, p_rep=0.0)
        assert config.p_var == 0.0
        config = SGAConfig(p_var=1.0, p_mute=1.0, p_cro=1.0, p_rep=1.0)
        assert config.p_var == 1.0

    def test_zero_width_accepted(self) -> None:
        """Width=0 should be accepted (degenerate but not config's job to reject)."""
        config = SGAConfig(width=0)
        assert config.width == 0

    def test_zero_depth_accepted(self) -> None:
        """Depth=0 should be accepted at config level."""
        config = SGAConfig(depth=0)
        assert config.depth == 0


# ===========================================================================
# Derivative operator grammar: d / d^2 in operator pools
# ===========================================================================


class TestDerivativeOperatorGrammar:
    """d and d^2 must be present as binary (arity=2) operators in pools."""

    def test_d_in_ops(self) -> None:
        from kd2.search.sga.config import OPS

        op_dict = dict(OPS)
        assert "d" in op_dict
        assert op_dict["d"] == 2

    def test_d2_in_ops(self) -> None:
        from kd2.search.sga.config import OPS

        op_dict = dict(OPS)
        assert "d^2" in op_dict
        assert op_dict["d^2"] == 2

    def test_d_in_root(self) -> None:
        from kd2.search.sga.config import ROOT

        root_dict = dict(ROOT)
        assert "d" in root_dict
        assert root_dict["d"] == 2

    def test_d2_in_root(self) -> None:
        from kd2.search.sga.config import ROOT

        root_dict = dict(ROOT)
        assert "d^2" in root_dict
        assert root_dict["d^2"] == 2

    def test_d_in_op2(self) -> None:
        from kd2.search.sga.config import OP2

        op2_dict = dict(OP2)
        assert "d" in op2_dict
        assert op2_dict["d"] == 2

    def test_d2_in_op2(self) -> None:
        from kd2.search.sga.config import OP2

        op2_dict = dict(OP2)
        assert "d^2" in op2_dict
        assert op2_dict["d^2"] == 2

    def test_d_not_in_op1(self) -> None:
        """d and d^2 are binary; they should NOT be in the unary pool OP1."""
        from kd2.search.sga.config import OP1

        op1_names = {name for name, _ in OP1}
        assert "d" not in op1_names
        assert "d^2" not in op1_names


# ===========================================================================
# build_den: derivative denominator construction
# ===========================================================================


class TestBuildDen:
    """build_den constructs the set of valid derivative denominator axes."""

    def test_build_den_importable(self) -> None:
        from kd2.search.sga.config import build_den # noqa: F401

    def test_excludes_lhs_axis(self) -> None:
        """den must NOT contain lhs_axis (e.g. 't' for u_t = RHS)."""
        from kd2.search.sga.config import build_den

        axes = ["x", "t", "y"]
        den = build_den(axes=axes, lhs_axis="t")
        den_names = {name for name, _ in den}
        assert "t" not in den_names

    def test_includes_non_lhs_axes(self) -> None:
        """den should include all axes except lhs_axis."""
        from kd2.search.sga.config import build_den

        axes = ["x", "t", "y"]
        den = build_den(axes=axes, lhs_axis="t")
        den_names = {name for name, _ in den}
        assert den_names == {"x", "y"}

    def test_single_spatial_axis(self) -> None:
        """Common case: axes=['x', 't'], lhs_axis='t' -> den = [('x', 0)]."""
        from kd2.search.sga.config import build_den

        den = build_den(axes=["x", "t"], lhs_axis="t")
        assert len(den) == 1
        assert den[0] == ("x", 0)

    def test_den_entries_are_leaves(self) -> None:
        """All den entries must have arity 0 (they are leaf coordinate variables)."""
        from kd2.search.sga.config import build_den

        den = build_den(axes=["x", "y", "z", "t"], lhs_axis="t")
        for name, arity in den:
            assert arity == 0, f"den entry '{name}' has arity {arity}, expected 0"

    def test_empty_axes_raises(self) -> None:
        """If no axes remain after filtering, should raise ValueError."""
        from kd2.search.sga.config import build_den

        with pytest.raises(ValueError):
            build_den(axes=["t"], lhs_axis="t")

    def test_returns_tuple(self) -> None:
        """build_den should return a tuple of (name, arity) tuples."""
        from kd2.search.sga.config import build_den

        den = build_den(axes=["x", "t"], lhs_axis="t")
        assert isinstance(den, (list, tuple))
        for entry in den:
            assert isinstance(entry, tuple)
            assert len(entry) == 2
