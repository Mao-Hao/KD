"""Tests for derivative naming utilities.

Tests build_derivative_name, parse_derivative_name, parse_compound_derivative
from kd2.core.expr.naming.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kd2.core.expr.naming import (
    build_derivative_name,
    parse_compound_derivative,
    parse_derivative_name,
)

# Strategies for property-based tests

# Single-letter axis names (most common in PDE context)
_axis_st = st.sampled_from(list("xyzt"))
# Multi-char field names (realistic PDE fields)
_field_st = st.sampled_from(["u", "v", "w", "p", "phi", "psi", "vel", "rho"])
# Derivative order 1..5 (practical range)
_order_st = st.integers(min_value=1, max_value=5)


# ===========================================================================
# build_derivative_name
# ===========================================================================


class TestBuildDerivativeNameSmoke:
    """Smoke tests: function exists, is callable, and signature works."""

    @pytest.mark.smoke
    def test_callable(self) -> None:
        assert callable(build_derivative_name)

    @pytest.mark.smoke
    def test_returns_string(self) -> None:
        result = build_derivative_name("u", "x", 1)
        assert isinstance(result, str)


class TestBuildDerivativeNameBasic:
    """Happy-path tests for build_derivative_name."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("field", "axis", "order", "expected"),
        [
            ("u", "x", 1, "u_x"),
            ("u", "x", 2, "u_xx"),
            ("u", "x", 3, "u_xxx"),
            ("u", "t", 1, "u_t"),
            ("u", "t", 2, "u_tt"),
            ("v", "y", 1, "v_y"),
            ("v", "y", 4, "v_yyyy"),
        ],
    )
    def test_basic_cases(
        self, field: str, axis: str, order: int, expected: str
    ) -> None:
        assert build_derivative_name(field, axis, order) == expected

    @pytest.mark.unit
    def test_default_order_is_one(self) -> None:
        assert build_derivative_name("u", "x") == "u_x"

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("field", "axis", "order", "expected"),
        [
            ("phi", "x", 1, "phi_x"),
            ("phi", "x", 2, "phi_xx"),
            ("psi", "t", 3, "psi_ttt"),
            ("vel", "z", 1, "vel_z"),
            ("rho", "y", 2, "rho_yy"),
        ],
    )
    def test_multi_char_field(
        self, field: str, axis: str, order: int, expected: str
    ) -> None:
        assert build_derivative_name(field, axis, order) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("field", "axis", "order", "expected"),
        [
            # compound: appending derivative to already-derived field
            ("u_x", "y", 1, "u_x_y"),
            ("u_x", "x", 1, "u_x_x"),
            ("u_xx", "y", 2, "u_xx_yy"),
            ("phi_t", "x", 1, "phi_t_x"),
        ],
    )
    def test_compound_building(
        self, field: str, axis: str, order: int, expected: str
    ) -> None:
        assert build_derivative_name(field, axis, order) == expected


class TestBuildDerivativeNameValidation:
    """Negative tests: invalid input should raise ValueError."""

    @pytest.mark.unit
    def test_empty_field_raises(self) -> None:
        with pytest.raises(ValueError):
            build_derivative_name("", "x", 1)

    @pytest.mark.unit
    def test_empty_axis_raises(self) -> None:
        with pytest.raises(ValueError):
            build_derivative_name("u", "", 1)

    @pytest.mark.unit
    def test_order_zero_raises(self) -> None:
        with pytest.raises(ValueError):
            build_derivative_name("u", "x", 0)

    @pytest.mark.unit
    def test_negative_order_raises(self) -> None:
        with pytest.raises(ValueError):
            build_derivative_name("u", "x", -1)

    @pytest.mark.unit
    def test_negative_large_order_raises(self) -> None:
        with pytest.raises(ValueError):
            build_derivative_name("u", "x", -10)


class TestBuildDerivativeNameProperties:
    """Property-based tests for build_derivative_name."""

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_result_contains_field_prefix(
        self, field: str, axis: str, order: int
    ) -> None:
        """Result must start with the field name."""
        result = build_derivative_name(field, axis, order)
        assert result.startswith(field)

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_result_contains_underscore_separator(
        self, field: str, axis: str, order: int
    ) -> None:
        """Result must contain an underscore after field name."""
        result = build_derivative_name(field, axis, order)
        # After the field, there should be _<axis repeated>
        suffix = result[len(field):]
        assert suffix.startswith("_")

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_result_ends_with_axis_repeated(
        self, field: str, axis: str, order: int
    ) -> None:
        """The trailing segment must be axis repeated `order` times."""
        result = build_derivative_name(field, axis, order)
        expected_suffix = axis * order
        assert result.endswith(expected_suffix)

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_higher_order_produces_longer_name(
        self, field: str, axis: str, order: int
    ) -> None:
        """Increasing order by 1 adds exactly one character to the name."""
        r1 = build_derivative_name(field, axis, order)
        r2 = build_derivative_name(field, axis, order + 1)
        assert len(r2) == len(r1) + 1


# ===========================================================================
# parse_derivative_name
# ===========================================================================


class TestParseDerivativeNameSmoke:
    """Smoke: function exists and returns the right type."""

    @pytest.mark.smoke
    def test_callable(self) -> None:
        assert callable(parse_derivative_name)

    @pytest.mark.smoke
    def test_returns_tuple_or_none(self) -> None:
        result = parse_derivative_name("u_x")
        assert result is None or (isinstance(result, tuple) and len(result) == 3)


class TestParseDerivativeNameBasic:
    """Happy-path: same-axis terminal derivatives."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("u_x", ("u", "x", 1)),
            ("u_xx", ("u", "x", 2)),
            ("u_xxx", ("u", "x", 3)),
            ("u_t", ("u", "t", 1)),
            ("u_tt", ("u", "t", 2)),
            ("v_y", ("v", "y", 1)),
            ("v_yy", ("v", "y", 2)),
            ("w_z", ("w", "z", 1)),
        ],
    )
    def test_single_char_field(self, name: str, expected: tuple[str, str, int]) -> None:
        assert parse_derivative_name(name) == expected

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("phi_x", ("phi", "x", 1)),
            ("phi_xx", ("phi", "x", 2)),
            ("phi_tt", ("phi", "t", 2)),
            ("vel_t", ("vel", "t", 1)),
            ("rho_yy", ("rho", "y", 2)),
            ("psi_zzz", ("psi", "z", 3)),
        ],
    )
    def test_multi_char_field(self, name: str, expected: tuple[str, str, int]) -> None:
        assert parse_derivative_name(name) == expected


class TestParseDerivativeNameRejects:
    """Negative tests: things that are NOT same-axis terminal derivatives."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "name",
        [
            "u_xy", # mixed partial (two different axis chars)
            "u_xt", # mixed partial
            "u_x_y", # compound format (two underscores)
            "u_xx_y", # compound format
            "u", # no underscore, just a field
            "x", # single char, no underscore — coordinate
            "", # empty string
            "123", # numeric, not a derivative
            "42", # numeric
            "_x", # leading underscore, no field
            "u_", # trailing underscore, no axis
        ],
    )
    def test_rejects_non_derivatives(self, name: str) -> None:
        assert parse_derivative_name(name) is None


class TestParseDerivativeNameDisambiguation:
    """Tests with known_fields and known_axes for disambiguation."""

    @pytest.mark.unit
    def test_known_axes_single_char(self) -> None:
        """Default behavior: "u_xx" with known_axes={"x"} -> ("u", "x", 2)."""
        result = parse_derivative_name("u_xx", known_axes={"x"})
        assert result == ("u", "x", 2)

    @pytest.mark.unit
    def test_known_axes_multi_char_axis(self) -> None:
        """Ambiguity resolution: "u_xx" with known_axes={"xx"} -> ("u", "xx", 1)."""
        result = parse_derivative_name("u_xx", known_axes={"xx"})
        assert result == ("u", "xx", 1)

    @pytest.mark.unit
    def test_known_fields_helps_boundary(self) -> None:
        """known_fields={"phi"} helps identify field boundary in "phi_x"."""
        result = parse_derivative_name("phi_x", known_fields={"phi"})
        assert result == ("phi", "x", 1)

    @pytest.mark.unit
    def test_known_fields_changes_parse(self) -> None:
        """known_fields={"phi_x"} should affect how "phi_x_y" is parsed.

        Without known_fields, "phi_x" might be parsed as field=phi, axis=x.
        But this is a same-axis test, so "phi_x_y" -> None (compound).
        """
        # parse_derivative_name only handles same-axis, compound -> None
        result = parse_derivative_name("phi_x_y")
        assert result is None

    @pytest.mark.unit
    def test_known_axes_no_match(self) -> None:
        """known_axes is validation context, so mismatches should reject."""
        result = parse_derivative_name("u_xx", known_axes={"y"})
        assert result is None

    @pytest.mark.unit
    def test_known_fields_changes_field_boundary(self) -> None:
        """known_fields actually changes the parse result.

        "phi_x" default parse: field="phi", axis="x", order=1
        But with known_fields={"ph"}, field="ph" and axis part="i_x"
        which doesn't look like a same-axis derivative -> None.

        This verifies known_fields has real effect on disambiguation.
        """
        # Default: phi is the field
        default = parse_derivative_name("phi_x")
        assert default == ("phi", "x", 1)

        # With known_fields={"ph"}: field boundary shifts
        # "ph" is the field, remaining is "i_x" — not a valid same-axis pattern
        shifted = parse_derivative_name("phi_x", known_fields={"ph"})
        assert shifted is None

    @pytest.mark.unit
    def test_known_fields_confirms_default(self) -> None:
        """known_fields={"phi"} confirms the default parse of "phi_x"."""
        result = parse_derivative_name("phi_x", known_fields={"phi"})
        assert result == ("phi", "x", 1)


class TestParseDerivativeNameRoundTrip:
    """Round-trip: parse(build(f, a, o)) == (f, a, o)."""

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_roundtrip_build_then_parse(
        self, field: str, axis: str, order: int
    ) -> None:
        """Building then parsing should recover the original components."""
        name = build_derivative_name(field, axis, order)
        result = parse_derivative_name(name)
        assert result == (field, axis, order)


# ===========================================================================
# parse_compound_derivative
# ===========================================================================


class TestParseCompoundDerivativeSmoke:
    """Smoke: function exists and returns correct type."""

    @pytest.mark.smoke
    def test_callable(self) -> None:
        assert callable(parse_compound_derivative)

    @pytest.mark.smoke
    def test_returns_tuple_or_none(self) -> None:
        result = parse_compound_derivative("u_x")
        assert result is None or isinstance(result, tuple)


class TestParseCompoundDerivativeSameAxis:
    """Same-axis cases: compound parser is a superset of same-axis parser."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("u_x", ("u", [("x", 1)])),
            ("u_xx", ("u", [("x", 2)])),
            ("u_xxx", ("u", [("x", 3)])),
            ("v_t", ("v", [("t", 1)])),
            ("v_tt", ("v", [("t", 2)])),
            ("phi_yy", ("phi", [("y", 2)])),
        ],
    )
    def test_same_axis_derivatives(
        self, name: str, expected: tuple[str, list[tuple[str, int]]]
    ) -> None:
        assert parse_compound_derivative(name) == expected


class TestParseCompoundDerivativeMixed:
    """Mixed partial derivatives in compound format."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("u_x_y", ("u", [("x", 1), ("y", 1)])),
            ("u_xx_y", ("u", [("x", 2), ("y", 1)])),
            ("u_x_yy", ("u", [("x", 1), ("y", 2)])),
            ("u_xx_yy", ("u", [("x", 2), ("y", 2)])),
            ("u_xxx_yy", ("u", [("x", 3), ("y", 2)])),
        ],
    )
    def test_two_axis_mixed(
        self, name: str, expected: tuple[str, list[tuple[str, int]]]
    ) -> None:
        assert parse_compound_derivative(name) == expected

    @pytest.mark.unit
    def test_same_axis_repeated_segments_not_merged(self) -> None:
        """u_x_x (two separate segments) != u_xx (one merged segment).

        Each underscore-delimited segment represents one FD application.
        build("u_x", "x", 1) produces "u_x_x" — two distinct d/dx steps.
        The compound parser preserves this: [("x",1), ("x",1)].
        Consumers (e.g., total-order computation) merge if needed.
        """
        # "u_x_x" = two sequential d/dx — NOT merged into order 2
        result_separate = parse_compound_derivative("u_x_x")
        assert result_separate == ("u", [("x", 1), ("x", 1)])

        # "u_xx" = single segment with order 2
        result_merged = parse_compound_derivative("u_xx")
        assert result_merged == ("u", [("x", 2)])

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("u_x_y_z", ("u", [("x", 1), ("y", 1), ("z", 1)])),
            ("u_xx_y_z", ("u", [("x", 2), ("y", 1), ("z", 1)])),
            ("u_x_yy_zz", ("u", [("x", 1), ("y", 2), ("z", 2)])),
        ],
    )
    def test_three_axis_mixed(
        self, name: str, expected: tuple[str, list[tuple[str, int]]]
    ) -> None:
        assert parse_compound_derivative(name) == expected

    @pytest.mark.unit
    def test_multi_char_field_mixed(self) -> None:
        result = parse_compound_derivative("phi_x_y")
        assert result == ("phi", [("x", 1), ("y", 1)])

    @pytest.mark.unit
    def test_multi_char_field_higher_order_mixed(self) -> None:
        result = parse_compound_derivative("phi_xx_tt")
        assert result == ("phi", [("x", 2), ("t", 2)])


class TestParseCompoundDerivativeRejects:
    """Negative tests: things that are NOT compound derivatives."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "name",
        [
            "u", # bare field
            "x", # coordinate
            "", # empty string
            "123", # numeric
            "42", # numeric
            "_x", # leading underscore, no field
            "u_", # trailing underscore, empty axis segment
            "u__x", # double underscore, empty segment in between
            "u_x_", # trailing underscore after valid segment
        ],
    )
    def test_rejects_non_derivatives(self, name: str) -> None:
        assert parse_compound_derivative(name) is None


class TestParseCompoundDerivativeSingleSegmentMixed:
    """Single-segment mixed axis: "u_xy" without underscore separation."""

    @pytest.mark.unit
    def test_single_segment_mixed_axis_rejected(self) -> None:
        """u_xy (no underscore between axes) is not parseable without known_axes.

        Without disambiguation context, a single segment "xy" could be:
        - axis="x" order=1 + axis="y" order=1 (mixed partial)
        - axis="xy" order=1 (exotic multi-char axis)
        Since we can't tell, return None.
        """
        assert parse_compound_derivative("u_xy") is None

    @pytest.mark.unit
    def test_single_segment_mixed_axis_with_known_axes_still_rejected(self) -> None:
        """Even with known_axes={"x","y"}, "u_xy" is rejected.

        Single-segment splitting (breaking "xy" into "x"+"y") is not
        supported. Users must use compound format "u_x_y" for mixed
        partials. This avoids ambiguity with multi-char axis names.
        """
        result = parse_compound_derivative("u_xy", known_axes={"x", "y"})
        assert result is None

    @pytest.mark.unit
    def test_single_segment_mixed_axis_partial_known(self) -> None:
        """With known_axes={"x"} only, "u_xy" is still ambiguous.

        "y" could be the second character of a multi-char axis "xy",
        not necessarily a separate axis. Without "y" in known_axes,
        we cannot split confidently.
        """
        # Only "x" known — "y" is ambiguous, so reject
        assert parse_compound_derivative("u_xy", known_axes={"x"}) is None


class TestParseCompoundDerivativeDisambiguation:
    """Tests with known_axes / known_fields."""

    @pytest.mark.unit
    def test_known_axes_affects_segment_parsing(self) -> None:
        """With known_axes={"x"}, "u_xx" -> ("u", [("x", 2)])."""
        result = parse_compound_derivative("u_xx", known_axes={"x"})
        assert result == ("u", [("x", 2)])

    @pytest.mark.unit
    def test_known_axes_multi_char(self) -> None:
        """With known_axes={"xx"}, "u_xx" -> ("u", [("xx", 1)])."""
        result = parse_compound_derivative("u_xx", known_axes={"xx"})
        assert result == ("u", [("xx", 1)])

    @pytest.mark.unit
    def test_known_fields_compound(self) -> None:
        """known_fields={"phi"} helps parse "phi_x_y" correctly."""
        result = parse_compound_derivative("phi_x_y", known_fields={"phi"})
        assert result == ("phi", [("x", 1), ("y", 1)])


class TestParseCompoundDerivativeRoundTrip:
    """Round-trip: build a compound name via sequential builds, then parse."""

    @pytest.mark.unit
    def test_roundtrip_single_axis(self) -> None:
        """build u_xx then compound-parse -> ("u", [("x", 2)])."""
        name = build_derivative_name("u", "x", 2)
        result = parse_compound_derivative(name)
        assert result is not None
        field, derivs = result
        assert field == "u"
        assert derivs == [("x", 2)]

    @pytest.mark.unit
    def test_roundtrip_two_axis_sequential_build(self) -> None:
        """Build u_x, then build u_x_y (compound), parse back."""
        step1 = build_derivative_name("u", "x", 1) # "u_x"
        step2 = build_derivative_name(step1, "y", 1) # "u_x_y"
        result = parse_compound_derivative(step2)
        assert result is not None
        field, derivs = result
        assert field == "u"
        assert derivs == [("x", 1), ("y", 1)]

    @pytest.mark.unit
    def test_roundtrip_higher_order_compound(self) -> None:
        """Build u_xx, then build u_xx_yy (compound), parse back."""
        step1 = build_derivative_name("u", "x", 2) # "u_xx"
        step2 = build_derivative_name(step1, "y", 2) # "u_xx_yy"
        result = parse_compound_derivative(step2)
        assert result is not None
        field, derivs = result
        assert field == "u"
        assert derivs == [("x", 2), ("y", 2)]


# ===========================================================================
# Cross-function consistency
# ===========================================================================


class TestCrossFunctionConsistency:
    """Verify consistency between parse_derivative_name and parse_compound_derivative."""

    @pytest.mark.unit
    @given(field=_field_st, axis=_axis_st, order=_order_st)
    @settings(max_examples=50)
    def test_same_axis_parse_agrees_with_compound(
        self, field: str, axis: str, order: int
    ) -> None:
        """For same-axis derivatives, both parsers should agree on the field and axis."""
        name = build_derivative_name(field, axis, order)
        simple = parse_derivative_name(name)
        compound = parse_compound_derivative(name)

        assert simple is not None
        assert compound is not None

        s_field, s_axis, s_order = simple
        c_field, c_derivs = compound

        assert s_field == c_field
        # compound should produce single-element list with same axis/order
        assert len(c_derivs) == 1
        assert c_derivs[0] == (s_axis, s_order)

    @pytest.mark.unit
    def test_compound_rejects_implies_simple_rejects(self) -> None:
        """If compound parse returns None for a bare name, simple should too."""
        for name in ["u", "x", "", "123"]:
            assert parse_derivative_name(name) is None
            assert parse_compound_derivative(name) is None

    @pytest.mark.unit
    def test_mixed_partial_simple_rejects_compound_accepts(self) -> None:
        """Mixed partials: simple parser rejects, compound parser accepts."""
        name = "u_x_y"
        assert parse_derivative_name(name) is None
        result = parse_compound_derivative(name)
        assert result is not None
        assert result == ("u", [("x", 1), ("y", 1)])


# ===========================================================================
# Integration: naming matches executor.py conventions
# ===========================================================================


class TestExecutorConventionCompatibility:
    """Verify that parse results match what executor.py recognizes.

    executor.py uses _DERIVATIVE_PATTERN = r'^([a-zA-Z]+)_([a-zA-Z])([a-zA-Z]*)$'
    and validates that rest chars all equal first_axis.

    Our parse_derivative_name should produce the same field/axis/order
    for the same inputs that executor.py's _try_parse_terminal_derivative handles.
    """

    @pytest.mark.unit
    @pytest.mark.parametrize(
        ("name", "field", "axis", "order"),
        [
            # Executor recognizes these terminal derivatives:
            ("u_x", "u", "x", 1),
            ("u_xx", "u", "x", 2),
            ("u_xxx", "u", "x", 3),
            ("v_t", "v", "t", 1),
            ("v_tt", "v", "t", 2),
            ("phi_yy", "phi", "y", 2),
        ],
    )
    def test_matches_executor_terminal_derivatives(
        self, name: str, field: str, axis: str, order: int
    ) -> None:
        result = parse_derivative_name(name)
        assert result is not None
        assert result == (field, axis, order)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "name",
        [
            # Executor rejects these (mixed axes):
            "u_xy",
            "u_xt",
            "v_yz",
        ],
    )
    def test_rejects_what_executor_rejects(self, name: str) -> None:
        """Mixed-axis names that executor.py would reject as terminal derivatives."""
        assert parse_derivative_name(name) is None


class TestSympyBridgeCompatibility:
    """Verify compound parse handles names produced by sympy_bridge.

    sympy_bridge._derivative_symbol_name produces: "{field}_{axis*order}"
    and compound derivatives like diff_y(u_x) produce "u_x_y".
    """

    @pytest.mark.unit
    def test_sympy_bridge_simple_derivative(self) -> None:
        """sympy_bridge produces "u_xx" for diff2_x(u) — parse should handle."""
        result = parse_compound_derivative("u_xx")
        assert result == ("u", [("x", 2)])

    @pytest.mark.unit
    def test_sympy_bridge_compound_derivative(self) -> None:
        """diff_y(u_x) produces "u_x_y" — compound parse should handle."""
        result = parse_compound_derivative("u_x_y")
        assert result == ("u", [("x", 1), ("y", 1)])

    @pytest.mark.unit
    def test_sympy_bridge_higher_compound(self) -> None:
        """diff2_y(u_xx) produces "u_xx_yy" — compound parse should handle."""
        result = parse_compound_derivative("u_xx_yy")
        assert result == ("u", [("x", 2), ("y", 2)])
