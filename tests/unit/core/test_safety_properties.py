"""Property-based tests for numerical safety utilities.

Uses hypothesis to automatically find edge cases that hand-written tests miss.
Each test describes a mathematical PROPERTY that must hold for ALL inputs,
and hypothesis generates hundreds of random inputs to verify.
"""

import pytest
import torch
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from kd2.core.safety import safe_div, safe_exp, safe_log

# Strategy: finite floats in a wide range (excludes NaN/Inf)
finite_floats = st.floats(
    min_value=-1e10, max_value=1e10,
    allow_nan=False, allow_infinity=False,
)

# Strategy: positive finite floats
positive_floats = st.floats(
    min_value=1e-10, max_value=1e10,
    allow_nan=False, allow_infinity=False,
)

# Strategy: non-zero finite floats
nonzero_floats = st.floats(
    min_value=-1e10, max_value=1e10,
    allow_nan=False, allow_infinity=False,
).filter(lambda x: abs(x) > 1e-15)


@pytest.mark.numerical
class TestSafeDivProperties:
    """Mathematical properties of safe_div."""

    @given(a=finite_floats, b=finite_floats)
    @settings(max_examples=500)
    def test_finite_input_finite_output(self, a: float, b: float) -> None:
        """Core guarantee: finite inputs always produce finite output."""
        result = safe_div(torch.tensor(a), torch.tensor(b))
        assert torch.isfinite(result).all()

    @given(a=positive_floats, b=positive_floats)
    @settings(max_examples=200)
    def test_positive_div_positive_is_positive(self, a: float, b: float) -> None:
        """Sign preservation: positive / positive = positive."""
        result = safe_div(torch.tensor(a), torch.tensor(b))
        assert result.item() > 0

    @given(a=finite_floats, b=nonzero_floats)
    @settings(max_examples=200)
    def test_approximate_inverse(self, a: float, b: float) -> None:
        """When b is not near zero, safe_div(a, b) * b ≈ a."""
        assume(abs(b) > 1e-3) # only test when b is reasonably large
        result = safe_div(torch.tensor(a), torch.tensor(b))
        reconstructed = result * torch.tensor(b)
        torch.testing.assert_close(
            reconstructed, torch.tensor(a), rtol=1e-4, atol=1e-6,
        )

    @given(a=finite_floats)
    @settings(max_examples=200)
    def test_div_by_one(self, a: float) -> None:
        """safe_div(a, 1) ≈ a."""
        result = safe_div(torch.tensor(a), torch.tensor(1.0))
        torch.testing.assert_close(result, torch.tensor(a), rtol=1e-6, atol=1e-8)


@pytest.mark.numerical
class TestSafeExpProperties:
    """Mathematical properties of safe_exp."""

    @given(x=finite_floats)
    @settings(max_examples=500)
    def test_always_positive_finite(self, x: float) -> None:
        """Core guarantee: output is always positive and finite."""
        result = safe_exp(torch.tensor(x))
        assert result.item() > 0
        assert torch.isfinite(result).all()

    @given(x1=finite_floats, x2=finite_floats)
    @settings(max_examples=200)
    def test_monotonicity(self, x1: float, x2: float) -> None:
        """Monotonicity: x1 < x2 → safe_exp(x1) <= safe_exp(x2)."""
        assume(x1 < x2)
        r1 = safe_exp(torch.tensor(x1))
        r2 = safe_exp(torch.tensor(x2))
        assert r1.item() <= r2.item()

    @given(x=st.floats(min_value=-40.0, max_value=40.0))
    @settings(max_examples=200)
    def test_matches_torch_exp_in_safe_range(self, x: float) -> None:
        """Within clamp range, safe_exp matches torch.exp exactly."""
        t = torch.tensor(x)
        torch.testing.assert_close(safe_exp(t), torch.exp(t))


@pytest.mark.numerical
class TestSafeLogProperties:
    """Mathematical properties of safe_log."""

    @given(x=finite_floats)
    @settings(max_examples=500)
    def test_finite_input_finite_output(self, x: float) -> None:
        """Core guarantee: finite input always produces finite output."""
        result = safe_log(torch.tensor(x))
        assert torch.isfinite(result).all()

    @given(x1=positive_floats, x2=positive_floats)
    @settings(max_examples=200)
    def test_monotonicity_positive(self, x1: float, x2: float) -> None:
        """Monotonicity for positive inputs: x1 < x2 → safe_log(x1) < safe_log(x2)."""
        assume(x1 < x2)
        assume(x1 > 1e-8) # above eps threshold for strict monotonicity
        assume(abs(x2 - x1) > x1 * 1e-6) # ensure distinguishable in float32
        r1 = safe_log(torch.tensor(x1))
        r2 = safe_log(torch.tensor(x2))
        assert r1.item() < r2.item()

    @given(x=st.floats(min_value=1e-5, max_value=1e8))
    @settings(max_examples=200)
    def test_matches_torch_log_for_positive(self, x: float) -> None:
        """For positive values above eps, safe_log matches torch.log."""
        t = torch.tensor(x)
        torch.testing.assert_close(
            safe_log(t), torch.log(t), rtol=1e-5, atol=1e-8,
        )

    @given(x=st.floats(min_value=0.01, max_value=40.0))
    @settings(max_examples=200)
    def test_exp_log_roundtrip(self, x: float) -> None:
        """safe_log(safe_exp(x)) ≈ x for positive values in safe range."""
        t = torch.tensor(x)
        roundtrip = safe_log(safe_exp(t))
        torch.testing.assert_close(roundtrip, t, rtol=1e-4, atol=1e-6)
