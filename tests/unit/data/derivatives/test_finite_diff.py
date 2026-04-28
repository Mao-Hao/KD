"""Tests for finite difference derivative provider.

TDD TDD red phase: These tests define the expected behavior.
All tests should fail until implementation is complete.
"""

from __future__ import annotations

import math

import pytest
import torch

from kd2.data.derivatives import (
    DerivativeProvider,
    FiniteDiffProvider,
)
from kd2.data.derivatives.finite_diff import central_diff
from kd2.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType

# =============================================================================
# Smoke Tests
# =============================================================================


class TestSmoke:
    """Smoke tests - basic functionality checks."""

    @pytest.mark.smoke
    def test_provider_init(self, simple_1d_dataset: PDEDataset) -> None:
        """Provider can be initialized with valid dataset."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        assert provider is not None
        assert isinstance(provider, DerivativeProvider)

    @pytest.mark.smoke
    def test_get_derivative_basic(self, simple_1d_dataset: PDEDataset) -> None:
        """Provider returns tensor for basic derivative request."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        result = provider.get_derivative("u", "x", order=1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == simple_1d_dataset.get_shape()


# =============================================================================
# Central Difference Algorithm Tests
# =============================================================================


class TestCentralDiff:
    """Tests for central_diff function."""

    @pytest.mark.unit
    def test_central_diff_order_1_sin(self) -> None:
        """First derivative of sin(x) should be cos(x)."""
        n = 100
        x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        dx = x[1] - x[0]
        f = torch.sin(x)

        df = central_diff(f, dx.item(), axis=0, order=1)
        expected = torch.cos(x)

        # Interior points should be accurate
        # Skip boundary points (first and last)
        assert torch.allclose(df[2:-2], expected[2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    def test_central_diff_order_2_sin(self) -> None:
        """Second derivative of sin(x) should be -sin(x)."""
        n = 100
        x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        dx = x[1] - x[0]
        f = torch.sin(x)

        d2f = central_diff(f, dx.item(), axis=0, order=2)
        expected = -torch.sin(x)

        # Interior points should be accurate
        assert torch.allclose(d2f[2:-2], expected[2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    def test_central_diff_order_3_sin(self) -> None:
        """Third derivative of sin(x) should be -cos(x)."""
        n = 200 # More points for higher order accuracy
        x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
        dx = x[1] - x[0]
        f = torch.sin(x)

        d3f = central_diff(f, dx.item(), axis=0, order=3)
        expected = -torch.cos(x)

        # Interior points (skip more for 3rd order)
        assert torch.allclose(d3f[4:-4], expected[4:-4], rtol=1e-2, atol=1e-5)

    @pytest.mark.unit
    def test_central_diff_preserves_shape(self) -> None:
        """Output shape should match input shape."""
        f = torch.randn(50, dtype=torch.float64)
        dx = 0.1
        df = central_diff(f, dx, axis=0, order=1)
        assert df.shape == f.shape

    @pytest.mark.unit
    def test_central_diff_2d_axis_0(self) -> None:
        """Central diff along axis 0 in 2D tensor."""
        n_x, n_t = 64, 32
        x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
        t = torch.linspace(0, 1, n_t, dtype=torch.float64)
        dx = x[1] - x[0]

        X, T = torch.meshgrid(x, t, indexing="ij")
        f = torch.sin(X) # Only depends on x

        df = central_diff(f, dx.item(), axis=0, order=1)
        expected = torch.cos(X)

        # Check interior points
        assert torch.allclose(df[2:-2,:], expected[2:-2,:], rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    def test_central_diff_2d_axis_1(self) -> None:
        """Central diff along axis 1 in 2D tensor."""
        n_x, n_t = 64, 32
        x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
        t = torch.linspace(0, 1, n_t, dtype=torch.float64)
        dt = t[1] - t[0]

        X, T = torch.meshgrid(x, t, indexing="ij")
        f = torch.exp(-T) # Only depends on t

        df = central_diff(f, dt.item(), axis=1, order=1)
        expected = -torch.exp(-T)

        # Check interior points
        assert torch.allclose(df[:, 2:-2], expected[:, 2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    def test_central_diff_invalid_order(self) -> None:
        """Invalid order should raise ValueError."""
        f = torch.randn(50, dtype=torch.float64)
        dx = 0.1

        with pytest.raises(ValueError):
            central_diff(f, dx, axis=0, order=0)

        with pytest.raises(ValueError):
            central_diff(f, dx, axis=0, order=4)

        with pytest.raises(ValueError):
            central_diff(f, dx, axis=0, order=-1)

    @pytest.mark.unit
    def test_central_diff_invalid_axis(self) -> None:
        """Invalid axis should raise ValueError."""
        f = torch.randn(50, dtype=torch.float64)
        dx = 0.1

        with pytest.raises((ValueError, IndexError)):
            central_diff(f, dx, axis=1, order=1) # 1D tensor, no axis 1


# =============================================================================
# Provider Tests
# =============================================================================


class TestProviderInit:
    """Tests for FiniteDiffProvider initialization."""

    @pytest.mark.unit
    def test_provider_requires_grid_topology(
        self, scattered_dataset: PDEDataset
    ) -> None:
        """Provider should reject non-Grid topology."""
        with pytest.raises(ValueError, match="[Gg]rid"):
            FiniteDiffProvider(scattered_dataset)

    @pytest.mark.unit
    def test_provider_accepts_grid_topology(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        """Provider should accept Grid topology."""
        assert simple_1d_dataset.topology == DataTopology.GRID
        provider = FiniteDiffProvider(simple_1d_dataset)
        assert provider is not None

    @pytest.mark.unit
    def test_provider_max_order_default(self, simple_1d_dataset: PDEDataset) -> None:
        """Default max_order should be 3."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        # Should be able to get up to order 3
        derivatives = provider.available_derivatives()
        orders = [order for _, _, order in derivatives]
        assert max(orders) == 3

    @pytest.mark.unit
    def test_provider_max_order_custom(self, simple_1d_dataset: PDEDataset) -> None:
        """Custom max_order should be respected."""
        provider = FiniteDiffProvider(simple_1d_dataset, max_order=2)
        derivatives = provider.available_derivatives()
        orders = [order for _, _, order in derivatives]
        assert max(orders) == 2


class TestProviderGetDerivative:
    """Tests for FiniteDiffProvider.get_derivative method."""

    @pytest.mark.unit
    def test_provider_get_derivative_shape(self, simple_1d_dataset: PDEDataset) -> None:
        """Derivative should have same shape as field."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        deriv = provider.get_derivative("u", "x", order=1)
        expected_shape = simple_1d_dataset.get_shape()
        assert deriv.shape == expected_shape

    @pytest.mark.unit
    def test_provider_get_derivative_dtype(self, simple_1d_dataset: PDEDataset) -> None:
        """Derivative should preserve dtype."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        deriv = provider.get_derivative("u", "x", order=1)
        original_dtype = simple_1d_dataset.get_field("u").dtype
        assert deriv.dtype == original_dtype

    @pytest.mark.unit
    def test_provider_get_derivative_2d(self, simple_2d_dataset: PDEDataset) -> None:
        """Provider should handle 2D data correctly."""
        provider = FiniteDiffProvider(simple_2d_dataset)

        # Derivative along x
        u_x = provider.get_derivative("u", "x", order=1)
        assert u_x.shape == simple_2d_dataset.get_shape()

        # Derivative along t
        u_t = provider.get_derivative("u", "t", order=1)
        assert u_t.shape == simple_2d_dataset.get_shape()


class TestProviderAvailableDerivatives:
    """Tests for FiniteDiffProvider.available_derivatives method."""

    @pytest.mark.unit
    def test_available_derivatives_1d(self, simple_1d_dataset: PDEDataset) -> None:
        """1D dataset should have derivatives for single axis."""
        provider = FiniteDiffProvider(simple_1d_dataset, max_order=3)
        derivatives = provider.available_derivatives()

        # Should have u_x, u_xx, u_xxx
        expected = [("u", "x", 1), ("u", "x", 2), ("u", "x", 3)]
        for deriv in expected:
            assert deriv in derivatives

    @pytest.mark.unit
    def test_available_derivatives_2d(self, simple_2d_dataset: PDEDataset) -> None:
        """2D dataset should have derivatives for both axes."""
        provider = FiniteDiffProvider(simple_2d_dataset, max_order=2)
        derivatives = provider.available_derivatives()

        # Should have derivatives along x and t
        expected_contains = [
            ("u", "x", 1),
            ("u", "x", 2),
            ("u", "t", 1),
            ("u", "t", 2),
        ]
        for deriv in expected_contains:
            assert deriv in derivatives


class TestProviderDiff:
    """Tests for FiniteDiffProvider.diff method.

    diff() computes finite difference derivatives on arbitrary runtime tensor
    expressions (not just precomputed field derivatives). It should delegate
    to central_diff() with the correct axis_idx, dx, and is_periodic.
    """

    # Smoke: callable and returns tensor

    @pytest.mark.smoke
    def test_diff_returns_tensor(self, simple_1d_dataset: PDEDataset) -> None:
        """diff() returns a Tensor of the same shape as input."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        field_values = simple_1d_dataset.get_field("u")
        result = provider.diff(field_values, "x", order=1)
        assert isinstance(result, torch.Tensor)
        assert result.shape == field_values.shape

    @pytest.mark.smoke
    def test_diff_preserves_dtype(self, simple_1d_dataset: PDEDataset) -> None:
        """diff() preserves the dtype of the input expression."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        field_values = simple_1d_dataset.get_field("u")
        result = provider.diff(field_values, "x", order=1)
        assert result.dtype == field_values.dtype

    # Consistency: diff(field_data) == get_derivative(field)

    @pytest.mark.unit
    def test_diff_matches_get_derivative_1d_order1(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        """diff(field_data, 'x', 1) should match get_derivative('u', 'x', 1).

        Applying FD to the raw field tensor should give the same result
        as precomputed derivatives, since both use central_diff internally.
        """
        provider = FiniteDiffProvider(simple_1d_dataset)
        field_values = simple_1d_dataset.get_field("u")

        diff_result = provider.diff(field_values, "x", order=1)
        precomputed = provider.get_derivative("u", "x", order=1)

        torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    @pytest.mark.unit
    def test_diff_matches_get_derivative_1d_order2(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        """diff(field_data, 'x', 2) should match get_derivative('u', 'x', 2)."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        field_values = simple_1d_dataset.get_field("u")

        diff_result = provider.diff(field_values, "x", order=2)
        precomputed = provider.get_derivative("u", "x", order=2)

        torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    @pytest.mark.unit
    def test_diff_matches_get_derivative_1d_order3(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        """diff(field_data, 'x', 3) should match get_derivative('u', 'x', 3)."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        field_values = simple_1d_dataset.get_field("u")

        diff_result = provider.diff(field_values, "x", order=3)
        precomputed = provider.get_derivative("u", "x", order=3)

        torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    @pytest.mark.unit
    def test_diff_matches_get_derivative_2d_both_axes(
        self, simple_2d_dataset: PDEDataset
    ) -> None:
        """diff(field_data, axis, 1) matches get_derivative for both axes in 2D.

        Tests that axis_idx mapping is correct for multi-dimensional data.
        """
        provider = FiniteDiffProvider(simple_2d_dataset)
        field_values = simple_2d_dataset.get_field("u")

        for axis_name in ["x", "t"]:
            diff_result = provider.diff(field_values, axis_name, order=1)
            precomputed = provider.get_derivative("u", axis_name, order=1)
            torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    # Composite expression: diff(u**2, 'x', 1) ~ 2*u*u_x

    @pytest.mark.unit
    def test_diff_composite_u_squared(self, simple_2d_dataset: PDEDataset) -> None:
        """diff(u**2, 'x', 1) should approximate 2*u*u_x (chain rule).

        u = sin(x) * exp(-t)
        d/dx[u^2] = 2*u*u_x = 2*sin(x)*cos(x)*exp(-2t) = sin(2x)*exp(-2t)

        FD truncation error is present, so tolerance is looser than
        the consistency test.
        """
        provider = FiniteDiffProvider(simple_2d_dataset)
        u = simple_2d_dataset.get_field("u") # sin(x)*exp(-t), shape (64, 32)
        u_x = provider.get_derivative("u", "x", order=1)

        # Compute d/dx[u^2] via diff()
        u_sq = u**2
        d_u_sq = provider.diff(u_sq, "x", order=1)

        # Analytical reference: 2*u*u_x
        expected = 2 * u * u_x

        # Interior points only (skip 2 boundary points on each side along x-axis)
        torch.testing.assert_close(
            d_u_sq[2:-2,:], expected[2:-2,:], rtol=1e-3, atol=1e-5
        )

    @pytest.mark.unit
    def test_diff_composite_polynomial(self, polynomial_1d_dataset: PDEDataset) -> None:
        """diff(u**2, 'x', 1) for u=x^3: d/dx[x^6] = 6*x^5.

        Polynomial test gives exact FD results for low-enough degree
        relative to the stencil accuracy.
        """
        provider = FiniteDiffProvider(polynomial_1d_dataset)
        u = polynomial_1d_dataset.get_field("u") # x^3
        x = polynomial_1d_dataset.get_coords("x")

        u_sq = u**2 # x^6
        d_u_sq = provider.diff(u_sq, "x", order=1)

        expected = 6 * x**5

        # Interior points, moderate tolerance for 6th degree polynomial
        torch.testing.assert_close(d_u_sq[4:-4], expected[4:-4], rtol=1e-2, atol=1e-4)

    # Order tests: order 1, 2, 3 all work

    @pytest.mark.unit
    def test_diff_all_orders_produce_finite_results(
        self, simple_2d_dataset: PDEDataset
    ) -> None:
        """diff() at orders 1, 2, 3 all produce finite tensors."""
        provider = FiniteDiffProvider(simple_2d_dataset)
        expr = simple_2d_dataset.get_field("u")

        for order in [1, 2, 3]:
            result = provider.diff(expr, "x", order=order)
            assert torch.isfinite(result).all(), (
                f"diff(expr, 'x', {order}) produced NaN/Inf"
            )
            assert result.shape == expr.shape

    @pytest.mark.unit
    def test_diff_order2_sin(self, simple_1d_dataset: PDEDataset) -> None:
        """diff(sin(x), 'x', 2) should approximate -sin(x).

        Tests that order=2 correctly applies the second derivative stencil.
        """
        provider = FiniteDiffProvider(simple_1d_dataset)
        u = simple_1d_dataset.get_field("u") # sin(x)
        x = simple_1d_dataset.get_coords("x")

        result = provider.diff(u, "x", order=2)
        expected = -torch.sin(x)

        torch.testing.assert_close(result[2:-2], expected[2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.unit
    def test_diff_order3_sin(self, simple_1d_dataset: PDEDataset) -> None:
        """diff(sin(x), 'x', 3) should approximate -cos(x).

        Tests that order=3 correctly applies the third derivative stencil.
        """
        provider = FiniteDiffProvider(simple_1d_dataset)
        u = simple_1d_dataset.get_field("u") # sin(x)
        x = simple_1d_dataset.get_coords("x")

        result = provider.diff(u, "x", order=3)
        expected = -torch.cos(x)

        # 3rd order is 2nd-order accurate, needs larger tolerance
        torch.testing.assert_close(result[4:-4], expected[4:-4], rtol=1e-2, atol=1e-4)

    # Invalid axis: unknown axis name -> KeyError

    @pytest.mark.unit
    def test_diff_invalid_axis_raises_key_error(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        """diff() with unknown axis name should raise KeyError."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        expr = torch.randn(100, dtype=torch.float64)

        with pytest.raises(KeyError):
            provider.diff(expr, "nonexistent_axis", order=1)

    @pytest.mark.unit
    def test_diff_invalid_axis_y_on_xt_dataset(
        self, simple_2d_dataset: PDEDataset
    ) -> None:
        """diff() with axis 'y' on a dataset with only 'x' and 't' axes."""
        provider = FiniteDiffProvider(simple_2d_dataset)
        expr = simple_2d_dataset.get_field("u")

        with pytest.raises(KeyError):
            provider.diff(expr, "y", order=1)

    # Periodic axis: diff() should use wrap-around padding

    @pytest.mark.numerical
    def test_diff_periodic_matches_get_derivative(self) -> None:
        """diff() on periodic axis should produce same result as get_derivative.

        Uses the same periodic dataset as TestPeriodicBoundary to verify
        that diff() correctly passes is_periodic to central_diff.
        """
        n = 128
        x_full = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
        x = x_full[:-1] # [0, 1) for periodic domain
        u = torch.sin(2 * math.pi * x)

        ds = PDEDataset(
            name="periodic_diff_test",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x, is_periodic=True)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

        provider = FiniteDiffProvider(ds)
        field_values = ds.get_field("u")

        diff_result = provider.diff(field_values, "x", order=1)
        precomputed = provider.get_derivative("u", "x", order=1)

        # Must be bit-identical since same code path
        torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    @pytest.mark.numerical
    def test_diff_periodic_boundary_accuracy(self) -> None:
        """diff() on periodic data should have uniform accuracy across all points.

        u = sin(2*pi*x) on [0, 1)
        u_x = 2*pi*cos(2*pi*x)

        If is_periodic is correctly passed, boundary points should use
        wrap-around stencil and match interior accuracy.
        """
        n = 128
        x_full = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
        x = x_full[:-1]
        u = torch.sin(2 * math.pi * x)

        ds = PDEDataset(
            name="periodic_diff_accuracy",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x, is_periodic=True)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

        provider = FiniteDiffProvider(ds)
        result = provider.diff(u, "x", order=1)

        expected = 2 * math.pi * torch.cos(2 * math.pi * x)
        abs_err = (result - expected).abs()

        interior_max_err = abs_err[4:-4].max().item()
        boundary_max_err = max(
            abs_err[:2].max().item(),
            abs_err[-2:].max().item(),
        )

        # With periodic stencil, boundary should be within 10x of interior
        assert boundary_max_err < 10 * interior_max_err, (
            f"Periodic diff() boundary error ({boundary_max_err:.2e}) is much "
            f"worse than interior ({interior_max_err:.2e}). "
            f"is_periodic may not be passed to central_diff."
        )

    # Negative tests: invalid input, error handling (>= 20%)

    @pytest.mark.numerical
    def test_diff_nan_input_raises(self, simple_1d_dataset: PDEDataset) -> None:
        """diff() with NaN in expression should raise ValueError.

        Delegates to central_diff which validates input for NaN.
        """
        provider = FiniteDiffProvider(simple_1d_dataset)
        expr = torch.randn(100, dtype=torch.float64)
        expr[10] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            provider.diff(expr, "x", order=1)

    @pytest.mark.numerical
    def test_diff_inf_input_raises(self, simple_1d_dataset: PDEDataset) -> None:
        """diff() with Inf in expression should raise ValueError.

        Delegates to central_diff which validates input for Inf.
        """
        provider = FiniteDiffProvider(simple_1d_dataset)
        expr = torch.randn(100, dtype=torch.float64)
        expr[10] = float("inf")

        with pytest.raises(ValueError, match="Inf"):
            provider.diff(expr, "x", order=1)

    @pytest.mark.unit
    def test_diff_invalid_order_raises(self, simple_1d_dataset: PDEDataset) -> None:
        """diff() with invalid order (0, 4, -1) should raise ValueError.

        Delegates to central_diff which validates order range.
        """
        provider = FiniteDiffProvider(simple_1d_dataset)
        expr = simple_1d_dataset.get_field("u")

        with pytest.raises(ValueError):
            provider.diff(expr, "x", order=0)
        with pytest.raises(ValueError):
            provider.diff(expr, "x", order=4)
        with pytest.raises(ValueError):
            provider.diff(expr, "x", order=-1)

    @pytest.mark.unit
    def test_diff_wrong_shape_tensor(self, simple_2d_dataset: PDEDataset) -> None:
        """diff() with tensor whose size doesn't match grid should fail.

        Shape guard catches mismatched axis dimension before central_diff
        is called, preventing silent wrong-dx computation.
        """
        provider = FiniteDiffProvider(simple_2d_dataset)
        # simple_2d_dataset has shape (64, 32)

        # Too small — caught by shape guard
        wrong_shape = torch.randn(3, 3, dtype=torch.float64)
        with pytest.raises(ValueError, match="doesn't match"):
            provider.diff(wrong_shape, "x", order=1)

        # Right number of dims but wrong sizes — caught by shape guard
        wrong_size = torch.randn(100, 32, dtype=torch.float64)
        with pytest.raises(ValueError, match="doesn't match"):
            provider.diff(wrong_size, "x", order=1)

        # Wrong on second axis
        wrong_axis2 = torch.randn(64, 100, dtype=torch.float64)
        with pytest.raises(ValueError, match="doesn't match"):
            provider.diff(wrong_axis2, "t", order=1)


class TestProviderErrors:
    """Tests for error handling in FiniteDiffProvider."""

    @pytest.mark.unit
    def test_provider_invalid_field(self, simple_1d_dataset: PDEDataset) -> None:
        """Getting derivative of nonexistent field should raise KeyError."""
        provider = FiniteDiffProvider(simple_1d_dataset)

        with pytest.raises(KeyError):
            provider.get_derivative("nonexistent", "x", order=1)

    @pytest.mark.unit
    def test_provider_invalid_axis(self, simple_1d_dataset: PDEDataset) -> None:
        """Getting derivative along nonexistent axis should raise KeyError."""
        provider = FiniteDiffProvider(simple_1d_dataset)

        with pytest.raises(KeyError):
            provider.get_derivative("u", "nonexistent", order=1)

    @pytest.mark.unit
    def test_provider_invalid_order_zero(self, simple_1d_dataset: PDEDataset) -> None:
        """Order 0 should raise ValueError."""
        provider = FiniteDiffProvider(simple_1d_dataset)

        with pytest.raises(ValueError):
            provider.get_derivative("u", "x", order=0)

    @pytest.mark.unit
    def test_provider_invalid_order_negative(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        """Negative order should raise ValueError."""
        provider = FiniteDiffProvider(simple_1d_dataset)

        with pytest.raises(ValueError):
            provider.get_derivative("u", "x", order=-1)

    @pytest.mark.unit
    def test_provider_order_exceeds_max(self, simple_1d_dataset: PDEDataset) -> None:
        """Order exceeding max_order should raise ValueError."""
        provider = FiniteDiffProvider(simple_1d_dataset, max_order=2)

        with pytest.raises(ValueError):
            provider.get_derivative("u", "x", order=3)


# =============================================================================
# Numerical Accuracy Tests
# =============================================================================


class TestNumericalAccuracy:
    """Numerical accuracy tests for derivative computation."""

    @pytest.mark.numerical
    def test_accuracy_sin(self, simple_1d_dataset: PDEDataset) -> None:
        """Verify sin(x) derivative accuracy.

        u = sin(x)
        u_x = cos(x)
        u_xx = -sin(x)
        """
        provider = FiniteDiffProvider(simple_1d_dataset)
        x = simple_1d_dataset.get_coords("x")

        # First derivative
        u_x = provider.get_derivative("u", "x", order=1)
        expected_u_x = torch.cos(x)
        # Interior points (skip boundary)
        assert torch.allclose(u_x[2:-2], expected_u_x[2:-2], rtol=1e-3, atol=1e-6)

        # Second derivative
        u_xx = provider.get_derivative("u", "x", order=2)
        expected_u_xx = -torch.sin(x)
        assert torch.allclose(u_xx[2:-2], expected_u_xx[2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.numerical
    def test_accuracy_polynomial(self, polynomial_1d_dataset: PDEDataset) -> None:
        """Verify polynomial derivative accuracy.

        u = x^3
        u_x = 3*x^2
        u_xx = 6*x
        u_xxx = 6
        """
        provider = FiniteDiffProvider(polynomial_1d_dataset)
        x = polynomial_1d_dataset.get_coords("x")

        # First derivative: 3*x^2
        u_x = provider.get_derivative("u", "x", order=1)
        expected_u_x = 3 * x**2
        assert torch.allclose(u_x[2:-2], expected_u_x[2:-2], rtol=1e-3, atol=1e-6)

        # Second derivative: 6*x
        u_xx = provider.get_derivative("u", "x", order=2)
        expected_u_xx = 6 * x
        assert torch.allclose(u_xx[2:-2], expected_u_xx[2:-2], rtol=1e-3, atol=1e-6)

        # Third derivative: 6 (constant)
        u_xxx = provider.get_derivative("u", "x", order=3)
        expected_u_xxx = torch.full_like(x, 6.0)
        # Lower tolerance for 3rd order
        assert torch.allclose(u_xxx[4:-4], expected_u_xxx[4:-4], rtol=1e-2, atol=1e-4)

    @pytest.mark.numerical
    def test_accuracy_exp(self) -> None:
        """Verify exp(x) derivative accuracy.

        u = exp(x)
        u_x = exp(x)
        u_xx = exp(x)
        """
        # Create dataset with exp function
        n_points = 100
        x = torch.linspace(
            0, 1, n_points, dtype=torch.float64
        ) # Small range to avoid overflow
        u = torch.exp(x)

        from kd2.data.schema import (
            DataTopology,
            PDEDataset,
            TaskType,
        )

        dataset = PDEDataset(
            name="test_exp",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

        provider = FiniteDiffProvider(dataset)

        # First derivative
        u_x = provider.get_derivative("u", "x", order=1)
        expected = torch.exp(x)
        assert torch.allclose(u_x[2:-2], expected[2:-2], rtol=1e-3, atol=1e-6)

        # Second derivative
        u_xx = provider.get_derivative("u", "x", order=2)
        assert torch.allclose(u_xx[2:-2], expected[2:-2], rtol=1e-3, atol=1e-6)

    @pytest.mark.numerical
    def test_accuracy_2d_mixed(self, simple_2d_dataset: PDEDataset) -> None:
        """Verify 2D derivative accuracy.

        u = sin(x) * exp(-t)
        u_x = cos(x) * exp(-t)
        u_t = -sin(x) * exp(-t)
        """
        provider = FiniteDiffProvider(simple_2d_dataset)
        x = simple_2d_dataset.get_coords("x")
        t = simple_2d_dataset.get_coords("t")

        X, T = torch.meshgrid(x, t, indexing="ij")

        # Derivative along x
        u_x = provider.get_derivative("u", "x", order=1)
        expected_u_x = torch.cos(X) * torch.exp(-T)
        # Interior points
        assert torch.allclose(
            u_x[2:-2, 2:-2], expected_u_x[2:-2, 2:-2], rtol=1e-3, atol=1e-6
        )

        # Derivative along t
        u_t = provider.get_derivative("u", "t", order=1)
        expected_u_t = -torch.sin(X) * torch.exp(-T)
        assert torch.allclose(
            u_t[2:-2, 2:-2], expected_u_t[2:-2, 2:-2], rtol=1e-3, atol=1e-6
        )

    @pytest.mark.numerical
    def test_relative_error_bounds(self, simple_1d_dataset: PDEDataset) -> None:
        """Relative error should be within acceptable bounds."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        x = simple_1d_dataset.get_coords("x")

        u_x = provider.get_derivative("u", "x", order=1)
        expected = torch.cos(x)

        # Compute relative error (avoid division by zero near zeros of cos)
        mask = torch.abs(expected) > 0.1
        if mask.any():
            relative_error = torch.abs(u_x[mask] - expected[mask]) / torch.abs(
                expected[mask]
            )
            # Max relative error should be < 1e-3 for interior points
            interior_mask = mask.clone()
            interior_mask[:2] = False
            interior_mask[-2:] = False
            if interior_mask.any():
                max_rel_error = relative_error[interior_mask[mask]].max()
                assert max_rel_error < 1e-3, f"Max relative error: {max_rel_error}"


# =============================================================================
# Adversarial Attack Tests (High Priority)
# =============================================================================


class TestCentralDiffAttack:
    """Adversarial tests for central_diff - attacking edge cases and invalid inputs."""

    @pytest.mark.numerical
    def test_attack_dx_zero(self) -> None:
        """dx=0 should raise ValueError, not divide by zero.

        H1: Division by zero when dx=0.
        """
        f = torch.randn(50, dtype=torch.float64)
        with pytest.raises((ValueError, ZeroDivisionError)):
            central_diff(f, dx=0.0, axis=0, order=1)

    @pytest.mark.numerical
    def test_attack_dx_tiny(self) -> None:
        """dx extremely small should raise ValueError.

        H1: Near-zero dx causes numerical overflow (1/dx -> Inf).
        PyTorch does not raise FloatingPointError, so only ValueError is expected.
        """
        f = torch.randn(50, dtype=torch.float64)
        # 1e-320 is below float64 subnormal threshold
        with pytest.raises(ValueError, match="dx"):
            central_diff(f, dx=1e-320, axis=0, order=1)

    @pytest.mark.numerical
    def test_attack_dx_huge(self) -> None:
        """dx extremely large causes precision loss but results should be finite.

        H2: Large dx causes all derivatives to approach zero.
        Mathematically, derivative = (f[i+1] - f[i-1]) / (2*dx).
        When dx=1e15, the division produces extremely small (near-zero) values,
        but they should remain finite and computable.
        """
        f = torch.sin(torch.linspace(0, 2 * math.pi, 50, dtype=torch.float64))
        result = central_diff(f, dx=1e15, axis=0, order=1)
        # Result should be finite (precision loss, not error)
        assert torch.isfinite(result).all(), "dx=1e15 should produce finite results"
        # Result is near zero due to precision loss
        assert result.abs().max() < 1e-10, "Large dx produces near-zero derivatives"

    @pytest.mark.numerical
    def test_attack_dx_inf(self) -> None:
        """dx=inf should raise ValueError.

        H2b: Infinite dx is invalid input and should be rejected early.
        """
        f = torch.randn(50, dtype=torch.float64)
        with pytest.raises(ValueError, match="dx"):
            central_diff(f, dx=float("inf"), axis=0, order=1)

    @pytest.mark.numerical
    def test_attack_insufficient_points_order1(self) -> None:
        """Order 1 with < 3 points should fail gracefully.

        H3: Central difference needs 3 points for 1st order (f[i-1], f[i], f[i+1]).
        """
        f = torch.tensor([1.0, 2.0], dtype=torch.float64) # Only 2 points
        with pytest.raises((ValueError, IndexError)):
            central_diff(f, dx=0.1, axis=0, order=1)

    @pytest.mark.numerical
    def test_attack_insufficient_points_order2(self) -> None:
        """Order 2 with < 3 points should fail gracefully.

        H3: Second-order central difference needs at least 3 points.
        """
        f = torch.tensor([1.0, 2.0], dtype=torch.float64) # Only 2 points
        with pytest.raises((ValueError, IndexError)):
            central_diff(f, dx=0.1, axis=0, order=2)

    @pytest.mark.numerical
    def test_attack_insufficient_points_order3(self) -> None:
        """Order 3 with < 5 points should fail gracefully.

        H3: Third-order central difference needs 5 points for stencil.
        """
        f = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64) # Only 4 points
        with pytest.raises((ValueError, IndexError)):
            central_diff(f, dx=0.1, axis=0, order=3)

    @pytest.mark.numerical
    def test_attack_nan_input(self) -> None:
        """NaN input should raise ValueError.

        H4: NaN values should be detected and rejected early.
        """
        f = torch.tensor([1.0, float("nan"), 3.0, 4.0, 5.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            central_diff(f, dx=0.1, axis=0, order=1)

    @pytest.mark.numerical
    def test_attack_inf_input(self) -> None:
        """Inf input should raise ValueError.

        H4: Infinity values should be detected and rejected early.
        """
        f = torch.tensor([1.0, float("inf"), 3.0, 4.0, 5.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            central_diff(f, dx=0.1, axis=0, order=1)


class TestProviderAttack:
    """Adversarial tests for FiniteDiffProvider."""

    @pytest.mark.unit
    def test_attack_nonuniform_grid(self) -> None:
        """Non-uniform grid should raise ValueError.

        H5: Finite difference assumes uniform spacing. Non-uniform grids
        would produce silently wrong results if not detected.
        """
        from kd2.data.schema import (
            DataTopology,
            PDEDataset,
            TaskType,
        )

        # Non-uniform spacing
        x = torch.tensor(
            [0.0, 0.1, 0.5, 0.6, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5],
            dtype=torch.float64,
        )
        u = torch.sin(x)
        dataset = PDEDataset(
            name="nonuniform",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )
        # Should raise on init due to non-uniform grid
        with pytest.raises(ValueError, match="[Uu]niform"):
            FiniteDiffProvider(dataset)

    @pytest.mark.unit
    def test_attack_max_order_exceeds_supported(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        """max_order > 3 should raise ValueError.

        H6: central_diff only supports orders 1, 2, 3.
        Requesting max_order=4+ should fail at init time.
        """
        with pytest.raises(ValueError, match="max_order"):
            FiniteDiffProvider(simple_1d_dataset, max_order=4)

    @pytest.mark.unit
    def test_attack_max_order_exceeds_supported_large(
        self, simple_1d_dataset: PDEDataset
    ) -> None:
        """max_order much larger than 3 should raise ValueError.

        H6: Edge case with very large max_order.
        """
        with pytest.raises(ValueError, match="max_order"):
            FiniteDiffProvider(simple_1d_dataset, max_order=100)

    @pytest.mark.unit
    def test_attack_max_order_zero(self, simple_1d_dataset: PDEDataset) -> None:
        """max_order=0 should raise ValueError.

        H7: Zero max_order makes no sense (no derivatives to compute).
        """
        with pytest.raises(ValueError, match="max_order"):
            FiniteDiffProvider(simple_1d_dataset, max_order=0)

    @pytest.mark.unit
    def test_attack_max_order_negative(self, simple_1d_dataset: PDEDataset) -> None:
        """max_order=-1 should raise ValueError.

        H7: Negative max_order is invalid.
        """
        with pytest.raises(ValueError, match="max_order"):
            FiniteDiffProvider(simple_1d_dataset, max_order=-1)


# =============================================================================
# Codex Review Issue Tests
# =============================================================================


class TestBoundaryAccuracy:
    """Tests for boundary point accuracy - Codex CX-H1.

    Codex found: 2nd order derivative endpoint formula (f[2] - 2f[1] + f[0])/dx^2
    was only 1st-order accurate. After fix, boundary should be 2nd-order accurate
    using the formula: (2f0 - 5f1 + 4f2 - f3)/dx^2.
    """

    @pytest.mark.numerical
    def test_second_order_boundary_accuracy(self) -> None:
        """Boundary points of 2nd derivative should have reasonable accuracy.

        Uses u = x^3 (u'' = 6x) to test boundary accuracy.
        After fix: should be 2nd-order accurate at boundaries.
        """
        from kd2.data.schema import (
            DataTopology,
            PDEDataset,
            TaskType,
        )

        # Use x^3 function: u'' = 6x
        n = 100
        x = torch.linspace(0, 1, n, dtype=torch.float64)
        u = x**3 # u'' = 6x
        dx = 1.0 / (n - 1)

        # Create dataset
        dataset = PDEDataset(
            name="cubic",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )
        provider = FiniteDiffProvider(dataset)
        u_xx = provider.get_derivative("u", "x", order=2)

        expected = 6 * x # Analytical solution

        # Boundary points should be finite and close to analytical solution
        # Left boundary x=0 -> u''=0
        assert torch.isfinite(u_xx[0]), "Left boundary should be finite"
        assert abs(u_xx[0].item() - expected[0].item()) < 0.5, (
            f"Left boundary error too large: {abs(u_xx[0].item() - expected[0].item())}"
        )

        # Right boundary x=1 -> u''=6
        assert torch.isfinite(u_xx[-1]), "Right boundary should be finite"
        assert abs(u_xx[-1].item() - expected[-1].item()) < 0.5, (
            f"Right boundary error too large: {abs(u_xx[-1].item() - expected[-1].item())}"
        )

    @pytest.mark.numerical
    def test_second_order_boundary_convergence(self) -> None:
        """Boundary formula error should converge at 2nd order with grid refinement.

        If the formula is 2nd-order accurate, halving dx should reduce error by ~4x.
        This tests the convergence rate, not just the error magnitude.
        """
        from kd2.data.schema import (
            DataTopology,
            PDEDataset,
            TaskType,
        )

        def compute_boundary_error(n: int) -> tuple[float, float]:
            """Compute boundary errors for u=x^3 with n points."""
            x = torch.linspace(0, 1, n, dtype=torch.float64)
            u = x**3
            expected = 6 * x

            dataset = PDEDataset(
                name="cubic",
                task_type=TaskType.PDE,
                topology=DataTopology.GRID,
                axes={"x": AxisInfo(name="x", values=x)},
                axis_order=["x"],
                fields={"u": FieldData(name="u", values=u)},
                lhs_field="u",
                lhs_axis="x",
            )
            provider = FiniteDiffProvider(dataset)
            u_xx = provider.get_derivative("u", "x", order=2)

            left_err = abs(u_xx[0].item() - expected[0].item())
            right_err = abs(u_xx[-1].item() - expected[-1].item())
            return left_err, right_err

        # Compute errors at different resolutions
        _, err_coarse = compute_boundary_error(50)
        _, err_fine = compute_boundary_error(100)

        # For 2nd-order accuracy: error ratio should be ~4 when n doubles
        # Allow some tolerance (ratio should be > 3 for 2nd order)
        if err_coarse > 1e-10: # Only check if error is not too small
            ratio = err_coarse / err_fine
            assert ratio > 3.0, (
                f"Boundary convergence rate too low: ratio={ratio:.2f}, "
                f"expected >3 for 2nd-order accuracy. "
                f"Errors: coarse={err_coarse:.2e}, fine={err_fine:.2e}"
            )

    @pytest.mark.numerical
    def test_boundary_not_nan_or_inf(self, simple_1d_dataset: PDEDataset) -> None:
        """All boundary points should be finite (not NaN/Inf)."""
        provider = FiniteDiffProvider(simple_1d_dataset)

        for order in [1, 2, 3]:
            deriv = provider.get_derivative("u", "x", order=order)
            assert torch.isfinite(deriv).all(), (
                f"Order {order} derivative contains NaN/Inf at boundary"
            )


class TestAccuracyParameter:
    """Tests for accuracy parameter - Codex CX-H2.

    Codex found: accuracy parameter accepts 2/4/6 but implementation
    always uses 4th-order formulas. Fix: only support accuracy=4,
    raise ValueError for other values.
    """

    @pytest.mark.unit
    def test_accuracy_default_is_4(self, simple_1d_dataset: PDEDataset) -> None:
        """Default accuracy should be 4 (matching actual implementation)."""
        provider = FiniteDiffProvider(simple_1d_dataset)
        # After fix: default should be 4, not 2
        assert provider._accuracy == 4, (
            f"Default accuracy should be 4, got {provider._accuracy}"
        )

    @pytest.mark.unit
    def test_accuracy_4_accepted(self, simple_1d_dataset: PDEDataset) -> None:
        """accuracy=4 should be accepted without error."""
        provider = FiniteDiffProvider(simple_1d_dataset, accuracy=4)
        assert provider._accuracy == 4

    @pytest.mark.unit
    def test_accuracy_2_raises(self, simple_1d_dataset: PDEDataset) -> None:
        """accuracy=2 should raise ValueError (not implemented)."""
        with pytest.raises(ValueError, match="accuracy"):
            FiniteDiffProvider(simple_1d_dataset, accuracy=2)

    @pytest.mark.unit
    def test_accuracy_6_raises(self, simple_1d_dataset: PDEDataset) -> None:
        """accuracy=6 should raise ValueError (not implemented)."""
        with pytest.raises(ValueError, match="accuracy"):
            FiniteDiffProvider(simple_1d_dataset, accuracy=6)

    @pytest.mark.unit
    def test_accuracy_invalid_raises(self, simple_1d_dataset: PDEDataset) -> None:
        """Invalid accuracy values should raise ValueError."""
        with pytest.raises(ValueError, match="accuracy"):
            FiniteDiffProvider(simple_1d_dataset, accuracy=3)
        with pytest.raises(ValueError, match="accuracy"):
            FiniteDiffProvider(simple_1d_dataset, accuracy=0)
        with pytest.raises(ValueError, match="accuracy"):
            FiniteDiffProvider(simple_1d_dataset, accuracy=-1)


# =============================================================================
# Regression: DATA/M1 — FiniteDiffProvider must respect is_periodic
# =============================================================================


class TestPeriodicBoundary:
    """DATA/M1: FiniteDiffProvider ignores is_periodic flag.

    When AxisInfo.is_periodic=True, boundary derivatives should use
    wrap-around stencil (connecting f[-1] to f[0]) instead of one-sided
    formulas. Currently central_diff has no periodic support, so
    boundary accuracy on periodic data is degraded.
    """

    @staticmethod
    def _make_periodic_dataset(
        n: int = 128,
        is_periodic: bool = True,
    ) -> PDEDataset:
        """Build a 1D dataset of sin(2*pi*x) on [0, 1).

        For periodic data the domain must be [0, 1) not [0, 1], because
        the last point wraps to the first. We use linspace with n+1 points
        then drop the last to get proper periodicity.
        """

        # [0, 1) with n points — proper periodic domain
        x_full = torch.linspace(0.0, 1.0, n + 1, dtype=torch.float64)
        x = x_full[:-1] # drop the duplicate endpoint
        u = torch.sin(2 * math.pi * x)

        return PDEDataset(
            name="periodic_sin",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x, is_periodic=is_periodic)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

    @pytest.mark.numerical
    def test_periodic_boundary_matches_interior_accuracy(self) -> None:
        """With is_periodic=True, boundary error should be comparable to interior.

        Analytical derivative: u'(x) = 2*pi*cos(2*pi*x).
        Interior 4th-order stencils give ~O(dx^4) error.
        Periodic wrap-around should give the same order at boundaries.
        If non-periodic one-sided formula is used, boundary error will be
        orders of magnitude worse than interior error.
        """
        ds = self._make_periodic_dataset(n=128, is_periodic=True)
        provider = FiniteDiffProvider(ds)
        u_x = provider.get_derivative("u", "x", order=1)

        x = ds.get_coords("x")
        expected = 2 * math.pi * torch.cos(2 * math.pi * x)

        abs_err = (u_x - expected).abs()

        # Interior error (indices 4:-4, well away from boundary)
        interior_max_err = abs_err[4:-4].max().item()

        # Boundary error (first 2 and last 2 points)
        boundary_max_err = max(
            abs_err[:2].max().item(),
            abs_err[-2:].max().item(),
        )

        # Key assertion: boundary accuracy should be within 10x of interior
        # With non-periodic stencil the ratio is typically 100x-1000x worse
        assert boundary_max_err < 10 * interior_max_err, (
            f"Periodic boundary error ({boundary_max_err:.2e}) is much worse "
            f"than interior ({interior_max_err:.2e}). "
            f"Ratio = {boundary_max_err / max(interior_max_err, 1e-30):.0f}x. "
            f"is_periodic=True was likely ignored."
        )

    @pytest.mark.numerical
    def test_periodic_second_derivative_boundary(self) -> None:
        """Second derivative with periodic BC should also have uniform accuracy.

        u = sin(2*pi*x) → u'' = -(2*pi)^2 * sin(2*pi*x)
        """
        ds = self._make_periodic_dataset(n=128, is_periodic=True)
        provider = FiniteDiffProvider(ds)
        u_xx = provider.get_derivative("u", "x", order=2)

        x = ds.get_coords("x")
        expected = -((2 * math.pi) ** 2) * torch.sin(2 * math.pi * x)

        abs_err = (u_xx - expected).abs()
        interior_max_err = abs_err[4:-4].max().item()
        boundary_max_err = max(
            abs_err[:2].max().item(),
            abs_err[-2:].max().item(),
        )

        assert boundary_max_err < 10 * interior_max_err, (
            f"Periodic 2nd derivative boundary error ({boundary_max_err:.2e}) "
            f"vs interior ({interior_max_err:.2e}), "
            f"ratio = {boundary_max_err / max(interior_max_err, 1e-30):.0f}x"
        )

    @pytest.mark.numerical
    def test_nonperiodic_boundary_is_worse(self) -> None:
        """Confirm non-periodic one-sided formula gives worse boundary accuracy.

        Same data (sin(2*pi*x)) but is_periodic=False.
        Boundary error should be notably larger than interior because
        one-sided formulas are lower order. This test documents the
        baseline that periodic should improve upon.
        """
        ds = self._make_periodic_dataset(n=128, is_periodic=False)
        provider = FiniteDiffProvider(ds)
        u_x = provider.get_derivative("u", "x", order=1)

        x = ds.get_coords("x")
        expected = 2 * math.pi * torch.cos(2 * math.pi * x)

        abs_err = (u_x - expected).abs()
        interior_max_err = abs_err[4:-4].max().item()
        boundary_max_err = max(
            abs_err[:2].max().item(),
            abs_err[-2:].max().item(),
        )

        # Non-periodic: boundary should be significantly worse (>10x)
        # This test is expected to PASS (it documents current behavior)
        assert boundary_max_err > 10 * interior_max_err, (
            f"Expected non-periodic boundary to be much worse than interior, "
            f"but ratio is only {boundary_max_err / max(interior_max_err, 1e-30):.1f}x"
        )

    @pytest.mark.numerical
    def test_periodic_vs_nonperiodic_boundary_gap(self) -> None:
        """Periodic BC must produce substantially better boundary derivatives.

        Build the same dataset twice (periodic and non-periodic), compute
        boundary errors, and assert periodic is at least 5x better.
        """
        n = 128
        ds_periodic = self._make_periodic_dataset(n=n, is_periodic=True)
        ds_nonperiodic = self._make_periodic_dataset(n=n, is_periodic=False)

        provider_p = FiniteDiffProvider(ds_periodic)
        provider_np = FiniteDiffProvider(ds_nonperiodic)

        u_x_p = provider_p.get_derivative("u", "x", order=1)
        u_x_np = provider_np.get_derivative("u", "x", order=1)

        x = ds_periodic.get_coords("x")
        expected = 2 * math.pi * torch.cos(2 * math.pi * x)

        # Boundary indices
        boundary = list(range(2)) + list(range(n - 2, n))
        err_p = (u_x_p[boundary] - expected[boundary]).abs().max().item()
        err_np = (u_x_np[boundary] - expected[boundary]).abs().max().item()

        # Periodic should be at least 5x more accurate at boundaries
        assert err_p < err_np / 5, (
            f"Periodic boundary error ({err_p:.2e}) should be <5x better "
            f"than non-periodic ({err_np:.2e}), "
            f"ratio = {err_np / max(err_p, 1e-30):.1f}x"
        )

    @pytest.mark.numerical
    def test_periodic_third_derivative_boundary(self) -> None:
        """Third derivative with periodic BC should have uniform accuracy.

        u = sin(2*pi*x) → u''' = -(2*pi)^3 * cos(2*pi*x)

        3rd-order one-sided formula is only 1st-order accurate, so this is
        the order where periodic wrap-around matters most.
        """
        ds = self._make_periodic_dataset(n=128, is_periodic=True)
        provider = FiniteDiffProvider(ds)
        u_xxx = provider.get_derivative("u", "x", order=3)

        x = ds.get_coords("x")
        expected = -((2 * math.pi) ** 3) * torch.cos(2 * math.pi * x)

        abs_err = (u_xxx - expected).abs()
        interior_max_err = abs_err[4:-4].max().item()
        boundary_max_err = max(
            abs_err[:2].max().item(),
            abs_err[-2:].max().item(),
        )

        # 3rd-order one-sided formula is only O(dx), so without periodic
        # wrap-around the boundary/interior ratio is enormous
        assert boundary_max_err < 10 * interior_max_err, (
            f"Periodic 3rd derivative boundary error ({boundary_max_err:.2e}) "
            f"vs interior ({interior_max_err:.2e}), "
            f"ratio = {boundary_max_err / max(interior_max_err, 1e-30):.0f}x"
        )

    @pytest.mark.numerical
    def test_periodic_2d_mixed_axes(self) -> None:
        """2D dataset: x periodic, t non-periodic.

        u = sin(2*pi*x) * exp(-t)
        u_x = 2*pi*cos(2*pi*x) * exp(-t) → x boundary should be accurate
        u_t = -sin(2*pi*x) * exp(-t) → t boundary uses one-sided (worse)

        Verifies _precompute_derivatives respects is_periodic per axis.
        """
        n_x, n_t = 128, 32
        # x: periodic domain [0, 1)
        x_full = torch.linspace(0.0, 1.0, n_x + 1, dtype=torch.float64)
        x = x_full[:-1]
        # t: non-periodic domain [0, 1]
        t = torch.linspace(0.0, 1.0, n_t, dtype=torch.float64)

        X, T = torch.meshgrid(x, t, indexing="ij")
        u = torch.sin(2 * math.pi * X) * torch.exp(-T)

        ds = PDEDataset(
            name="periodic_2d",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=True),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )

        provider = FiniteDiffProvider(ds)

        # --- x derivative: periodic axis → boundary should match interior ---
        u_x = provider.get_derivative("u", "x", order=1)
        expected_u_x = 2 * math.pi * torch.cos(2 * math.pi * X) * torch.exp(-T)

        # Compare at interior t columns (avoid t boundary effects)
        t_interior = slice(4, -4)
        x_err = (u_x[:, t_interior] - expected_u_x[:, t_interior]).abs()
        x_interior_err = x_err[4:-4,:].max().item()
        x_boundary_err = max(
            x_err[:2,:].max().item(),
            x_err[-2:,:].max().item(),
        )

        assert x_boundary_err < 10 * x_interior_err, (
            f"x (periodic) boundary error ({x_boundary_err:.2e}) should be "
            f"close to interior ({x_interior_err:.2e}), "
            f"ratio = {x_boundary_err / max(x_interior_err, 1e-30):.0f}x"
        )

        # --- t derivative: non-periodic → boundary naturally worse ---
        u_t = provider.get_derivative("u", "t", order=1)
        expected_u_t = -torch.sin(2 * math.pi * X) * torch.exp(-T)

        # Compare at interior x rows (avoid x boundary effects)
        x_interior = slice(4, -4)
        t_err = (u_t[x_interior,:] - expected_u_t[x_interior,:]).abs()
        t_interior_err = t_err[:, 4:-4].max().item()
        t_boundary_err = max(
            t_err[:, :2].max().item(),
            t_err[:, -2:].max().item(),
        )

        # t is non-periodic: boundary should be notably worse than interior
        assert t_boundary_err > 5 * t_interior_err, (
            f"t (non-periodic) boundary should be worse than interior, "
            f"but ratio is only {t_boundary_err / max(t_interior_err, 1e-30):.1f}x"
        )


# =============================================================================
# Uniform Grid Check — Tolerance + Degenerate Spacing
# =============================================================================
#
# Regression: ``UNIFORM_GRID_RTOL`` was 1e-6 — too tight for ordinary
# ``torch.linspace(dtype=float32)`` grids (rel deviation reaches ~6e-5 at
# n=1000). The integrator's check was relaxed to 1e-4 in commit 27b17e4,
# but the FD provider was missed. Now both paths share rtol=1e-4 + a
# ``DX_ZERO_FLOOR`` guard against constant-coordinate axes.


class TestUniformGridTolerance:
    """Tolerance behavior of ``_check_uniform_grid``."""

    @pytest.mark.unit
    @pytest.mark.parametrize("n", [100, 500, 1000])
    def test_accepts_float32_linspace(self, n: int) -> None:
        """Ordinary float32 linspace grids must be accepted up to ~n=1000.

        Float32 has ~7 decimal digits, so successive linspace differences
        drift up to ~6e-5 (relative) at n=1000. The relaxed rtol=1e-4
        accepts this; the previous 1e-6 falsely rejected it.
        """
        from kd2.data.derivatives.finite_diff import _check_uniform_grid

        x = torch.linspace(0.0, 1.0, n, dtype=torch.float32)
        dx = _check_uniform_grid(x, "x")
        # Spacing should be close to ideal 1/(n-1)
        assert dx == pytest.approx(1.0 / (n - 1), rel=1e-3)

    @pytest.mark.unit
    def test_accepts_float64_linspace_large(self) -> None:
        """Float64 grids should comfortably pass at much larger n."""
        from kd2.data.derivatives.finite_diff import _check_uniform_grid

        x = torch.linspace(0.0, 1.0, 100_000, dtype=torch.float64)
        dx = _check_uniform_grid(x, "x")
        assert dx == pytest.approx(1.0 / (100_000 - 1), rel=1e-9)

    @pytest.mark.unit
    def test_rejects_geometric_spacing(self) -> None:
        """Genuine non-uniform grid (geometric spacing) is still rejected.

        Guards against the relaxed tolerance accepting real non-uniform
        grids. Geometric spacing has rel deviation O(1).
        """
        from kd2.data.derivatives.finite_diff import _check_uniform_grid

        # x = [1, 2, 4, 8, 16, ...] — purely geometric
        x = torch.tensor([2.0**i for i in range(10)], dtype=torch.float64)
        with pytest.raises(ValueError, match="non-uniform"):
            _check_uniform_grid(x, "x")

    @pytest.mark.unit
    def test_rejects_log_spacing(self) -> None:
        """Log-spaced grid is rejected as non-uniform."""
        from kd2.data.derivatives.finite_diff import _check_uniform_grid

        x = torch.logspace(0.0, 2.0, 50, dtype=torch.float64)
        with pytest.raises(ValueError, match="non-uniform"):
            _check_uniform_grid(x, "x")

    @pytest.mark.unit
    def test_rejects_degenerate_dx_zero(self) -> None:
        """Constant-coordinate axis (all values equal) is rejected.

        ``dx = 0`` would divide by zero downstream. The new
        ``DX_ZERO_FLOOR`` guard catches this with a ``degenerate spacing``
        diagnostic instead of letting the relative check muddle the
        message.
        """
        from kd2.data.derivatives.finite_diff import _check_uniform_grid

        x = torch.zeros(10, dtype=torch.float64)
        with pytest.raises(ValueError, match="degenerate spacing"):
            _check_uniform_grid(x, "x")

    @pytest.mark.unit
    def test_rejects_subnormal_dx(self) -> None:
        """Spacing below ``DX_ZERO_FLOOR=1e-30`` is treated as degenerate."""
        from kd2.data.derivatives.finite_diff import _check_uniform_grid

        # All ten values within 1e-31 of each other — dx underflows the floor.
        x = torch.linspace(0.0, 1e-31, 10, dtype=torch.float64)
        with pytest.raises(ValueError, match="degenerate spacing"):
            _check_uniform_grid(x, "x")


class TestProviderFloat32:
    """End-to-end: FiniteDiffProvider must accept float32 datasets.

    The public ``Model.fit()`` instantiates ``FiniteDiffProvider`` for any
    dataset, including float32 ones (e.g. on Apple Silicon MPS where the
    burgers generator forces float32). The previous tight tolerance
    rejected such datasets at the very first axis check.
    """

    @pytest.mark.unit
    def test_provider_accepts_float32_dataset(self) -> None:
        """Build a float32 1D dataset and instantiate the provider."""
        nx, nt = 64, 32
        x = torch.linspace(-1.0, 1.0, nx + 1, dtype=torch.float32)[:-1]
        t = torch.linspace(0.0, 1.0, nt, dtype=torch.float32)
        # Simple non-trivial field: sin(pi*x) * exp(-t)
        u = torch.sin(torch.pi * x).unsqueeze(1) * torch.exp(-t).unsqueeze(0)

        ds = PDEDataset(
            name="float32_test",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=True),
                "t": AxisInfo(name="t", values=t),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        provider = FiniteDiffProvider(ds, max_order=2)
        # Smoke-check the precomputed cache covers both axes
        u_x = provider.get_derivative("u", "x", order=1)
        u_t = provider.get_derivative("u", "t", order=1)
        assert u_x.shape == ds.get_shape()
        assert u_t.shape == ds.get_shape()
