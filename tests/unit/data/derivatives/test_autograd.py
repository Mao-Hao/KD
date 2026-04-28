"""Tests for AutogradProvider - autograd-based derivative computation.

TDD TDD red phase: These tests define the expected behavior of AutogradProvider.
All tests should fail until implementation is complete.

Test model: We use exact-function models (SinModel, PolyModel) that don't
require training, so derivatives have known analytical solutions.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd2.data.derivatives.autograd import AutogradProvider
from kd2.data.derivatives.base import DerivativeProvider
from kd2.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)

# =============================================================================
# Test Models (exact functions, no training needed)
# =============================================================================


class SinModel(nn.Module):
    """Exact sin(x) model for derivative testing.

    u = sin(x), independent of t.
    u_x = cos(x), u_xx = -sin(x), u_xxx = -cos(x)
    """

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.sin(x)


class PolyModel(nn.Module):
    """Exact x^2 model for derivative testing.

    u = x^2, independent of t.
    u_x = 2x, u_xx = 2
    """

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return x**2


class MultiFieldModel(nn.Module):
    """Model returning two fields: u = sin(x), v = cos(x).

    Used to test multi-field get_field / get_derivative.
    """

    def forward(self, x: Tensor, t: Tensor) -> dict[str, Tensor]:
        return {"u": torch.sin(x), "v": torch.cos(x)}


class SinProductModel(nn.Module):
    """u = sin(x) * exp(-t).

    u_x = cos(x) * exp(-t)
    u_t = -sin(x) * exp(-t)
    """

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        return torch.sin(x) * torch.exp(-t)


# =============================================================================
# Fixtures
# =============================================================================


def _make_coords_1d(n: int = 50) -> dict[str, Tensor]:
    """Create 1D coords with requires_grad=True."""
    x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64, requires_grad=True)
    t = torch.zeros(n, dtype=torch.float64, requires_grad=True)
    return {"x": x, "t": t}


def _make_coords_2d(n_x: int = 50, n_t: int = 30) -> dict[str, Tensor]:
    """Create 2D coords with requires_grad=True (flattened for NN input)."""
    x_1d = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t_1d = torch.linspace(0, 1, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x_1d, t_1d, indexing="ij")
    x = X.reshape(-1).clone().detach().requires_grad_(True)
    t = T.reshape(-1).clone().detach().requires_grad_(True)
    return {"x": x, "t": t}


def _make_dataset_1d(n: int = 50) -> PDEDataset:
    """Create minimal 1D PDEDataset for metadata."""
    x = torch.linspace(0, 2 * math.pi, n, dtype=torch.float64)
    u = torch.sin(x)
    return PDEDataset(
        name="test_autograd_1d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="x",
    )


def _make_dataset_2d(n_x: int = 50, n_t: int = 30) -> PDEDataset:
    """Create minimal 2D PDEDataset for metadata."""
    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(X) * torch.exp(-T)
    return PDEDataset(
        name="test_autograd_2d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def sin_provider() -> AutogradProvider:
    """AutogradProvider with SinModel and 1D coords."""
    model = SinModel().double()
    coords = _make_coords_1d(50)
    dataset = _make_dataset_1d(50)
    return AutogradProvider(model=model, coords=coords, dataset=dataset)


@pytest.fixture
def poly_provider() -> AutogradProvider:
    """AutogradProvider with PolyModel and 1D coords."""
    model = PolyModel().double()
    coords = _make_coords_1d(50)
    dataset = _make_dataset_1d(50)
    return AutogradProvider(model=model, coords=coords, dataset=dataset)


@pytest.fixture
def sin_product_provider() -> AutogradProvider:
    """AutogradProvider with SinProductModel and 2D coords."""
    model = SinProductModel().double()
    coords = _make_coords_2d(50, 30)
    dataset = _make_dataset_2d(50, 30)
    return AutogradProvider(model=model, coords=coords, dataset=dataset)


# =============================================================================
# Smoke Tests
# =============================================================================


class TestSmoke:
    """Smoke tests - basic instantiation and interface checks."""

    @pytest.mark.smoke
    def test_autograd_provider_importable(self) -> None:
        """AutogradProvider can be imported."""
        from kd2.data.derivatives.autograd import AutogradProvider # noqa: F811

        assert AutogradProvider is not None

    @pytest.mark.smoke
    def test_provider_init(self, sin_provider: AutogradProvider) -> None:
        """Provider can be initialized with valid arguments."""
        assert sin_provider is not None
        assert isinstance(sin_provider, DerivativeProvider)

    @pytest.mark.smoke
    def test_get_field_returns_tensor(self, sin_provider: AutogradProvider) -> None:
        """get_field returns a Tensor."""
        result = sin_provider.get_field("u")
        assert isinstance(result, torch.Tensor)

    @pytest.mark.smoke
    def test_diff_returns_tensor(self, sin_provider: AutogradProvider) -> None:
        """diff returns a Tensor."""
        field = sin_provider.get_field("u")
        result = sin_provider.diff(field, "x", order=1)
        assert isinstance(result, torch.Tensor)

    @pytest.mark.smoke
    def test_get_derivative_returns_tensor(
        self, sin_provider: AutogradProvider
    ) -> None:
        """get_derivative returns a Tensor."""
        result = sin_provider.get_derivative("u", "x", order=1)
        assert isinstance(result, torch.Tensor)


# =============================================================================
# Init Tests
# =============================================================================


class TestInit:
    """Tests for AutogradProvider initialization."""

    @pytest.mark.unit
    def test_init_stores_model(self) -> None:
        """Provider stores the model reference."""
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
        assert provider.model is model

    @pytest.mark.unit
    def test_init_stores_coords(self) -> None:
        """Provider stores the coords reference."""
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
        assert provider.coords is coords

    @pytest.mark.unit
    def test_init_stores_dataset(self) -> None:
        """Provider stores the dataset reference."""
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
        assert provider.dataset is dataset

    @pytest.mark.unit
    def test_init_coords_requires_grad(self) -> None:
        """Coords without requires_grad should raise ValueError or be handled."""
        model = SinModel().double()
        # Coords WITHOUT requires_grad
        x = torch.linspace(0, 2 * math.pi, 50, dtype=torch.float64)
        t = torch.zeros(50, dtype=torch.float64)
        coords = {"x": x, "t": t}
        dataset = _make_dataset_1d()

        # Should either raise ValueError or auto-set requires_grad
        # Prefer explicit error so user knows to set requires_grad
        with pytest.raises(ValueError, match="requires_grad"):
            AutogradProvider(model=model, coords=coords, dataset=dataset)

    @pytest.mark.unit
    def test_init_coords_keys_match_dataset_axes(self) -> None:
        """Coords keys should match dataset axis_order."""
        model = SinModel().double()
        # Coords with wrong keys
        coords = {
            "y": torch.linspace(0, 1, 50, dtype=torch.float64, requires_grad=True),
            "z": torch.zeros(50, dtype=torch.float64, requires_grad=True),
        }
        dataset = _make_dataset_1d()

        with pytest.raises((KeyError, ValueError)):
            AutogradProvider(model=model, coords=coords, dataset=dataset)


# =============================================================================
# get_field Tests
# =============================================================================


class TestGetField:
    """Tests for AutogradProvider.get_field method."""

    @pytest.mark.unit
    def test_get_field_sin(self, sin_provider: AutogradProvider) -> None:
        """get_field('u') returns sin(x) for SinModel."""
        result = sin_provider.get_field("u")
        x = sin_provider.coords["x"]
        expected = torch.sin(x)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.unit
    def test_get_field_poly(self, poly_provider: AutogradProvider) -> None:
        """get_field('u') returns x^2 for PolyModel."""
        result = poly_provider.get_field("u")
        x = poly_provider.coords["x"]
        expected = x**2
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.unit
    def test_get_field_invalid_name(self, sin_provider: AutogradProvider) -> None:
        """get_field with invalid name should raise KeyError."""
        with pytest.raises(KeyError):
            sin_provider.get_field("nonexistent")

    @pytest.mark.unit
    def test_get_field_result_in_computation_graph(
        self, sin_provider: AutogradProvider
    ) -> None:
        """get_field result should be part of computation graph (grad_fn is set)."""
        result = sin_provider.get_field("u")
        assert result.requires_grad, "get_field result should require grad"


# =============================================================================
# diff Tests (core)
# =============================================================================


class TestDiff:
    """Tests for AutogradProvider.diff method - core autograd differentiation."""

    @pytest.mark.unit
    def test_diff_sin_order1(self, sin_provider: AutogradProvider) -> None:
        """d/dx sin(x) = cos(x)."""
        u = sin_provider.get_field("u")
        u_x = sin_provider.diff(u, "x", order=1)
        x = sin_provider.coords["x"]
        expected = torch.cos(x)
        torch.testing.assert_close(u_x, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_sin_order2(self, sin_provider: AutogradProvider) -> None:
        """d²/dx² sin(x) = -sin(x)."""
        u = sin_provider.get_field("u")
        u_xx = sin_provider.diff(u, "x", order=2)
        x = sin_provider.coords["x"]
        expected = -torch.sin(x)
        torch.testing.assert_close(u_xx, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_sin_order3(self, sin_provider: AutogradProvider) -> None:
        """d³/dx³ sin(x) = -cos(x)."""
        u = sin_provider.get_field("u")
        u_xxx = sin_provider.diff(u, "x", order=3)
        x = sin_provider.coords["x"]
        expected = -torch.cos(x)
        torch.testing.assert_close(u_xxx, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_poly_order1(self, poly_provider: AutogradProvider) -> None:
        """d/dx x^2 = 2x."""
        u = poly_provider.get_field("u")
        u_x = poly_provider.diff(u, "x", order=1)
        x = poly_provider.coords["x"]
        expected = 2 * x
        torch.testing.assert_close(u_x, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_poly_order2(self, poly_provider: AutogradProvider) -> None:
        """d²/dx² x^2 = 2 (constant)."""
        u = poly_provider.get_field("u")
        u_xx = poly_provider.diff(u, "x", order=2)
        expected = torch.full_like(u_xx, 2.0)
        torch.testing.assert_close(u_xx, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_preserves_shape(self, sin_provider: AutogradProvider) -> None:
        """diff result should have the same shape as the input expression."""
        u = sin_provider.get_field("u")
        u_x = sin_provider.diff(u, "x", order=1)
        assert u_x.shape == u.shape

    @pytest.mark.unit
    def test_diff_create_graph(self, sin_provider: AutogradProvider) -> None:
        """diff result should remain in the computation graph (supports higher-order).

        This tests that create_graph=True is used, so the result can be
        differentiated again.
        """
        u = sin_provider.get_field("u")
        u_x = sin_provider.diff(u, "x", order=1)
        # u_x should require grad (in computation graph)
        assert u_x.requires_grad, "diff result should remain in computation graph"

        # Should be able to differentiate u_x again manually
        u_xx = sin_provider.diff(u_x, "x", order=1)
        x = sin_provider.coords["x"]
        expected = -torch.sin(x)
        torch.testing.assert_close(u_xx, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_2d_partial_x(self, sin_product_provider: AutogradProvider) -> None:
        """d/dx [sin(x)*exp(-t)] = cos(x)*exp(-t)."""
        u = sin_product_provider.get_field("u")
        u_x = sin_product_provider.diff(u, "x", order=1)
        x = sin_product_provider.coords["x"]
        t = sin_product_provider.coords["t"]
        expected = torch.cos(x) * torch.exp(-t)
        torch.testing.assert_close(u_x, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_2d_partial_t(self, sin_product_provider: AutogradProvider) -> None:
        """d/dt [sin(x)*exp(-t)] = -sin(x)*exp(-t)."""
        u = sin_product_provider.get_field("u")
        u_t = sin_product_provider.diff(u, "t", order=1)
        x = sin_product_provider.coords["x"]
        t = sin_product_provider.coords["t"]
        expected = -torch.sin(x) * torch.exp(-t)
        torch.testing.assert_close(u_t, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_diff_invalid_axis(self, sin_provider: AutogradProvider) -> None:
        """diff with nonexistent axis should raise KeyError."""
        u = sin_provider.get_field("u")
        with pytest.raises(KeyError):
            sin_provider.diff(u, "nonexistent", order=1)

    @pytest.mark.unit
    def test_diff_invalid_order_zero(self, sin_provider: AutogradProvider) -> None:
        """diff with order=0 should raise ValueError."""
        u = sin_provider.get_field("u")
        with pytest.raises(ValueError, match="order"):
            sin_provider.diff(u, "x", order=0)

    @pytest.mark.unit
    def test_diff_invalid_order_negative(self, sin_provider: AutogradProvider) -> None:
        """diff with negative order should raise ValueError."""
        u = sin_provider.get_field("u")
        with pytest.raises(ValueError, match="order"):
            sin_provider.diff(u, "x", order=-1)

    @pytest.mark.unit
    def test_diff_open_form_expression(self, sin_provider: AutogradProvider) -> None:
        """diff should work on arbitrary expressions, not just model output.

        Test: d/dx (u * u) = 2u * u_x = 2*sin(x)*cos(x) = sin(2x)
        """
        u = sin_provider.get_field("u")
        expr = u * u # u^2 = sin^2(x)
        d_expr = sin_provider.diff(expr, "x", order=1)
        x = sin_provider.coords["x"]
        expected = torch.sin(2 * x) # 2*sin(x)*cos(x) = sin(2x)
        torch.testing.assert_close(d_expr, expected, rtol=1e-4, atol=1e-6)


# =============================================================================
# get_derivative Tests
# =============================================================================


class TestGetDerivative:
    """Tests for AutogradProvider.get_derivative method."""

    @pytest.mark.unit
    def test_get_derivative_matches_diff(self, sin_provider: AutogradProvider) -> None:
        """get_derivative should give same result as diff(get_field(...))."""
        result = sin_provider.get_derivative("u", "x", order=1)
        u = sin_provider.get_field("u")
        expected = sin_provider.diff(u, "x", order=1)
        torch.testing.assert_close(result, expected, rtol=1e-5, atol=1e-8)

    @pytest.mark.unit
    def test_get_derivative_sin_order1(self, sin_provider: AutogradProvider) -> None:
        """get_derivative('u', 'x', 1) = cos(x) for sin model."""
        result = sin_provider.get_derivative("u", "x", order=1)
        x = sin_provider.coords["x"]
        expected = torch.cos(x)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_get_derivative_sin_order2(self, sin_provider: AutogradProvider) -> None:
        """get_derivative('u', 'x', 2) = -sin(x) for sin model."""
        result = sin_provider.get_derivative("u", "x", order=2)
        x = sin_provider.coords["x"]
        expected = -torch.sin(x)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    def test_get_derivative_invalid_field(self, sin_provider: AutogradProvider) -> None:
        """get_derivative with invalid field should raise KeyError."""
        with pytest.raises(KeyError):
            sin_provider.get_derivative("nonexistent", "x", order=1)

    @pytest.mark.unit
    def test_get_derivative_invalid_axis(self, sin_provider: AutogradProvider) -> None:
        """get_derivative with invalid axis should raise KeyError."""
        with pytest.raises(KeyError):
            sin_provider.get_derivative("u", "nonexistent", order=1)

    @pytest.mark.unit
    def test_get_derivative_invalid_order(self, sin_provider: AutogradProvider) -> None:
        """get_derivative with order < 1 should raise ValueError."""
        with pytest.raises(ValueError):
            sin_provider.get_derivative("u", "x", order=0)
        with pytest.raises(ValueError):
            sin_provider.get_derivative("u", "x", order=-1)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "field,axis,order",
        [
            ("u", "x", 1),
            ("u", "x", 2),
            ("u", "t", 1),
        ],
    )
    def test_get_derivative_2d(
        self,
        sin_product_provider: AutogradProvider,
        field: str,
        axis: str,
        order: int,
    ) -> None:
        """get_derivative should work for 2D coords with different axes."""
        result = sin_product_provider.get_derivative(field, axis, order=order)
        assert isinstance(result, torch.Tensor)
        assert torch.isfinite(result).all()


# =============================================================================
# available_derivatives Tests
# =============================================================================


class TestAvailableDerivatives:
    """Tests for AutogradProvider.available_derivatives method."""

    @pytest.mark.unit
    def test_available_derivatives_returns_list(
        self, sin_provider: AutogradProvider
    ) -> None:
        """available_derivatives should return a list of tuples."""
        result = sin_provider.available_derivatives()
        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 3

    @pytest.mark.unit
    def test_available_derivatives_contains_expected(
        self, sin_provider: AutogradProvider
    ) -> None:
        """available_derivatives should contain expected (field, axis, order) tuples."""
        result = sin_provider.available_derivatives()
        # At minimum should include first-order derivatives of all fields along all axes
        assert ("u", "x", 1) in result

    @pytest.mark.unit
    def test_available_derivatives_2d(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        """2D provider should have derivatives along both axes."""
        result = sin_product_provider.available_derivatives()
        assert ("u", "x", 1) in result
        assert ("u", "t", 1) in result

    @pytest.mark.unit
    def test_available_derivatives_format(self, sin_provider: AutogradProvider) -> None:
        """Each tuple should be (str, str, int)."""
        result = sin_provider.available_derivatives()
        for field, axis, order in result:
            assert isinstance(field, str)
            assert isinstance(axis, str)
            assert isinstance(order, int)
            assert order >= 1


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrors:
    """Tests for error handling edge cases."""

    @pytest.mark.unit
    def test_diff_order_type_check(self, sin_provider: AutogradProvider) -> None:
        """diff with non-integer order should raise TypeError or ValueError."""
        u = sin_provider.get_field("u")
        with pytest.raises((TypeError, ValueError)):
            sin_provider.diff(u, "x", order=1.5) # type: ignore[arg-type]

    @pytest.mark.unit
    def test_get_field_lhs_field_valid(self, sin_provider: AutogradProvider) -> None:
        """get_field should work for the lhs_field defined in dataset."""
        lhs_field = sin_provider.dataset.lhs_field
        result = sin_provider.get_field(lhs_field)
        assert isinstance(result, torch.Tensor)


# =============================================================================
# Numerical Stability Tests
# =============================================================================


class TestNumericalStability:
    """Numerical stability tests for autograd derivatives."""

    @pytest.mark.numerical
    def test_diff_result_finite(self, sin_provider: AutogradProvider) -> None:
        """All derivative values should be finite (no NaN/Inf)."""
        u = sin_provider.get_field("u")
        for order in [1, 2, 3]:
            result = sin_provider.diff(u, "x", order=order)
            assert torch.isfinite(result).all(), (
                f"Order {order} derivative contains NaN/Inf"
            )

    @pytest.mark.numerical
    def test_diff_high_order_stability(self) -> None:
        """Higher-order derivatives should remain finite and accurate.

        sin(x) has a known cycle: derivatives repeat every 4 orders.
        Order 4: d^4/dx^4 sin(x) = sin(x). Needs max_order=4 at
        construction since the default is 3.
        """
        model = SinModel().double()
        coords = _make_coords_1d(50)
        dataset = _make_dataset_1d(50)
        provider = AutogradProvider(
            model=model, coords=coords, dataset=dataset, max_order=4
        )
        u = provider.get_field("u")
        u_4 = provider.diff(u, "x", order=4)
        x = provider.coords["x"]
        expected = torch.sin(x) # 4th derivative of sin = sin
        assert torch.isfinite(u_4).all(), "4th order derivative has NaN/Inf"
        torch.testing.assert_close(u_4, expected, rtol=1e-3, atol=1e-5)

    @pytest.mark.numerical
    def test_diff_dtype_preserved(self, sin_provider: AutogradProvider) -> None:
        """Derivative should preserve the dtype of the input."""
        u = sin_provider.get_field("u")
        u_x = sin_provider.diff(u, "x", order=1)
        assert u_x.dtype == u.dtype

    @pytest.mark.numerical
    def test_get_derivative_all_finite(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        """All precomputed derivatives should be finite."""
        for field, axis, order in sin_product_provider.available_derivatives():
            result = sin_product_provider.get_derivative(field, axis, order=order)
            assert torch.isfinite(result).all(), (
                f"Derivative ({field}, {axis}, {order}) contains NaN/Inf"
            )


# =============================================================================
# Parametrized Accuracy Tests
# =============================================================================


class TestAccuracyParametrized:
    """Parametrized accuracy tests comparing autograd vs analytical solutions."""

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "order,expected_fn",
        [
            (1, lambda x: torch.cos(x)),
            (2, lambda x: -torch.sin(x)),
            (3, lambda x: -torch.cos(x)),
        ],
        ids=["sin_order1", "sin_order2", "sin_order3"],
    )
    def test_sin_derivatives(
        self,
        sin_provider: AutogradProvider,
        order: int,
        expected_fn: object,
    ) -> None:
        """Verify sin(x) derivatives at multiple orders."""
        u = sin_provider.get_field("u")
        result = sin_provider.diff(u, "x", order=order)
        x = sin_provider.coords["x"]
        expected = expected_fn(x) # type: ignore[operator]
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)

    @pytest.mark.unit
    @pytest.mark.parametrize(
        "order,expected_fn",
        [
            (1, lambda x: 2 * x),
            (2, lambda x: torch.full_like(x, 2.0)),
        ],
        ids=["poly_order1", "poly_order2"],
    )
    def test_poly_derivatives(
        self,
        poly_provider: AutogradProvider,
        order: int,
        expected_fn: object,
    ) -> None:
        """Verify x^2 derivatives at multiple orders."""
        u = poly_provider.get_field("u")
        result = poly_provider.diff(u, "x", order=order)
        x = poly_provider.coords["x"]
        expected = expected_fn(x) # type: ignore[operator]
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)


# =============================================================================
# Review Fix Tests
# =============================================================================


class TestReviewFixes:
    """Tests for issues found during code review."""

    @pytest.mark.unit
    def test_diff_disconnected_graph_raises_value_error(
        self, sin_provider: AutogradProvider
    ) -> None:
        """diff on expression disconnected from coords should raise ValueError."""
        # Create a tensor with no connection to any coord
        disconnected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="not connected"):
            sin_provider.diff(disconnected, "x", order=1)

    @pytest.mark.unit
    def test_init_max_order_validation(self) -> None:
        """max_order < 1 should raise ValueError."""
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        with pytest.raises(ValueError, match="max_order"):
            AutogradProvider(model=model, coords=coords, dataset=dataset, max_order=0)

    @pytest.mark.unit
    def test_init_max_order_negative(self) -> None:
        """Negative max_order should raise ValueError."""
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        with pytest.raises(ValueError, match="max_order"):
            AutogradProvider(model=model, coords=coords, dataset=dataset, max_order=-1)

    @pytest.mark.unit
    def test_diff_rejects_order_above_max_order_default(
        self, sin_provider: AutogradProvider
    ) -> None:
        """diff with order > max_order (default=3) must raise ValueError.

        Without this guard, create_graph=True compounds the autograd graph
        for every order step — an unbounded order would cause exponential
        memory blow-up. Contract parity with FiniteDiffProvider.
        """
        u = sin_provider.get_field("u")
        with pytest.raises(ValueError, match=r"exceeds max_order"):
            sin_provider.diff(u, "x", order=4)

    @pytest.mark.unit
    def test_diff_rejects_order_above_custom_max_order(self) -> None:
        """Custom max_order should be enforced by diff()."""
        model = SinModel().double()
        coords = _make_coords_1d()
        dataset = _make_dataset_1d()
        provider = AutogradProvider(
            model=model, coords=coords, dataset=dataset, max_order=1
        )
        u = provider.get_field("u")
        with pytest.raises(ValueError, match=r"exceeds max_order 1"):
            provider.diff(u, "x", order=2)

    @pytest.mark.unit
    def test_diff_at_max_order_still_accepted(
        self, sin_provider: AutogradProvider
    ) -> None:
        """Boundary: order == max_order is valid (no off-by-one)."""
        u = sin_provider.get_field("u")
        result = sin_provider.diff(u, "x", order=3)
        assert result.shape == u.shape
