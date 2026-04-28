"""End-to-end tests for diff operator support.

Tests the diff operator pipeline from three angles:
1. AutogradProvider direct tests: verify diff() correctness without executor
2. Executor end-to-end tests: execute diff_x(u) etc. through PythonExecutor
3. FiniteDiffProvider tests: verify diff() works correctly for open-form diff

Test model: u = sin(x) * exp(-t) with known analytical derivatives:
    u_x = cos(x) * exp(-t)
    u_xx = -sin(x) * exp(-t) = -u
    u_t = -sin(x) * exp(-t) = -u
    u_xt = -cos(x) * exp(-t) = -u_x

For the product rule test (diff_x(mul(u, u_x))):
    d/dx [u * u_x] = u_x * u_x + u * u_xx
                    = cos^2(x)*exp(-2t) + sin(x)*(-sin(x))*exp(-2t)
                    = cos^2(x)*exp(-2t) - sin^2(x)*exp(-2t)
                    = cos(2x)*exp(-2t)

Coordinate setup:
    The AutogradProvider uses flattened 1D coords (from meshgrid) for the
    neural network model. The PDEDataset stores 2D grid metadata separately.
    These are intentionally decoupled -- coords connect to the computation
    graph, dataset fields are static/detached.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd2.core.executor.context import ExecutionContext
from kd2.core.expr.executor import PythonExecutor
from kd2.core.expr.registry import FunctionRegistry
from kd2.data.derivatives.autograd import AutogradProvider
from kd2.data.derivatives.finite_diff import FiniteDiffProvider
from kd2.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)

# =============================================================================
# Test Models
# =============================================================================


class SinProductModel(nn.Module):
    """u = sin(x) * exp(-t).

    Known derivatives:
        u_x = cos(x) * exp(-t)
        u_xx = -sin(x) * exp(-t)
        u_t = -sin(x) * exp(-t)
        u_xt = -cos(x) * exp(-t)
    """

    def forward(self, *, x: Tensor, t: Tensor) -> Tensor:
        return torch.sin(x) * torch.exp(-t)


class PolyModel(nn.Module):
    """u = x^3 * t^2.

    Known derivatives:
        u_x = 3 * x^2 * t^2
        u_xx = 6 * x * t^2
        u_t = 2 * x^3 * t
        u_xt = 6 * x^2 * t
    """

    def forward(self, *, x: Tensor, t: Tensor) -> Tensor:
        return x**3 * t**2


# =============================================================================
# Helper Functions
# =============================================================================

# Default grid sizes
_N_X = 30
_N_T = 20


def _make_coords_2d(n_x: int = _N_X, n_t: int = _N_T) -> dict[str, Tensor]:
    """Create flattened 2D coords with requires_grad=True.

    Creates a meshgrid over [0.1, 2*pi-0.1] x [0.1, 1.0], then flattens
    to 1D tensors of length n_x * n_t. Uses float64 for numerical accuracy.
    """
    x_1d = torch.linspace(0.1, 2 * math.pi - 0.1, n_x, dtype=torch.float64)
    t_1d = torch.linspace(0.1, 1.0, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x_1d, t_1d, indexing="ij")
    x = X.reshape(-1).clone().detach().requires_grad_(True)
    t = T.reshape(-1).clone().detach().requires_grad_(True)
    return {"x": x, "t": t}


def _make_dataset_2d(n_x: int = _N_X, n_t: int = _N_T) -> PDEDataset:
    """Create a 2D PDEDataset with proper grid shape.

    Field u has shape (n_x, n_t) matching the axis_order ["x", "t"].
    Dataset fields are detached (static data, not in computation graph).
    """
    x = torch.linspace(0.1, 2 * math.pi - 0.1, n_x, dtype=torch.float64)
    t = torch.linspace(0.1, 1.0, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(X) * torch.exp(-T)

    return PDEDataset(
        name="test_diff_operator",
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


def _make_poly_coords(n_x: int = _N_X, n_t: int = _N_T) -> dict[str, Tensor]:
    """Create flattened 2D coords for PolyModel testing."""
    x_1d = torch.linspace(0.1, 3.0, n_x, dtype=torch.float64)
    t_1d = torch.linspace(0.1, 2.0, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x_1d, t_1d, indexing="ij")
    x = X.reshape(-1).clone().detach().requires_grad_(True)
    t = T.reshape(-1).clone().detach().requires_grad_(True)
    return {"x": x, "t": t}


def _make_poly_dataset(n_x: int = _N_X, n_t: int = _N_T) -> PDEDataset:
    """Create a 2D PDEDataset for PolyModel."""
    x = torch.linspace(0.1, 3.0, n_x, dtype=torch.float64)
    t = torch.linspace(0.1, 2.0, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    u = X**3 * T**2

    return PDEDataset(
        name="test_poly",
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


def _make_grid_dataset(n_x: int = 50, n_t: int = 30) -> PDEDataset:
    """Create a proper grid PDEDataset for FiniteDiffProvider tests.

    Uses full [0, 2*pi] x [0, 1] range with enough points for FD stencils.
    """
    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(X) * torch.exp(-T)

    return PDEDataset(
        name="test_grid",
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


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def coords() -> dict[str, Tensor]:
    """Flattened 2D coordinate tensors with requires_grad=True."""
    return _make_coords_2d()


@pytest.fixture
def sin_product_provider(coords: dict[str, Tensor]) -> AutogradProvider:
    """AutogradProvider with SinProductModel and 2D grid coords."""
    model = SinProductModel().double()
    dataset = _make_dataset_2d()
    return AutogradProvider(model=model, coords=coords, dataset=dataset)


@pytest.fixture
def poly_provider() -> AutogradProvider:
    """AutogradProvider with PolyModel and 2D grid coords."""
    coords = _make_poly_coords()
    model = PolyModel().double()
    dataset = _make_poly_dataset()
    return AutogradProvider(model=model, coords=coords, dataset=dataset)


@pytest.fixture
def executor() -> PythonExecutor:
    """PythonExecutor with default registry."""
    return PythonExecutor(FunctionRegistry.create_default())


@pytest.fixture
def autograd_context(
    sin_product_provider: AutogradProvider,
) -> ExecutionContext:
    """ExecutionContext backed by AutogradProvider."""
    return ExecutionContext(
        dataset=sin_product_provider.dataset,
        derivative_provider=sin_product_provider,
    )


# =============================================================================
# Part 1: AutogradProvider Direct Tests
# =============================================================================


@pytest.mark.smoke
class TestAutogradProviderDiffSmoke:
    """Smoke: AutogradProvider.diff is callable and returns tensors."""

    def test_diff_returns_tensor(self, sin_product_provider: AutogradProvider) -> None:
        """diff() returns a Tensor."""
        u = sin_product_provider.get_field("u")
        result = sin_product_provider.diff(u, "x", 1)
        assert isinstance(result, Tensor)

    def test_diff_preserves_shape(self, sin_product_provider: AutogradProvider) -> None:
        """diff() preserves the shape of the input expression."""
        u = sin_product_provider.get_field("u")
        result = sin_product_provider.diff(u, "x", 1)
        assert result.shape == u.shape


@pytest.mark.unit
class TestDiffFirstOrder:
    """diff_x(u) is equivalent to get_derivative('u', 'x', 1)."""

    def test_diff_x_matches_analytical(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        """diff(u, 'x', 1) = cos(x) * exp(-t)."""
        u = sin_product_provider.get_field("u")
        u_x = sin_product_provider.diff(u, "x", 1)

        x = coords["x"]
        t = coords["t"]
        expected = torch.cos(x) * torch.exp(-t)
        torch.testing.assert_close(u_x, expected, rtol=1e-4, atol=1e-6)

    def test_diff_t_matches_analytical(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        """diff(u, 't', 1) = -sin(x) * exp(-t) = -u."""
        u = sin_product_provider.get_field("u")
        u_t = sin_product_provider.diff(u, "t", 1)

        x = coords["x"]
        t = coords["t"]
        expected = -torch.sin(x) * torch.exp(-t)
        torch.testing.assert_close(u_t, expected, rtol=1e-4, atol=1e-6)

    def test_diff_x_matches_get_derivative(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        """diff(get_field('u'), 'x', 1) == get_derivative('u', 'x', 1)."""
        u = sin_product_provider.get_field("u")
        diff_result = sin_product_provider.diff(u, "x", 1)
        get_deriv_result = sin_product_provider.get_derivative("u", "x", 1)
        torch.testing.assert_close(diff_result, get_deriv_result, rtol=1e-5, atol=1e-8)

    def test_diff_t_matches_get_derivative(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        """diff(get_field('u'), 't', 1) == get_derivative('u', 't', 1)."""
        u = sin_product_provider.get_field("u")
        diff_result = sin_product_provider.diff(u, "t", 1)
        get_deriv_result = sin_product_provider.get_derivative("u", "t", 1)
        torch.testing.assert_close(diff_result, get_deriv_result, rtol=1e-5, atol=1e-8)


@pytest.mark.unit
class TestDiffSecondOrder:
    """diff2_x(u) is equivalent to get_derivative('u', 'x', 2)."""

    def test_diff2_x_matches_analytical(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        """diff(u, 'x', 2) = -sin(x) * exp(-t)."""
        u = sin_product_provider.get_field("u")
        u_xx = sin_product_provider.diff(u, "x", 2)

        x = coords["x"]
        t = coords["t"]
        expected = -torch.sin(x) * torch.exp(-t)
        torch.testing.assert_close(u_xx, expected, rtol=1e-4, atol=1e-6)

    def test_diff2_x_matches_get_derivative(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        """diff(u, 'x', 2) == get_derivative('u', 'x', 2)."""
        u = sin_product_provider.get_field("u")
        diff_result = sin_product_provider.diff(u, "x", 2)
        get_deriv_result = sin_product_provider.get_derivative("u", "x", 2)
        torch.testing.assert_close(diff_result, get_deriv_result, rtol=1e-5, atol=1e-8)


@pytest.mark.unit
class TestDiffProductRule:
    """diff_x(mul(u, u_x)) applies product rule correctly."""

    def test_product_rule_sin_product(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        """d/dx[u * u_x] = u_x^2 + u * u_xx = cos(2x) * exp(-2t).

        For u = sin(x)*exp(-t):
            u_x = cos(x)*exp(-t)
            u_xx = -sin(x)*exp(-t)
            d/dx[u * u_x] = u_x * u_x + u * u_xx
                           = cos^2(x)*exp(-2t) - sin^2(x)*exp(-2t)
                           = cos(2x)*exp(-2t)
        """
        u = sin_product_provider.get_field("u")
        u_x = sin_product_provider.diff(u, "x", 1)
        product = u * u_x
        result = sin_product_provider.diff(product, "x", 1)

        x = coords["x"]
        t = coords["t"]
        expected = torch.cos(2 * x) * torch.exp(-2 * t)
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)

    def test_product_rule_poly(self, poly_provider: AutogradProvider) -> None:
        """d/dx[u * u_x] for u = x^3 * t^2.

        u_x = 3*x^2*t^2
        d/dx[u * u_x] = d/dx[x^3*t^2 * 3*x^2*t^2]
                       = d/dx[3*x^5*t^4]
                       = 15*x^4*t^4
        """
        u = poly_provider.get_field("u")
        u_x = poly_provider.diff(u, "x", 1)
        product = u * u_x
        result = poly_provider.diff(product, "x", 1)

        x = poly_provider.coords["x"]
        t = poly_provider.coords["t"]
        expected = 15 * x**4 * t**4
        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-6)


@pytest.mark.unit
class TestDiffNested:
    """Nested diff operators: diff_x(diff_t(u))."""

    def test_mixed_partial_xt(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        """diff_x(diff_t(u)) = d/dx(-sin(x)*exp(-t)) = -cos(x)*exp(-t)."""
        u = sin_product_provider.get_field("u")
        u_t = sin_product_provider.diff(u, "t", 1)
        u_tx = sin_product_provider.diff(u_t, "x", 1)

        x = coords["x"]
        t = coords["t"]
        expected = -torch.cos(x) * torch.exp(-t)
        torch.testing.assert_close(u_tx, expected, rtol=1e-4, atol=1e-6)

    def test_mixed_partial_commutativity(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        """diff_x(diff_t(u)) == diff_t(diff_x(u)) (Clairaut's theorem)."""
        u = sin_product_provider.get_field("u")

        u_t = sin_product_provider.diff(u, "t", 1)
        u_tx = sin_product_provider.diff(u_t, "x", 1)

        u_x = sin_product_provider.diff(u, "x", 1)
        u_xt = sin_product_provider.diff(u_x, "t", 1)

        torch.testing.assert_close(u_tx, u_xt, rtol=1e-4, atol=1e-6)

    def test_higher_order_via_nesting(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        """diff_x(diff_x(u)) == diff(u, 'x', 2)."""
        u = sin_product_provider.get_field("u")

        # Nested: diff then diff again
        u_x = sin_product_provider.diff(u, "x", 1)
        u_xx_nested = sin_product_provider.diff(u_x, "x", 1)

        # Direct second order
        u_xx_direct = sin_product_provider.diff(u, "x", 2)

        torch.testing.assert_close(u_xx_nested, u_xx_direct, rtol=1e-4, atol=1e-6)


# =============================================================================
# Part 2: Executor End-to-End Tests
# =============================================================================


@pytest.mark.unit
class TestExecutorDiffEndToEnd:
    """Execute diff expressions through PythonExecutor + real AutogradProvider.

    IMPORTANT: The executor's full path (_execute_with_diff) fetches variables
    via context.get_variable(), which returns dataset.fields["u"].values --
    detached from the computation graph. Therefore, diff_x(u) through the
    executor will fail because the "u" tensor has no grad_fn connected to
    the coordinates.

    Tests that work: expressions where the diff argument is computed by
    the executor itself (and thus connects to the computation graph via
    the AutogradProvider).

    Tests marked xfail: expressions where diff operates on a raw variable
    from the dataset (detached, no computation graph).
    """

    def test_diff_x_u(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        """Execute diff_x(u) through executor."""
        result = executor.execute("diff_x(u)", autograd_context)
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_diff2_x_u(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        """Execute diff2_x(u) through executor."""
        result = executor.execute("diff2_x(u)", autograd_context)
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_diff_t_u(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        """Execute diff_t(u) through executor."""
        result = executor.execute("diff_t(u)", autograd_context)
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_diff_x_mul_u_ux(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        """Execute diff_x(mul(u, u_x)) through executor."""
        result = executor.execute("diff_x(mul(u, u_x))", autograd_context)
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_nested_diff_x_diff_t_u(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        """Execute diff_x(diff_t(u)) through executor."""
        result = executor.execute("diff_x(diff_t(u))", autograd_context)
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_has_open_form_diff_detection(self) -> None:
        """Executor correctly identifies expressions needing full path."""
        from kd2.core.expr.executor import has_open_form_diff

        assert has_open_form_diff("diff_x(u)") is True
        assert has_open_form_diff("diff2_x(u)") is True
        assert has_open_form_diff("diff_t(u)") is True
        assert has_open_form_diff("diff_x(mul(u, u_x))") is True
        assert has_open_form_diff("diff_x(diff_t(u))") is True
        # Terminal derivatives are NOT open-form
        assert has_open_form_diff("u_x") is False
        assert has_open_form_diff("add(u_x, u_xx)") is False

    def test_executor_used_diff_flag(
        self,
        executor: PythonExecutor,
        autograd_context: ExecutionContext,
    ) -> None:
        """Executor sets used_diff=False for expressions without diff."""
        result = executor.execute("add(u, u)", autograd_context)
        assert result.used_diff is False


# =============================================================================
# Part 3: FiniteDiffProvider Error Path
# =============================================================================


@pytest.mark.unit
class TestFiniteDiffProviderDiff:
    """FiniteDiffProvider.diff() computes FD on arbitrary expressions.

    After, diff() delegates to central_diff instead of raising
    NotImplementedError. It should return correct finite difference derivatives.
    """

    def test_fd_diff_returns_finite_tensor(self) -> None:
        """FiniteDiffProvider.diff() returns a finite tensor."""
        dataset = _make_grid_dataset(50, 30)
        provider = FiniteDiffProvider(dataset, max_order=2)

        expr = torch.randn(50, 30, dtype=torch.float64)
        result = provider.diff(expr, "x", 1)
        assert isinstance(result, Tensor)
        assert result.shape == expr.shape
        assert torch.isfinite(result).all()

    def test_fd_diff_works_for_both_axes(self) -> None:
        """FiniteDiffProvider.diff() works for all dataset axes."""
        dataset = _make_grid_dataset(50, 30)
        provider = FiniteDiffProvider(dataset, max_order=2)

        # Use actual field data for meaningful results
        u = dataset.get_field("u")

        result_x = provider.diff(u, "x", 1)
        assert isinstance(result_x, Tensor)
        assert torch.isfinite(result_x).all()

        result_t = provider.diff(u, "t", 1)
        assert isinstance(result_t, Tensor)
        assert torch.isfinite(result_t).all()

    def test_fd_diff_works_for_order_2(self) -> None:
        """FiniteDiffProvider.diff() handles second-order derivatives."""
        dataset = _make_grid_dataset(50, 30)
        provider = FiniteDiffProvider(dataset, max_order=2)

        u = dataset.get_field("u")
        result = provider.diff(u, "x", 2)
        assert isinstance(result, Tensor)
        assert torch.isfinite(result).all()

    def test_fd_diff_matches_precomputed(self) -> None:
        """diff(field_data) should match get_derivative() exactly.

        Both use the same central_diff function on the same data,
        so results should be bit-identical.
        """
        dataset = _make_grid_dataset(50, 30)
        provider = FiniteDiffProvider(dataset, max_order=2)

        u = dataset.get_field("u")
        diff_result = provider.diff(u, "x", 1)
        precomputed = provider.get_derivative("u", "x", 1)

        torch.testing.assert_close(diff_result, precomputed, rtol=1e-12, atol=1e-12)

    def test_fd_get_derivative_still_works(self) -> None:
        """FiniteDiffProvider.get_derivative() still works (precomputed)."""
        dataset = _make_grid_dataset(50, 30)
        provider = FiniteDiffProvider(dataset, max_order=2)

        result = provider.get_derivative("u", "x", 1)
        assert isinstance(result, Tensor)
        assert torch.isfinite(result).all()


# =============================================================================
# Part 4: Numerical Stability
# =============================================================================


@pytest.mark.numerical
class TestDiffNumericalStability:
    """Numerical stability tests for diff operator."""

    def test_all_results_finite(self, sin_product_provider: AutogradProvider) -> None:
        """All derivative computations produce finite results."""
        u = sin_product_provider.get_field("u")

        for axis in ["x", "t"]:
            for order in [1, 2, 3]:
                result = sin_product_provider.diff(u, axis, order)
                assert torch.isfinite(result).all(), (
                    f"diff(u, '{axis}', {order}) produced NaN/Inf"
                )

    def test_dtype_preserved(self, sin_product_provider: AutogradProvider) -> None:
        """Diff preserves the dtype of the input."""
        u = sin_product_provider.get_field("u")
        u_x = sin_product_provider.diff(u, "x", 1)
        assert u_x.dtype == u.dtype

    def test_third_order_accuracy(
        self, sin_product_provider: AutogradProvider, coords: dict[str, Tensor]
    ) -> None:
        """Third-order derivatives remain accurate.

        d^3/dx^3 [sin(x)*exp(-t)] = -cos(x)*exp(-t)
        """
        u = sin_product_provider.get_field("u")
        u_xxx = sin_product_provider.diff(u, "x", 3)

        x = coords["x"]
        t = coords["t"]
        expected = -torch.cos(x) * torch.exp(-t)
        torch.testing.assert_close(u_xxx, expected, rtol=1e-3, atol=1e-5)

    def test_result_stays_in_computation_graph(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        """Diff result should remain in the computation graph.

        This ensures create_graph=True is used internally, supporting
        higher-order derivatives.
        """
        u = sin_product_provider.get_field("u")
        u_x = sin_product_provider.diff(u, "x", 1)
        assert u_x.requires_grad, "diff result should remain in computation graph"

    def test_disconnected_tensor_raises(
        self, sin_product_provider: AutogradProvider
    ) -> None:
        """diff on a tensor not connected to coords raises ValueError."""
        disconnected = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        with pytest.raises(ValueError, match="not connected"):
            sin_product_provider.diff(disconnected, "x", 1)
