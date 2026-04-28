"""Tests for ExecutionContext.

Test coverage:
- smoke: Basic instantiation and interface existence
- unit: Context variable/derivative/constant lookup

Note: This module tests only ExecutionContext from context.py.
TreeExecutor tests were removed as part of IR refactor (Task 012.6).
"""

from __future__ import annotations

import math
from unittest.mock import Mock

import pytest
import torch

from kd2.core.executor import ExecutionContext
from kd2.data import PDEDataset
from kd2.data.derivatives import FiniteDiffProvider
from kd2.data.schema import AxisInfo, DataTopology, FieldData, TaskType

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_2d_dataset() -> PDEDataset:
    """2D grid dataset (x, t) for context testing.

    Creates a dataset with:
    - x: 32 points in [0, 2*pi]
    - t: 16 points in [0, 1]
    - Field u = sin(x) * exp(-t)
    - Field v = cos(x) * exp(-t)
    """
    n_x = 32
    n_t = 16

    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)

    # Create 2D meshgrid: shape (n_x, n_t)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(xx) * torch.exp(-tt)
    v = torch.cos(xx) * torch.exp(-tt)

    return PDEDataset(
        name="test_2d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={
            "u": FieldData(name="u", values=u),
            "v": FieldData(name="v", values=v),
        },
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def derivative_provider(simple_2d_dataset: PDEDataset) -> FiniteDiffProvider:
    """Finite difference provider for the 2D dataset."""
    return FiniteDiffProvider(simple_2d_dataset, max_order=3)


@pytest.fixture
def execution_context(
    simple_2d_dataset: PDEDataset,
    derivative_provider: FiniteDiffProvider,
) -> ExecutionContext:
    """Execution context for testing."""
    return ExecutionContext(
        dataset=simple_2d_dataset,
        derivative_provider=derivative_provider,
        constants={"pi": math.pi, "nu": 0.1},
    )


# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.smoke
class TestContextSmoke:
    """Smoke tests: basic instantiation and interface existence."""

    def test_execution_context_can_be_created(
        self, execution_context: ExecutionContext
    ) -> None:
        """ExecutionContext can be instantiated."""
        assert execution_context is not None
        assert execution_context.dataset is not None
        assert execution_context.derivative_provider is not None


# =============================================================================
# Unit Tests - ExecutionContext Variables
# =============================================================================


@pytest.mark.unit
class TestExecutionContextVariables:
    """Unit tests for ExecutionContext variable access."""

    def test_get_field_variable(self, execution_context: ExecutionContext) -> None:
        """get_variable returns field values for field names."""
        u = execution_context.get_variable("u")
        assert u is not None
        assert u.shape == (32, 16)
        # Verify it's the correct field (sin(x)*exp(-t) at x=0, t=0 is 0)
        torch.testing.assert_close(
            u[0, 0],
            torch.tensor(0.0, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_get_second_field(self, execution_context: ExecutionContext) -> None:
        """get_variable works for multiple fields."""
        v = execution_context.get_variable("v")
        assert v is not None
        assert v.shape == (32, 16)
        # cos(0)*exp(0) = 1
        torch.testing.assert_close(
            v[0, 0],
            torch.tensor(1.0, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )

    def test_get_coordinate_x(self, execution_context: ExecutionContext) -> None:
        """get_variable returns coordinate values for axis names."""
        x = execution_context.get_variable("x")
        assert x is not None
        # x should broadcast to (32, 16)
        assert x.shape == (32, 16)

    def test_get_coordinate_t(self, execution_context: ExecutionContext) -> None:
        """get_variable returns t coordinate values."""
        t = execution_context.get_variable("t")
        assert t is not None
        # t should broadcast to (32, 16)
        assert t.shape == (32, 16)

    def test_get_variable_not_found(self, execution_context: ExecutionContext) -> None:
        """get_variable raises KeyError for unknown variables."""
        with pytest.raises(KeyError):
            execution_context.get_variable("nonexistent")


# =============================================================================
# Unit Tests - ExecutionContext Derivatives
# =============================================================================


@pytest.mark.unit
class TestExecutionContextDerivatives:
    """Unit tests for ExecutionContext derivative access."""

    def test_get_first_derivative(self, execution_context: ExecutionContext) -> None:
        """get_derivative returns first-order derivative."""
        u_x = execution_context.get_derivative("u", "x", 1)
        assert u_x is not None
        assert u_x.shape == (32, 16)

    def test_get_second_derivative(self, execution_context: ExecutionContext) -> None:
        """get_derivative returns second-order derivative."""
        u_xx = execution_context.get_derivative("u", "x", 2)
        assert u_xx is not None
        assert u_xx.shape == (32, 16)

    def test_get_time_derivative(self, execution_context: ExecutionContext) -> None:
        """get_derivative returns time derivative."""
        u_t = execution_context.get_derivative("u", "t", 1)
        assert u_t is not None
        assert u_t.shape == (32, 16)

    def test_get_derivative_field_not_found(
        self, execution_context: ExecutionContext
    ) -> None:
        """get_derivative raises KeyError for unknown field."""
        with pytest.raises(KeyError):
            execution_context.get_derivative("w", "x", 1)

    def test_get_derivative_axis_not_found(
        self, execution_context: ExecutionContext
    ) -> None:
        """get_derivative raises KeyError for unknown axis."""
        with pytest.raises(KeyError):
            execution_context.get_derivative("u", "y", 1)

    def test_get_derivative_order_zero_raises(
        self, execution_context: ExecutionContext
    ) -> None:
        """get_derivative raises ValueError for order=0."""
        with pytest.raises(ValueError, match="order must be >= 1"):
            execution_context.get_derivative("u", "x", 0)

    def test_get_derivative_negative_order_raises(
        self, execution_context: ExecutionContext
    ) -> None:
        """get_derivative raises ValueError for negative order."""
        with pytest.raises(ValueError, match="order must be >= 1"):
            execution_context.get_derivative("u", "x", -1)

    def test_get_derivative_order_exceeds_max_raises(
        self, execution_context: ExecutionContext
    ) -> None:
        """get_derivative raises ValueError when order exceeds max_order.

        The fixture creates FiniteDiffProvider with max_order=3,
        so order=4 should raise ValueError.
        """
        with pytest.raises(ValueError, match="exceeds max_order"):
            execution_context.get_derivative("u", "x", 4)


# =============================================================================
# Unit Tests - ExecutionContext Constants
# =============================================================================


@pytest.mark.unit
class TestExecutionContextConstants:
    """Unit tests for ExecutionContext constant access."""

    def test_get_constant(self, execution_context: ExecutionContext) -> None:
        """get_constant returns named constant value."""
        pi = execution_context.get_constant("pi")
        assert abs(pi - math.pi) < 1e-10

    def test_get_constant_nu(self, execution_context: ExecutionContext) -> None:
        """get_constant returns nu constant."""
        nu = execution_context.get_constant("nu")
        assert abs(nu - 0.1) < 1e-10

    def test_get_constant_not_found(
        self, execution_context: ExecutionContext
    ) -> None:
        """get_constant raises KeyError for unknown constant."""
        with pytest.raises(KeyError):
            execution_context.get_constant("nonexistent")


# =============================================================================
# Unit Tests - Spatial Axes
# =============================================================================


@pytest.mark.unit
class TestExecutionContextSpatialAxes:
    """Unit tests for spatial axis derivation from dataset metadata."""

    def test_1d_pde_excludes_lhs_axis(
        self, execution_context: ExecutionContext
    ) -> None:
        """axis_order=['x', 't'] and lhs_axis='t' derives ['x']."""
        assert execution_context.spatial_axes == ["x"]

    @pytest.mark.parametrize(
        ("axis_order", "expected"),
        [
            (["x", "y", "t"], ["x", "y"]),
            (["t", "x", "y", "z"], ["x", "y", "z"]),
        ],
    )
    def test_multidimensional_pde_preserves_axis_order(
        self,
        axis_order: list[str],
        expected: list[str],
    ) -> None:
        """Spatial axes preserve dataset.axis_order after excluding lhs_axis."""
        dataset = PDEDataset(
            name="test_spatial_axes",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes=None,
            axis_order=axis_order,
            fields=None,
            lhs_field="",
            lhs_axis="t",
        )
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        assert context.spatial_axes == expected

    def test_empty_lhs_axis_returns_empty_list(self) -> None:
        """Missing lhs_axis means spatial axes are undefined at context level."""
        dataset = PDEDataset(
            name="test_empty_lhs_axis",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes=None,
            axis_order=["x", "t"],
            fields=None,
            lhs_field="",
            lhs_axis="",
        )
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        assert context.spatial_axes == []

    def test_missing_axis_order_returns_empty_list(self) -> None:
        """Missing axis_order is a valid regression/incomplete-data fallback."""
        dataset = PDEDataset(
            name="test_missing_axis_order",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes=None,
            axis_order=None,
            fields=None,
            lhs_field="",
            lhs_axis="t",
        )
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        assert context.spatial_axes == []

    def test_scattered_mode_uses_axis_order_without_axes_or_fields(self) -> None:
        """SCATTERED datasets can expose axis_order without axes or fields."""
        dataset = PDEDataset(
            name="test_scattered_spatial_axes",
            task_type=TaskType.PDE,
            topology=DataTopology.SCATTERED,
            axes=None,
            axis_order=["x", "y", "t"],
            fields=None,
            lhs_field="",
            lhs_axis="t",
        )
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        assert context.spatial_axes == ["x", "y"]

    def test_spatial_axes_is_derived_on_each_access(self) -> None:
        """Mutating dataset.lhs_axis is reflected by the next property access."""
        dataset = PDEDataset(
            name="test_spatial_axes_not_cached",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes=None,
            axis_order=["x", "y", "t"],
            fields=None,
            lhs_field="",
            lhs_axis="t",
        )
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        assert context.spatial_axes == ["x", "y"]
        dataset.lhs_axis = "y"
        assert context.spatial_axes == ["x", "t"]


# =============================================================================
# Unit Tests - Coordinate Broadcasting
# =============================================================================


@pytest.mark.unit
class TestCoordinateBroadcasting:
    """Unit tests for coordinate broadcasting behavior."""

    def test_x_coordinate_broadcasts_correctly(
        self, execution_context: ExecutionContext
    ) -> None:
        """x coordinate broadcasts along x-axis (varies across first dim)."""
        x = execution_context.get_variable("x")
        # Each row should have constant x value
        assert x.shape == (32, 16)
        # First row: all values should be 0
        torch.testing.assert_close(
            x[0,:],
            torch.full((16,), 0.0, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        # x varies across first dimension
        assert x[1, 0] > x[0, 0]

    def test_t_coordinate_broadcasts_correctly(
        self, execution_context: ExecutionContext
    ) -> None:
        """t coordinate broadcasts along t-axis (varies across second dim)."""
        t = execution_context.get_variable("t")
        # Each column should have constant t value
        assert t.shape == (32, 16)
        # First column: all values should be 0
        torch.testing.assert_close(
            t[:, 0],
            torch.full((32,), 0.0, dtype=torch.float64),
            atol=1e-10,
            rtol=1e-10,
        )
        # t varies across second dimension
        assert t[0, 1] > t[0, 0]


# =============================================================================
# Unit Tests - Edge Cases
# =============================================================================


@pytest.mark.unit
class TestContextEdgeCases:
    """Edge case tests for ExecutionContext."""

    def test_context_with_empty_constants(
        self,
        simple_2d_dataset: PDEDataset,
        derivative_provider: FiniteDiffProvider,
    ) -> None:
        """Context works with empty constants dict."""
        context = ExecutionContext(
            dataset=simple_2d_dataset,
            derivative_provider=derivative_provider,
            constants={},
        )
        # Should raise KeyError for any constant
        with pytest.raises(KeyError):
            context.get_constant("pi")

    def test_context_without_axis_order_raises(self) -> None:
        """Context with missing axis_order raises on coordinate access."""
        x = torch.linspace(0, 1, 10, dtype=torch.float64)
        u = torch.randn(10, dtype=torch.float64)

        dataset = PDEDataset(
            name="test_missing_axis_order",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x)},
            axis_order=None, # Missing axis_order
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="",
        )

        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=Mock(),
            constants={},
        )

        # Coordinate access should raise ValueError
        with pytest.raises(ValueError, match="axis_order"):
            context.get_variable("x")
