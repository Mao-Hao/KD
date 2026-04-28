"""Unit tests for kd2.core.integrator — TDD red phase.

Tests interface contracts, dataclass construction, input validation,
and edge cases for the PDE integrator.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy
import torch

from kd2.core.expr.naming import parse_compound_derivative
from kd2.core.expr.sympy_bridge import to_sympy
from kd2.core.integrator import (
    IntegrationResult,
    _finite_diff,
    _mol_rhs,
    _ParsedSymbols,
    _SpatialAxisInfo,
    integrate_pde,
)
from kd2.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)

# Fixtures


@pytest.fixture
def simple_grid_dataset() -> PDEDataset:
    """Minimal 1D grid dataset (x, t) with u field."""
    nx, nt = 32, 10
    x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.sin(x).unsqueeze(-1).expand(nx, nt).clone()
    return PDEDataset(
        name="test-simple",
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


@pytest.fixture
def scattered_dataset() -> PDEDataset:
    """Scattered topology dataset — integrator should reject."""
    nx, nt = 16, 5
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    u = torch.randn(nx, nt, dtype=torch.float64)
    return PDEDataset(
        name="test-scattered",
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
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
def mixed_partial_dataset() -> PDEDataset:
    """2D periodic dataset for u_t = u_xy with analytic solution exp(-t)sin(x+y)."""
    nx, ny, nt = 16, 16, 9
    x = torch.arange(nx, dtype=torch.float64) * (2 * torch.pi / nx)
    y = torch.arange(ny, dtype=torch.float64) * (2 * torch.pi / ny)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    x_grid, y_grid, t_grid = torch.meshgrid(x, y, t, indexing="ij")
    u = torch.sin(x_grid + y_grid) * torch.exp(-t_grid)
    return PDEDataset(
        name="test-mixed-partial",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "y": AxisInfo(name="y", values=y, is_periodic=True),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


# IntegrationResult dataclass


class TestIntegrationResult:
    """Tests for the IntegrationResult dataclass."""

    @pytest.mark.smoke
    def test_success_result_construction(self) -> None:
        """Successful result holds a tensor and has correct defaults."""
        field = torch.randn(10, 5)
        result = IntegrationResult(success=True, predicted_field=field)
        assert result.success is True
        assert result.predicted_field is not None
        assert result.warning == ""
        assert result.diverged_at_t is None

    @pytest.mark.smoke
    def test_failure_result_construction(self) -> None:
        """Failed result carries warning and optional divergence time."""
        result = IntegrationResult(
            success=False,
            predicted_field=None,
            warning="NaN detected",
            diverged_at_t=0.42,
        )
        assert result.success is False
        assert result.predicted_field is None
        assert "NaN" in result.warning
        assert result.diverged_at_t == pytest.approx(0.42)


# integrate_pde — smoke / interface


class TestIntegratePdeSmoke:
    """Smoke tests: function is callable with the documented signature."""

    @pytest.mark.smoke
    def test_callable(self) -> None:
        """integrate_pde exists and is callable."""
        assert callable(integrate_pde)

    @pytest.mark.smoke
    def test_returns_integration_result(self, simple_grid_dataset: PDEDataset) -> None:
        """Return type is IntegrationResult (once implemented)."""
        rhs = sympy.Symbol("u_x")
        result = integrate_pde(rhs, simple_grid_dataset)
        assert isinstance(result, IntegrationResult)


# integrate_pde — SCATTERED topology rejection


class TestScatteredRejection:
    """SCATTERED topology must be rejected gracefully."""

    def test_scattered_returns_failure(self, scattered_dataset: PDEDataset) -> None:
        """SCATTERED → IntegrationResult(success=False) with warning."""
        rhs = sympy.Symbol("u_x")
        result = integrate_pde(rhs, scattered_dataset)
        assert isinstance(result, IntegrationResult)
        assert result.success is False
        assert result.predicted_field is None
        assert result.warning # non-empty warning string

    def test_scattered_does_not_raise(self, scattered_dataset: PDEDataset) -> None:
        """SCATTERED must not raise an exception."""
        rhs = sympy.Symbol("u_x")
        # Should not raise
        integrate_pde(rhs, scattered_dataset)


# integrate_pde — output tensor properties


class TestOutputTensorProperties:
    """Properties of predicted_field on successful integration."""

    def test_output_is_torch_tensor(self, simple_grid_dataset: PDEDataset) -> None:
        """predicted_field must be a torch.Tensor, not numpy."""
        rhs = sympy.Symbol("u_x")
        result = integrate_pde(rhs, simple_grid_dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert isinstance(result.predicted_field, torch.Tensor)

    def test_output_shape_matches_dataset(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        """predicted_field shape must match dataset grid shape."""
        rhs = sympy.Symbol("u_x")
        result = integrate_pde(rhs, simple_grid_dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        expected_shape = simple_grid_dataset.get_shape()
        assert result.predicted_field.shape == expected_shape

    def test_output_is_finite(self, simple_grid_dataset: PDEDataset) -> None:
        """predicted_field must not contain NaN or Inf."""
        rhs = sympy.Symbol("u_x")
        result = integrate_pde(rhs, simple_grid_dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        assert torch.isfinite(result.predicted_field).all()


# integrate_pde — edge cases


class TestEdgeCases:
    """Edge cases for integrate_pde."""

    def test_zero_rhs_preserves_initial_condition(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        """RHS = 0 means u_t = 0 → u should stay at initial condition."""
        rhs = sympy.Integer(0)
        result = integrate_pde(rhs, simple_grid_dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        ic = simple_grid_dataset.get_field("u")[:, 0]
        # Every time slice should equal the initial condition
        for t_idx in range(result.predicted_field.shape[-1]):
            torch.testing.assert_close(
                result.predicted_field[:, t_idx],
                ic,
                rtol=1e-4,
                atol=1e-6,
            )

    def test_constant_rhs_linear_growth(self, simple_grid_dataset: PDEDataset) -> None:
        """RHS = constant means u_t = c → u grows linearly in time."""
        c_val = 2.0
        rhs = sympy.Float(c_val)
        result = integrate_pde(rhs, simple_grid_dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        t_vals = simple_grid_dataset.get_coords("t")
        ic = simple_grid_dataset.get_field("u")[:, 0]
        # At each time t: u(x, t) ≈ u(x, 0) + c * t
        for t_idx in range(result.predicted_field.shape[-1]):
            expected = ic + c_val * t_vals[t_idx]
            torch.testing.assert_close(
                result.predicted_field[:, t_idx],
                expected,
                rtol=1e-4,
                atol=1e-6,
            )

    def test_unsupported_lap_function_rejected_before_lambdify(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        """lap(...) is executor-only; integrator should fail before solve_ivp."""
        result = integrate_pde(to_sympy("lap(u)"), simple_grid_dataset)

        assert result.success is False
        assert "Unsupported function calls" in result.warning
        assert "lap" in result.warning
        assert "Integration failed" not in result.warning

    def test_nested_derivative_placeholder_rejected_cleanly(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        """: ``d1_x`` placeholders must fail cleanly before lambdify.

        The bridge now avoids this placeholder for the linear nested subset,
        but persisted or manually built SymPy expressions may still contain it.
        """
        # Synthesise a legacy placeholder symbol that lambdify cannot resolve.
        rhs = sympy.Symbol("u") + sympy.Symbol("d1_x")
        result = integrate_pde(rhs, simple_grid_dataset)

        assert result.success is False
        assert "" in result.warning
        assert "d1_x" in result.warning
        # Make sure the message stays concise (one short sentence is enough).
        assert len(result.warning) < 300, (
            f"Warning too long ({len(result.warning)} chars), "
            "should stay one terse sentence"
        )

    def test_unknown_non_placeholder_symbol_also_rejected(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        """Non-placeholder unknown symbols should also bail cleanly
        (not just the d{N}_{axis} pattern)."""
        rhs = sympy.Symbol("u") + sympy.Symbol("foobar")
        result = integrate_pde(rhs, simple_grid_dataset)

        assert result.success is False
        assert "" in result.warning
        assert "foobar" in result.warning

    def test_linear_nested_diff_integrates_as_compound_derivative(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        """: linear nested diff expands to u_x_x and remains integrable."""
        rhs = to_sympy("diff_x(add(u_x, diff_x(u)))")
        result = integrate_pde(rhs, simple_grid_dataset, max_step=0.05)

        assert rhs == 2 * sympy.Symbol("u_x_x")
        assert result.success is True, result.warning
        assert result.predicted_field is not None

    def test_explicit_coordinate_symbol_works(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        """ D-domain check: an RHS with explicit coordinate ``x`` as
        a scalar variable (no nested-derivative placeholder) should
        integrate successfully via the existing coord-grid path; no fix
        was needed for this case."""
        rhs = sympy.Symbol("u") + sympy.Symbol("x")
        result = integrate_pde(rhs, simple_grid_dataset)

        assert result.success is True, (
            f"Expected coord-as-variable to work natively; got: {result.warning}"
        )
        assert result.predicted_field is not None

    def test_method_parameter_accepted(self, simple_grid_dataset: PDEDataset) -> None:
        """Different method strings should be accepted."""
        rhs = sympy.Integer(0)
        for method in ["RK45", "Radau", "BDF", "RK23"]:
            result = integrate_pde(rhs, simple_grid_dataset, method=method)
            assert isinstance(result, IntegrationResult)

    def test_max_step_parameter_accepted(self, simple_grid_dataset: PDEDataset) -> None:
        """max_step kwarg should be accepted."""
        rhs = sympy.Integer(0)
        result = integrate_pde(rhs, simple_grid_dataset, max_step=0.01)
        assert isinstance(result, IntegrationResult)


# integrate_pde — non-uniform spacing rejection


class TestNonUniformSpatialSpacing:
    """integrate_pde must reject non-uniform spatial grids cleanly.

    The current FD stencils (``_central_diff_*``) use a single ``dx``
    value derived from ``vals[1] - vals[0]``. Non-uniform grids would
    silently produce wrong derivatives, so the integrator must short-
    circuit with a clear warning instead.
    """

    def test_rejects_non_uniform_spatial_axis(self) -> None:
        """Geometric (non-uniform) x-axis → success=False with diagnostic."""
        # Geometric spacing: x = [0, 0.1, 0.3, 0.7, 1.5, ...]
        nx, nt = 16, 5
        x_vals = torch.tensor(
            [0.1 * (2.0**i - 1.0) for i in range(nx)], dtype=torch.float64
        )
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.sin(x_vals).unsqueeze(-1).expand(nx, nt).clone()
        ds = PDEDataset(
            name="non-uniform",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        result = integrate_pde(sympy.Symbol("u_x"), ds)
        assert isinstance(result, IntegrationResult)
        assert result.success is False
        assert result.warning
        msg = result.warning.lower()
        assert "non-uniform" in msg or "uniform" in msg
        assert "x" in result.warning # axis name surfaced

    def test_does_not_raise_on_non_uniform(self) -> None:
        """Non-uniform spacing must not raise — only return failure."""
        nx, nt = 8, 3
        x_vals = torch.tensor(
            [0.0, 0.1, 0.4, 1.0, 1.2, 1.25, 1.5, 1.51], dtype=torch.float64
        )
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.zeros(nx, nt, dtype=torch.float64)
        ds = PDEDataset(
            name="non-uniform",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        # Must not raise
        integrate_pde(sympy.Symbol("u_x"), ds)

    def test_accepts_uniform_with_floating_point_drift(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        """``torch.linspace`` produces uniform spacing within float drift —
        the uniformity check must accept this as uniform."""
        rhs = sympy.Symbol("u_x")
        result = integrate_pde(rhs, simple_grid_dataset)
        # simple_grid_dataset uses torch.linspace which has tiny float
        # spacing variation but is conceptually uniform; integration
        # must succeed (or fail for unrelated reasons, not uniformity).
        if not result.success:
            assert "uniform" not in (result.warning or "").lower()

    def test_rejects_constant_axis_dx_zero(self) -> None:
        """Constant-coordinate axis (dx=0) must be rejected — naive
        ``np.allclose([0,...], 0)`` would pass uniformity but downstream
        FD divides by ``dx^k`` and produces NaN/Inf (cross-model review)."""
        nx, nt = 8, 3
        x_vals = torch.zeros(nx, dtype=torch.float64) # all zeros → dx=0
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.zeros(nx, nt, dtype=torch.float64)
        ds = PDEDataset(
            name="degenerate",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        result = integrate_pde(sympy.Symbol("u_x"), ds)
        assert isinstance(result, IntegrationResult)
        assert result.success is False
        msg = (result.warning or "").lower()
        assert "x" in (result.warning or "")
        assert "dx" in msg or "degenerate" in msg or "zero" in msg

    def test_accepts_float32_linspace(self) -> None:
        """``torch.linspace(dtype=float32)`` at n≥100 has rel drift up to
        ~1e-5 — the tolerance must accept it (cross-model review)."""
        nx, nt = 100, 3
        x_vals = torch.linspace(0.0, 1.0, nx, dtype=torch.float32)
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float32)
        u = torch.zeros(nx, nt, dtype=torch.float32)
        ds = PDEDataset(
            name="float32-grid",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        result = integrate_pde(sympy.Symbol("u"), ds)
        # Must not be rejected for non-uniformity (other failures are OK).
        if not result.success:
            assert "non-uniform" not in (result.warning or "").lower()
            assert "uniform" not in (result.warning or "").lower()


# integrate_pde — divergence handling


class TestDivergenceHandling:
    """Integration must handle divergent equations gracefully."""

    def test_explosive_rhs_does_not_crash(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        """A huge positive RHS should not crash, should report failure."""
        # u_t = 1e10 * u → exponential blowup
        u_sym = sympy.Symbol("u")
        rhs = 1e10 * u_sym
        result = integrate_pde(rhs, simple_grid_dataset)
        assert isinstance(result, IntegrationResult)
        assert not result.success, "Explosive RHS should fail"
        assert result.warning, "Failed integration must have a warning"

    def test_diverged_at_t_is_set_on_failure(
        self, simple_grid_dataset: PDEDataset
    ) -> None:
        """When divergence is detected, diverged_at_t should be populated."""
        u_sym = sympy.Symbol("u")
        rhs = 1e10 * u_sym
        result = integrate_pde(rhs, simple_grid_dataset)
        assert not result.success, "Explosive RHS should fail"
        # Radau may fail in Jacobian computation (diverged_at_t=None)
        # or detect divergence (diverged_at_t=float) — both are valid
        if result.diverged_at_t is not None:
            assert isinstance(result.diverged_at_t, float)
            assert result.diverged_at_t >= 0.0
        else:
            assert "fail" in result.warning.lower() or "nan" in result.warning.lower()


# integrate_pde — multi-dimensional


class TestMultiDimensional:
    """Tests for datasets with multiple spatial dimensions."""

    def test_2d_dataset_accepted(self) -> None:
        """2D grid (x, y, t) should be accepted."""
        nx, ny, nt = 16, 16, 5
        x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
        y = torch.linspace(0.0, 2 * torch.pi, ny, dtype=torch.float64)
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.randn(nx, ny, nt, dtype=torch.float64)
        dataset = PDEDataset(
            name="test-2d",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=True),
                "y": AxisInfo(name="y", values=y, is_periodic=True),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "y", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        rhs = sympy.Symbol("u_x") + sympy.Symbol("u_y")
        result = integrate_pde(rhs, dataset)
        assert isinstance(result, IntegrationResult)

    def test_mixed_partial_symbol_from_sympy_bridge_integrates(
        self, mixed_partial_dataset: PDEDataset
    ) -> None:
        """diff_y(u_x) integrates as nested finite differences."""
        rhs = to_sympy("diff_y(u_x)")
        result = integrate_pde(
            rhs,
            mixed_partial_dataset,
            method="RK45",
            max_step=0.1,
        )
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None

        predicted = result.predicted_field.to(torch.float64)
        analytical = mixed_partial_dataset.get_field("u").to(torch.float64)
        initial = analytical[:, :, 0]
        final_predicted = predicted[:, :, -1]
        final_analytical = analytical[:, :, -1]

        corr = torch.corrcoef(
            torch.stack([final_predicted.flatten(), final_analytical.flatten()])
        )[0, 1]
        assert corr.item() > 0.99, f"Final-slice correlation {corr.item():.4f} < 0.99"

        decay_ratio = (final_predicted.norm() / initial.norm()).item()
        assert decay_ratio == pytest.approx(math.exp(-1.0), rel=0.15, abs=0.05)


# integrate_pde — Dirichlet boundary conditions


class TestDirichletBoundary:
    """Tests for non-periodic (Dirichlet) boundary conditions."""

    def test_dirichlet_dataset_accepted(self) -> None:
        """Non-periodic spatial axis should be handled (Dirichlet BC)."""
        nx, nt = 32, 10
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.zeros(nx, nt, dtype=torch.float64)
        # IC: u(x, 0) = sin(pi*x), BCs: u(0,t)=0, u(1,t)=0
        u[:, 0] = torch.sin(torch.pi * x)
        dataset = PDEDataset(
            name="test-dirichlet",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        rhs = sympy.Symbol("u_xx")
        result = integrate_pde(rhs, dataset)
        assert isinstance(result, IntegrationResult)

    def test_dirichlet_boundaries_preserved(self) -> None:
        """Boundary values should remain fixed for Dirichlet BC."""
        nx, nt = 32, 10
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
        u = torch.zeros(nx, nt, dtype=torch.float64)
        u[:, 0] = torch.sin(torch.pi * x)
        dataset = PDEDataset(
            name="test-dirichlet-bc",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=False),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )
        rhs = sympy.Symbol("u_xx")
        result = integrate_pde(rhs, dataset)
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        # Dirichlet: boundary values from IC
        left_bc = u[0, 0].item()
        right_bc = u[-1, 0].item()
        torch.testing.assert_close(
            result.predicted_field[0,:],
            torch.full((nt,), left_bc, dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )
        torch.testing.assert_close(
            result.predicted_field[-1,:],
            torch.full((nt,), right_bc, dtype=torch.float64),
            rtol=1e-4,
            atol=1e-6,
        )


# INTG/H1: Cross-field derivative guard


class TestCrossFieldGuard:
    """Regression tests for INTG/H1: cross-field derivative bypass.

    When RHS contains derivatives of a field other than lhs_field (e.g.,
    lhs_field="u" but RHS uses v_x), _mol_rhs would silently compute
    the derivative using LHS field data, producing wrong results.
    The integrator must detect this and return success=False.
    """

    @staticmethod
    def _two_field_dataset() -> PDEDataset:
        """Create a 1D periodic dataset with fields u and v."""
        nx, nt = 32, 10
        x = torch.linspace(0.0, 2 * torch.pi, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
        u = torch.sin(x).unsqueeze(-1).expand(nx, nt).clone()
        v = torch.cos(x).unsqueeze(-1).expand(nx, nt).clone()
        return PDEDataset(
            name="test-two-field",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x, is_periodic=True),
                "t": AxisInfo(name="t", values=t, is_periodic=False),
            },
            axis_order=["x", "t"],
            fields={
                "u": FieldData(name="u", values=u),
                "v": FieldData(name="v", values=v),
            },
            lhs_field="u",
            lhs_axis="t",
        )

    def test_cross_field_derivative_rejected(self) -> None:
        """RHS = v_x with lhs_field=u must be rejected (success=False).

        The bug: _mol_rhs always uses lhs_field data for FD computation,
        so v_x would compute d(u)/dx instead of d(v)/dx. The guard must
        detect that v_x references field 'v' which is not the lhs_field.
        """
        ds = self._two_field_dataset()
        rhs = sympy.Symbol("v_x")
        result = integrate_pde(rhs, ds)
        assert result.success is False, (
            "Cross-field derivative v_x should be rejected, "
            "but integration returned success=True"
        )
        # Warning should mention the cross-field issue
        assert result.warning, "Failed result must include a warning"

    def test_cross_field_compound_derivative_rejected(self) -> None:
        """RHS = v_xx with lhs_field=u must also be rejected."""
        ds = self._two_field_dataset()
        rhs = sympy.Symbol("v_xx")
        result = integrate_pde(rhs, ds)
        assert result.success is False, (
            "Cross-field derivative v_xx should be rejected, "
            "but integration returned success=True"
        )
        assert result.warning, "Failed result must include a warning"

    def test_same_field_derivative_accepted(self) -> None:
        """RHS = u_x with lhs_field=u must succeed (no false positive).

        This confirms the guard does not over-reject: derivatives of
        the same field as lhs_field must still be integrated normally.
        """
        ds = self._two_field_dataset()
        rhs = sympy.Symbol("u_x")
        result = integrate_pde(rhs, ds)
        assert result.success is True, (
            f"Same-field derivative u_x should be accepted, "
            f"but got failure: {result.warning}"
        )
        assert result.predicted_field is not None

    def test_multi_field_state_var_still_rejected(self) -> None:
        """RHS = v (direct reference, not derivative) must be rejected.

        The existing guard checks len(state_vars) > 1, but when RHS = v
        with lhs_field=u, state_vars = {"v"} (size 1) — so the guard
        does not trigger. This is also part of the cross-field bug:
        any reference to a non-lhs field should be rejected.
        """
        ds = self._two_field_dataset()
        rhs = sympy.Symbol("v")
        result = integrate_pde(rhs, ds)
        assert result.success is False, (
            "Direct cross-field reference 'v' should be rejected, "
            "but integration returned success=True"
        )

    def test_mol_rhs_defensive_assert_on_cross_field_derivative(self) -> None:
        """Defense-in-depth: _mol_rhs asserts single-field if guard bypassed.

        integrate_pde's cross-field guard rejects multi-field RHS before
        reaching _mol_rhs. But _mol_rhs itself reuses lhs_field data for
        every derivative name — if the guard is ever weakened without
        updating _mol_rhs, silent wrong values would result. This assert
        pins the invariant and makes any contract break fail loudly.
        """
        parsed = _ParsedSymbols(
            state_vars={"u"},
            derivatives={"v_x": ("v", [("x", 1)])},
        )
        spatial_info = [
            _SpatialAxisInfo(
                name="x",
                values=np.linspace(0.0, 1.0, 8),
                dx=0.125,
                periodic=True,
                axis_index=0,
            )
        ]
        with pytest.raises(AssertionError, match="single-field invariant"):
            _mol_rhs(
                np.ones(8),
                lambda u, v_x: u, # dummy rhs_func, never reached
                parsed,
                spatial_info,
                (8,),
                [sympy.Symbol("u"), sympy.Symbol("v_x")],
                "u",
            )

    def test_mol_rhs_defensive_assert_on_cross_field_state_var(self) -> None:
        """Defense-in-depth: cross-field state var also trips the assert."""
        parsed = _ParsedSymbols(
            state_vars={"v"},
            derivatives={},
        )
        spatial_info = [
            _SpatialAxisInfo(
                name="x",
                values=np.linspace(0.0, 1.0, 8),
                dx=0.125,
                periodic=True,
                axis_index=0,
            )
        ]
        with pytest.raises(AssertionError, match="single-field invariant"):
            _mol_rhs(
                np.ones(8),
                lambda v: v,
                parsed,
                spatial_info,
                (8,),
                [sympy.Symbol("v")],
                "u",
            )


# M3R/M4: Finite difference minimum points guard


class TestFDMinimumPoints:
    """Regression tests for M3R/M4: _finite_diff with too few points.

    Without a minimum-points guard:
    - order 1 periodic on 2 pts: wraps around, finite but wrong
    - order 2 periodic on 2-3 pts: wraps around, finite but wrong
    - order 3 non-periodic on 4 pts: IndexError at u[-5]
    - order 3 periodic on 3-4 pts: wraps around, wrong

    Expected minimum points: {order 1: 3, order 2: 3, order 3: 5}.
    """

    def test_order1_too_few_points(self) -> None:
        """Order-1 FD on 2 points must raise ValueError."""
        u = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match=r"(?i).*(point|size|too\s*(few|small))"):
            _finite_diff(u, 0.1, order=1, periodic=False)

    def test_order2_too_few_points(self) -> None:
        """Order-2 FD on 2 points must raise ValueError."""
        u = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match=r"(?i).*(point|size|too\s*(few|small))"):
            _finite_diff(u, 0.1, order=2, periodic=False)

    def test_order3_too_few_points_4pts(self) -> None:
        """Order-3 FD on 4 points must raise ValueError (needs 5)."""
        u = np.ones(4)
        with pytest.raises(ValueError, match=r"(?i).*(point|size|too\s*(few|small))"):
            _finite_diff(u, 0.1, order=3, periodic=False)

    def test_order1_minimum_points_accepted(self) -> None:
        """Order-1 FD on 3 points must succeed (minimum for central diff)."""
        u = np.array([0.0, 1.0, 4.0])
        result = _finite_diff(u, 1.0, order=1, periodic=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == u.shape
        assert np.all(np.isfinite(result))

    def test_order2_minimum_points_accepted(self) -> None:
        """Order-2 FD on 3 points must succeed."""
        u = np.array([0.0, 1.0, 4.0])
        result = _finite_diff(u, 1.0, order=2, periodic=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == u.shape
        assert np.all(np.isfinite(result))

    def test_order3_minimum_points_accepted(self) -> None:
        """Order-3 FD on 5 points must succeed (minimum for 5-pt stencil)."""
        u = np.array([0.0, 1.0, 8.0, 27.0, 64.0])
        result = _finite_diff(u, 1.0, order=3, periodic=False)
        assert isinstance(result, np.ndarray)
        assert result.shape == u.shape
        assert np.all(np.isfinite(result))

    def test_periodic_too_few_points_wrong_values(self) -> None:
        """Periodic order-2 FD on 2 points must raise ValueError.

        Without the guard, periodic wrapping produces [2, -2] for
        input [0, 1] with dx=1 — finite but numerically wrong.
        """
        u = np.array([0.0, 1.0])
        with pytest.raises(ValueError, match=r"(?i).*(point|size|too\s*(few|small))"):
            _finite_diff(u, 1.0, order=2, periodic=True)

    def test_periodic_order1_too_few_points(self) -> None:
        """Periodic order-1 FD on 2 points must raise ValueError.

        Without the guard, periodic wrapping produces finite but wrong
        values due to stencil width exceeding array size.
        """
        u = np.array([1.0, 2.0])
        with pytest.raises(ValueError, match=r"(?i).*(point|size|too\s*(few|small))"):
            _finite_diff(u, 1.0, order=1, periodic=True)

    def test_periodic_order3_too_few_points(self) -> None:
        """Periodic order-3 FD on 4 points must raise ValueError (needs 5).

        Without the guard, periodic wrapping with pad=2 on 4 points
        creates a padded array of length 8, and the stencil produces
        finite but numerically wrong results.
        """
        u = np.array([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(ValueError, match=r"(?i).*(point|size|too\s*(few|small))"):
            _finite_diff(u, 1.0, order=3, periodic=True)


# M2a/H1: Mixed partial derivative confirmation


class TestMixedPartialConfirmation:
    """Confirmation tests for M2a/H1: mixed partial u_x_y.

    The naming utility parse_compound_derivative correctly parses u_x_y
    as ("u", [("x", 1), ("y", 1)]), and _mol_rhs iterates through
    axis_orders applying FD sequentially. These tests confirm the
    end-to-end behavior is correct.

    Analytic solution: u = sin(x+y)*exp(-t)
    -> u_xy = d/dy(d/dx[sin(x+y)*exp(-t)])
             = d/dy[cos(x+y)*exp(-t)]
             = -sin(x+y)*exp(-t)
             = -u
    So u_t = u_xy means u_t = -u, giving exponential decay.
    """

    def test_mixed_partial_parse_correct(self) -> None:
        """parse_compound_derivative must parse u_x_y as mixed partial.

        This verifies the naming layer correctly identifies the field
        and both derivative axes, which is a prerequisite for _mol_rhs
        to apply sequential FD correctly.
        """
        result = parse_compound_derivative(
            "u_x_y",
            known_fields={"u"},
            known_axes={"x", "y"},
        )
        assert result is not None, "u_x_y should parse as a compound derivative"
        field, axis_orders = result
        assert field == "u"
        assert len(axis_orders) == 2
        assert ("x", 1) in axis_orders
        assert ("y", 1) in axis_orders

    def test_mixed_partial_correct_values(
        self, mixed_partial_dataset: PDEDataset
    ) -> None:
        """Integrated u_t = u_xy must match analytic exponential decay.

        u = sin(x+y)*exp(-t), so u_xy = -u, meaning u_t = -u.
        At t=1 the field should decay by factor exp(-1) ~ 0.368.
        We verify:
        1. Integration succeeds
        2. Pointwise error at final time is small relative to signal
        3. Energy decay matches exp(-1) within tolerance
        """
        rhs = to_sympy("diff_y(u_x)")
        result = integrate_pde(
            rhs,
            mixed_partial_dataset,
            method="RK45",
            max_step=0.05,
        )
        assert result.success, f"Integration must succeed: {result.warning}"
        assert result.predicted_field is not None

        predicted = result.predicted_field.to(torch.float64)
        analytical = mixed_partial_dataset.get_field("u").to(torch.float64)

        # Check decay ratio at final time: should be exp(-1) ~ 0.368
        # Using L2 norm ratio as a robust aggregate measure
        initial_norm = analytical[:, :, 0].norm()
        final_predicted_norm = predicted[:, :, -1].norm()
        final_analytical_norm = analytical[:, :, -1].norm()

        # The predicted decay should be close to the analytical decay
        predicted_ratio = (final_predicted_norm / initial_norm).item()
        analytical_ratio = (final_analytical_norm / initial_norm).item()

        # Analytical ratio is exp(-1) ~ 0.368; predicted should be close
        torch.testing.assert_close(
            torch.tensor(predicted_ratio),
            torch.tensor(analytical_ratio),
            rtol=0.05,
            atol=0.02,
        )

        # Pointwise relative error at final time should be small
        final_err = (predicted[:, :, -1] - analytical[:, :, -1]).norm()
        assert final_err / final_analytical_norm < 0.1, (
            f"Relative pointwise error {final_err / final_analytical_norm:.4f} "
            f"exceeds 10% threshold"
        )


# _check_spatial_uniformity — small-axis consistency with FD (Issue A)


class TestCheckSpatialUniformitySmallAxis:
    """Round-2 fix: integrator must reject size<2 spatial axes consistently
    with FD's `_check_uniform_grid`. Earlier the helper had a `len<=2:
    continue` early exit that silently passed grids FD would reject.
    """

    @staticmethod
    def _build_dataset(x_vals: torch.Tensor, nt: int = 4) -> PDEDataset:
        t = torch.linspace(0.0, 1.0, nt, dtype=x_vals.dtype)
        u = torch.zeros(x_vals.shape[0], nt, dtype=x_vals.dtype)
        return PDEDataset(
            name="small-axis-check",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals),
                "t": AxisInfo(name="t", values=t),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="t",
        )

    @pytest.mark.unit
    def test_size_one_axis_emits_warning(self) -> None:
        """size==1 has no diffs to verify uniformity → must warn, not pass."""
        from kd2.core.integrator import _check_spatial_uniformity

        dataset = self._build_dataset(torch.tensor([0.0], dtype=torch.float64))
        warning = _check_spatial_uniformity(dataset, ["x"])
        assert warning is not None, "size==1 must be reported, not silently skipped"
        assert ">=2" in warning or "must have" in warning, (
            f"Warning should mention the size requirement, got: {warning!r}"
        )

    @pytest.mark.unit
    def test_size_two_axis_accepts_when_uniform(self) -> None:
        """size==2 has exactly one diff, which is trivially uniform.

        Mirrors FD's behavior: `_check_uniform_grid` accepts size==2 and
        returns the single dx. The integrator must agree.
        """
        from kd2.core.integrator import _check_spatial_uniformity

        dataset = self._build_dataset(
            torch.tensor([0.0, 0.1], dtype=torch.float64),
        )
        warning = _check_spatial_uniformity(dataset, ["x"])
        assert warning is None, (
            f"size==2 uniform grid should be accepted by both FD and "
            f"integrator, got warning: {warning!r}"
        )
