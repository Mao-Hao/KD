"""Integration tests for kd2.core.integrator — TDD red phase.

Tests integrate_pde against synthetic PDE datasets with known
analytical solutions. Uses real components (synthetic data generators,
sympy bridge) — no mocks.
"""

from __future__ import annotations

import pytest
import sympy
import torch

from kd2.core.integrator import IntegrationResult, integrate_pde
from kd2.data.schema import DataTopology, PDEDataset
from kd2.data.synthetic import (
    generate_advection_data,
    generate_burgers_data,
    generate_diffusion_data,
)

# Fixtures


@pytest.fixture
def advection_dataset() -> PDEDataset:
    """1D advection dataset: u_t = -c*u_x, c=1, periodic BC."""
    return generate_advection_data(
        speeds=(1.0,),
        waves=(1.0,),
        grid_sizes=(128,),
        nt=51,
        seed=42,
    )


@pytest.fixture
def diffusion_dataset() -> PDEDataset:
    """1D diffusion dataset: u_t = 0.1*u_xx, periodic BC."""
    return generate_diffusion_data(
        alpha=0.1,
        waves=(1.0,),
        grid_sizes=(128,),
        nt=51,
        seed=42,
    )


@pytest.fixture
def burgers_dataset() -> PDEDataset:
    """Burgers dataset: u_t = -u*u_x + 0.1*u_xx, periodic BC."""
    return generate_burgers_data(
        nx=128,
        nt=51,
        nu=0.1,
        seed=42,
    )


# Advection equation: u_t = -c * u_x


class TestAdvectionIntegration:
    """Integration of the advection equation with known analytical solution."""

    def test_advection_returns_success(self, advection_dataset: PDEDataset) -> None:
        """Advection with correct RHS should succeed."""
        # u_t = -1.0 * u_x → RHS = neg(u_x) = -u_x
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert isinstance(result, IntegrationResult)
        assert result.success is True
        assert result.predicted_field is not None

    def test_advection_output_is_tensor(self, advection_dataset: PDEDataset) -> None:
        """Output must be torch.Tensor."""
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert result.success
        assert isinstance(result.predicted_field, torch.Tensor)

    def test_advection_output_shape(self, advection_dataset: PDEDataset) -> None:
        """Output shape must match dataset grid shape."""
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert result.success
        assert result.predicted_field is not None
        assert result.predicted_field.shape == advection_dataset.get_shape()

    def test_advection_correlation_with_analytical(
        self, advection_dataset: PDEDataset
    ) -> None:
        """Predicted field should correlate > 0.95 with analytical solution.

        The analytical solution is stored in the dataset's field data.
        We compare flattened predicted vs analytical via Pearson correlation.
        """
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert result.success
        assert result.predicted_field is not None

        analytical = advection_dataset.get_field("u").to(torch.float64)
        predicted = result.predicted_field.to(torch.float64)

        # Pearson correlation on flattened arrays
        a_flat = analytical.flatten()
        p_flat = predicted.flatten()
        a_centered = a_flat - a_flat.mean()
        p_centered = p_flat - p_flat.mean()
        corr = (a_centered * p_centered).sum() / (a_centered.norm() * p_centered.norm())
        assert corr.item() > 0.95, f"Correlation {corr.item():.4f} < 0.95"

    def test_advection_output_is_finite(self, advection_dataset: PDEDataset) -> None:
        """Predicted field must not contain NaN or Inf."""
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert result.success
        assert result.predicted_field is not None
        assert torch.isfinite(result.predicted_field).all()


# Diffusion equation: u_t = D * u_xx


class TestDiffusionIntegration:
    """Integration of the diffusion equation with known analytical solution."""

    def test_diffusion_returns_success(self, diffusion_dataset: PDEDataset) -> None:
        """Diffusion with correct RHS should succeed."""
        u_xx = sympy.Symbol("u_xx")
        rhs = 0.1 * u_xx
        result = integrate_pde(rhs, diffusion_dataset)
        assert isinstance(result, IntegrationResult)
        assert result.success is True
        assert result.predicted_field is not None

    def test_diffusion_output_shape(self, diffusion_dataset: PDEDataset) -> None:
        """Output shape must match dataset."""
        u_xx = sympy.Symbol("u_xx")
        rhs = 0.1 * u_xx
        result = integrate_pde(rhs, diffusion_dataset)
        assert result.success
        assert result.predicted_field is not None
        assert result.predicted_field.shape == diffusion_dataset.get_shape()

    def test_diffusion_correlation_with_analytical(
        self, diffusion_dataset: PDEDataset
    ) -> None:
        """Predicted field should correlate > 0.95 with analytical solution."""
        u_xx = sympy.Symbol("u_xx")
        rhs = 0.1 * u_xx
        result = integrate_pde(rhs, diffusion_dataset)
        assert result.success
        assert result.predicted_field is not None

        analytical = diffusion_dataset.get_field("u").to(torch.float64)
        predicted = result.predicted_field.to(torch.float64)

        a_flat = analytical.flatten()
        p_flat = predicted.flatten()
        a_centered = a_flat - a_flat.mean()
        p_centered = p_flat - p_flat.mean()
        corr = (a_centered * p_centered).sum() / (a_centered.norm() * p_centered.norm())
        assert corr.item() > 0.95, f"Correlation {corr.item():.4f} < 0.95"

    def test_diffusion_decays_over_time(self, diffusion_dataset: PDEDataset) -> None:
        """Diffusion should smooth/damp the solution over time.

        Property: L2 norm of the solution should decrease (or stay constant)
        from first to last time step.
        """
        u_xx = sympy.Symbol("u_xx")
        rhs = 0.1 * u_xx
        result = integrate_pde(rhs, diffusion_dataset)
        assert result.success
        assert result.predicted_field is not None

        field = result.predicted_field.to(torch.float64)
        norm_first = field[:, 0].norm()
        norm_last = field[:, -1].norm()
        assert norm_last <= norm_first * 1.05 # small tolerance


# Burgers equation: u_t = -u*u_x + nu*u_xx


class TestBurgersIntegration:
    """Integration of Burgers equation (nonlinear PDE)."""

    def test_burgers_does_not_diverge(self, burgers_dataset: PDEDataset) -> None:
        """Short-time Burgers integration should not diverge.

        Uses correct equation RHS = -u*u_x + 0.1*u_xx.
        We only check that it completes without NaN/Inf.
        """
        u = sympy.Symbol("u")
        u_x = sympy.Symbol("u_x")
        u_xx = sympy.Symbol("u_xx")
        rhs = -u * u_x + 0.1 * u_xx
        result = integrate_pde(rhs, burgers_dataset)
        assert isinstance(result, IntegrationResult)
        assert result.success, f"Burgers integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        assert torch.isfinite(result.predicted_field).all()

    def test_burgers_output_shape(self, burgers_dataset: PDEDataset) -> None:
        """Output shape must match Burgers dataset."""
        u = sympy.Symbol("u")
        u_x = sympy.Symbol("u_x")
        u_xx = sympy.Symbol("u_xx")
        rhs = -u * u_x + 0.1 * u_xx
        result = integrate_pde(rhs, burgers_dataset)
        assert result.success, f"Burgers integration must succeed: {result.warning}"
        assert result.predicted_field is not None
        assert result.predicted_field.shape == burgers_dataset.get_shape()


# SCATTERED topology rejection


class TestScatteredTopology:
    """SCATTERED topology must be rejected by integrate_pde."""

    def test_scattered_returns_failure(self) -> None:
        """SCATTERED dataset → IntegrationResult(success=False)."""
        nx, nt = 32, 10
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
        u = torch.randn(nx, nt, dtype=torch.float64)
        from kd2.data.schema import AxisInfo, FieldData, TaskType

        dataset = PDEDataset(
            name="scattered-reject",
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
        rhs = sympy.Symbol("u_x")
        result = integrate_pde(rhs, dataset)
        assert result.success is False
        assert result.predicted_field is None
        assert "SCATTERED" in result.warning or "scattered" in result.warning.lower()


# Wrong / divergent equations


class TestWrongEquations:
    """Wrong equations should not crash the integrator."""

    def test_wrong_equation_does_not_crash(self, advection_dataset: PDEDataset) -> None:
        """Using a completely wrong RHS should not raise an exception."""
        # Apply diffusion equation to advection data
        u_xx = sympy.Symbol("u_xx")
        rhs = 100.0 * u_xx # wrong equation, huge coefficient
        result = integrate_pde(rhs, advection_dataset)
        assert isinstance(result, IntegrationResult)
        # It may succeed or fail, but must not crash

    def test_unknown_symbols_in_rhs(self, advection_dataset: PDEDataset) -> None:
        """RHS with symbols not in the dataset should be handled gracefully.

        E.g., u_y when dataset only has x axis. The integrator should
        either treat it as zero, raise a clear error, or return failure.
        """
        u_y = sympy.Symbol("u_y")
        rhs = u_y # "y" axis doesn't exist in 1D advection
        result = integrate_pde(rhs, advection_dataset)
        assert isinstance(result, IntegrationResult)
        # Must not crash; either fail gracefully or succeed with degenerate result


# Initial condition preservation


class TestInitialCondition:
    """The predicted field at t=0 should match the dataset's initial condition."""

    def test_ic_matches_at_t0(self, advection_dataset: PDEDataset) -> None:
        """predicted_field[:, 0] should match dataset field at t=0."""
        u_x = sympy.Symbol("u_x")
        rhs = -u_x
        result = integrate_pde(rhs, advection_dataset)
        assert result.success
        assert result.predicted_field is not None

        ic_dataset = advection_dataset.get_field("u")[:, 0].to(torch.float64)
        ic_predicted = result.predicted_field[:, 0].to(torch.float64)
        torch.testing.assert_close(ic_predicted, ic_dataset, rtol=1e-5, atol=1e-8)

    def test_diffusion_ic_matches_at_t0(self, diffusion_dataset: PDEDataset) -> None:
        """Diffusion: predicted_field[:, 0] should match IC."""
        u_xx = sympy.Symbol("u_xx")
        rhs = 0.1 * u_xx
        result = integrate_pde(rhs, diffusion_dataset)
        assert result.success
        assert result.predicted_field is not None

        ic_dataset = diffusion_dataset.get_field("u")[:, 0].to(torch.float64)
        ic_predicted = result.predicted_field[:, 0].to(torch.float64)
        torch.testing.assert_close(ic_predicted, ic_dataset, rtol=1e-5, atol=1e-8)
