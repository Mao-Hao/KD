"""Integration: Evaluator + AutogradProvider / FiniteDiffProvider.

: Validates that AutogradProvider can compute derivatives
correctly when used through the Evaluator pipeline, even though
Evaluator.evaluate_terms() wraps execution in torch.no_grad().

 D1: AutogradProvider.get_derivative() uses
torch.enable_grad() internally to override the outer no_grad context.

Test strategy:
- Use exact-function NN models (no training needed) so derivatives
  have known analytical forms.
- Verify the full pipeline: NN model -> AutogradProvider -> ExecutionContext
  -> PythonExecutor -> Evaluator -> EvaluationResult with valid coefficients.
- Regression: FiniteDiffProvider + Evaluator still works unchanged.
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd2.core.evaluator import Evaluator
from kd2.core.executor import ExecutionContext
from kd2.core.expr import FunctionRegistry, PythonExecutor
from kd2.core.linear_solve import LeastSquaresSolver
from kd2.data.derivatives.autograd import AutogradProvider
from kd2.data.derivatives.finite_diff import FiniteDiffProvider
from kd2.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)

# Exact-function NN models (no training required)


class LinearFieldModel(nn.Module):
    """u(x) = 2*x. u_x = 2.

    Simplest possible model for testing autograd through Evaluator.
    Single spatial axis, no time dependence.
    """

    def forward(self, *, x: Tensor) -> Tensor:
        return 2.0 * x


class QuadraticFieldModel(nn.Module):
    """u(x, t) = x^2 + t.

    u_x = 2*x
    u_t = 1
    u_xx = 2
    """

    def forward(self, *, x: Tensor, t: Tensor) -> Tensor:
        return x**2 + t


# Fixtures

_N = 32


@pytest.fixture
def registry() -> FunctionRegistry:
    return FunctionRegistry.create_default()


@pytest.fixture
def executor(registry: FunctionRegistry) -> PythonExecutor:
    return PythonExecutor(registry)


@pytest.fixture
def solver() -> LeastSquaresSolver:
    return LeastSquaresSolver()


def _make_1d_autograd_components() -> tuple[nn.Module, dict[str, Tensor], PDEDataset]:
    """Build model, coords, dataset for the 1D linear model."""
    x_vals = torch.linspace(0.1, 2.0, _N, dtype=torch.float64)
    x = x_vals.clone().detach().requires_grad_(True)

    dataset = PDEDataset(
        name="linear_1d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x_vals)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=2.0 * x_vals)},
        lhs_field="u",
        lhs_axis="x",
    )
    model = LinearFieldModel().double()
    coords = {"x": x}
    return model, coords, dataset


def _make_2d_autograd_components() -> tuple[nn.Module, dict[str, Tensor], PDEDataset]:
    """Build model, coords, dataset for the 2D quadratic model."""
    x_1d = torch.linspace(0.1, 2.0, _N, dtype=torch.float64)
    t_1d = torch.linspace(0.0, 1.0, _N // 2, dtype=torch.float64)
    X, T = torch.meshgrid(x_1d, t_1d, indexing="ij")
    u_grid = X**2 + T

    x_flat = X.reshape(-1).clone().detach().requires_grad_(True)
    t_flat = T.reshape(-1).clone().detach().requires_grad_(True)

    dataset = PDEDataset(
        name="quad_2d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x_1d),
            "t": AxisInfo(name="t", values=t_1d),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_grid)},
        lhs_field="u",
        lhs_axis="t",
    )
    model = QuadraticFieldModel().double()
    coords = {"x": x_flat, "t": t_flat}
    return model, coords, dataset


# Tests


class TestAutogradEvaluatorIntegration:
    """Evaluator + AutogradProvider integration"""

    @pytest.mark.integration
    def test_autograd_evaluator_computes_derivative(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        """AutogradProvider derivatives work through the full Evaluator pipeline.

        Setup: u(x) = 2*x, so u_x = 2.
        If we evaluate_terms(["u_x"]) against lhs = u (= 2x),
        we expect coefficient ~ 1 (since u_x = 2 and u = 2x, but the
        linear fit of u_x against u is: u = x * u_x, coeff = x).

        Simpler test: just verify evaluate_terms returns is_valid=True
        and produces finite coefficients, confirming the autograd path
        works end-to-end inside Evaluator's no_grad context.
        """
        model, coords, dataset = _make_2d_autograd_components()
        provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
        context = ExecutionContext(dataset=dataset, derivative_provider=provider)

        # LHS: u_t (for u = x^2 + t, u_t = 1 everywhere)
        lhs = provider.get_derivative("u", "t", order=1)
        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs,
        )

        result = evaluator.evaluate_terms(["u_x"])

        # Core assertion: the pipeline did not crash and produced a valid result
        assert result.is_valid, (
            f"Evaluator + AutogradProvider failed: {result.error_message}"
        )
        assert result.coefficients is not None
        assert torch.isfinite(result.coefficients).all(), "Coefficients must be finite"

    @pytest.mark.integration
    def test_autograd_evaluator_no_grad_does_not_block(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        """AutogradProvider derivatives work even when an outer no_grad exists.

        This directly tests D1: AutogradProvider's internal
        enable_grad() must override the Evaluator's no_grad() context.

        Verification approach: call evaluate_terms (which wraps in no_grad)
        and confirm that derivative-based terms produce valid, non-trivial
        results. If enable_grad is NOT implemented, the autograd.grad call
        inside AutogradProvider will fail because the NN forward pass
        produces tensors without grad_fn.
        """
        model, coords, dataset = _make_2d_autograd_components()
        provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
        context = ExecutionContext(dataset=dataset, derivative_provider=provider)

        # u = x^2 + t => u_t = 1 (constant), u_xx = 2 (constant)
        lhs = provider.get_derivative("u", "t", order=1)
        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs,
        )

        # Evaluate u_xx: a constant (= 2) fitted against u_t (= 1)
        # => coefficient should be 0.5
        result = evaluator.evaluate_terms(["u_xx"])

        assert result.is_valid, f"no_grad blocked autograd: {result.error_message}"
        assert result.coefficients is not None

        # u_t = c * u_xx => 1 = c * 2 => c = 0.5
        # This is a property check: the coefficient should be close to 0.5,
        # verifying that the derivative values are actually computed correctly,
        # not just "some number".
        torch.testing.assert_close(
            result.coefficients.flatten(),
            torch.tensor([0.5], dtype=result.coefficients.dtype),
            rtol=1e-4,
            atol=1e-6,
        )

    @pytest.mark.integration
    def test_finitediff_evaluator_still_works(
        self,
        executor: PythonExecutor,
        solver: LeastSquaresSolver,
    ) -> None:
        """FiniteDiffProvider + Evaluator regression test.

        Ensures the fix does not break the existing FD path.
        Uses the same mathematical relationship: u = sin(x)*exp(-t),
        u_t = -u, so evaluating ["u"] against u_t should give coeff ~ -1.
        """
        n_x, n_t = 64, 32
        x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
        t = torch.linspace(0, 1, n_t, dtype=torch.float64)
        X, T = torch.meshgrid(x, t, indexing="ij")
        u = torch.sin(X) * torch.exp(-T)

        dataset = PDEDataset(
            name="sinexp",
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
        provider = FiniteDiffProvider(dataset, max_order=2)
        context = ExecutionContext(dataset=dataset, derivative_provider=provider)
        lhs = provider.get_derivative("u", "t", order=1)

        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs,
        )

        # u_t = -sin(x)*exp(-t) = -u, so coefficient of u should be ~ -1
        result = evaluator.evaluate_terms(["u"])

        assert result.is_valid, f"FiniteDiff + Evaluator broke: {result.error_message}"
        assert result.coefficients is not None

        # Coefficient should be close to -1 (property: u_t = -u)
        coeff = result.coefficients.flatten()[0].item()
        assert abs(coeff - (-1.0)) < 0.05, f"Expected coefficient ~ -1.0, got {coeff}"

        # R^2 should be very high for this exact relationship
        assert result.r2 > 0.95, f"R^2 too low: {result.r2}"


class TestAutogradCoordinateDerivative:
    """``diff_x(x)`` must equal 1 in autograd mode

    Before the fix, ``_resolve_name_for_diff`` resolved bare coordinate
    names through ``context.get_variable``, which returns the dataset's
    detached coord values. The resulting tensor was outside the
    autograd graph, so ``diff_x(x)`` raised "expression is not
    connected to coordinate 'x'" instead of returning 1.0.
    """

    @pytest.mark.integration
    def test_diff_of_coordinate_is_one_1d(self, executor: PythonExecutor) -> None:
        model, coords, dataset = _make_1d_autograd_components()
        provider = AutogradProvider(model, coords, dataset, max_order=1)
        ctx = ExecutionContext(dataset=dataset, derivative_provider=provider)
        result = executor.execute("diff_x(x)", ctx)
        torch.testing.assert_close(
            result.value,
            torch.ones(_N, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.integration
    def test_diff_of_coordinate_is_one_2d(self, executor: PythonExecutor) -> None:
        model, coords, dataset = _make_2d_autograd_components()
        provider = AutogradProvider(model, coords, dataset, max_order=1)
        ctx = ExecutionContext(dataset=dataset, derivative_provider=provider)
        # 2D coords are flattened to (_N * _N//2,) in the fixture.
        n_pts = _N * (_N // 2)
        result = executor.execute("diff_x(x)", ctx)
        torch.testing.assert_close(
            result.value,
            torch.ones(n_pts, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
        result_t = executor.execute("diff_t(t)", ctx)
        torch.testing.assert_close(
            result_t.value,
            torch.ones(n_pts, dtype=torch.float64),
            rtol=1e-12,
            atol=1e-12,
        )
