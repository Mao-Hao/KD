"""Integration tests for PythonExecutor with existing system components.

This module tests the integration of PythonExecutor with:
- ExecutionContext (variable/derivative access)
- DerivativeProvider (terminal derivatives vs open-form diff)
- prefix_to_python (prefix conversion + execution)
- Solver (expression results as solver input)

Test coverage:
- integration: PythonExecutor + ExecutionContext
- integration: Terminal derivatives vs open-form diff
- integration: prefix -> python -> execute
- smoke: Solver input compatibility

Note: TreeExecutor equivalence tests were removed as part of IR refactor (Task 012.6).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import pytest
import torch
from torch import Tensor

from kd2.core.executor import ExecutionContext
from kd2.core.expr.executor import ExecutorResult, PythonExecutor
from kd2.core.expr.registry import FunctionRegistry
from kd2.core.linear_solve import LeastSquaresSolver, SolveResult
from kd2.data import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd2.data.derivatives import DerivativeProvider, FiniteDiffProvider

if TYPE_CHECKING:
    pass


# =============================================================================
# Fixtures - Shared Test Data
# =============================================================================


@pytest.fixture
def simple_1d_dataset() -> PDEDataset:
    """Simple 1D dataset for basic tests.

    Creates:
    - x: 32 points in [0, 2*pi]
    - u = sin(x)
    - v = cos(x)

    Analytical derivatives:
    - u_x = cos(x)
    - u_xx = -sin(x)
    """
    n_x = 32
    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    u = torch.sin(x)
    v = torch.cos(x)

    return PDEDataset(
        name="test_1d",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={
            "u": FieldData(name="u", values=u),
            "v": FieldData(name="v", values=v),
        },
        lhs_field="u",
        lhs_axis="x",
    )


@pytest.fixture
def simple_2d_dataset() -> PDEDataset:
    """2D dataset for more complex tests.

    Creates:
    - x: 32 points in [0, 2*pi]
    - t: 16 points in [0, 1]
    - u = sin(x) * exp(-t)

    Analytical derivatives:
    - u_x = cos(x) * exp(-t)
    - u_t = -sin(x) * exp(-t) = -u
    - u_xx = -sin(x) * exp(-t) = -u
    """
    n_x = 32
    n_t = 16

    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)

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
def derivative_provider_1d(simple_1d_dataset: PDEDataset) -> FiniteDiffProvider:
    """Finite difference provider for 1D dataset."""
    return FiniteDiffProvider(simple_1d_dataset, max_order=2)


@pytest.fixture
def derivative_provider_2d(simple_2d_dataset: PDEDataset) -> FiniteDiffProvider:
    """Finite difference provider for 2D dataset."""
    return FiniteDiffProvider(simple_2d_dataset, max_order=3)


@pytest.fixture
def execution_context_1d(
    simple_1d_dataset: PDEDataset,
    derivative_provider_1d: FiniteDiffProvider,
) -> ExecutionContext:
    """ExecutionContext for 1D dataset."""
    return ExecutionContext(
        dataset=simple_1d_dataset,
        derivative_provider=derivative_provider_1d,
        constants={"pi": math.pi, "C": 1.5},
    )


@pytest.fixture
def execution_context_2d(
    simple_2d_dataset: PDEDataset,
    derivative_provider_2d: FiniteDiffProvider,
) -> ExecutionContext:
    """ExecutionContext for 2D dataset."""
    return ExecutionContext(
        dataset=simple_2d_dataset,
        derivative_provider=derivative_provider_2d,
        constants={"pi": math.pi, "nu": 0.1},
    )


@pytest.fixture
def default_registry() -> FunctionRegistry:
    """FunctionRegistry with default operators."""
    return FunctionRegistry.create_default()


@pytest.fixture
def python_executor(default_registry: FunctionRegistry) -> PythonExecutor:
    """PythonExecutor with default registry."""
    return PythonExecutor(default_registry)


# =============================================================================
# Integration Tests: PythonExecutor + ExecutionContext
# =============================================================================


@pytest.mark.integration
class TestPythonExecutorWithExecutionContext:
    """Test PythonExecutor using ExecutionContext for variable access.

    This tests the integration between PythonExecutor (which uses compile+eval)
    and ExecutionContext (which provides variable and derivative values).
    """

    def test_execute_with_context_variable(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """PythonExecutor can access variables from ExecutionContext.

        NOTE: This may require an adapter/bridge to convert ExecutionContext
        interface to what PythonExecutor expects.
        """
        # Get u directly from context for comparison
        expected_u = execution_context_1d.get_variable("u")

        # Execute "u" expression using PythonExecutor
        # This requires the executor to access context.get_variable("u")
        result = python_executor.execute("u", execution_context_1d)

        assert isinstance(result, ExecutorResult)
        assert result.value is not None
        torch.testing.assert_close(result.value, expected_u, rtol=1e-10, atol=1e-10)

    def test_execute_with_context_coordinate(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """PythonExecutor can access coordinate values from ExecutionContext."""
        expected_x = execution_context_1d.get_variable("x")

        result = python_executor.execute("x", execution_context_1d)

        assert result.value is not None
        torch.testing.assert_close(result.value, expected_x, rtol=1e-10, atol=1e-10)

    def test_execute_expression_with_context_variables(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """PythonExecutor evaluates expressions using context variables."""
        u = execution_context_1d.get_variable("u")
        v = execution_context_1d.get_variable("v")
        expected = u + v

        result = python_executor.execute("add(u, v)", execution_context_1d)

        assert result.value is not None
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_with_terminal_derivatives(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """PythonExecutor can access terminal derivatives (u_x) from context.

        Terminal derivatives are variables that look like u_x, u_xx, etc.
        They should be resolved by looking up in the derivative_provider.

        XFAIL: _VariableAccessDict.__missing__() only calls context.get_variable().
        Need to extend to try context.get_derivative() for derivative patterns.
        """
        # Get u_x from context (via derivative_provider)
        expected_u_x = execution_context_1d.get_derivative("u", "x", 1)

        # Execute "u_x" - should resolve to derivative
        result = python_executor.execute("u_x", execution_context_1d)

        assert result.value is not None
        torch.testing.assert_close(result.value, expected_u_x, rtol=1e-5, atol=1e-8)

    def test_execute_expression_with_derivatives(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:
        """PythonExecutor evaluates expression with terminal derivatives.

        XFAIL: _VariableAccessDict.__missing__() only calls context.get_variable().
        Need to extend to try context.get_derivative() for derivative patterns (u_x, u_xx).
        """
        u = execution_context_2d.get_variable("u")
        u_x = execution_context_2d.get_derivative("u", "x", 1)
        u_xx = execution_context_2d.get_derivative("u", "x", 2)

        # mul(u, u_x) + u_xx
        expected = u * u_x + u_xx

        result = python_executor.execute("add(mul(u, u_x), u_xx)", execution_context_2d)

        assert result.value is not None
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_with_constant(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:
        """PythonExecutor can access named constants from context.

        XFAIL: _VariableAccessDict doesn't expose context.constants.
        Need to check context.get_constant() in __missing__().
        """
        u = execution_context_2d.get_variable("u")
        nu = execution_context_2d.get_constant("nu")

        # mul(nu, u) = 0.1 * u
        expected = nu * u

        result = python_executor.execute("mul(nu, u)", execution_context_2d)

        assert result.value is not None
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)


# =============================================================================
# Integration Tests: Terminal Derivatives vs Open-Form Diff
# =============================================================================


@pytest.mark.integration
class TestDerivativeResolution:
    """Test terminal derivative resolution vs open-form diff.

    Terminal derivatives (u_x, u_xx): Variables resolved from precomputed cache
    Open-form diff (diff_x(expr)): Computed at runtime via DerivativeProvider.diff()

    For the base field u:
    - u_x (terminal) should equal diff_x(u) (open-form)
    """

    def test_terminal_derivative_from_context(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """Terminal u_x is resolved from precomputed derivatives.

        XFAIL: _VariableAccessDict.__missing__() only calls context.get_variable().
        Need to extend to try context.get_derivative() for derivative patterns.
        """
        result = python_executor.execute("u_x", execution_context_1d)

        assert result.value is not None
        assert result.used_diff is False # Terminal, not open-form diff

    def test_open_form_diff_uses_provider(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """Open-form diff_x(u) calls DerivativeProvider.diff().

        FiniteDiffProvider.diff() is implemented ().
        This test verifies the integration path works end-to-end.
        """
        result = python_executor.execute("diff_x(u)", execution_context_1d)
        assert result.value is not None
        assert result.used_diff is True
        assert torch.isfinite(result.value).all()

    def test_compound_terminal_matches_nested_call_numerically(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """``u_x_x`` terminal must match ``diff_x(diff_x(u))`` exactly.

        Guards the B' invariant: the compound-terminal path
        (``get_derivative(u, x, 1)`` then ``context.diff(_, x, 1)``) walks the
        same nested-FD stencil as the open-form nested call. Folding ``u_x_x``
        into direct-order ``u_xx`` would break this equivalence — that path
        uses a different stencil and is forbidden by design.
        """
        compound = python_executor.execute("u_x_x", execution_context_1d)
        nested = python_executor.execute("diff_x(diff_x(u))", execution_context_1d)

        assert compound.value is not None
        assert nested.value is not None
        torch.testing.assert_close(compound.value, nested.value, rtol=0.0, atol=0.0)

    def test_terminal_and_open_form_equivalence_with_mock_provider(
        self,
        simple_1d_dataset: PDEDataset,
        default_registry: FunctionRegistry,
    ) -> None:
        """Terminal u_x equals diff_x(u) when using a provider that supports diff().

        This test uses a mock provider that returns IDENTICAL analytical values
        for both get_derivative() and diff(). This verifies interface equivalence
        between terminal resolution and open-form diff, not numerical accuracy.

        XFAIL: Requires:
        1. _VariableAccessDict extension for derivative patterns (u_x)
        2. diff_x function registration and routing to DerivativeProvider.diff()
        """

        class MockDerivativeProvider(DerivativeProvider):
            """Mock provider returning identical analytical values for both interfaces.

            Design: Both get_derivative() and diff() return the same analytical
            derivative values. This tests that the execution paths are equivalent,
            not that the derivative computation is accurate.
            """

            def __init__(self, dataset: PDEDataset) -> None:
                self._dataset = dataset
                self._cache: dict[tuple[str, str, int], Tensor] = {}
                # Precompute analytical derivatives
                if dataset.axes is not None:
                    x = dataset.axes["x"].values
                    u = torch.sin(x)
                    u_x = torch.cos(x)
                    u_xx = -torch.sin(x)
                    # Store both field and derivatives
                    self._field_cache = {"u": u}
                    self._cache[("u", "x", 1)] = u_x
                    self._cache[("u", "x", 2)] = u_xx

            def get_derivative(self, field: str, axis: str, order: int) -> torch.Tensor:
                key = (field, axis, order)
                if key not in self._cache:
                    raise KeyError(f"Derivative {key} not found")
                return self._cache[key].clone()

            def diff(
                self, expression: torch.Tensor, axis: str, order: int
            ) -> torch.Tensor:
                """Return same analytical value as get_derivative for base field u.

                For interface equivalence testing: when expression == u,
                diff(u, x, 1) should return same value as get_derivative(u, x, 1).
                """
                if self._dataset.axes is None:
                    raise ValueError("No axes")

                # Check if expression matches base field u
                u = self._field_cache.get("u")
                if u is not None and torch.allclose(
                    expression, u, rtol=1e-10, atol=1e-10
                ):
                    # Return same analytical derivative as get_derivative
                    key = ("u", axis, order)
                    if key in self._cache:
                        return self._cache[key].clone()

                # For other expressions, raise NotImplementedError
                raise NotImplementedError(
                    f"diff() only supports base field u for testing, got expression with shape {expression.shape}"
                )

            def available_derivatives(self) -> list[tuple[str, str, int]]:
                return list(self._cache.keys())

        provider = MockDerivativeProvider(simple_1d_dataset)
        context = ExecutionContext(
            dataset=simple_1d_dataset,
            derivative_provider=provider,
            constants={},
        )

        executor = PythonExecutor(default_registry)

        # Get terminal u_x
        terminal_result = executor.execute("u_x", context)
        # Get open-form diff_x(u)
        diff_result = executor.execute("diff_x(u)", context)

        # Both should be close (exact equivalence for base field)
        assert terminal_result.value is not None
        assert diff_result.value is not None

        # Interface equivalence: both paths should return identical analytical values
        # Using tight tolerance since both use the same underlying cached value
        torch.testing.assert_close(
            terminal_result.value, diff_result.value, rtol=1e-10, atol=1e-10
        )

    def test_open_form_diff_on_expression(
        self,
        simple_1d_dataset: PDEDataset,
        default_registry: FunctionRegistry,
    ) -> None:
        """Open-form diff on complex expression: diff_x(mul(u, u)).

        d/dx(u^2) = 2u * u_x by chain rule.

        This test uses a custom AnalyticalDiffProvider that implements diff()
        using finite differences, bypassing the FiniteDiffProvider limitation.
        """

        class AnalyticalDiffProvider(DerivativeProvider):
            """Provider with analytical diff for testing."""

            def __init__(self, dataset: PDEDataset) -> None:
                self._dataset = dataset
                if dataset.axes is None:
                    raise ValueError("No axes")
                x = dataset.axes["x"].values
                self._dx = (x[-1] - x[0]).item() / (len(x) - 1)

            def get_derivative(self, field: str, axis: str, order: int) -> torch.Tensor:
                raise KeyError(f"No precomputed {field}_{axis}")

            def diff(
                self, expression: torch.Tensor, axis: str, order: int
            ) -> torch.Tensor:
                if order != 1:
                    raise NotImplementedError
                # Central difference on the expression
                result = torch.zeros_like(expression)
                result[1:-1] = (expression[2:] - expression[:-2]) / (2 * self._dx)
                result[0] = (expression[1] - expression[0]) / self._dx
                result[-1] = (expression[-1] - expression[-2]) / self._dx
                return result

            def available_derivatives(self) -> list[tuple[str, str, int]]:
                return []

        provider = AnalyticalDiffProvider(simple_1d_dataset)
        context = ExecutionContext(
            dataset=simple_1d_dataset,
            derivative_provider=provider,
            constants={},
        )

        executor = PythonExecutor(default_registry)

        # Execute diff_x(mul(u, u)) = d/dx(u^2)
        result = executor.execute("diff_x(mul(u, u))", context)

        assert result.value is not None
        assert result.used_diff is True

        # Verify: d/dx(sin^2(x)) = 2*sin(x)*cos(x) = sin(2x)
        if simple_1d_dataset.axes is not None:
            x = simple_1d_dataset.axes["x"].values
            expected = torch.sin(2 * x)
            # Allow larger tolerance due to finite difference
            # (boundary has higher error) - skip boundary points
            torch.testing.assert_close(
                result.value[1:-1], expected[1:-1], rtol=0.15, atol=0.15
            )


# =============================================================================
# Integration Tests: Prefix Conversion + Execution
# =============================================================================


@pytest.mark.integration
class TestPrefixConversionExecution:
    """Test prefix -> python -> execute pipeline.

    This validates that:
    1. Prefix tokens can be converted to Python expressions
    2. The resulting expressions execute correctly
    3. Results match direct Python execution
    """

    def test_simple_prefix_to_execution(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
        default_registry: FunctionRegistry,
    ) -> None:
        """prefix_to_python then execute produces correct result."""
        from kd2.core.compat.prefix import prefix_to_python

        # Prefix: add u v -> add(u, v)
        prefix_tokens = ["add", "u", "v"]
        python_expr = prefix_to_python(prefix_tokens, default_registry)

        assert python_expr == "add(u, v)"

        # Execute the converted expression
        result = python_executor.execute(python_expr, execution_context_2d)

        # Verify correctness
        u = execution_context_2d.get_variable("u")
        v = execution_context_2d.get_variable("v")
        expected = u + v

        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_nested_prefix_to_execution(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
        default_registry: FunctionRegistry,
    ) -> None:
        """Nested prefix expression converts and executes correctly.

        XFAIL: PythonExecutor needs _VariableAccessDict extension for u_x, u_xx.
        """
        from kd2.core.compat.prefix import prefix_to_python

        # Prefix: add mul u u_x u_xx
        # = add(mul(u, u_x), u_xx)
        prefix_tokens = ["add", "mul", "u", "u_x", "u_xx"]
        python_expr = prefix_to_python(prefix_tokens, default_registry)

        assert python_expr == "add(mul(u, u_x), u_xx)"

        result = python_executor.execute(python_expr, execution_context_2d)

        u = execution_context_2d.get_variable("u")
        u_x = execution_context_2d.get_derivative("u", "x", 1)
        u_xx = execution_context_2d.get_derivative("u", "x", 2)
        expected = u * u_x + u_xx

        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_roundtrip_python_to_prefix_to_python(
        self,
        default_registry: FunctionRegistry,
    ) -> None:
        """python -> prefix -> python roundtrip preserves semantics."""
        from kd2.core.compat.prefix import prefix_to_python, python_to_prefix

        original = "add(mul(u, v), sin(u))"

        # Python -> Prefix
        prefix = python_to_prefix(original)
        # Prefix -> Python
        reconstructed = prefix_to_python(prefix, default_registry)

        # Note: exact string equality may not hold, but semantic equivalence should
        # For well-formed expressions, they should be equal
        assert reconstructed == original

    def test_prefix_execution_matches_direct_execution(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
        default_registry: FunctionRegistry,
    ) -> None:
        """Prefix-converted expression gives same result as direct execution."""
        from kd2.core.compat.prefix import prefix_to_python

        # Direct Python expression
        direct_expr = "mul(sin(u), cos(v))"

        # Convert via prefix
        from kd2.core.compat.prefix import python_to_prefix

        prefix = python_to_prefix(direct_expr)
        converted_expr = prefix_to_python(prefix, default_registry)

        # Execute both
        direct_result = python_executor.execute(direct_expr, execution_context_2d)
        converted_result = python_executor.execute(converted_expr, execution_context_2d)

        torch.testing.assert_close(
            direct_result.value, converted_result.value, rtol=1e-10, atol=1e-10
        )


# =============================================================================
# Smoke Tests: Solver Integration
# =============================================================================


@pytest.mark.smoke
class TestSolverIntegration:
    """Smoke tests for expression execution results as Solver input.

    The Solver expects:
    - theta: Feature matrix (n_samples, n_terms) - flattened expression values
    - y: Target vector (n_samples,) - flattened LHS values
    """

    def test_expression_result_as_solver_input(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:
        """Expression execution result can be used as Solver input.

        XFAIL: PythonExecutor needs _VariableAccessDict extension for u_x, u_xx.
        """
        # Execute multiple expressions to build theta matrix
        expressions = ["u", "u_x", "u_xx", "mul(u, u_x)"]
        results = []

        for expr in expressions:
            result = python_executor.execute(expr, execution_context_2d)
            assert result.value is not None
            # Flatten for solver input
            results.append(result.value.flatten())

        # Build theta matrix (n_samples, n_terms)
        theta = torch.stack(results, dim=1)

        # Get LHS (target) - u_t for Burgers-like equation
        u_t = execution_context_2d.get_derivative("u", "t", 1)
        y = u_t.flatten()

        # Verify shapes are compatible
        assert theta.shape[0] == y.shape[0]
        assert theta.shape[1] == len(expressions)

        # Verify types
        assert isinstance(theta, Tensor)
        assert isinstance(y, Tensor)

    def test_solver_with_expression_results(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:
        """LeastSquaresSolver can solve with expression results.

        XFAIL: PythonExecutor needs _VariableAccessDict extension for u_xx.
        """
        # Build theta from expression results
        expressions = ["u", "u_xx"]
        results = []

        for expr in expressions:
            result = python_executor.execute(expr, execution_context_2d)
            results.append(result.value.flatten())

        theta = torch.stack(results, dim=1)

        # Target: use u_t as LHS
        u_t = execution_context_2d.get_derivative("u", "t", 1)
        y = u_t.flatten()

        # Create and run solver
        solver = LeastSquaresSolver()
        solve_result = solver.solve(theta, y)

        # Verify result structure
        assert isinstance(solve_result, SolveResult)
        assert solve_result.coefficients.shape == (len(expressions),)
        assert isinstance(solve_result.r2, float)
        assert isinstance(solve_result.residual, float)

    def test_solver_with_numerical_stability(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:
        """Solver handles expressions that might have numerical issues."""
        # div(u, v) uses safe_div, should not produce NaN/Inf
        result = python_executor.execute("div(u, v)", execution_context_2d)
        assert result.value is not None
        assert torch.isfinite(result.value).all()

        # Build theta with potentially problematic term
        theta = result.value.flatten().unsqueeze(1)
        y = torch.randn_like(theta[:, 0])

        solver = LeastSquaresSolver()
        solve_result = solver.solve(theta, y)

        # Result should be finite
        assert torch.isfinite(solve_result.coefficients).all()


# =============================================================================
# Numerical Tests: Edge Cases in Integration
# =============================================================================


@pytest.mark.numerical
class TestIntegrationNumericalEdgeCases:
    """Numerical edge case tests for integration scenarios."""

    def test_expression_with_zero_values(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """Expression execution handles zero values correctly.

        u = sin(x) has zeros at x = 0, pi, 2*pi.
        Operations with zeros should be numerically stable.
        """
        # div(v, u) where u has zeros
        result = python_executor.execute("div(v, u)", execution_context_1d)
        assert result.value is not None
        # safe_div should prevent Inf
        assert torch.isfinite(result.value).all()

    def test_expression_with_small_values(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """Expression with very small values is stable."""
        # exp(-n2(u)) = exp(-sin^2(x)), can be very small
        result = python_executor.execute("exp(neg(n2(u)))", execution_context_1d)
        assert result.value is not None
        assert torch.isfinite(result.value).all()

    def test_derivative_at_boundary(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """Derivative values at boundaries are finite.

        XFAIL: PythonExecutor needs _VariableAccessDict extension for u_x.
        """
        result = python_executor.execute("u_x", execution_context_1d)
        assert result.value is not None
        assert torch.isfinite(result.value).all()

        # Check boundary values specifically
        assert torch.isfinite(result.value[0])
        assert torch.isfinite(result.value[-1])

    def test_nested_operations_numerical_stability(
        self,
        python_executor: PythonExecutor,
        execution_context_2d: ExecutionContext,
    ) -> None:
        """Deeply nested operations maintain numerical stability."""
        # sin(cos(exp(neg(u)))) - many nested operations
        result = python_executor.execute("sin(cos(exp(neg(u))))", execution_context_2d)
        assert result.value is not None
        assert torch.isfinite(result.value).all()

        # Verify result is in valid range for sin(cos(...))
        assert (result.value >= -1.0).all()
        assert (result.value <= 1.0).all()

    def test_empty_expression_error(
        self,
        python_executor: PythonExecutor,
        execution_context_1d: ExecutionContext,
    ) -> None:
        """Empty or whitespace-only expressions should raise an error.

        H2: Empty input handling - must raise ValueError or SyntaxError.
        """
        # Empty string
        with pytest.raises((ValueError, SyntaxError)):
            python_executor.execute("", execution_context_1d)

        # Whitespace-only string
        with pytest.raises((ValueError, SyntaxError)):
            python_executor.execute(" ", execution_context_1d)

        # Only tabs/newlines
        with pytest.raises((ValueError, SyntaxError)):
            python_executor.execute("\t\n", execution_context_1d)

    def test_extreme_values(
        self,
        default_registry: FunctionRegistry,
    ) -> None:
        """Test numerical stability with extreme input values.

        H3: Must handle 1e100, NaN, Inf without crashing.

        NOTE: PDEDataset/FieldData validates against NaN/Inf in input data,
        so we test executor behavior by:
        1. Testing operations that PRODUCE extreme values
        2. Directly testing with custom variable dict (bypassing validation)
        """
        from kd2.data.derivatives import DerivativeProvider

        # Create minimal dataset with normal values
        n = 10
        x = torch.linspace(0, 1, n, dtype=torch.float64)
        u = torch.sin(x * math.pi) # Normal values

        dataset = PDEDataset(
            name="extreme_test",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={"x": AxisInfo(name="x", values=x)},
            axis_order=["x"],
            fields={"u": FieldData(name="u", values=u)},
            lhs_field="u",
            lhs_axis="x",
        )

        class MinimalProvider(DerivativeProvider):
            def get_derivative(self, field: str, axis: str, order: int) -> Tensor:
                raise KeyError(f"No derivative for {field}_{axis}")

            def diff(self, expr: Tensor, axis: str, order: int) -> Tensor:
                raise NotImplementedError("No diff support")

            def available_derivatives(self) -> list[tuple[str, str, int]]:
                return []

        provider = MinimalProvider()
        context = ExecutionContext(
            dataset=dataset,
            derivative_provider=provider,
            constants={},
        )

        executor = PythonExecutor(default_registry)

        # Test 1: Operations that produce very large values
        # exp(u * 100) could overflow, but safe_exp should handle it
        # Note: u ranges from 0 to ~1, so u * 100 ranges to ~100
        # exp(100) = 2.7e43 which is large but finite in float64
        result_large = executor.execute("exp(mul(u, add(u, u)))", context)
        assert result_large.value is not None
        # Should be finite or clamped, not crash
        assert not torch.isnan(result_large.value).any()

        # Test 2: Division that could produce Inf
        # div(1, sub(u, u)) would be 1/0, but safe_div should prevent Inf
        # Using a small constant to test near-zero division
        result_div = executor.execute("div(u, mul(u, u))", context)
        assert result_div.value is not None
        # safe_div should prevent Inf
        assert torch.isfinite(result_div.value).all()

        # Test 3: Operations that stay finite
        # n2(u) = u^2 >= 0, always finite for finite input
        result_n2 = executor.execute("n2(u)", context) # u^2
        assert result_n2.value is not None
        assert not torch.isnan(result_n2.value).any()
        assert torch.isfinite(result_n2.value).all()

        # Test 4: Nested operations that amplify numerical issues
        # Multiple exp/log cycles
        result_nested = executor.execute("exp(neg(exp(neg(u))))", context)
        assert result_nested.value is not None
        assert torch.isfinite(result_nested.value).all()

        # Test 5: Direct extreme value test via registry functions
        # Test that registry functions can handle extreme values

        # Get functions from registry
        add_fn = default_registry.get_func("add")
        div_fn = default_registry.get_func("div")

        # Test add with extreme values
        extreme_tensor = torch.tensor([1e100, 1e-100, 0.0], dtype=torch.float64)
        result_extreme_add = add_fn(extreme_tensor, extreme_tensor)
        # 1e100 + 1e100 = 2e100, still finite in float64
        assert torch.isfinite(result_extreme_add).all() or result_extreme_add[
            0
        ] == float("inf")

        # Test safe_div with zero
        zero_tensor = torch.zeros(3, dtype=torch.float64)
        one_tensor = torch.ones(3, dtype=torch.float64)
        result_safe_div = div_fn(one_tensor, zero_tensor)
        # safe_div should prevent Inf
        assert torch.isfinite(result_safe_div).all()
