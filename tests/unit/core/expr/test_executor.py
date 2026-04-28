"""Tests for PythonExecutor.

Test coverage:
- smoke: Basic instantiation and method existence
- unit: Two-path execution strategy, has_open_form_diff detection
- numerical: Tensor operations, derivative computation, edge cases

TDD Status: RED - Tests written before implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, PropertyMock

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from kd2.core.executor.context import ExecutionContext
from kd2.core.expr.executor import (
    _SPECIAL_OPERATOR_STUBS,
    _SPECIAL_OPERATORS,
    ExecutorResult,
    PythonExecutor,
    _should_use_full_path,
    has_open_form_diff,
)
from kd2.core.expr.registry import FunctionRegistry
from kd2.data.derivatives.autograd import AutogradProvider
from kd2.data.derivatives.finite_diff import FiniteDiffProvider
from kd2.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def default_registry() -> FunctionRegistry:
    """Registry with default operators."""
    return FunctionRegistry.create_default()


@pytest.fixture
def mock_context() -> MagicMock:
    """Mock ExecutionContext for testing.

    The mock provides:
    - get_variable(name) -> tensor values for u, v, w, u_x, u_xx
    - device property -> cpu
    - derivative_provider.diff(expr, axis, order) -> mock derivative
    """
    context = MagicMock()

    # Device
    type(context).device = PropertyMock(return_value=torch.device("cpu"))

    # Variable values (all same shape for compatibility)
    variables = {
        "u": torch.tensor([1.0, 2.0, 3.0]),
        "v": torch.tensor([4.0, 5.0, 6.0]),
        "w": torch.tensor([0.5, 0.5, 0.5]),
        "u_x": torch.tensor([0.1, 0.2, 0.3]),
        "u_xx": torch.tensor([0.01, 0.02, 0.03]),
        "C": torch.tensor([1.0, 1.0, 1.0]), # Constant placeholder
    }

    def get_variable(name: str) -> Tensor:
        if name not in variables:
            raise KeyError(f"Variable '{name}' not found")
        return variables[name]

    context.get_variable = MagicMock(side_effect=get_variable)

    # Precomputed derivatives (terminal derivatives like u_x, u_xx)
    derivatives = {
        ("u", "x", 1): torch.tensor([0.1, 0.2, 0.3]),
        ("u", "x", 2): torch.tensor([0.01, 0.02, 0.03]),
    }

    def get_derivative(field: str, axis: str, order: int) -> Tensor:
        key = (field, axis, order)
        if key not in derivatives:
            raise KeyError(f"Derivative {field}_{axis}*{order} not found")
        return derivatives[key]

    context.get_derivative = MagicMock(side_effect=get_derivative)

    # Named constants
    constants: dict[str, float] = {
        "C": 1.0,
    }

    def get_constant(name: str) -> float:
        if name not in constants:
            raise KeyError(f"Constant '{name}' not found")
        return constants[name]

    context.get_constant = MagicMock(side_effect=get_constant)

    # Derivative provider for open-form diff
    def mock_diff(expression: Tensor, axis: str, order: int) -> Tensor:
        # Simple mock: just multiply by a factor based on axis/order
        factor = 0.1 * order
        return expression * factor

    context.derivative_provider = MagicMock()
    context.derivative_provider.diff = MagicMock(side_effect=mock_diff)

    # context.diff delegates to derivative_provider.diff (mirrors ExecutionContext)
    context.diff = MagicMock(side_effect=mock_diff)

    return context


def _make_lap_mock_context(
    spatial_axes: list[str],
    *,
    u: Tensor | None = None,
    axis_factors: dict[str, float] | None = None,
) -> MagicMock:
    """Mock context whose diff() scales expressions by an axis factor."""
    context = MagicMock()
    type(context).device = PropertyMock(return_value=torch.device("cpu"))

    context.dataset = MagicMock()
    context.dataset.fields = {"u": object(), "v": object()}
    context.dataset.axes = {}
    context.spatial_axes = list(spatial_axes)

    values = {
        "u": u if u is not None else torch.tensor([1.0, 2.0, 3.0]),
        "v": torch.tensor([4.0, 5.0, 6.0]),
    }

    def get_variable(name: str) -> Tensor:
        if name not in values:
            raise KeyError(f"Variable '{name}' not found")
        return values[name]

    context.get_variable = MagicMock(side_effect=get_variable)
    context.get_derivative = MagicMock(side_effect=KeyError("No derivatives"))
    context.get_constant = MagicMock(side_effect=KeyError("No constants"))

    context.derivative_provider = MagicMock()
    context.derivative_provider.coords = {}

    factors = axis_factors or {axis: 2.0 for axis in spatial_axes}

    def diff(expression: Tensor, axis: str, order: int) -> Tensor:
        if order != 2:
            raise ValueError(f"Expected lap to request order=2, got {order}")
        return expression * factors[axis]

    context.diff = MagicMock(side_effect=diff)
    return context


@pytest.fixture
def executor(default_registry: FunctionRegistry) -> PythonExecutor:
    """PythonExecutor with default registry."""
    return PythonExecutor(default_registry)


# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.smoke
class TestPythonExecutorSmoke:
    """Smoke tests: basic instantiation and method existence."""

    def test_executor_can_be_instantiated(
        self, default_registry: FunctionRegistry
    ) -> None:
        """PythonExecutor can be instantiated."""
        executor = PythonExecutor(default_registry)
        assert executor is not None

    def test_executor_with_custom_max_depth(
        self, default_registry: FunctionRegistry
    ) -> None:
        """PythonExecutor accepts custom max_depth."""
        executor = PythonExecutor(default_registry, max_depth=500)
        assert executor is not None

    def test_execute_method_exists(self, default_registry: FunctionRegistry) -> None:
        """execute method exists on PythonExecutor."""
        executor = PythonExecutor(default_registry)
        assert hasattr(executor, "execute")
        assert callable(executor.execute)

    def test_execute_simple_variable(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute a simple variable expression."""
        result = executor.execute("u", mock_context)
        assert isinstance(result, ExecutorResult)
        assert isinstance(result.value, Tensor)
        assert result.used_diff is False

    def test_execute_simple_function_call(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute a simple function call."""
        result = executor.execute("add(u, v)", mock_context)
        assert isinstance(result, ExecutorResult)
        assert isinstance(result.value, Tensor)


@pytest.mark.smoke
class TestHasOpenFormDiffSmoke:
    """Smoke tests for has_open_form_diff function."""

    def test_function_exists(self) -> None:
        """has_open_form_diff function exists."""
        assert callable(has_open_form_diff)

    def test_returns_bool(self) -> None:
        """has_open_form_diff returns a boolean."""
        result = has_open_form_diff("u")
        assert isinstance(result, bool)


# =============================================================================
# Unit Tests - has_open_form_diff
# =============================================================================


@pytest.mark.unit
class TestHasOpenFormDiff:
    """Unit tests for open-form diff detection."""

    def test_no_diff_simple_variable(self) -> None:
        """Simple variable has no open-form diff."""
        assert has_open_form_diff("u") is False

    def test_no_diff_constant(self) -> None:
        """Constant has no open-form diff."""
        assert has_open_form_diff("1.0") is False

    def test_no_diff_function_call(self) -> None:
        """Regular function call has no open-form diff."""
        assert has_open_form_diff("add(u, v)") is False
        assert has_open_form_diff("sin(u)") is False

    def test_no_diff_nested_function(self) -> None:
        """Nested regular functions have no open-form diff."""
        assert has_open_form_diff("add(mul(u, v), w)") is False

    def test_no_diff_terminal_derivatives(self) -> None:
        """Terminal derivatives (u_x, u_xx) are NOT open-form diff.

        u_x and u_xx are variables (terminals), not function calls.
        They are precomputed and looked up from context.
        """
        assert has_open_form_diff("u_x") is False
        assert has_open_form_diff("u_xx") is False
        assert has_open_form_diff("add(u_x, u_xx)") is False

    def test_has_diff_simple(self) -> None:
        """diff_x(u) is open-form diff."""
        assert has_open_form_diff("diff_x(u)") is True

    def test_has_diff2(self) -> None:
        """diff2_x(u) is open-form diff."""
        assert has_open_form_diff("diff2_x(u)") is True

    def test_has_diff_other_axis(self) -> None:
        """diff_t(u) is open-form diff."""
        assert has_open_form_diff("diff_t(u)") is True

    def test_has_diff_in_nested_expr(self) -> None:
        """Open-form diff inside nested expression is detected."""
        assert has_open_form_diff("add(diff_x(u), v)") is True
        assert has_open_form_diff("mul(u, diff_x(v))") is True
        assert has_open_form_diff("sin(diff_x(u))") is True

    def test_has_diff_mixed_with_terminal(self) -> None:
        """Mixed open-form diff and terminal derivative."""
        # u_x is terminal, diff_x(u) is open-form
        assert has_open_form_diff("add(diff_x(u), u_x)") is True

    def test_has_diff_nested_diff(self) -> None:
        """Nested open-form diff is detected."""
        assert has_open_form_diff("diff_x(diff_x(u))") is True

    def test_has_diff_complex_expression(self) -> None:
        """Complex expression with open-form diff."""
        expr = "add(mul(diff_x(u), u_x), mul(C, diff2_x(u)))"
        assert has_open_form_diff(expr) is True

    def test_diff3_and_higher(self) -> None:
        """Higher order diff operators are detected."""
        assert has_open_form_diff("diff3_x(u)") is True

    def test_lap_is_context_aware_special_operator(self) -> None:
        """lap(u) must route to full path even though it is not diff_*."""
        assert has_open_form_diff("lap(u)") is True

    def test_lap_inside_regular_expression_detected(self) -> None:
        """Nested lap calls must also route to full path."""
        assert has_open_form_diff("add(lap(u), v)") is True

    def test_lap_zero_args_still_routes_to_full_path(self) -> None:
        """lap() routes to dispatcher so arity errors are ValueError."""
        assert has_open_form_diff("lap()") is True


@pytest.mark.unit
class TestHasOpenFormDiffErrors:
    """Error handling for has_open_form_diff."""

    def test_empty_string_raises(self) -> None:
        """Empty string raises ValueError."""
        with pytest.raises(ValueError):
            has_open_form_diff("")

    def test_syntax_error_raises(self) -> None:
        """Syntax error raises ValueError."""
        with pytest.raises(ValueError):
            has_open_form_diff("add(u,")

        with pytest.raises(ValueError):
            has_open_form_diff("(((")

    def test_whitespace_only_raises(self) -> None:
        """Whitespace-only string raises ValueError."""
        with pytest.raises(ValueError):
            has_open_form_diff(" ")


# =============================================================================
# Unit Tests - Fast Path (compile + eval)
# =============================================================================


@pytest.mark.unit
class TestFastPathExecution:
    """Unit tests for fast path execution (no open-form diff)."""

    def test_execute_constant(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute a constant expression."""
        result = executor.execute("1.0", mock_context)
        # Constants should become tensors
        assert result.used_diff is False
        torch.testing.assert_close(
            result.value,
            torch.tensor(1.0, device=mock_context.device),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_execute_variable(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute a variable expression."""
        result = executor.execute("u", mock_context)
        assert result.used_diff is False
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_unary_function(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute a unary function."""
        result = executor.execute("sin(u)", mock_context)
        assert result.used_diff is False
        expected = torch.sin(torch.tensor([1.0, 2.0, 3.0]))
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_binary_function(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute a binary function."""
        result = executor.execute("add(u, v)", mock_context)
        assert result.used_diff is False
        expected = torch.tensor([5.0, 7.0, 9.0]) # [1+4, 2+5, 3+6]
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_nested_expression(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute a nested expression."""
        result = executor.execute("add(mul(u, v), w)", mock_context)
        assert result.used_diff is False
        # mul(u, v) = [4, 10, 18], add(..., w) = [4.5, 10.5, 18.5]
        expected = torch.tensor([4.5, 10.5, 18.5])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_terminal_derivatives(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute expression with terminal derivatives (u_x, u_xx).

        Terminal derivatives are variables from context, not diff calls.
        """
        result = executor.execute("add(u_x, u_xx)", mock_context)
        assert result.used_diff is False
        # u_x = [0.1, 0.2, 0.3], u_xx = [0.01, 0.02, 0.03]
        expected = torch.tensor([0.11, 0.22, 0.33])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_compound_terminal_derivative_preserves_sequence(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """u_x_x means d/dx of u_x, not direct second derivative u_xx."""
        result = executor.execute("u_x_x", mock_context)

        assert result.used_diff is False
        expected = torch.tensor([0.01, 0.02, 0.03])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)
        mock_context.get_derivative.assert_called_with("u", "x", 1)
        mock_context.diff.assert_called()

    def test_execute_complex_fast_path(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute complex expression on fast path."""
        # add(mul(u, u_x), mul(C, u_xx))
        # = mul(u, u_x) + mul(C, u_xx)
        # = [0.1, 0.4, 0.9] + [0.01, 0.02, 0.03]
        # = [0.11, 0.42, 0.93]
        result = executor.execute("add(mul(u, u_x), mul(C, u_xx))", mock_context)
        assert result.used_diff is False
        expected = torch.tensor([0.11, 0.42, 0.93])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)


# =============================================================================
# Unit Tests - Full Path (AST traversal + diff)
# =============================================================================


@pytest.mark.unit
class TestFullPathExecution:
    """Unit tests for full path execution (with open-form diff)."""

    def test_execute_simple_diff(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute simple diff_x(u)."""
        result = executor.execute("diff_x(u)", mock_context)
        assert result.used_diff is True
        # Mock diff multiplies by 0.1 * order = 0.1
        # u = [1, 2, 3], diff_x(u) = [0.1, 0.2, 0.3]
        expected = torch.tensor([0.1, 0.2, 0.3])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_diff2(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute second-order derivative diff2_x(u)."""
        result = executor.execute("diff2_x(u)", mock_context)
        assert result.used_diff is True
        # Mock diff multiplies by 0.1 * order = 0.2
        # u = [1, 2, 3], diff2_x(u) = [0.2, 0.4, 0.6]
        expected = torch.tensor([0.2, 0.4, 0.6])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_diff_mixed_with_regular(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute expression mixing diff and regular functions."""
        # add(diff_x(u), v)
        # diff_x(u) = [0.1, 0.2, 0.3], v = [4, 5, 6]
        # result = [4.1, 5.2, 6.3]
        result = executor.execute("add(diff_x(u), v)", mock_context)
        assert result.used_diff is True
        expected = torch.tensor([4.1, 5.2, 6.3])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_diff_mixed_with_terminal(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute expression with both diff and terminal derivatives."""
        # add(diff_x(u), u_x)
        # diff_x(u) = [0.1, 0.2, 0.3], u_x = [0.1, 0.2, 0.3]
        # result = [0.2, 0.4, 0.6]
        result = executor.execute("add(diff_x(u), u_x)", mock_context)
        assert result.used_diff is True
        expected = torch.tensor([0.2, 0.4, 0.6])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_diff_of_expression(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute diff of a complex expression."""
        # diff_x(mul(u, u))
        # mul(u, u) = [1, 4, 9]
        # diff_x(...) = [0.1, 0.4, 0.9]
        result = executor.execute("diff_x(mul(u, u))", mock_context)
        assert result.used_diff is True
        expected = torch.tensor([0.1, 0.4, 0.9])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_execute_nested_diff(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Execute nested diff operators."""
        # diff_x(diff_x(u))
        # diff_x(u) = [0.1, 0.2, 0.3]
        # diff_x(diff_x(u)) = [0.01, 0.02, 0.03]
        result = executor.execute("diff_x(diff_x(u))", mock_context)
        assert result.used_diff is True
        expected = torch.tensor([0.01, 0.02, 0.03])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_diff_with_extra_positional_args_raises(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """diff operators accept exactly one positional arg

        ``diff_x(u, u_x)`` previously routed through the executor by
        silently consuming only the first arg, hiding malformed RHS
        expressions. Reject the call instead.
        """
        with pytest.raises(ValueError, match="diff_x.*1 argument"):
            executor.execute("diff_x(u, u_x)", mock_context)

    def test_diff_with_keyword_args_raises(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """diff operators do not accept keyword arguments"""
        with pytest.raises(ValueError, match="diff_x.*keyword"):
            executor.execute("diff_x(arg=u)", mock_context)

    def test_diff_with_zero_args_raises_uniform_value_error(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """``diff_x()`` raises a single ValueError regardless of topology

        Previously the GRID fast path leaked a registry KeyError because
        ``diff_x`` is not a registered function, while the SCATTERED full
        path raised ValueError; callers couldn't write one ``except``
        clause that handles both. Now both routes share the same
        ValueError contract via the full-path dispatcher.
        """
        with pytest.raises(ValueError, match="diff_x.*1 argument"):
            executor.execute("diff_x()", mock_context)


@pytest.mark.unit
class TestLapSpecialOperator:
    """Unit tests for lap(expr) special-operator dispatch."""

    def test_lap_1d_matches_second_derivative(self, executor: PythonExecutor) -> None:
        """lap(u) over one spatial axis computes one second derivative."""
        context = _make_lap_mock_context(["x"])

        result = executor.execute("lap(u)", context)

        assert result.used_diff is True
        torch.testing.assert_close(
            result.value,
            torch.tensor([2.0, 4.0, 6.0]),
            rtol=1e-5,
            atol=1e-8,
        )
        context.diff.assert_called_once()
        assert context.diff.call_args.args[1:] == ("x", 2)

    def test_lap_2d_sums_spatial_second_derivatives(
        self, executor: PythonExecutor
    ) -> None:
        """lap(u) sums d2/dx2 and d2/dy2 in spatial_axes order."""
        context = _make_lap_mock_context(
            ["x", "y"],
            axis_factors={"x": 2.0, "y": 3.0},
        )

        result = executor.execute("lap(u)", context)

        torch.testing.assert_close(
            result.value,
            torch.tensor([5.0, 10.0, 15.0]),
            rtol=1e-5,
            atol=1e-8,
        )
        assert [call.args[1] for call in context.diff.call_args_list] == ["x", "y"]

    def test_lap_3d_sums_all_spatial_axes(self, executor: PythonExecutor) -> None:
        """lap(u) supports three spatial axes."""
        context = _make_lap_mock_context(
            ["x", "y", "z"],
            axis_factors={"x": 2.0, "y": 3.0, "z": 4.0},
        )

        result = executor.execute("lap(u)", context)

        torch.testing.assert_close(
            result.value,
            torch.tensor([9.0, 18.0, 27.0]),
            rtol=1e-5,
            atol=1e-8,
        )
        assert [call.args[1] for call in context.diff.call_args_list] == [
            "x",
            "y",
            "z",
        ]

    def test_lap_of_expression_evaluates_inner_first(
        self, executor: PythonExecutor
    ) -> None:
        """lap(mul(u, u)) differentiates the evaluated inner tensor."""
        context = _make_lap_mock_context(["x"], axis_factors={"x": 2.0})

        result = executor.execute("lap(mul(u, u))", context)

        torch.testing.assert_close(
            result.value,
            torch.tensor([2.0, 8.0, 18.0]),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_lap_inside_composite_expression(self, executor: PythonExecutor) -> None:
        """lap can be combined with regular registry operators."""
        context = _make_lap_mock_context(["x"], axis_factors={"x": 2.0})

        result = executor.execute("add(lap(u), mul(u, u))", context)

        torch.testing.assert_close(
            result.value,
            torch.tensor([3.0, 8.0, 15.0]),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_nested_lap_keeps_expression_chain(self, executor: PythonExecutor) -> None:
        """lap(lap(u)) applies the dispatcher recursively."""
        context = _make_lap_mock_context(
            ["x", "y"],
            axis_factors={"x": 2.0, "y": 3.0},
        )

        result = executor.execute("lap(lap(u))", context)

        torch.testing.assert_close(
            result.value,
            torch.tensor([25.0, 50.0, 75.0]),
            rtol=1e-5,
            atol=1e-8,
        )
        assert context.diff.call_count == 4

    def test_lap_requires_one_argument(self, executor: PythonExecutor) -> None:
        """lap() raises a dispatcher ValueError, not a registry error."""
        context = _make_lap_mock_context(["x"])

        with pytest.raises(ValueError, match="lap expects 1 argument, got 0"):
            executor.execute("lap()", context)

    def test_lap_rejects_multiple_arguments(self, executor: PythonExecutor) -> None:
        """lap(u, v) reports the received arity."""
        context = _make_lap_mock_context(["x"])

        with pytest.raises(ValueError, match="lap expects 1 argument, got 2"):
            executor.execute("lap(u, v)", context)

    def test_lap_rejects_keyword_arguments(self, executor: PythonExecutor) -> None:
        """lap(u, axis="x") must raise, not silently swallow the kwarg.

        Mirrors the diff_* arity contract so dispatcher
        semantics stay symmetric across special operators. Without this
        check the kwarg drops on the floor and the user sees a result
        computed as if they wrote ``lap(u)`` — a silent semantic change.
        """
        context = _make_lap_mock_context(["x"])

        with pytest.raises(ValueError, match="does not accept keyword arguments"):
            executor.execute("lap(u, axis='x')", context)

    def test_lap_requires_spatial_axes(self, executor: PythonExecutor) -> None:
        """lap(u) without spatial axes points to lhs_axis setup."""
        context = _make_lap_mock_context([])

        with pytest.raises(ValueError, match="spatial_axes.*lhs_axis"):
            executor.execute("lap(u)", context)

    def test_lap_registry_override_is_rejected(self) -> None:
        """A registered custom lap must not be silently shadowed."""

        def custom_lap(x: Tensor) -> Tensor:
            return x * 100.0

        registry = FunctionRegistry()
        registry.register("lap", custom_lap, arity=1)
        executor = PythonExecutor(registry)
        context = _make_lap_mock_context(["x"])

        with pytest.raises(ValueError, match="lap.*registry.*special operator"):
            executor.execute("lap(u)", context)


# =============================================================================
# Unit Tests - Error Handling
# =============================================================================


@pytest.mark.unit
class TestExecutorErrors:
    """Error handling tests for PythonExecutor."""

    def test_empty_expression_raises(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Empty expression raises ValueError."""
        with pytest.raises(ValueError):
            executor.execute("", mock_context)

    def test_syntax_error_raises(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Syntax error raises ValueError."""
        with pytest.raises(ValueError):
            executor.execute("add(u,", mock_context)

    def test_unknown_variable_raises(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Unknown variable raises KeyError."""
        with pytest.raises(KeyError):
            executor.execute("unknown_var", mock_context)

    def test_unknown_function_raises(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Unknown function raises KeyError (or similar)."""
        with pytest.raises((KeyError, NameError)):
            executor.execute("unknown_func(u)", mock_context)

    def test_invalid_max_depth_raises(self, default_registry: FunctionRegistry) -> None:
        """Invalid max_depth raises ValueError."""
        with pytest.raises(ValueError):
            PythonExecutor(default_registry, max_depth=0)

        with pytest.raises(ValueError):
            PythonExecutor(default_registry, max_depth=-1)


# =============================================================================
# Unit Tests - ExecutorResult
# =============================================================================


@pytest.mark.unit
class TestExecutorResult:
    """Unit tests for ExecutorResult dataclass."""

    def test_result_has_value(self) -> None:
        """ExecutorResult has value attribute."""
        value = torch.tensor([1.0, 2.0])
        result = ExecutorResult(value=value, used_diff=False)
        assert result.value is value

    def test_result_has_used_diff(self) -> None:
        """ExecutorResult has used_diff attribute."""
        value = torch.tensor([1.0])
        result = ExecutorResult(value=value, used_diff=True)
        assert result.used_diff is True


# =============================================================================
# Numerical Tests - Correctness
# =============================================================================


@pytest.mark.numerical
class TestExecutorNumericalCorrectness:
    """Numerical correctness tests for executor."""

    def test_add_correctness(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """add(u, v) computes correct result."""
        result = executor.execute("add(u, v)", mock_context)
        u = mock_context.get_variable("u")
        v = mock_context.get_variable("v")
        expected = u + v
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_mul_correctness(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """mul(u, v) computes correct result."""
        result = executor.execute("mul(u, v)", mock_context)
        u = mock_context.get_variable("u")
        v = mock_context.get_variable("v")
        expected = u * v
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_sub_correctness(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """sub(u, v) computes correct result."""
        result = executor.execute("sub(u, v)", mock_context)
        u = mock_context.get_variable("u")
        v = mock_context.get_variable("v")
        expected = u - v
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_div_correctness(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """div(u, v) computes safe division."""
        result = executor.execute("div(u, v)", mock_context)
        # div uses safe_div, so should not produce NaN/Inf
        assert torch.isfinite(result.value).all()

    def test_sin_correctness(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """sin(u) computes correct result."""
        result = executor.execute("sin(u)", mock_context)
        u = mock_context.get_variable("u")
        expected = torch.sin(u)
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_cos_correctness(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """cos(u) computes correct result."""
        result = executor.execute("cos(u)", mock_context)
        u = mock_context.get_variable("u")
        expected = torch.cos(u)
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_exp_correctness(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """exp(u) computes correct result (clamped)."""
        result = executor.execute("exp(u)", mock_context)
        # exp uses safe_exp with clamping
        assert torch.isfinite(result.value).all()

    def test_n2_correctness(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """n2(u) computes u^2."""
        result = executor.execute("n2(u)", mock_context)
        u = mock_context.get_variable("u")
        expected = u * u
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_n3_correctness(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """n3(u) computes u^3."""
        result = executor.execute("n3(u)", mock_context)
        u = mock_context.get_variable("u")
        expected = u * u * u
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_neg_correctness(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """neg(u) computes -u."""
        result = executor.execute("neg(u)", mock_context)
        u = mock_context.get_variable("u")
        expected = -u
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)


# =============================================================================
# Numerical Tests - Edge Cases
# =============================================================================


@pytest.mark.numerical
class TestExecutorNumericalEdgeCases:
    """Numerical edge case tests for executor."""

    @pytest.fixture
    def edge_case_context(self) -> MagicMock:
        """Context with edge case values."""
        context = MagicMock()
        type(context).device = PropertyMock(return_value=torch.device("cpu"))

        variables = {
            "zero": torch.tensor([0.0, 0.0, 0.0]),
            "small": torch.tensor([1e-15, 1e-15, 1e-15]),
            "large": torch.tensor([1e30, 1e30, 1e30]),
            "mixed": torch.tensor([-1.0, 0.0, 1.0]),
            "u": torch.tensor([1.0, 2.0, 3.0]),
        }

        def get_variable(name: str) -> Tensor:
            if name not in variables:
                raise KeyError(f"Variable '{name}' not found")
            return variables[name]

        context.get_variable = MagicMock(side_effect=get_variable)
        context.derivative_provider = MagicMock()
        mock_diff_fn = lambda expr, axis, order: expr * 0.1 * order # noqa: E731
        context.derivative_provider.diff = MagicMock(side_effect=mock_diff_fn)
        context.diff = MagicMock(side_effect=mock_diff_fn)

        return context

    def test_div_by_zero_safe(
        self, executor: PythonExecutor, edge_case_context: MagicMock
    ) -> None:
        """Division by zero is safe (no NaN/Inf)."""
        result = executor.execute("div(u, zero)", edge_case_context)
        assert torch.isfinite(result.value).all()

    def test_div_by_small_safe(
        self, executor: PythonExecutor, edge_case_context: MagicMock
    ) -> None:
        """Division by very small number is safe."""
        result = executor.execute("div(u, small)", edge_case_context)
        assert torch.isfinite(result.value).all()

    def test_exp_large_safe(
        self, executor: PythonExecutor, edge_case_context: MagicMock
    ) -> None:
        """Exponential of large value is safe (clamped)."""
        result = executor.execute("exp(large)", edge_case_context)
        assert torch.isfinite(result.value).all()

    def test_nested_edge_case_safe(
        self, executor: PythonExecutor, edge_case_context: MagicMock
    ) -> None:
        """Nested expression with edge cases is safe."""
        result = executor.execute("exp(div(u, small))", edge_case_context)
        assert torch.isfinite(result.value).all()

    def test_zero_operations(
        self, executor: PythonExecutor, edge_case_context: MagicMock
    ) -> None:
        """Operations with zero work correctly."""
        # add(zero, u)
        result = executor.execute("add(zero, u)", edge_case_context)
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

        # mul(zero, u)
        result = executor.execute("mul(zero, u)", edge_case_context)
        expected = torch.tensor([0.0, 0.0, 0.0])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)


# =============================================================================
# Numerical Tests - Empty Tensor
# =============================================================================


@pytest.mark.numerical
class TestExecutorEmptyTensor:
    """Tests for empty tensor handling."""

    @pytest.fixture
    def empty_context(self) -> MagicMock:
        """Context with empty tensor."""
        context = MagicMock()
        type(context).device = PropertyMock(return_value=torch.device("cpu"))

        variables = {
            "empty": torch.tensor([]),
        }

        def get_variable(name: str) -> Tensor:
            if name not in variables:
                raise KeyError(f"Variable '{name}' not found")
            return variables[name]

        context.get_variable = MagicMock(side_effect=get_variable)
        context.derivative_provider = MagicMock()

        return context

    def test_unary_on_empty(
        self, executor: PythonExecutor, empty_context: MagicMock
    ) -> None:
        """Unary function on empty tensor returns empty tensor."""
        result = executor.execute("sin(empty)", empty_context)
        assert result.value.shape == torch.Size([0])

    def test_n2_on_empty(
        self, executor: PythonExecutor, empty_context: MagicMock
    ) -> None:
        """n2 on empty tensor returns empty tensor."""
        result = executor.execute("n2(empty)", empty_context)
        assert result.value.shape == torch.Size([0])


# =============================================================================
# Integration-like Tests - Path Selection
# =============================================================================


@pytest.mark.unit
class TestPathSelection:
    """Tests for correct path selection (fast vs full)."""

    def test_fast_path_not_call_diff(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Fast path should NOT call context.diff."""
        # Expression without open-form diff
        executor.execute("add(u, v)", mock_context)
        # context.diff should not be called
        mock_context.diff.assert_not_called()

    def test_full_path_calls_diff(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Full path should call context.diff."""
        # Expression with open-form diff
        executor.execute("diff_x(u)", mock_context)
        # context.diff should be called
        mock_context.diff.assert_called()

    def test_used_diff_flag_fast_path(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Fast path sets used_diff=False."""
        result = executor.execute("add(u, v)", mock_context)
        assert result.used_diff is False

    def test_used_diff_flag_full_path(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Full path sets used_diff=True."""
        result = executor.execute("diff_x(u)", mock_context)
        assert result.used_diff is True

    def test_lap_routes_to_full_path(self, executor: PythonExecutor) -> None:
        """lap(u) must not use fast-path registry evaluation."""
        context = _make_lap_mock_context(["x"])

        result = executor.execute("lap(u)", context)

        assert result.used_diff is True

    def test_should_use_full_path_for_lap(self) -> None:
        """Path selection treats lap as context-aware."""
        context = _make_lap_mock_context(["x"])

        assert _should_use_full_path("lap(u)", context, force_diff_path=False) is True
        assert _should_use_full_path("lap()", context, force_diff_path=False) is True
        assert (
            _should_use_full_path("add(u, v)", context, force_diff_path=False) is False
        )


# =============================================================================
# Unit Tests - Max Depth Enforcement (H001)
# =============================================================================


def _build_nested_expression(base: str, wrapper: str, depth: int) -> str:
    """Build a deeply nested expression.

    Example: _build_nested_expression("u", "sin", 3) -> "sin(sin(sin(u)))"
    """
    expr = base
    for _ in range(depth):
        expr = f"{wrapper}({expr})"
    return expr


@pytest.mark.unit
class TestMaxDepthEnforcement:
    """Test max_depth enforcement during execution."""

    def test_deep_nesting_within_limit(
        self, default_registry: FunctionRegistry, mock_context: MagicMock
    ) -> None:
        """Deep nesting within limit should succeed.

        Generate depth-50 nested sin(sin(...sin(u)...)) expression.
        With max_depth=100, execution should succeed.
        """
        executor = PythonExecutor(default_registry, max_depth=100)
        deep_expr = _build_nested_expression("u", "sin", 50)

        # Should not raise
        result = executor.execute(deep_expr, mock_context)
        assert isinstance(result, ExecutorResult)
        assert torch.isfinite(result.value).all()

    def test_deep_nesting_exceeds_limit(
        self, default_registry: FunctionRegistry, mock_context: MagicMock
    ) -> None:
        """Deep nesting exceeding limit should raise RuntimeError.

        Generate depth-150 nested expression with max_depth=100.
        Should raise RuntimeError for exceeding depth limit.
        """
        executor = PythonExecutor(default_registry, max_depth=100)
        deep_expr = _build_nested_expression("u", "sin", 150)

        with pytest.raises(RuntimeError, match="depth"):
            executor.execute(deep_expr, mock_context)

    def test_custom_max_depth_respected(
        self, default_registry: FunctionRegistry, mock_context: MagicMock
    ) -> None:
        """Custom max_depth should be respected.

        With max_depth=10, depth-15 nesting should fail.
        """
        executor = PythonExecutor(default_registry, max_depth=10)
        deep_expr = _build_nested_expression("u", "sin", 15)

        with pytest.raises(RuntimeError, match="depth"):
            executor.execute(deep_expr, mock_context)

    def test_depth_check_full_path(
        self, default_registry: FunctionRegistry, mock_context: MagicMock
    ) -> None:
        """Max depth should also be enforced on full path (with diff).

        Build deep nesting containing diff operators.
        """
        executor = PythonExecutor(default_registry, max_depth=10)
        # sin(sin(...diff_x(u)...))
        deep_expr = _build_nested_expression("diff_x(u)", "sin", 15)

        with pytest.raises(RuntimeError, match="depth"):
            executor.execute(deep_expr, mock_context)


# =============================================================================
# Unit Tests - has_open_form_diff Edge Cases (H002)
# =============================================================================


@pytest.mark.unit
class TestHasOpenFormDiffEdgeCases:
    """Edge cases for has_open_form_diff detection."""

    def test_diff_as_variable_not_detected(self) -> None:
        """Variable named 'diff' should not be detected as open-form diff.

        'diff' is a valid variable name, not a diff operator call.
        """
        assert has_open_form_diff("diff") is False # just a variable
        assert has_open_form_diff("add(diff, u)") is False # diff is variable

    def test_similar_name_not_detected(self) -> None:
        """Function with similar name should not be detected.

        Names like 'mydiff_x', 'diff123', 'xdiff_x' are NOT diff operators.
        """
        assert has_open_form_diff("mydiff_x(u)") is False
        assert has_open_form_diff("diff123(u)") is False
        assert has_open_form_diff("xdiff_x(u)") is False
        assert has_open_form_diff("prediff_x(u)") is False

    def test_exact_diff_pattern_detected(self) -> None:
        """Only exact diff_axis pattern should be detected.

        Valid patterns: diff_x, diff2_x, diff_t, diff3_y, etc.
        """
        assert has_open_form_diff("diff_x(u)") is True
        assert has_open_form_diff("diff2_x(u)") is True
        assert has_open_form_diff("diff_t(u)") is True
        assert has_open_form_diff("diff3_y(u)") is True

    def test_diff_without_axis_not_detected(self) -> None:
        """'diff(u)' without axis should not be detected (invalid format).

        diff_axis requires the underscore and axis suffix.
        """
        assert has_open_form_diff("diff(u)") is False

    def test_diff_underscore_only_not_detected(self) -> None:
        """'diff_(u)' with underscore but no axis should not be detected."""
        assert has_open_form_diff("diff_(u)") is False

    def test_diff_uppercase_not_detected(self) -> None:
        """Uppercase DIFF_x should not be detected (case sensitive)."""
        assert has_open_form_diff("DIFF_x(u)") is False
        assert has_open_form_diff("Diff_x(u)") is False

    def test_diff_in_string_literal_not_detected(self) -> None:
        """String containing 'diff_x' should not trigger false positive.

        Note: This tests that we parse AST, not just regex match.
        """
        # This is a string literal, not a function call
        # Python AST should not detect this as a Call node
        assert has_open_form_diff("'diff_x(u)'") is False

    def test_diff_with_empty_parens(self) -> None:
        """``diff_x()`` routes to the full path

        Even a malformed zero-arg diff call must hit the full-path
        dispatcher so it gets a uniform ValueError, instead of leaking
        a fast-path KeyError on GRID topology while raising ValueError
        on SCATTERED.
        """
        assert has_open_form_diff("diff_x()") is True

    def test_lap_name_variable_not_detected(self) -> None:
        """Bare variable named lap is not a special-operator call."""
        assert has_open_form_diff("lap") is False

    def test_lap_call_detected_independent_of_arg_count(self) -> None:
        """lap calls always route to the full dispatcher path."""
        assert has_open_form_diff("lap()") is True
        assert has_open_form_diff("lap(u)") is True

    def test_special_operator_set_is_derived_from_stub_map(self) -> None:
        """Special operator routing and conflict checks share one name source."""
        assert frozenset(_SPECIAL_OPERATOR_STUBS) == _SPECIAL_OPERATORS


# =============================================================================
# Unit Tests - force_diff_path and auto-detect fields=None (Change 6)
# =============================================================================


def _make_pinn_context(
    field_values: dict[str, Tensor] | None = None,
) -> MagicMock:
    """Create a PINN-like context where dataset.fields=None.

    In PINN mode, field values come from provider.get_field(), not
    from dataset.fields. The dataset has fields=None (SCATTERED topology).
    """
    context = MagicMock()
    type(context).device = PropertyMock(return_value=torch.device("cpu"))

    # PINN: dataset.fields is None (no observed field data)
    context.dataset = MagicMock()
    context.dataset.fields = None
    context.dataset.axes = None

    # Provider resolves fields from NN model
    values = field_values or {
        "u": torch.tensor([1.0, 2.0, 3.0]),
    }

    def get_field(name: str) -> Tensor:
        if name not in values:
            raise KeyError(f"Field '{name}' not found in provider")
        return values[name]

    context.derivative_provider = MagicMock()
    context.derivative_provider.get_field = MagicMock(side_effect=get_field)

    def mock_diff(expression: Tensor, axis: str, order: int) -> Tensor:
        return expression * 0.1 * order

    context.derivative_provider.diff = MagicMock(side_effect=mock_diff)
    context.diff = MagicMock(side_effect=mock_diff)

    # get_variable fails (no observed data)
    def get_variable(name: str) -> Tensor:
        raise KeyError(f"Variable '{name}' not found in dataset")

    context.get_variable = MagicMock(side_effect=get_variable)

    # No constants
    def get_constant(name: str) -> float:
        raise KeyError(f"Constant '{name}' not found")

    context.get_constant = MagicMock(side_effect=get_constant)

    return context


class _ScatteredLinearModel(nn.Module):
    """Simple autograd model for SCATTERED executor tests."""

    def forward(self, *, x: Tensor, t: Tensor) -> Tensor:
        return x + 2.0 * t


class _ScatteredQuarticModel(nn.Module):
    """Quartic autograd model with nonzero nested Laplacian."""

    def forward(self, *, x: Tensor, y: Tensor, t: Tensor) -> Tensor:
        return x**4 + y**4 + t


def _make_scattered_autograd_context() -> tuple[ExecutionContext, dict[str, Tensor]]:
    """Create a SCATTERED execution context backed by AutogradProvider."""
    coords = {
        "x": torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64).requires_grad_(True),
        "t": torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64).requires_grad_(True),
    }
    dataset = PDEDataset(
        name="scattered_executor",
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
        axis_order=["x", "t"],
        fields=None,
        lhs_field="u",
        lhs_axis="t",
    )
    provider = AutogradProvider(
        model=_ScatteredLinearModel().double(),
        coords=coords,
        dataset=dataset,
    )
    return ExecutionContext(dataset=dataset, derivative_provider=provider), coords


def _make_scattered_quartic_context() -> tuple[ExecutionContext, dict[str, Tensor]]:
    """Create a SCATTERED context where lap(lap(u)) is analytically known."""
    coords = {
        "x": torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64).requires_grad_(True),
        "y": torch.tensor([0.25, 0.75, 1.25], dtype=torch.float64).requires_grad_(True),
        "t": torch.tensor([0.2, 0.4, 0.6], dtype=torch.float64).requires_grad_(True),
    }
    dataset = PDEDataset(
        name="scattered_quartic_lap",
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
        axis_order=["x", "y", "t"],
        fields=None,
        lhs_field="u",
        lhs_axis="t",
    )
    provider = AutogradProvider(
        model=_ScatteredQuarticModel().double(),
        coords=coords,
        dataset=dataset,
    )
    return ExecutionContext(dataset=dataset, derivative_provider=provider), coords


def _make_grid_context_2d() -> ExecutionContext:
    """Create a 2D grid execution context for regression coverage."""
    x = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    dataset = PDEDataset(
        name="grid_executor",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=torch.sin(xx) * torch.exp(-tt))},
        lhs_field="u",
        lhs_axis="t",
    )
    provider = FiniteDiffProvider(dataset, max_order=1)
    return ExecutionContext(dataset=dataset, derivative_provider=provider)


@pytest.mark.unit
class TestForceDiffPath:
    """Tests for force_diff_path parameter (kd2 Change 6).

    When force_diff_path=True, the executor always uses the AST traversal
    path which resolves fields from provider.get_field(). This is needed
    for PINN mode where dataset.fields=None.
    """

    def test_force_diff_path_resolves_field_from_provider(
        self, executor: PythonExecutor
    ) -> None:
        """force_diff_path=True resolves bare 'u' from provider, not dataset."""
        ctx = _make_pinn_context()
        result = executor.execute("u", ctx, force_diff_path=True)
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)
        # Provider's get_field must have been called
        ctx.derivative_provider.get_field.assert_called()

    def test_force_diff_path_sets_used_diff_true(
        self, executor: PythonExecutor
    ) -> None:
        """force_diff_path=True sets used_diff=True in result."""
        ctx = _make_pinn_context()
        result = executor.execute("u", ctx, force_diff_path=True)
        assert result.used_diff is True

    def test_force_diff_path_with_function_call(self, executor: PythonExecutor) -> None:
        """force_diff_path=True works with function expressions like n2(u)."""
        ctx = _make_pinn_context()
        result = executor.execute("n2(u)", ctx, force_diff_path=True)
        expected = torch.tensor([1.0, 4.0, 9.0])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_force_diff_path_false_is_default(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Default force_diff_path=False preserves existing fast-path behavior."""
        result = executor.execute("add(u, v)", mock_context)
        assert result.used_diff is False

    def test_force_diff_path_backward_compatible(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Existing call sites (no force_diff_path arg) still work."""
        # This should work without the keyword argument
        result = executor.execute("add(u, v)", mock_context)
        assert isinstance(result, ExecutorResult)


@pytest.mark.unit
class TestAutoDetectFieldsNone:
    """Tests for automatic full-path routing when dataset.fields is None.

    When the context's dataset has fields=None (PINN/SCATTERED), the
    executor should automatically route to the full path even without
    force_diff_path=True. This avoids KeyError in the fast path.
    """

    def test_auto_routes_full_path_when_fields_none(
        self, executor: PythonExecutor
    ) -> None:
        """Auto-detect: fields=None routes to full path without explicit flag."""
        ctx = _make_pinn_context()
        # Without force_diff_path — should auto-detect
        result = executor.execute("u", ctx)
        expected = torch.tensor([1.0, 2.0, 3.0])
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)

    def test_auto_detect_sets_used_diff_true(self, executor: PythonExecutor) -> None:
        """Auto-detect marks used_diff=True when fields=None."""
        ctx = _make_pinn_context()
        result = executor.execute("u", ctx)
        assert result.used_diff is True

    def test_auto_detect_does_not_affect_normal_context(
        self, executor: PythonExecutor, mock_context: MagicMock
    ) -> None:
        """Normal context (fields exist) still uses fast path."""
        # mock_context doesn't set dataset.fields to None,
        # so fast path should be used
        result = executor.execute("add(u, v)", mock_context)
        assert result.used_diff is False

    def test_auto_detect_with_diff_expression(self, executor: PythonExecutor) -> None:
        """Auto-detect + diff expression both trigger full path (no conflict)."""
        ctx = _make_pinn_context()
        result = executor.execute("diff_x(u)", ctx)
        assert result.used_diff is True
        ctx.diff.assert_called()


@pytest.mark.unit
class TestScatteredCoordinateResolution:
    """Regression coverage for bare coordinate resolution in SCATTERED mode."""

    def test_scattered_bare_coordinate_resolves_from_provider(
        self, executor: PythonExecutor
    ) -> None:
        """Bare coordinate names resolve from provider coords when dataset has none."""
        context, coords = _make_scattered_autograd_context()

        result = executor.execute("t", context)

        torch.testing.assert_close(result.value, coords["t"], rtol=1e-5, atol=1e-8)
        assert result.used_diff is True

    def test_scattered_bare_field_still_resolves_from_model(
        self, executor: PythonExecutor
    ) -> None:
        """Bare field lookup still prefers provider.get_field()."""
        context, coords = _make_scattered_autograd_context()
        provider = context.derivative_provider
        provider.get_field = MagicMock(wraps=provider.get_field)

        result = executor.execute("u", context)
        expected = coords["x"] + 2.0 * coords["t"]

        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)
        provider.get_field.assert_called_once_with("u")

    def test_scattered_compound_expression_with_coord_and_diff(
        self, executor: PythonExecutor
    ) -> None:
        """Mixed coord/diff expressions work in SCATTERED mode."""
        context, coords = _make_scattered_autograd_context()

        result = executor.execute("mul(t, diff_x(u))", context)

        torch.testing.assert_close(result.value, coords["t"], rtol=1e-5, atol=1e-8)

    def test_grid_mode_keeps_broadcast_coordinate_behavior(
        self, executor: PythonExecutor
    ) -> None:
        """Grid-mode full-path execution still uses context-broadcasted coords."""
        context = _make_grid_context_2d()
        expected = context.get_variable("x") * context.diff(
            context.get_variable("u"),
            "t",
            1,
        )

        result = executor.execute("mul(x, diff_t(u))", context)

        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)


@pytest.mark.unit
class TestLapAutogradPath:
    """Autograd coverage for nested lap dispatch in SCATTERED mode."""

    def test_nested_lap_keeps_autograd_chain(self, executor: PythonExecutor) -> None:
        """For u=x^4+y^4+t, lap(lap(u)) is 48 over spatial axes x,y."""
        context, coords = _make_scattered_quartic_context()

        result = executor.execute("lap(lap(u))", context)

        expected = torch.full_like(coords["x"], 48.0)
        assert result.used_diff is True
        torch.testing.assert_close(result.value, expected, rtol=1e-5, atol=1e-8)
