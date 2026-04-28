"""PythonExecutor: expression execution with context-aware derivative support.

This module implements the PythonExecutor which executes Python AST expressions
using a routed execution strategy:

1. Fast path (90%): compile() + eval() for expressions without open-form diff
2. Full path (10%): AST traversal for expressions with diff_x(), diff2_x(), etc.
3. Special operators: context-aware scalar operators such as lap(expr)

Design principles:
- Device-aware tensor operations
- Numerical safety via FunctionRegistry safe functions
- Clear separation between terminal derivatives (u_x), open-form diff
  (diff_x(expr)), and context-aware special operators (lap(expr))
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import Tensor

from kd2.core.expr.naming import parse_compound_derivative, parse_derivative_name
from kd2.core.expr.registry import FunctionRegistry, _lap_stub

if TYPE_CHECKING:
    from kd2.core.executor.context import ExecutionContext


# Pattern for valid diff operators: diff_x, diff2_x, diff3_t, etc.
# Format: diff[order]_axis where order is optional digits and axis is letters
_DIFF_PATTERN = re.compile(r"^diff([0-9]*)_([a-z]+)$")

_SPECIAL_OPERATOR_STUBS: dict[str, Any] = {"lap": _lap_stub}
# Context-aware scalar-to-scalar operators outside the diff_* naming family.
_SPECIAL_OPERATORS: frozenset[str] = frozenset(_SPECIAL_OPERATOR_STUBS)


@dataclass
class ExecutorResult:
    """Result of expression execution.

    Attributes:
        value: The computed tensor value.
        used_diff: Whether the full AST traversal path was used. True when
            the expression contains open-form diff operators or special
            operators, force_diff_path was set, or dataset.fields is None
            (PINN/SCATTERED mode).
    """

    value: Tensor
    used_diff: bool


class PythonExecutor:
    """Execute Python AST expressions with routed derivative support.

    Fast path: compile() + eval() for expressions without open-form diff
    Full path: AST traversal for expressions with diff_x(), diff2_x(), etc.
    Special operators: context-aware scalar operators dispatched in full path.

    The executor uses FunctionRegistry to resolve all function calls and
    ExecutionContext to access variables and derivative providers.

    Examples:
        >>> registry = FunctionRegistry.create_default()
        >>> executor = PythonExecutor(registry)
        >>> result = executor.execute("add(u, v)", context)
        >>> print(result.value, result.used_diff)
    """

    def __init__(
        self,
        registry: FunctionRegistry,
        max_depth: int = 1000,
    ) -> None:
        """Initialize executor.

        Args:
            registry: Function registry with all operators.
            max_depth: Maximum recursion depth for AST traversal.

        Raises:
            ValueError: If max_depth is not positive.
        """
        if max_depth <= 0:
            raise ValueError(f"max_depth must be positive, got {max_depth}")

        self._registry = registry
        self._max_depth = max_depth

    @property
    def registry(self) -> FunctionRegistry:
        """Public access to the function registry."""
        return self._registry

    def execute(
        self,
        code: str,
        context: ExecutionContext,
        force_diff_path: bool = False,
    ) -> ExecutorResult:
        """Execute expression and return result.

        Uses two-path strategy:
        1. If no open-form diff operators: fast path with compile() + eval()
        2. If open-form diff operators present: full path with AST traversal

        Args:
            code: Python expression string, e.g., "add(mul(u, u_x), C)"
            context: Execution context with variables and derivative provider.
            force_diff_path: Force full AST traversal even without open-form diff.

        Returns:
            ExecutorResult with computed tensor and execution metadata.

        Raises:
            ValueError: If expression is invalid (empty, syntax error).
            KeyError: If unknown variable or function is referenced.
            RuntimeError: If execution fails (depth exceeded, etc.).
        """
        # Validate input
        if not code or not code.strip():
            raise ValueError("Expression cannot be empty")

        # Parse the expression
        try:
            tree = ast.parse(code, mode="eval")
        except SyntaxError as e:
            raise ValueError(f"Syntax error in expression: {e}") from e

        # Check depth limit before execution
        ast_depth = _get_ast_depth(tree.body)
        if ast_depth > self._max_depth:
            raise RuntimeError(f"Maximum recursion depth ({self._max_depth}) exceeded")

        use_full_path = _should_use_full_path(
            code,
            context,
            force_diff_path=force_diff_path,
        )

        if use_full_path:
            # Full path: AST traversal with diff support
            value = self._execute_with_diff(tree.body, context, depth=0)
        else:
            # Fast path: compile + eval
            value = self._execute_simple(code, context)

        return ExecutorResult(value=value, used_diff=use_full_path)

    def _execute_simple(
        self,
        code: str,
        context: ExecutionContext,
    ) -> Tensor:
        """Execute expression using fast path (compile + eval).

        This path reads field values from the dataset (via context.get_variable),
        NOT from the derivative provider's model. For expressions without
        open-form diff operators, this is the standard execution path.

        Args:
            code: Python expression string.
            context: Execution context with variables.

        Returns:
            Computed tensor value.

        Raises:
            KeyError: If unknown variable is referenced.
        """
        # Compile the expression
        compiled = compile(ast.parse(code, mode="eval"), "<expr>", "eval")

        # Build evaluation context with builtins disabled for security.
        # Setting __builtins__ to {} prevents access to dangerous functions
        # like open(), exec(), __import__(), etc.
        eval_ctx: dict[str, Any] = {"__builtins__": {}}
        eval_ctx.update(self._registry.get_context()) # Registered functions

        # Add variables from context - wrap get_variable for direct access
        # We need to intercept variable access through a custom namespace
        known_fields, known_axes = _context_name_sets(context)
        eval_ctx = _VariableAccessDict(
            eval_ctx,
            context,
            known_fields=known_fields,
            known_axes=known_axes,
        )

        # Execute with restricted globals to prevent code injection.
        # SECURITY: globals MUST have __builtins__: {} to disable built-in functions.
        # An empty dict {} would be auto-populated with default builtins by Python!
        try:
            result = eval(compiled, {"__builtins__": {}}, eval_ctx) # noqa: S307
        except NameError as e:
            # Convert NameError to KeyError for unknown variables
            # Extract variable name from error message
            raise KeyError(str(e)) from e

        # Ensure result is a tensor
        if not isinstance(result, Tensor):
            result = torch.tensor(result, device=context.device, dtype=torch.float32)

        return result

    def _execute_with_diff(
        self,
        node: ast.expr,
        context: ExecutionContext,
        depth: int,
    ) -> Tensor:
        """Execute expression using full path (AST traversal with diff).

        NOTE: In the full path (diff-enabled), field variables are fetched from
        the derivative provider's model (via get_field) rather than the dataset.
        This is necessary to maintain computation graph connectivity for autograd.
        As a result, field values may differ slightly from the fast path (which
        reads from dataset) if the model has not perfectly fitted the data.

        Args:
            node: AST expression node.
            context: Execution context.
            depth: Current recursion depth.

        Returns:
            Computed tensor value.

        Raises:
            RuntimeError: If maximum depth is exceeded.
            KeyError: If unknown variable or function is referenced.
        """
        # Check depth limit (use >= to ensure depth never exceeds max_depth)
        if depth >= self._max_depth:
            raise RuntimeError(f"Maximum recursion depth ({self._max_depth}) exceeded")

        if isinstance(node, ast.Call):
            return self._execute_call(node, context, depth)

        elif isinstance(node, ast.Name):
            # Variable reference (with full derivative-aware resolution)
            return self._resolve_name_for_diff(node.id, context)

        elif isinstance(node, ast.Constant):
            # Numeric constant
            return torch.tensor(node.value, device=context.device, dtype=torch.float32)

        elif isinstance(node, ast.UnaryOp):
            if isinstance(node.op, ast.USub):
                # Unary minus: -x
                operand = self._execute_with_diff(node.operand, context, depth + 1)
                return -operand
            elif isinstance(node.op, ast.UAdd):
                # Unary plus: +x
                return self._execute_with_diff(node.operand, context, depth + 1)
            else:
                raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        elif isinstance(node, ast.BinOp):
            # Binary operations (shouldn't normally appear in our DSL,
            # but handle for robustness)
            left = self._execute_with_diff(node.left, context, depth + 1)
            right = self._execute_with_diff(node.right, context, depth + 1)

            if isinstance(node.op, ast.Add):
                return left + right
            elif isinstance(node.op, ast.Sub):
                return left - right
            elif isinstance(node.op, ast.Mult):
                return left * right
            elif isinstance(node.op, ast.Div):
                from kd2.core.safety import safe_div

                return safe_div(left, right)
            else:
                raise ValueError(f"Unsupported binary operator: {type(node.op)}")

        else:
            raise ValueError(f"Unsupported AST node type: {type(node)}")

    def _execute_call(
        self,
        node: ast.Call,
        context: ExecutionContext,
        depth: int,
    ) -> Tensor:
        """Execute a function call node.

        Args:
            node: AST Call node.
            context: Execution context.
            depth: Current recursion depth.

        Returns:
            Computed tensor value.
        """
        # Get function name
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls are supported")

        func_name = node.func.id

        if _is_diff_operator(func_name):
            # Open-form diff: compute derivative of inner expression.
            # Strict arity contract — no silent drop of extra positional or
            # keyword arguments: a malformed RHS like
            # ``diff_x(u, u_x)`` must surface as a ValueError instead of
            # being computed as ``diff_x(u)`` with ``u_x`` discarded.
            if node.keywords:
                raise ValueError(
                    f"Diff operator '{func_name}' does not accept keyword arguments"
                )
            if len(node.args) != 1:
                raise ValueError(
                    f"Diff operator '{func_name}' expects exactly 1 "
                    f"argument, got {len(node.args)}"
                )

            inner = self._execute_with_diff(node.args[0], context, depth + 1)
            axis, order = _parse_diff_name(func_name)
            return context.diff(inner, axis, order)

        if _is_special_operator(func_name):
            return self._dispatch_special_operator(func_name, node, context, depth)

        # Regular function call
        # Evaluate arguments first
        args = [self._execute_with_diff(arg, context, depth + 1) for arg in node.args]

        # Get function from registry
        try:
            func = self._registry.get_func(func_name)
        except KeyError as e:
            raise KeyError(f"Function '{func_name}' not found in registry") from e

        result: Tensor = func(*args)
        return result

    def _dispatch_special_operator(
        self,
        name: str,
        node: ast.Call,
        context: ExecutionContext,
        depth: int,
    ) -> Tensor:
        """Dispatch context-aware special operators.

        Args:
            name: Special operator name.
            node: Call node containing operator arguments.
            context: Execution context with PDE metadata and derivative access.
            depth: Current AST traversal depth.

        Returns:
            Tensor result of the dispatched operator.

        Raises:
            ValueError: If the operator arguments or context metadata are invalid.
        """
        self._raise_if_registry_conflicts_with_special_operator(name)

        if name == "lap":
            if node.keywords:
                raise ValueError("lap operator does not accept keyword arguments")
            if len(node.args) != 1:
                raise ValueError(f"lap expects 1 argument, got {len(node.args)}")

            spatial_axes = context.spatial_axes
            if not spatial_axes:
                raise ValueError(
                    "lap operator requires non-empty spatial_axes; "
                    "ensure dataset.lhs_axis is set and axis_order has spatial axes"
                )

            inner = self._execute_with_diff(node.args[0], context, depth + 1)
            result = context.diff(inner, spatial_axes[0], 2)
            for axis in spatial_axes[1:]:
                result = result + context.diff(inner, axis, 2)
            return result

        raise ValueError(f"Unknown special operator: {name}")

    def _raise_if_registry_conflicts_with_special_operator(self, name: str) -> None:
        """Reject registry callables shadowed by special-operator dispatch."""
        expected_stub = _SPECIAL_OPERATOR_STUBS.get(name)
        if expected_stub is None:
            return

        try:
            registered_func = self._registry.get_func(name)
        except KeyError:
            return

        if registered_func is not expected_stub:
            raise ValueError(
                f"{name} registry entry conflicts with context-aware special "
                f"operator '{name}'"
            )

    def _resolve_name_for_diff(
        self,
        name: str,
        context: ExecutionContext,
    ) -> Tensor:
        """Resolve a name to a tensor in diff-enabled execution.

        NOTE: In the full path (diff-enabled), field variables are fetched from
        the derivative provider's model (via get_field) rather than the dataset.
        This is necessary to maintain computation graph connectivity for autograd.
        As a result, field values may differ slightly from the fast path (which
        reads from dataset) if the model has not perfectly fitted the data.

        Lookup order:
        1. Provider fields (get_field) - maintains computation graph for autograd
        2. Provider coordinates (SCATTERED mode fallback)
        3. Context variables (fields, coordinates)
        4. Terminal derivatives (u_x, u_xx, etc.)
        5. Named constants

        Args:
            name: Symbol name to resolve.
            context: Execution context.

        Returns:
            Resolved tensor value.

        Raises:
            KeyError: If symbol cannot be resolved.
        """
        # 1. If provider supports get_field and name is a field,
        # use provider to maintain computation graph connectivity.
        # This is critical for AutogradProvider where fields must
        # stay connected to the coordinate computation graph.
        provider = context.derivative_provider
        get_field = getattr(provider, "get_field", None)
        known_fields, known_axes = _context_name_sets(context)
        if get_field is not None and callable(get_field):
            try:
                result = get_field(name)
                # Validate result is a real Tensor (not a mock or other type)
                if isinstance(result, Tensor):
                    return result
            except (KeyError, AttributeError, NotImplementedError):
                pass

        # 1.5 Try as coordinate from provider.
        #
        # Two routes feed this branch:
        # * SCATTERED mode: dataset.fields/axes are unavailable but the
        # provider still holds coordinate tensors for autograd.
        # * GRID + autograd: ``diff_x(x)`` must return 1.0 — the
        # coordinate tensor in ``provider.coords`` is a leaf with
        # ``requires_grad=True`` so autograd can differentiate it,
        # whereas ``context.get_variable`` would hand back the
        # dataset's detached values and break the graph
        #
        # When the dataset.axes value is also present we still defer to
        # ``ExecutionContext`` (broadcasted coords) so finite-difference
        # GRID execution keeps the field-shaped coord tensors it needs;
        # only autograd-style leaves with ``requires_grad`` jump the
        # queue.
        dataset_axes = getattr(getattr(context, "dataset", None), "axes", None)
        provider_coords = provider.coords
        if name in provider_coords:
            coord_t = provider_coords[name]
            in_dataset_axes = isinstance(dataset_axes, dict) and name in dataset_axes
            if not in_dataset_axes or coord_t.requires_grad:
                return coord_t

        # 2. Try as variable (field or coordinate)
        try:
            return context.get_variable(name)
        except KeyError:
            pass

        # 3. Try to parse as terminal derivative (u_x, u_xx, etc.)
        derivative = _try_parse_terminal_derivative(
            name,
            context,
            known_fields=known_fields,
            known_axes=known_axes,
        )
        if derivative is not None:
            return derivative

        # 4. Try as named constant
        try:
            value = context.get_constant(name)
            return torch.tensor(value, device=context.device, dtype=torch.float32)
        except KeyError:
            pass

        raise KeyError(f"Unknown symbol: {name}")


class _VariableAccessDict(dict[str, Any]):
    """Custom dict that intercepts missing key access to look up variables.

    This allows eval() to access context variables transparently.
    Lookup order:
    1. Variables (fields and coordinates)
    2. Terminal derivatives (u_x, u_xx, etc.)
    3. Named constants (nu, pi, etc.)
    """

    def __init__(
        self,
        base: dict[str, Any],
        context: ExecutionContext,
        *,
        known_fields: set[str] | None = None,
        known_axes: set[str] | None = None,
    ) -> None:
        super().__init__(base)
        self._context = context
        self._known_fields = known_fields
        self._known_axes = known_axes

    def __missing__(self, key: str) -> Tensor:
        """Called when a key is not found in the dict.

        Lookup order:
        1. Try as variable (field or coordinate)
        2. Try to parse as derivative (u_x, u_xx, etc.)
        3. Try as constant (nu, pi, etc.)

        Args:
            key: The variable/derivative/constant name to look up.

        Returns:
            Tensor value for the symbol.

        Raises:
            KeyError: If symbol not found in any lookup path.
        """
        # 1. Try as variable (field or coordinate)
        try:
            return self._context.get_variable(key)
        except KeyError:
            pass

        # 2. Try to parse as derivative (u_x, u_xx, etc.)
        derivative = _try_parse_terminal_derivative(
            key,
            self._context,
            known_fields=self._known_fields,
            known_axes=self._known_axes,
        )
        if derivative is not None:
            return derivative

        # 3. Try as constant (nu, pi, etc.)
        try:
            value = self._context.get_constant(key)
            # Convert to tensor on the correct device
            return torch.tensor(value, device=self._context.device, dtype=torch.float32)
        except KeyError:
            pass

        # 4. All lookups failed
        raise KeyError(f"Unknown symbol: {key}")


def _try_parse_terminal_derivative(
    name: str,
    context: ExecutionContext,
    *,
    known_fields: set[str] | None = None,
    known_axes: set[str] | None = None,
) -> Tensor | None:
    """Try to parse name as a terminal or compound derivative pattern.

    Patterns:
    - u_x -> field="u", axis="x", order=1
    - u_xx -> field="u", axis="x", order=2
    - u_xxx -> field="u", axis="x", order=3
    - u_x_x -> d/dx of u_x, preserving sequential derivative semantics

    Compact terminal names use one derivative segment. Compound names use
    multiple segments and are evaluated sequentially through ``context.diff``.

    Args:
        name: Potential derivative name.
        context: Execution context for derivative lookup.

    Returns:
        Tensor if successfully parsed, None otherwise.
    """
    parsed = parse_derivative_name(
        name,
        known_fields=known_fields,
        known_axes=known_axes,
    )
    if parsed is not None:
        field, axis, order = parsed
        try:
            return context.get_derivative(field, axis, order)
        except (KeyError, ValueError):
            return None

    compound = parse_compound_derivative(
        name,
        known_fields=known_fields,
        known_axes=known_axes,
    )
    if compound is None:
        return None
    field, segments = compound
    try:
        first_axis, first_order = segments[0]
        result = context.get_derivative(field, first_axis, first_order)
        for axis, order in segments[1:]:
            result = context.diff(result, axis, order)
        return result
    except (AttributeError, KeyError, NotImplementedError, ValueError):
        return None


def _context_name_sets(
    context: ExecutionContext,
) -> tuple[set[str] | None, set[str] | None]:
    """Extract known field/axis names when the context exposes a dataset."""
    dataset = getattr(context, "dataset", None)
    fields = getattr(dataset, "fields", None)
    axes = getattr(dataset, "axes", None)
    known_fields = set(fields) if isinstance(fields, dict) else None
    known_axes = set(axes) if isinstance(axes, dict) else None
    return known_fields, known_axes


def _should_use_full_path(
    code: str,
    context: ExecutionContext,
    *,
    force_diff_path: bool,
) -> bool:
    """Decide whether expression execution must use AST traversal."""
    return (
        force_diff_path or _context_fields_missing(context) or has_open_form_diff(code)
    )


def _context_fields_missing(context: ExecutionContext) -> bool:
    """Return True when dataset.fields is unavailable.

    Returns True when dataset is None (no data at all) or when
    dataset.fields is None (PINN/SCATTERED mode). In both cases
    the fast path cannot resolve field names.
    """
    dataset = getattr(context, "dataset", None)
    if dataset is None:
        return True
    return getattr(dataset, "fields", None) is None


def _get_ast_depth(node: ast.AST) -> int:
    """Calculate the maximum depth of an AST tree using iteration.

    Uses an iterative approach with an explicit stack to avoid Python's
    recursion limit (~1000). This is important because we use this function
    to check depth BEFORE execution, so a recursive implementation would
    fail with RecursionError before we could report the depth exceeded error.

    Args:
        node: AST node to measure depth from.

    Returns:
        Maximum depth of the tree (1 for a leaf node).
    """
    max_depth = 0
    # Stack contains (node, current_depth) pairs
    stack: list[tuple[ast.AST, int]] = [(node, 1)]

    while stack:
        current, depth = stack.pop()
        max_depth = max(max_depth, depth)

        # Add children to stack with incremented depth
        for child in ast.iter_child_nodes(current):
            stack.append((child, depth + 1))

    return max_depth


def _is_diff_operator(name: str) -> bool:
    """Check if name is a diff operator.

    Valid patterns: diff_x, diff2_x, diff_t, diff3_y, etc.

    Args:
        name: Function name to check.

    Returns:
        True if name matches diff operator pattern.
    """
    return _DIFF_PATTERN.match(name) is not None


def _is_special_operator(name: str) -> bool:
    """Check if name is a context-aware special operator."""
    return name in _SPECIAL_OPERATORS


def _parse_diff_name(name: str) -> tuple[str, int]:
    """Parse diff operator name to (axis, order).

    Args:
        name: Diff operator name (e.g., "diff_x", "diff2_x", "diff3_t").

    Returns:
        Tuple of (axis, order).

    Raises:
        ValueError: If name doesn't match diff pattern.
        ValueError: If order is 0 (d^0/dx^0 = identity, not a valid derivative).

    Examples:
        >>> _parse_diff_name("diff_x")
        ('x', 1)
        >>> _parse_diff_name("diff2_x")
        ('x', 2)
        >>> _parse_diff_name("diff3_t")
        ('t', 3)
    """
    match = _DIFF_PATTERN.match(name)
    if not match:
        raise ValueError(f"Invalid diff operator name: {name}")

    order_str, axis = match.groups()
    order = int(order_str) if order_str else 1

    # Reject order=0: d^0/dx^0 is identity, not a meaningful derivative operation
    if order < 1:
        raise ValueError(
            f"Diff order must be >= 1, got {order} in '{name}'. "
            f"Order 0 (identity) is not a valid derivative operation."
        )

    return axis, order


def has_open_form_diff(code: str) -> bool:
    """Check if expression needs full-path context-aware execution.

    Open-form diff: diff_x(expr), diff2_x(expr), etc. (function calls).
    NOT open-form: u_x, u_xx (terminals/variables, no arguments).
    Special operators: lap(expr) and lap() always route to full path so the
    dispatcher can apply context metadata and report arity errors.

    Any ``diff_*`` call — including malformed ones like ``diff_x()``,
    ``diff_x(u, u_x)`` or ``diff_x(arg=u)`` — also routes to the full
    path so the diff dispatcher can raise a uniform ValueError instead
    of leaking a fast-path KeyError on GRID topology while raising
    ValueError on SCATTERED.

    Args:
        code: Python expression string.

    Returns:
        True if expression contains open-form diff operators, False otherwise.

    Raises:
        ValueError: If code is empty or has syntax errors.

    Examples:
        >>> has_open_form_diff("add(u, v)")
        False
        >>> has_open_form_diff("add(u_x, u_xx)")
        False
        >>> has_open_form_diff("diff_x(u)")
        True
        >>> has_open_form_diff("add(diff_x(u), u_x)")
        True
    """
    # Validate input
    if not code or not code.strip():
        raise ValueError("Expression cannot be empty")

    # Parse the expression
    try:
        tree = ast.parse(code, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Syntax error in expression: {e}") from e

    # Walk the AST looking for full-path operators
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_name = node.func.id
            if _is_special_operator(func_name):
                return True
            # Any diff-shaped Call routes to full path — well-formed calls
            # are computed there and malformed ones (zero / multiple /
            # keyword args) get a uniform ValueError from the dispatcher.
            if _is_diff_operator(func_name):
                return True

    return False
