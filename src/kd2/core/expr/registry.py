"""Function registry for expression evaluation.

The FunctionRegistry manages callable functions (operators, terminals) and provides:
- Registration of functions with metadata (arity, commutativity)
- Context generation for eval()
- Query methods for function properties

This replaces the old Library class with a simpler, Python AST-focused design.
"""

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from kd2.core.safety import safe_div, safe_exp, safe_log

# Magnitude clamp for power operators (n2, n3).
# Inputs with |x| > _POWER_CLAMP are clamped before squaring/cubing.
# n2(1e6) = 1e12, n3(1e6) = 1e18 — both safe for float32 (max ~3.4e38).
# Matches threshold; consistent with safe_exp clamp pattern.
_POWER_CLAMP = 1e6

# Type alias for callable functions with any signature
# We use Any because functions can have varying arities (0, 1, 2 args)
AnyCallable = Callable[..., Any]


# Valid Python identifier pattern (ASCII only)
_VALID_NAME_PATTERN = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")

# Dangerous names that should be rejected for security
_DANGEROUS_NAMES = frozenset(
    {
        "__builtins__",
        "__class__",
        "__import__",
        "__globals__",
        "__code__",
        "__dict__",
        "__name__",
        "__module__",
        "__annotations__",
        "__doc__",
        "__slots__",
        "__init__",
        "__new__",
        "__del__",
        "__call__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__getattribute__",
        "eval",
        "exec",
        "compile",
        "open",
        "input",
        "breakpoint",
    }
)


@dataclass
class _FunctionInfo:
    """Internal storage for function metadata."""

    func: AnyCallable
    arity: int
    commutative: bool


class FunctionRegistry:
    """Registry of functions available for expression evaluation.

    The FunctionRegistry maintains a collection of callables and their metadata:
    - Functions (operators like add, mul, sin, etc.)
    - Terminals (variables like u_x, u_xx that resolve to values)

    All numerical operations use safe_* functions to prevent NaN/Inf.

    Examples:
        >>> reg = FunctionRegistry.create_default()
        >>> ctx = reg.get_context()
        >>> result = eval("add(mul(x, y), z)", ctx)

        >>> reg.register("custom_op", lambda a, b: a + b, arity=2)
        >>> reg.is_commutative("add")
        True
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._functions: dict[str, _FunctionInfo] = {}

    def _validate_name(self, name: str) -> None:
        """Validate function name for safety and correctness.

        Args:
            name: Function name to validate.

        Raises:
            ValueError: If name is empty, not a valid ASCII identifier,
                reserved (dunder), or dangerous.
        """
        if not name:
            raise ValueError("Function name cannot be empty")

        # Must be valid ASCII identifier
        if not _VALID_NAME_PATTERN.match(name):
            raise ValueError(f"Function name '{name}' must be a valid ASCII identifier")

        # Reject all dunder names
        if name.startswith("__") and name.endswith("__"):
            raise ValueError(
                f"Function name '{name}' is reserved (dunder names forbidden)"
            )

        # Reject specific dangerous names
        if name in _DANGEROUS_NAMES:
            raise ValueError(f"Function name '{name}' is dangerous/forbidden")

    def _validate_func(self, func: AnyCallable) -> None:
        """Validate that func is callable.

        Args:
            func: Function to validate.

        Raises:
            TypeError: If func is not callable.
        """
        if func is None or not callable(func):
            raise TypeError("func must be callable")

    def _validate_arity(self, arity: int) -> None:
        """Validate arity value.

        Args:
            arity: Arity to validate.

        Raises:
            ValueError: If arity is negative.
        """
        if arity < 0:
            raise ValueError(f"arity must be >= 0, got {arity}")

    def register(
        self,
        name: str,
        func: AnyCallable,
        arity: int,
        commutative: bool = False,
    ) -> None:
        """Register a function in the registry.

        Args:
            name: Function name (e.g., "add", "sin", "u_x").
            func: Callable implementing the function.
            arity: Number of arguments (0 for terminals, 1 for unary, 2 for binary).
            commutative: Whether the function is commutative (for arity=2 only).

        Raises:
            ValueError: If a function with the same name already exists.
            ValueError: If name is empty or dangerous.
            ValueError: If arity is negative.
            TypeError: If func is not callable.
        """
        # Validation
        self._validate_name(name)
        self._validate_func(func)
        self._validate_arity(arity)

        # Check for duplicates
        if name in self._functions:
            raise ValueError(f"Function '{name}' already exists in registry")

        self._functions[name] = _FunctionInfo(
            func=func,
            arity=arity,
            commutative=commutative,
        )

    def get_context(self) -> dict[str, AnyCallable]:
        """Get evaluation context for use with eval().

        Returns a dict mapping function names to callables, suitable for
        passing to eval() as the globals/locals dict.

        Returns:
            Dict mapping names to callables.

        Example:
            >>> ctx = registry.get_context()
            >>> result = eval("add(mul(2, 3), 4)", ctx)
        """
        return {name: info.func for name, info in self._functions.items()}

    def is_commutative(self, name: str) -> bool:
        """Check if a function is commutative.

        Args:
            name: Function name.

        Returns:
            True if the function is commutative, False otherwise.

        Raises:
            KeyError: If the function is not registered.
        """
        if name not in self._functions:
            raise KeyError(f"Function '{name}' not found in registry")
        return self._functions[name].commutative

    def get_arity(self, name: str) -> int:
        """Get the arity of a function.

        Args:
            name: Function name.

        Returns:
            Number of arguments (0, 1, or 2).

        Raises:
            KeyError: If the function is not registered.
        """
        if name not in self._functions:
            raise KeyError(f"Function '{name}' not found in registry")
        return self._functions[name].arity

    def get_func(self, name: str) -> AnyCallable:
        """Get the callable function by name.

        Args:
            name: Function name.

        Returns:
            The registered callable.

        Raises:
            KeyError: If the function is not registered.
        """
        if name not in self._functions:
            raise KeyError(f"Function '{name}' not found in registry")
        return self._functions[name].func

    def has(self, name: str) -> bool:
        """Check if a function is registered.

        Args:
            name: Function name.

        Returns:
            True if registered, False otherwise.
        """
        return name in self._functions

    def list_names(self) -> list[str]:
        """List all registered function names.

        Returns:
            List of all registered function names.
        """
        return list(self._functions.keys())

    def get_by_arity(self, arity: int) -> list[str]:
        """Get all function names with the specified arity.

        Args:
            arity: Number of arguments (0, 1, or 2).

        Returns:
            List of function names with the given arity.
        """
        return [name for name, info in self._functions.items() if info.arity == arity]

    @classmethod
    def create_default(cls) -> "FunctionRegistry":
        """Create a registry with default operators.

        Default operators:
        - Binary (arity=2): add, mul, sub, div (add/mul are commutative)
        - Unary (arity=1): sin, cos, exp, log, neg, n2, n3, lap

        All use safe_* functions for numerical stability.

        Returns:
            FunctionRegistry with default operators registered.
        """
        reg = cls()

        # Binary operators
        reg.register("add", torch.add, arity=2, commutative=True)
        reg.register("mul", torch.mul, arity=2, commutative=True)
        reg.register("sub", torch.sub, arity=2, commutative=False)
        reg.register("div", _safe_div_wrapper, arity=2, commutative=False)

        # Unary operators
        reg.register("sin", torch.sin, arity=1)
        reg.register("cos", torch.cos, arity=1)
        reg.register("exp", safe_exp, arity=1)
        reg.register("log", safe_log, arity=1)
        reg.register("neg", _neg, arity=1)
        reg.register("n2", _square, arity=1)
        reg.register("n3", _cube, arity=1)
        reg.register("lap", _lap_stub, arity=1)

        return reg


# Helper functions for default operators


def _safe_div_wrapper(a: Tensor, b: Tensor) -> Tensor:
    """Wrapper for safe_div with default eps."""
    return safe_div(a, b)


def _neg(x: Tensor) -> Tensor:
    """Compute -x."""
    return -x


def _square(x: Tensor) -> Tensor:
    """Compute x^2 with magnitude clamping for numerical safety.

    Clamps input to [-_POWER_CLAMP, _POWER_CLAMP] before squaring.
    Gradient vanishes outside the clamped range (saturation, not discontinuity).
    """
    x_c = torch.clamp(x, min=-_POWER_CLAMP, max=_POWER_CLAMP)
    return x_c * x_c


def _cube(x: Tensor) -> Tensor:
    """Compute x^3 with magnitude clamping for numerical safety.

    Clamps input to [-_POWER_CLAMP, _POWER_CLAMP] before cubing.
    Gradient vanishes outside the clamped range (saturation, not discontinuity).
    """
    x_c = torch.clamp(x, min=-_POWER_CLAMP, max=_POWER_CLAMP)
    return x_c * x_c * x_c


def _lap_stub(_x: Tensor) -> Tensor:
    """Prevent lap from being executed through registry fast path."""
    raise NotImplementedError(
        "lap requires a context-aware executor and cannot run via registry"
    )
