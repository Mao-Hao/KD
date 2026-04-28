"""PDE integrator: Method of Lines for discovered equations.

Given a discovered PDE in the form u_t = f(u, u_x, u_xx, ...),
integrates the equation forward in time using scipy's solve_ivp
with finite difference spatial derivatives.

Design:
- Lambdify rhs_expr to numpy callable
- Each timestep: FD compute spatial derivatives from u(t), call RHS, advance
- Boundary: periodic (padding FD) or Dirichlet (from dataset)
- Output: torch.Tensor predicted field
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy # type: ignore[import-untyped]
import torch
from numpy.typing import NDArray
from scipy.integrate import solve_ivp # type: ignore[import-untyped]
from sympy.core.function import AppliedUndef # type: ignore[import-untyped]
from torch import Tensor

from kd2.core.expr.naming import parse_compound_derivative
from kd2.data.derivatives.finite_diff import (
    DX_ZERO_FLOOR,
    UNIFORM_GRID_RTOL,
    is_uniform_grid,
)
from kd2.data.schema import DataTopology, PDEDataset

logger = logging.getLogger(__name__)

# Default solver method (handles stiff PDEs)
DEFAULT_METHOD = "Radau"


@dataclass
class IntegrationResult:
    """Result of PDE time integration.

    Attributes:
        success: Whether integration completed without errors.
        predicted_field: Predicted u(x, t) field as torch.Tensor,
            or None if integration failed.
        warning: Description of any issues encountered.
        diverged_at_t: Time at which divergence was detected, if any.
    """

    success: bool
    predicted_field: Tensor | None = None
    warning: str = ""
    diverged_at_t: float | None = None


@dataclass
class _ParsedSymbols:
    """Classified free symbols from the RHS expression."""

    state_vars: set[str] = field(default_factory=set)
    derivatives: dict[str, tuple[str, list[tuple[str, int]]]] = field(
        default_factory=dict
    )
    coordinates: set[str] = field(default_factory=set)
    unsupported_functions: set[str] = field(default_factory=set)
    unknown_symbols: set[str] = field(default_factory=set)


# Pattern for sympy_bridge derivative placeholders like ``d1_x``, ``d2_y``.
# These appear when SGA produces nested open-form derivatives such as
# ``diff_x(diff_x(u))`` or ``diff_x(u + u_x)`` for which there is no
_DERIV_PLACEHOLDER_RE = re.compile(r"^d\d+_[a-zA-Z]\w*$")


def _is_derivative_placeholder(name: str) -> bool:
    """Detect sympy_bridge placeholder symbols (``d{order}_{axis}``)."""
    return bool(_DERIV_PLACEHOLDER_RE.match(name))


def _classify_symbols(
    rhs_expr: sympy.Expr,
    field_names: set[str],
    coord_names: set[str],
) -> _ParsedSymbols:
    """Classify free symbols into state vars, derivatives, and coordinates."""
    parsed = _ParsedSymbols()
    for call in rhs_expr.atoms(AppliedUndef):
        parsed.unsupported_functions.add(str(call.func))

    for sym in rhs_expr.free_symbols:
        name = str(sym)
        derivative = parse_compound_derivative(
            name,
            known_fields=field_names,
            known_axes=coord_names,
        )
        if derivative is not None:
            parsed.derivatives[name] = derivative
            continue
        if name in field_names:
            parsed.state_vars.add(name)
            continue
        if name in coord_names:
            parsed.coordinates.add(name)
            continue
        # Unknown symbol — bucket it. The integrator will return a clean
        # IntegrationResult(success=False, warning=...) before lambdify,
        # rather than letting NameError leak through solve_ivp as a cryptic
        # "Cannot convert expression to float" message
        parsed.unknown_symbols.add(name)
    return parsed


def _central_diff_1(
    u: NDArray[np.floating[Any]],
    dx: float,
    periodic: bool,
) -> NDArray[np.floating[Any]]:
    """First-order central finite difference on a 1-D array."""
    if periodic:
        padded = np.concatenate(([u[-1]], u, [u[0]]))
        return (padded[2:] - padded[:-2]) / (2.0 * dx)
    result = np.empty_like(u)
    result[1:-1] = (u[2:] - u[:-2]) / (2.0 * dx)
    # Boundaries: second-order one-sided stencils
    result[0] = (-3.0 * u[0] + 4.0 * u[1] - u[2]) / (2.0 * dx)
    result[-1] = (3.0 * u[-1] - 4.0 * u[-2] + u[-3]) / (2.0 * dx)
    return result


def _central_diff_2(
    u: NDArray[np.floating[Any]],
    dx: float,
    periodic: bool,
) -> NDArray[np.floating[Any]]:
    """Second-order central finite difference on a 1-D array."""
    if periodic:
        padded = np.concatenate(([u[-1]], u, [u[0]]))
        return (padded[2:] - 2.0 * padded[1:-1] + padded[:-2]) / (dx * dx)
    result = np.empty_like(u)
    result[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (dx * dx)
    # Boundaries: copy interior-nearest value
    result[0] = result[1]
    result[-1] = result[-2]
    return result


def _central_diff_3(
    u: NDArray[np.floating[Any]],
    dx: float,
    periodic: bool,
) -> NDArray[np.floating[Any]]:
    """Third-order central finite difference on a 1-D array.

    Interior: (u[i+2] - 2*u[i+1] + 2*u[i-1] - u[i-2]) / (2*dx^3)
    Boundary: one-sided stencils.
    """
    dx3 = dx * dx * dx
    if periodic:
        padded = np.concatenate((u[-2:], u, u[:2]))
        return (padded[4:] - 2.0 * padded[3:-1] + 2.0 * padded[1:-3] - padded[:-4]) / (
            2.0 * dx3
        )
    result = np.empty_like(u)
    # Interior
    result[2:-2] = (u[4:] - 2.0 * u[3:-1] + 2.0 * u[1:-3] - u[:-4]) / (2.0 * dx3)
    # Left boundary: forward one-sided
    result[0] = (-u[0] + 3.0 * u[1] - 3.0 * u[2] + u[3]) / dx3
    result[1] = (-u[1] + 3.0 * u[2] - 3.0 * u[3] + u[4]) / dx3
    # Right boundary: backward one-sided
    result[-1] = (u[-1] - 3.0 * u[-2] + 3.0 * u[-3] - u[-4]) / dx3
    result[-2] = (u[-2] - 3.0 * u[-3] + 3.0 * u[-4] - u[-5]) / dx3
    return result


# Minimum number of grid points required per FD stencil order.
# order 1: central [i-1, i, i+1] -> 3 pts
# order 2: central [i-1, i, i+1] -> 3 pts
# order 3: 5-point [i-2, ..., i+2] -> 5 pts
_MIN_POINTS: dict[int, int] = {1: 3, 2: 3, 3: 5}


def _finite_diff(
    u: NDArray[np.floating[Any]],
    dx: float,
    order: int,
    periodic: bool,
) -> NDArray[np.floating[Any]]:
    """Dispatch to 1st, 2nd, or 3rd order FD."""
    min_pts = _MIN_POINTS.get(order)
    if min_pts is not None and len(u) < min_pts:
        raise ValueError(
            f"Need at least {min_pts} points for order-{order} FD, got {len(u)}"
        )
    if order == 1:
        return _central_diff_1(u, dx, periodic)
    if order == 2:
        return _central_diff_2(u, dx, periodic)
    if order == 3:
        return _central_diff_3(u, dx, periodic)
    raise ValueError(f"Unsupported derivative order: {order}")


@dataclass
class _SpatialAxisInfo:
    """Spatial axis metadata for FD computation."""

    name: str
    values: NDArray[np.floating[Any]]
    dx: float
    periodic: bool
    axis_index: int


def _check_spatial_uniformity(
    dataset: PDEDataset,
    spatial_axes: list[str],
) -> str | None:
    """Verify each spatial axis has positive, uniform spacing.

    The FD stencils in ``_central_diff_*`` use a single ``dx`` derived
    from ``vals[1] - vals[0]``; non-uniform grids would silently produce
    wrong derivatives, and a degenerate grid with ``dx=0`` divides by
    zero. Returns a warning string if any axis fails the check, ``None``
    if all are uniform and non-degenerate ( + cross-model review).

    Tolerance: ``rtol=1e-4`` accepts ``torch.linspace(dtype=float32)``
    drift (rel deviation up to ~1e-5 at n≥100) while still catching any
    real non-uniform grid (geometric / log spacing have rel deviation
    O(1)). The dx=0 guard runs before the relative-tolerance check so
    a constant-coordinate axis is reported, not silently passed.
    """
    assert dataset.axes is not None # noqa: S101
    for axis_name in spatial_axes:
        vals = dataset.axes[axis_name].values.detach().cpu().numpy().astype(np.float64)
        # Match FD's rejection for size<2 grids — `is_uniform_grid` returns
        # False for size<2 so we cannot rely on the predicate to produce a
        # meaningful diff/dx0 below. Report consistently with FD instead of
        # silently passing.
        if vals.size < 2:
            return (
                f"Spatial axis '{axis_name}' must have >=2 points to verify "
                f"uniformity, got {vals.size}"
            )
        diffs = np.diff(vals)
        dx0 = float(diffs[0])
        if abs(dx0) < DX_ZERO_FLOOR:
            return (
                f"Spatial axis '{axis_name}' has degenerate spacing dx={dx0:.6g}; "
                f"finite-difference stencils require nonzero dx"
            )
        # Delegate to the shared predicate so FD + integrator agree.
        if not is_uniform_grid(vals, rtol=UNIFORM_GRID_RTOL):
            max_dev = float(np.max(np.abs(diffs - dx0)))
            return (
                f"Spatial axis '{axis_name}' has non-uniform spacing "
                f"(dx[0]={dx0:.6g}, max deviation={max_dev:.6g}); "
                f"finite-difference stencils require a uniform grid"
            )
    return None


def _build_spatial_info(
    dataset: PDEDataset,
    spatial_axes: list[str],
) -> list[_SpatialAxisInfo]:
    """Extract spatial axis info from dataset."""
    assert dataset.axes is not None # noqa: S101
    info_list: list[_SpatialAxisInfo] = []
    for idx, axis_name in enumerate(spatial_axes):
        axis = dataset.axes[axis_name]
        vals = axis.values.detach().cpu().numpy().astype(np.float64)
        dx = float(vals[1] - vals[0]) if len(vals) > 1 else 1.0
        info_list.append(
            _SpatialAxisInfo(
                name=axis_name,
                values=vals,
                dx=dx,
                periodic=axis.is_periodic,
                axis_index=idx,
            )
        )
    return info_list


def _build_lambdify_args(
    parsed: _ParsedSymbols,
    coord_names: set[str],
) -> list[sympy.Symbol]:
    """Build ordered argument list for sympy.lambdify."""
    args: list[sympy.Symbol] = []
    for name in sorted(parsed.state_vars):
        args.append(sympy.Symbol(name))
    for name in sorted(parsed.derivatives.keys()):
        args.append(sympy.Symbol(name))
    for name in sorted(coord_names & parsed.coordinates):
        args.append(sympy.Symbol(name))
    return args


def _finite_diff_along_axis(
    u: NDArray[np.floating[Any]],
    axis_index: int,
    dx: float,
    order: int,
    periodic: bool,
) -> NDArray[np.floating[Any]]:
    """Apply 1-D FD along a specific axis of an N-D array."""
    moved = np.moveaxis(u, axis_index, 0)
    result = np.empty_like(moved)
    it = np.nditer(moved[0], flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        slc = (slice(None),) + idx
        result[slc] = _finite_diff(moved[slc], dx, order, periodic)
        it.iternext()
    return np.moveaxis(result, 0, axis_index)


def _mol_rhs(
    u_flat: NDArray[np.floating[Any]],
    rhs_func: Any,
    parsed: _ParsedSymbols,
    spatial_info: list[_SpatialAxisInfo],
    spatial_shape: tuple[int, ...],
    sym_args: list[sympy.Symbol],
    lhs_field: str,
) -> NDArray[np.floating[Any]]:
    """Compute du/dt for Method of Lines.

    Defense-in-depth: this function assumes a single-field RHS (all
    derivatives and state vars reference ``lhs_field``). ``integrate_pde``
    enforces this via a cross-field guard before calling here; the asserts
    below pin the contract so a future multi-field extension cannot silently
    reuse ``_mol_rhs`` with wrong semantics.
    """
    for _name, (fld, _orders) in parsed.derivatives.items():
        assert fld == lhs_field, ( # noqa: S101
            f"_mol_rhs single-field invariant violated: derivative "
            f"references '{fld}' but lhs_field is '{lhs_field}'"
        )
    for svar in parsed.state_vars:
        assert svar == lhs_field, ( # noqa: S101
            f"_mol_rhs single-field invariant violated: state var "
            f"'{svar}' does not match lhs_field '{lhs_field}'"
        )
    u = u_flat.reshape(spatial_shape)
    axis_lookup = {info.name: info for info in spatial_info}

    # Compute needed spatial derivatives
    deriv_values: dict[str, NDArray[np.floating[Any]]] = {}
    for name, (_fld, axis_orders) in parsed.derivatives.items():
        deriv = u
        for axis, order in axis_orders:
            axis_info = axis_lookup.get(axis)
            if axis_info is None:
                raise ValueError(
                    f"Spatial axis '{axis}' not found for derivative '{name}'",
                )
            if len(spatial_shape) == 1:
                deriv = _finite_diff(
                    deriv,
                    axis_info.dx,
                    order,
                    axis_info.periodic,
                )
            else:
                deriv = _finite_diff_along_axis(
                    deriv,
                    axis_info.axis_index,
                    axis_info.dx,
                    order,
                    axis_info.periodic,
                )
        deriv_values[name] = deriv

    # Build coordinate grids
    coord_grids: dict[str, NDArray[np.floating[Any]]] = {}
    for info in spatial_info:
        if info.name in parsed.coordinates:
            if len(spatial_info) == 1:
                coord_grids[info.name] = info.values
            else:
                shape = [1] * len(spatial_shape)
                shape[info.axis_index] = len(info.values)
                coord_grids[info.name] = np.broadcast_to(
                    info.values.reshape(shape),
                    spatial_shape,
                )

    # Assemble arguments in sym_args order. ``sym_args`` is built from
    # ``parsed.state_vars``, ``parsed.derivatives`` and the spatial subset of
    # ``parsed.coordinates`` (see ``_build_lambdify_args``). All those names
    # are filled into ``u``/``deriv_values``/``coord_grids`` above, and any
    # other symbol is short-circuited by ``integrate_pde`` via
    # ``parsed.unknown_symbols`` before we reach this function. The else
    # branch therefore signals a broken invariant rather than runtime input.
    call_args: list[Any] = []
    for sym in sym_args:
        name = str(sym)
        if name in parsed.state_vars:
            call_args.append(u)
        elif name in deriv_values:
            call_args.append(deriv_values[name])
        elif name in coord_grids:
            call_args.append(coord_grids[name])
        else:
            raise AssertionError( # pragma: no cover - guarded invariant
                f"_mol_rhs received unrecognised symbol '{name}' in "
                "sym_args; _classify_symbols should have routed it via "
                "parsed.unknown_symbols before "
                "lambdify."
            )

    dudt_raw: Any = rhs_func(*call_args)

    if np.isscalar(dudt_raw):
        dudt = np.full(spatial_shape, dudt_raw, dtype=np.float64)
    else:
        # np.array (not np.asarray) to guarantee a copy — avoids
        # mutating solve_ivp's internal state when setting Dirichlet BC below.
        dudt = np.array(dudt_raw, dtype=np.float64)

    # Enforce Dirichlet BC: du/dt = 0 at boundaries for non-periodic axes
    for info in spatial_info:
        if not info.periodic:
            idx_first: list[Any] = [slice(None)] * len(spatial_shape)
            idx_first[info.axis_index] = 0
            dudt[tuple(idx_first)] = 0.0

            idx_last: list[Any] = [slice(None)] * len(spatial_shape)
            idx_last[info.axis_index] = -1
            dudt[tuple(idx_last)] = 0.0

    return dudt.ravel()


def _check_divergence(
    y: NDArray[np.floating[Any]],
    t: NDArray[np.floating[Any]],
) -> float | None:
    """Return the first time at which NaN/Inf appears, or None."""
    for i in range(y.shape[1]):
        if not np.all(np.isfinite(y[:, i])):
            return float(t[i])
    return None


def _reconstruct_field(
    y: NDArray[np.floating[Any]],
    spatial_shape: tuple[int, ...],
    n_times: int,
    time_dim: int,
    device: torch.device,
) -> Tensor:
    """Reshape solve_ivp output to the dataset field shape."""
    output_shape = list(spatial_shape)
    output_shape.insert(time_dim, n_times)

    field_np = np.zeros(output_shape, dtype=np.float64)
    for i in range(n_times):
        u_spatial = y[:, i].reshape(spatial_shape)
        idx: list[Any] = [slice(None)] * len(output_shape)
        idx[time_dim] = i
        field_np[tuple(idx)] = u_spatial

    return torch.tensor(field_np, dtype=torch.float64, device=device)


def integrate_pde(
    rhs_expr: sympy.Expr,
    dataset: PDEDataset,
    *,
    method: str = DEFAULT_METHOD,
    max_step: float | None = None,
) -> IntegrationResult:
    """Integrate a PDE forward in time using Method of Lines.

    Converts the RHS sympy expression into a numpy callable,
    computes spatial derivatives via finite differences at each
    timestep, and uses scipy.integrate.solve_ivp to advance in time.

    Args:
        rhs_expr: RHS of the PDE (sympy expression). Free symbols
            may include field values (e.g., ``u``) and derivative
            symbols (e.g., ``u_x``, ``u_xx``).
        dataset: PDEDataset providing grid coordinates, boundary
            conditions (via AxisInfo.is_periodic), and initial
            conditions (field values at t=0).
        method: scipy solve_ivp integration method (default: "Radau").
        max_step: Maximum step size for the integrator. If None,
            the solver chooses automatically.

    Returns:
        IntegrationResult with predicted_field as torch.Tensor if
        successful, or with success=False and diagnostic info if not.
    """
    # Guard: GRID only
    if dataset.topology != DataTopology.GRID:
        return IntegrationResult(
            success=False,
            warning=(
                f"Integration requires GRID topology, got {dataset.topology.value}"
            ),
        )

    if dataset.axes is None or dataset.axis_order is None or dataset.fields is None:
        return IntegrationResult(
            success=False,
            warning="Dataset missing axes, axis_order, or fields",
        )

    if not dataset.lhs_field or not dataset.lhs_axis:
        return IntegrationResult(
            success=False,
            warning="Dataset missing lhs_field or lhs_axis",
        )

    time_axis = dataset.lhs_axis
    spatial_axes = dataset.spatial_axes

    if time_axis not in dataset.axes:
        return IntegrationResult(
            success=False,
            warning=f"Time axis '{time_axis}' not found in dataset axes",
        )

    t_vals = dataset.axes[time_axis].values.detach().cpu().numpy().astype(np.float64)
    t_span = (float(t_vals[0]), float(t_vals[-1]))

    # Field data for initial condition
    field_name = dataset.lhs_field
    field_data = (
        dataset.fields[field_name].values.detach().cpu().numpy().astype(np.float64)
    )

    non_uniform_warning = _check_spatial_uniformity(dataset, spatial_axes)
    if non_uniform_warning is not None:
        return IntegrationResult(success=False, warning=non_uniform_warning)

    spatial_info = _build_spatial_info(dataset, spatial_axes)

    # Classify symbols
    field_names = set(dataset.fields.keys())
    coord_names = set(spatial_axes)
    parsed = _classify_symbols(rhs_expr, field_names, coord_names)

    if parsed.unsupported_functions:
        unsupported = sorted(parsed.unsupported_functions)
        return IntegrationResult(
            success=False,
            warning=(
                "Unsupported function calls in RHS expression for integrate_pde: "
                f"{unsupported}. integrate_pde supports field, coordinate, and "
                "explicit derivative symbols only. Expand context-aware operators "
                "such as lap(...) to explicit derivative symbols, or add explicit "
                "lambdify support for the function."
            ),
        )

    # — clean degradation for nested open-form derivative placeholders
    # (e.g. ``d1_x``) and other unrecognised symbols. Without this guard the
    # downstream lambdify generates code that references the placeholder as a
    # free name, scipy.solve_ivp eventually fails with the cryptic
    # "Cannot convert expression to float" warning. Return a clear message
    # instead so VizEngine's plot-level error isolation can skip the
    # field-comparison and pde-residual plots cleanly.
    if parsed.unknown_symbols:
        unknowns = sorted(parsed.unknown_symbols)
        placeholders = [n for n in unknowns if _is_derivative_placeholder(n)]
        detail = (
            f"nested-derivative placeholders {placeholders}"
            if placeholders
            else f"unrecognised symbols {unknowns}"
        )
        return IntegrationResult(
            success=False,
            warning=(
                f"integrate_pde skipped: RHS contains {detail} "
                "Field-comparison and pde-residual plots skipped; "
                "other plots and metrics unaffected."
            ),
        )

    # Guard: cross-field reference — _mol_rhs maps all state_vars and
    # derivative fields to the single LHS field. If the RHS references a
    # different field (directly or via derivatives like v_x), the integrator
    # would silently compute wrong values. Reject early.
    cross_fields: list[str] = []
    for dname, (dfld, _axis_orders) in parsed.derivatives.items():
        if dfld != field_name:
            cross_fields.append(dname)
    for svar in sorted(parsed.state_vars):
        if svar != field_name:
            cross_fields.append(svar)

    if cross_fields:
        return IntegrationResult(
            success=False,
            warning=(
                f"Cross-field references not supported: "
                f"{cross_fields} reference fields other than "
                f"LHS field '{field_name}'"
            ),
        )

    # Lambdify
    sym_args = _build_lambdify_args(parsed, coord_names)
    try:
        rhs_func = sympy.lambdify(sym_args, rhs_expr, modules=["numpy"])
    except Exception as exc:
        return IntegrationResult(
            success=False,
            warning=f"Failed to lambdify RHS expression: {exc}",
        )

    # Initial condition: first time slice
    time_dim = dataset.axis_order.index(time_axis)
    u0 = np.take(field_data, 0, axis=time_dim).ravel().astype(np.float64)

    spatial_shape = tuple(dataset.axes[a].values.numel() for a in spatial_axes)

    # solve_ivp wrapper
    def ode_rhs(
        _t: float,
        u_flat: NDArray[np.floating[Any]],
    ) -> NDArray[np.floating[Any]]:
        return _mol_rhs(
            u_flat,
            rhs_func,
            parsed,
            spatial_info,
            spatial_shape,
            sym_args,
            field_name,
        )

    solve_kwargs: dict[str, Any] = {
        "method": method,
        "t_eval": t_vals,
        "dense_output": False,
    }
    if max_step is not None:
        solve_kwargs["max_step"] = max_step

    try:
        sol = solve_ivp(ode_rhs, t_span, u0, **solve_kwargs)
    except Exception as exc:
        return IntegrationResult(
            success=False,
            warning=f"solve_ivp failed: {exc}",
        )

    if sol.status == -1:
        return IntegrationResult(
            success=False,
            warning=f"solve_ivp integration failed: {sol.message}",
        )

    # Divergence check
    diverged_at_t = _check_divergence(sol.y, sol.t)

    device = torch.device("cpu")
    predicted = _reconstruct_field(
        sol.y,
        spatial_shape,
        len(sol.t),
        time_dim,
        device,
    )

    if diverged_at_t is not None:
        return IntegrationResult(
            success=False,
            predicted_field=predicted,
            diverged_at_t=diverged_at_t,
            warning=f"Integration diverged at t={diverged_at_t:.4g}",
        )

    return IntegrationResult(success=True, predicted_field=predicted)


__all__ = [
    "IntegrationResult",
    "integrate_pde",
]
