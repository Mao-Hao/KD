"""Internal helpers for ``PDEDataset.from_arrays``.

Kept separate from ``schema.py`` so the schema module stays focused on
the dataclass definitions and stays under the 500-line cap. These are
private helpers — not part of the public API.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

if TYPE_CHECKING:
    from kd2.data.schema import AxisInfo, FieldData


def to_float_tensor(
    values: Any,
    dtype: torch.dtype,
    *,
    kind: str,
    label: str,
) -> torch.Tensor:
    """Convert numpy/list/tensor input to a float tensor with ``dtype``.

    Args:
        values: Input array-like to convert.
        dtype: Target float dtype.
        kind: 'coord' or 'field' (used in error messages).
        label: Name of the entity (axis name or field name) for messages.

    Returns:
        torch.Tensor with the requested dtype.

    Raises:
        TypeError: If ``dtype`` is not a floating-point dtype.
        ValueError: If conversion fails.
    """
    if not is_floating_dtype(dtype):
        raise TypeError(f"dtype must be a floating-point torch.dtype, got {dtype!r}.")
    try:
        tensor = torch.as_tensor(values, dtype=dtype)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            f"Could not convert {kind} '{label}' to torch.Tensor (dtype={dtype}): {exc}"
        ) from exc
    return tensor


def is_floating_dtype(dtype: torch.dtype) -> bool:
    """Return True if ``dtype`` is a torch floating-point dtype.

    ``torch.dtype`` itself doesn't expose ``is_floating_point``; we probe
    via a zero-element tensor instead.
    """
    return torch.empty(0, dtype=dtype).is_floating_point()


def parse_lhs_spec(
    lhs: str,
    *,
    fields: dict[str, Any],
    coords: dict[str, Any],
) -> tuple[str, str]:
    """Parse ``"{field}_{axis}"`` into (field_name, axis_name).

    Splits on the LAST underscore so ``"my_field_x"`` -> ("my_field", "x").
    Validates that the parsed components exist in ``fields`` and ``coords``.

    The spec is interpreted as a **first-order** LHS derivative (the only
    LHS shape the public ``Model`` facade supports). There is no syntax
    here for higher-order LHS like ``u_tt``; if you need a wave equation,
    reduce it to a first-order-in-time system before calling.

    Raises:
        ValueError: If the spec is malformed or references missing entries.
    """
    if "_" not in lhs:
        raise ValueError(
            f"lhs spec '{lhs}' is malformed: expected '{{field}}_{{axis}}' "
            f"format (e.g., 'u_t'). Available fields: {list(fields.keys())}, "
            f"axes: {list(coords.keys())}."
        )
    lhs_field, lhs_axis = lhs.rsplit("_", 1)
    if not lhs_field or not lhs_axis:
        raise ValueError(
            f"lhs spec '{lhs}' is malformed: expected '{{field}}_{{axis}}' "
            f"format (e.g., 'u_t'). Available fields: {list(fields.keys())}, "
            f"axes: {list(coords.keys())}."
        )
    if lhs_field not in fields:
        raise ValueError(
            f"lhs field '{lhs_field}' (parsed from lhs='{lhs}') not "
            f"in fields. Available fields: {list(fields.keys())}."
        )
    if lhs_axis not in coords:
        raise ValueError(
            f"lhs axis '{lhs_axis}' (parsed from lhs='{lhs}') not "
            f"in coords. Available axes: {list(coords.keys())}."
        )
    return lhs_field, lhs_axis


def _check_strictly_increasing(tensor: torch.Tensor, axis_name: str) -> None:
    """Reject non-strictly-increasing 1D coordinates.

    Downstream finite-difference stencils assume a uniform, strictly
    increasing grid. Without this check, a reversed slice or a corrupted
    coordinate read silently feeds the FD code negative / zero / mixed
    spacing and produces meaningless derivatives — the SGA fit then
    happily reports an "equation" that is numerically valid but
    physically wrong. Single-element axes have nothing to compare and
    are accepted.
    """
    if tensor.numel() < 2:
        return
    diffs = torch.diff(tensor)
    if bool((diffs > 0).all()):
        return
    bad_idx = int((diffs <= 0).nonzero(as_tuple=False)[0].item())
    raise ValueError(
        f"coords['{axis_name}'] must be strictly increasing; "
        f"violation at index {bad_idx}: "
        f"values[{bad_idx}]={float(tensor[bad_idx]):.6g}, "
        f"values[{bad_idx + 1}]={float(tensor[bad_idx + 1]):.6g}. "
        f"If your data is reversed, apply np.flip / torch.flip "
        f"on both the coord and the corresponding field axis "
        f"before calling from_arrays."
    )


def build_axes_dict(
    coords: dict[str, torch.Tensor | np.ndarray | Sequence[float]],
    *,
    dtype: torch.dtype,
    periodic: Iterable[str] | None,
) -> dict[str, AxisInfo]:
    """Convert coords mapping into an ``{axis_name: AxisInfo}`` dict.

    Validates 1D shape, strict monotonicity, and that ``periodic``
    references known axes.
    """
    from kd2.data.schema import AxisInfo

    periodic_set: set[str] = set(periodic) if periodic is not None else set()
    unknown_periodic = periodic_set - set(coords.keys())
    if unknown_periodic:
        raise ValueError(
            f"periodic references unknown axes: {sorted(unknown_periodic)}. "
            f"Available axes: {list(coords.keys())}."
        )

    axes_dict: dict[str, AxisInfo] = {}
    for axis_name, axis_values in coords.items():
        tensor = to_float_tensor(axis_values, dtype, kind="coord", label=axis_name)
        if tensor.dim() != 1:
            raise ValueError(
                f"coords['{axis_name}'] must be 1D, got {tensor.dim()}D shape "
                f"{tuple(tensor.shape)}."
            )
        _check_strictly_increasing(tensor, axis_name)
        axes_dict[axis_name] = AxisInfo(
            name=axis_name,
            values=tensor,
            is_periodic=axis_name in periodic_set,
        )
    return axes_dict


def build_fields_dict(
    fields: dict[str, torch.Tensor | np.ndarray],
    *,
    dtype: torch.dtype,
) -> dict[str, FieldData]:
    """Convert fields mapping into an ``{field_name: FieldData}`` dict."""
    from kd2.data.schema import FieldData

    fields_dict: dict[str, FieldData] = {}
    for field_name, field_values in fields.items():
        tensor = to_float_tensor(field_values, dtype, kind="field", label=field_name)
        fields_dict[field_name] = FieldData(name=field_name, values=tensor)
    return fields_dict


def annotate_shape_error(
    exc: ValueError,
    *,
    axes_dict: dict[str, AxisInfo],
    fields_dict: dict[str, FieldData],
    axis_order: list[str],
) -> ValueError:
    """Wrap a __post_init__ ValueError with a richer per-axis/field summary."""
    shape_summary = ", ".join(
        f"{n}=len({axes_dict[n].values.numel()})" for n in axis_order
    )
    field_summary = ", ".join(
        f"{n}.shape={tuple(f.values.shape)}" for n, f in fields_dict.items()
    )
    return ValueError(
        f"PDEDataset.from_arrays failed: {exc} "
        f"[coords: {shape_summary}; fields: {field_summary}; "
        f"axis_order={axis_order}]"
    )
