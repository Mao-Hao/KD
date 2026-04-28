"""Derivative naming utilities.

Build and parse derivative names in the kd2 naming convention:
- Terminal (same-axis): field_axis[axis...], e.g., u_x, u_xx, phi_ttt
- Compound (mixed-partial): field_axis_axis_..., e.g., u_x_y, u_xx_yy

These utilities centralize the naming logic used by executor.py and
sympy_bridge.py for derivative symbol construction and parsing.
"""

from __future__ import annotations


def _match_axis_token(
    token: str,
    known_axes: set[str] | None,
) -> tuple[str, int] | None:
    """Parse one derivative token into (axis, order)."""
    if not token:
        return None
    if known_axes:
        matches: list[tuple[str, int]] = []
        for axis in known_axes:
            if not axis or "_" in axis or len(token) % len(axis) != 0:
                continue
            order = len(token) // len(axis)
            if order >= 1 and token == axis * order:
                matches.append((axis, order))
        if len(matches) == 1:
            return matches[0]
        return None
    axis = token[0]
    if token != axis * len(token):
        return None
    return axis, len(token)


def _parse_segments(
    field: str,
    tokens: list[str],
    known_axes: set[str] | None,
) -> tuple[str, list[tuple[str, int]]] | None:
    """Parse derivative segments after field extraction."""
    if not field or not field.strip():
        return None
    if not tokens or any(not token for token in tokens):
        return None
    parsed: list[tuple[str, int]] = []
    for token in tokens:
        result = _match_axis_token(token, known_axes)
        if result is None:
            return None
        parsed.append(result)
    return field, parsed


def _parse_with_known_fields(
    name: str,
    known_fields: set[str],
    known_axes: set[str] | None,
) -> tuple[str, list[tuple[str, int]]] | None:
    """Parse a derivative name using explicit field-name candidates."""
    for field in sorted(known_fields, key=len, reverse=True):
        prefix = f"{field}_"
        if not field or not name.startswith(prefix):
            continue
        tokens = name[len(prefix):].split("_")
        result = _parse_segments(field, tokens, known_axes)
        if result is not None:
            return result
    return None


def build_derivative_name(field: str, axis: str, order: int = 1) -> str:
    """Build terminal derivative name.

    Examples:
        build_derivative_name("u", "x", 1) -> "u_x"
        build_derivative_name("u", "x", 2) -> "u_xx"
        build_derivative_name("u_x", "y", 1) -> "u_x_y" (compound)

    Args:
        field: Field name (e.g., "u", "phi", "u_x" for compound).
        axis: Axis letter (e.g., "x", "t").
        order: Derivative order (>= 1).

    Returns:
        Derivative name string.

    Raises:
        ValueError: if field/axis empty, order < 1.
    """
    if not field:
        raise ValueError("field must be non-empty")
    if not axis or "_" in axis:
        raise ValueError("axis must be non-empty and must not contain underscores")
    if order < 1:
        raise ValueError("order must be >= 1")
    return f"{field}_{axis * order}"


def parse_derivative_name(
    name: str,
    *,
    known_fields: set[str] | None = None,
    known_axes: set[str] | None = None,
) -> tuple[str, str, int] | None:
    """Parse same-axis terminal derivative.

    Returns (field, axis, order) if name matches a same-axis terminal
    derivative pattern, or None otherwise.

    Examples:
        parse_derivative_name("u_xx") -> ("u", "x", 2)
        parse_derivative_name("u_x") -> ("u", "x", 1)
        parse_derivative_name("u_xxx") -> ("u", "x", 3)
        parse_derivative_name("phi_tt") -> ("phi", "t", 2)
        parse_derivative_name("u_xy") -> None (mixed partial)
        parse_derivative_name("u_x_y") -> None (compound format)
        parse_derivative_name("u") -> None (no underscore)
        parse_derivative_name("x") -> None (coordinate, not derivative)

    With known_axes for disambiguation:
        parse_derivative_name("u_xx", known_axes={"x"}) -> ("u", "x", 2)
        parse_derivative_name("u_xx", known_axes={"xx"}) -> ("u", "xx", 1)

    With known_fields:
        parse_derivative_name("phi_x", known_fields={"phi"}) -> ("phi", "x", 1)

    Args:
        name: Potential derivative name.
        known_fields: Optional set of known field names for disambiguation.
        known_axes: Optional set of known axis names for disambiguation.

    Returns:
        (field, axis, order) tuple or None if not a same-axis derivative.
    """
    parsed = parse_compound_derivative(
        name,
        known_fields=known_fields,
        known_axes=known_axes,
    )
    if parsed is None:
        return None
    field, segments = parsed
    if len(segments) != 1:
        return None
    axis, order = segments[0]
    return field, axis, order


def parse_compound_derivative(
    name: str,
    *,
    known_fields: set[str] | None = None,
    known_axes: set[str] | None = None,
) -> tuple[str, list[tuple[str, int]]] | None:
    """Parse derivative including mixed partials (compound format).

    Returns (field, [(axis, order), ...]) if name matches a derivative
    pattern, or None otherwise.

    Examples:
        parse_compound_derivative("u_xx") -> ("u", [("x", 2)])
        parse_compound_derivative("u_x") -> ("u", [("x", 1)])
        parse_compound_derivative("u_x_y") -> ("u", [("x", 1), ("y", 1)])
        parse_compound_derivative("u_xx_y") -> ("u", [("x", 2), ("y", 1)])
        parse_compound_derivative("u_xx_yy") -> ("u", [("x", 2), ("y", 2)])
        parse_compound_derivative("u") -> None
        parse_compound_derivative("x") -> None
        parse_compound_derivative("42") -> None

    Compound format: field_axisN_axisN_... where each segment is axis
    letter repeated N times.

    Args:
        name: Potential derivative name.
        known_fields: Optional set of known field names for disambiguation.
        known_axes: Optional set of known axis names for disambiguation.

    Returns:
        (field, derivatives_list) tuple or None if not parseable.
    """
    if not name or "_" not in name:
        return None
    if known_fields:
        return _parse_with_known_fields(name, known_fields, known_axes)
    field, *tokens = name.split("_")
    return _parse_segments(field, tokens, known_axes)


__all__ = [
    "build_derivative_name",
    "parse_compound_derivative",
    "parse_derivative_name",
]
