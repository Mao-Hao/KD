"""SGA tree/PDE to kd2 funcall expression converter.

Converts SGA tree structures (using operator names like ``+``, ``*``, ``^2``)
into kd2 function-call IR (using ``add(...)``, ``mul(...)``, etc.).
"""

from __future__ import annotations

import re

from kd2.search.sga.pde import PDE
from kd2.search.sga.tree import Node, Tree

# -- name mapping ------------------------------------------------------------
# Maps SGA operator names to kd2 funcall operator names.
_NAME_MAP: dict[str, str] = {
    "+": "add",
    "-": "sub",
    "*": "mul",
    "/": "div",
    "^2": "n2",
    "^3": "n3",
}

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


class Kd2Expression(str):
    """String subclass with token-aware default splitting for kd2 IR."""

    def split( # type: ignore[override]
        self,
        sep: str | None = None,
        maxsplit: int = -1,
    ) -> list[str]:
        if sep is not None:
            return super().split(sep, maxsplit)
        return _TOKEN_PATTERN.findall(self)


def _map_name(name: str) -> str:
    """Map an SGA name to kd2 funcall name, or pass through unchanged."""
    return _NAME_MAP.get(name, name)


def _axis_name(axis_node: Node) -> str:
    """Return the derivative axis name, requiring a leaf node."""
    if not axis_node.is_leaf:
        raise ValueError("Derivative axis must be a leaf node")
    return _map_name(axis_node.name)


def _derivative_funcall(name: str, node: Node) -> str:
    """Render SGA derivative operators into open-form diff IR."""
    if len(node.children) != 2:
        raise ValueError(f"Operator '{node.name}' requires exactly 2 children")
    axis = _axis_name(node.children[1])
    inner = _node_to_funcall(node.children[0])
    return f"{name}_{axis}({inner})"


def _node_to_funcall(node: Node) -> str:
    """Render a tree node into kd2 function-call syntax."""
    if node.name == "d" and len(node.children) == 2:
        return _derivative_funcall("diff", node)
    if node.name == "d^2" and len(node.children) == 2:
        return _derivative_funcall("diff2", node)
    if not node.children:
        return _map_name(node.name)
    children = ", ".join(_node_to_funcall(child) for child in node.children)
    return f"{_map_name(node.name)}({children})"


def tree_to_kd2_expr(tree: Tree) -> str:
    """Convert an SGA Tree to a kd2 function-call expression string.

    Performs recursive rendering with D1 name mapping:
    ``+`` -> ``add``, ``-`` -> ``sub``, ``*`` -> ``mul``,
    ``/`` -> ``div``, ``^2`` -> ``n2``, ``^3`` -> ``n3``.

    Derivative operators are rewritten into open-form diff IR:
    ``d(expr, x)`` -> ``diff_x(expr)``
    ``d^2(expr, x)`` -> ``diff2_x(expr)``

    Parameters
    ----------
    tree: Tree
        The SGA tree to convert.

    Returns
    -------
    str
        kd2 prefix format expression string.

    Examples
    --------
    >>> from kd2.search.sga.tree import Node, Tree
    >>> leaf = Node("u", 0)
    >>> tree = Tree(root=leaf)
    >>> tree_to_kd2_expr(tree)
    'u'
    """
    return Kd2Expression(_node_to_funcall(tree.root))


# PDE -> kd2 expression


def pde_to_kd2_expr(
    pde: PDE,
    coefficients: list[float] | None = None,
) -> str:
    """Convert a multi-term PDE to a complete kd2 expression.

    Without coefficients, terms are nested with ``add`` funcalls:
    ``"add(term1, add(term2, term3))"``

    With coefficients, each term is wrapped:
    ``"add(mul(c1, term1), mul(c2, term2))"``

    Special cases:
    - Single term without coeff: just the term string
    - Single term with coeff: ``"mul(c, term_str)"``
    - Empty PDE: ``""``

    Parameters
    ----------
    pde: PDE
        The PDE to convert.
    coefficients: list[float] | None
        Optional coefficients for each term. Must match the number of
        terms if provided.

    Returns
    -------
    str
        kd2 prefix format expression string.

    Raises
    ------
    ValueError
        If the number of coefficients does not match the number of terms.
    """
    if pde.width == 0:
        return Kd2Expression("")

    if coefficients is not None and len(coefficients) != pde.width:
        raise ValueError(
            f"Coefficient count ({len(coefficients)}) does not match "
            f"term count ({pde.width})"
        )

    term_strs = [tree_to_kd2_expr(tree) for tree in pde.terms]

    if coefficients is not None:
        # Historical note: this path used to emit ``*_const`` suffixes.
        # Keep the parameter for compatibility, but serialize plain literals
        # so the result remains valid Python funcall IR.
        term_strs = [
            f"mul({float(coefficient)}, {term})"
            for coefficient, term in zip(coefficients, term_strs, strict=True)
        ]

    if len(term_strs) == 1:
        return Kd2Expression(term_strs[0])

    result = term_strs[-1]
    for term in reversed(term_strs[:-1]):
        result = f"add({term}, {result})"

    return Kd2Expression(result)
