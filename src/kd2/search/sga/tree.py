"""SGA tree data structures: Node and Tree.

Node stores symbol name + arity + children
Tree wraps a root Node and delegates structural properties.
"""

from __future__ import annotations

import copy


class Node:
    """SGA tree node. Stores symbol name + structure only.

    Attributes:
        name: Operator or variable name (e.g. "mul", "u", "^2").
        arity: Number of children: 0=leaf, 1=unary, 2=binary.
        children: Child nodes (empty list for leaves).
    """

    __slots__ = ("name", "arity", "children")

    def __init__(
        self,
        name: str,
        arity: int,
        children: list[Node] | None = None,
    ) -> None:
        self.name = name
        self.arity = arity
        self.children: list[Node] = children if children is not None else []

    # -- properties -----------------------------------------------------------

    @property
    def is_leaf(self) -> bool:
        """True when arity is 0 (terminal node)."""
        return self.arity == 0

    @property
    def depth(self) -> int:
        """Max depth of the subtree rooted at this node (leaf = 0)."""
        if not self.children:
            return 0
        return 1 + max(child.depth for child in self.children)

    @property
    def size(self) -> int:
        """Total node count in the subtree rooted at this node."""
        return 1 + sum(child.size for child in self.children)

    # -- equality & representation -------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Structural equality: same name, arity, and children."""
        if not isinstance(other, Node):
            return NotImplemented
        return (
            self.name == other.name
            and self.arity == other.arity
            and self.children == other.children
        )

    def __str__(self) -> str:
        """Stable prefix notation: DFS pre-order, space-separated.

        Examples:
            - Leaf ``u`` -> ``"u"``
            - ``mul(u, x)`` -> ``"mul u x"``
            - ``^2(add(u, x))`` -> ``"^2 add u x"``
        """
        parts: list[str] = []
        self._prefix_collect(parts)
        return " ".join(parts)

    def _prefix_collect(self, parts: list[str]) -> None:
        """Collect prefix tokens via DFS pre-order traversal."""
        parts.append(self.name)
        for child in self.children:
            child._prefix_collect(parts)

    def __repr__(self) -> str:
        return f"Node({self.name!r}, arity={self.arity}, children={self.children!r})"


class Tree:
    """Single SGA term tree wrapping a root Node.

    Attributes:
        root: The root Node of this expression tree.
    """

    __slots__ = ("root",)

    def __init__(self, root: Node) -> None:
        self.root = root

    # -- properties -----------------------------------------------------------

    @property
    def depth(self) -> int:
        """Max depth of the tree (delegates to root)."""
        return self.root.depth

    @property
    def size(self) -> int:
        """Total node count in the tree."""
        return self.root.size

    # -- methods --------------------------------------------------------------

    def copy(self) -> Tree:
        """Deep copy: modifications to the copy do not affect the original."""
        return copy.deepcopy(self)

    # -- equality & representation -------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Structural equality via root comparison."""
        if not isinstance(other, Tree):
            return NotImplemented
        return self.root == other.root

    def __str__(self) -> str:
        """Prefix notation string (delegates to root)."""
        return str(self.root)

    def __repr__(self) -> str:
        return f"Tree({self.root!r})"
