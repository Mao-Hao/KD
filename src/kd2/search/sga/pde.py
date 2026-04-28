"""PDE candidate: a list of term Trees.

A PDE in SGA is represented as a sum of terms, where each term is an
expression Tree. Addition/subtraction semantics are handled by the
multi-term structure (not by individual tree roots).
"""

from __future__ import annotations

import copy

from kd2.search.sga.tree import Tree


class PDE:
    """PDE candidate = list of term Trees.

    Attributes:
        terms: List of Tree objects representing individual PDE terms.
    """

    __slots__ = ("terms",)

    def __init__(self, terms: list[Tree] | None = None) -> None:
        self.terms: list[Tree] = terms if terms is not None else []

    # -- properties -----------------------------------------------------------

    @property
    def width(self) -> int:
        """Number of terms in this PDE."""
        return len(self.terms)

    # -- methods --------------------------------------------------------------

    def copy(self) -> PDE:
        """Deep copy: modifications to the copy do not affect the original."""
        return copy.deepcopy(self)

    # -- equality & representation -------------------------------------------

    def __eq__(self, other: object) -> bool:
        """Structural equality: same terms in same order."""
        if not isinstance(other, PDE):
            return NotImplemented
        return self.terms == other.terms

    def __str__(self) -> str:
        """String representation: terms joined by ' + '.

        Example: ``"mul u x + ^2 u"`` for a 2-term PDE.
        Returns ``""`` for an empty PDE.
        """
        return " + ".join(str(t) for t in self.terms)

    def __repr__(self) -> str:
        return f"PDE(terms={self.terms!r})"
