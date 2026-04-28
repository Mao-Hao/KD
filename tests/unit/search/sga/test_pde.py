"""Tests for SGA PDE data model.

TDD red phase -- tests written against the interface spec before implementation.
A PDE is a candidate equation represented as a list of term Trees.
"""

from __future__ import annotations

import pytest
from kd2.search.sga.pde import PDE
from kd2.search.sga.tree import Node, Tree

# Helpers


def _make_leaf(name: str) -> Tree:
    """Build a single-leaf Tree for testing."""
    return Tree(root=Node(name=name, arity=0, children=[]))


def _make_mul_tree(left_name: str, right_name: str) -> Tree:
    """Build *(left, right) Tree."""
    left = Node(name=left_name, arity=0, children=[])
    right = Node(name=right_name, arity=0, children=[])
    root = Node(name="*", arity=2, children=[left, right])
    return Tree(root=root)


# ===========================================================================
# PDE construction
# ===========================================================================


class TestPDEConstruction:
    """PDE wraps a list of term Trees."""

    @pytest.mark.smoke
    def test_pde_creation_from_list(self) -> None:
        terms = [_make_leaf("u"), _make_mul_tree("u", "x")]
        pde = PDE(terms=terms)
        assert pde.terms is terms

    @pytest.mark.smoke
    def test_pde_empty_terms(self) -> None:
        """PDE with zero terms is allowed (degenerate but valid)."""
        pde = PDE(terms=[])
        assert pde.terms == []


# ===========================================================================
# PDE.width
# ===========================================================================


class TestPDEWidth:
    """width property returns the number of terms."""

    def test_width_matches_len_terms(self) -> None:
        terms = [_make_leaf("u"), _make_leaf("x"), _make_mul_tree("u", "x")]
        pde = PDE(terms=terms)
        assert pde.width == len(terms)

    def test_width_zero_for_empty(self) -> None:
        pde = PDE(terms=[])
        assert pde.width == 0

    def test_width_one(self) -> None:
        pde = PDE(terms=[_make_leaf("u")])
        assert pde.width == 1


# ===========================================================================
# PDE.copy (deep copy)
# ===========================================================================


class TestPDECopy:
    """PDE.copy() must be a deep copy -- modifying copy leaves original intact."""

    def test_copy_equals_original(self) -> None:
        terms = [_make_leaf("u"), _make_mul_tree("u", "x")]
        original = PDE(terms=terms)
        copied = original.copy()
        # Same content
        assert copied.width == original.width
        for orig_t, copy_t in zip(original.terms, copied.terms):
            assert orig_t == copy_t

    def test_copy_is_independent(self) -> None:
        """Mutating a copied PDE's term tree must not affect the original."""
        terms = [_make_mul_tree("u", "x")]
        original = PDE(terms=terms)
        copied = original.copy()

        # Mutate the copy's tree
        copied.terms[0].root.children[0] = Node(name="t", arity=0, children=[])

        # Original should still have "u" as the left child
        assert original.terms[0].root.children[0].name == "u"

    def test_copy_terms_are_different_objects(self) -> None:
        terms = [_make_leaf("u")]
        original = PDE(terms=terms)
        copied = original.copy()
        assert copied.terms[0] is not original.terms[0]

    def test_copy_list_is_different_object(self) -> None:
        """The terms list itself should be a new list."""
        terms = [_make_leaf("u")]
        original = PDE(terms=terms)
        copied = original.copy()
        copied.terms.append(_make_leaf("x"))
        assert original.width == 1 # original unaffected


# ===========================================================================
# PDE.__str__
# ===========================================================================


class TestPDEStr:
    """PDE should have a reasonable string representation."""

    def test_str_is_nonempty(self) -> None:
        """String repr should be non-empty for a non-empty PDE."""
        pde = PDE(terms=[_make_leaf("u"), _make_mul_tree("u", "x")])
        s = str(pde)
        assert len(s) > 0

    def test_str_contains_term_info(self) -> None:
        """String should contain info from each term."""
        pde = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        s = str(pde)
        # Both variable names should appear somewhere in the string
        assert "u" in s
        assert "x" in s

    def test_str_empty_pde(self) -> None:
        """Empty PDE should still produce a valid string (no crash)."""
        pde = PDE(terms=[])
        s = str(pde)
        assert isinstance(s, str)


# ===========================================================================
# Negative / edge-case tests (>= 20% negative)
# ===========================================================================


class TestPDENegative:
    """Error handling and edge cases for PDE."""

    def test_width_after_appending_term(self) -> None:
        """Width should reflect mutations to the terms list."""
        pde = PDE(terms=[_make_leaf("u")])
        assert pde.width == 1
        pde.terms.append(_make_leaf("x"))
        assert pde.width == 2

    def test_copy_of_empty_pde(self) -> None:
        """Copying an empty PDE should succeed."""
        pde = PDE(terms=[])
        copied = pde.copy()
        assert copied.width == 0
        assert copied.terms == []
