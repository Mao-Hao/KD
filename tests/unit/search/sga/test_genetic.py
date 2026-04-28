"""Tests for SGA genetic operators: random_tree, random_pde, mutate, crossover, replace.

TDD red phase -- tests written against the interface spec before implementation.
All operators must return NEW objects (inputs never modified).
"""

from __future__ import annotations

import pytest
import torch

from kd2.search.sga.config import OP1, OP2, OPS, ROOT, SGAConfig, build_den
from kd2.search.sga.genetic import (
    crossover,
    mutate,
    random_pde,
    random_tree,
    replace,
)
from kd2.search.sga.pde import PDE
from kd2.search.sga.tree import Node, Tree

# Helpers

VARS = ["u", "x", "t", "u_x", "u_t"]
# DEN for derivative denominator constraint (axes=["x","t"], lhs_axis="t")
DEN = build_den(axes=["x", "t"], lhs_axis="t")


def _make_rng(seed: int = 42) -> torch.Generator:
    """Create a seeded torch.Generator for reproducible tests."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def _make_leaf(name: str) -> Tree:
    return Tree(root=Node(name=name, arity=0, children=[]))


def _make_binary_tree(op: str, left: str, right: str) -> Tree:
    left_node = Node(name=left, arity=0, children=[])
    right_node = Node(name=right, arity=0, children=[])
    root = Node(name=op, arity=2, children=[left_node, right_node])
    return Tree(root=root)


def _make_unary_tree(op: str, child: str) -> Tree:
    child_node = Node(name=child, arity=0, children=[])
    root = Node(name=op, arity=1, children=[child_node])
    return Tree(root=root)


def _collect_nodes(node: Node) -> list[Node]:
    """Collect all nodes in a tree via DFS pre-order."""
    result = [node]
    for child in node.children:
        result.extend(_collect_nodes(child))
    return result


def _collect_leaves(node: Node) -> list[Node]:
    """Collect all leaf nodes."""
    return [n for n in _collect_nodes(node) if n.is_leaf]


def _collect_arity_map(node: Node) -> list[tuple[str, int]]:
    """Collect (name, arity) for every node -- used to verify structure."""
    return [(n.name, n.arity) for n in _collect_nodes(node)]


# ===========================================================================
# random_tree
# ===========================================================================


class TestRandomTree:
    """random_tree generates a well-formed expression tree."""

    @pytest.mark.smoke
    def test_returns_tree_instance(self) -> None:
        rng = _make_rng()
        tree = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        assert isinstance(tree, Tree)

    def test_root_from_root_pool(self) -> None:
        """Root node must come from the ROOT operator pool (no +/-)."""
        root_names = {name for name, _ in ROOT}
        rng = _make_rng()
        for seed in range(50):
            rng.manual_seed(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=3,
                p_var=0.5,
                rng=rng,
            )
            assert tree.root.name in root_names, (
                f"Root '{tree.root.name}' not in ROOT pool {root_names}"
            )

    def test_leaves_are_variables_or_den(self) -> None:
        """All leaf nodes must be variable names from vars or den."""
        var_set = set(VARS)
        den_set = {name for name, _ in DEN}
        allowed = var_set | den_set
        rng = _make_rng()
        for seed in range(50):
            rng.manual_seed(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.5,
                rng=rng,
            )
            leaves = _collect_leaves(tree.root)
            assert len(leaves) > 0, "Tree must have at least one leaf"
            for leaf in leaves:
                assert leaf.name in allowed, (
                    f"Leaf '{leaf.name}' not in vars|den {allowed}"
                )

    def test_depth_respects_constraint(self) -> None:
        """Generated tree depth must not exceed the specified depth."""
        rng = _make_rng()
        max_depth = 3
        for seed in range(50):
            rng.manual_seed(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=max_depth,
                p_var=0.5,
                rng=rng,
            )
            assert tree.depth <= max_depth, (
                f"Tree depth {tree.depth} exceeds max {max_depth}"
            )

    def test_leaf_arity_is_zero(self) -> None:
        """Every leaf node should have arity 0."""
        rng = _make_rng()
        tree = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=4,
            p_var=0.3,
            rng=rng,
        )
        for leaf in _collect_leaves(tree.root):
            assert leaf.arity == 0

    def test_operator_nodes_have_correct_children_count(self) -> None:
        """Each non-leaf node should have exactly arity children."""
        rng = _make_rng()
        for seed in range(30):
            rng.manual_seed(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.5,
                rng=rng,
            )
            for node in _collect_nodes(tree.root):
                if not node.is_leaf:
                    assert len(node.children) == node.arity, (
                        f"Node '{node.name}' has arity {node.arity} "
                        f"but {len(node.children)} children"
                    )


class TestRandomTreeReproducibility:
    """Same seed must produce the same tree."""

    def test_same_seed_same_tree(self) -> None:
        """Two calls with identical seed produce structurally equal trees."""
        seed = 123
        rng1 = _make_rng(seed)
        rng2 = _make_rng(seed)
        tree1 = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=4,
            p_var=0.5,
            rng=rng1,
        )
        tree2 = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=4,
            p_var=0.5,
            rng=rng2,
        )
        assert tree1 == tree2

    def test_different_seed_likely_different(self) -> None:
        """Different seeds should (almost surely) produce different trees."""
        trees = []
        for seed in range(10):
            rng = _make_rng(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.5,
                rng=rng,
            )
            trees.append(str(tree))
        # Not all 10 trees should be identical
        assert len(set(trees)) > 1


class TestRandomTreeEdgeCases:
    """Edge cases for random_tree generation."""

    def test_depth_one_is_just_root_with_leaf_children(self) -> None:
        """depth=1: root op + leaf children (bottom layer must be vars/den)."""
        rng = _make_rng()
        var_set = set(VARS)
        den_set = {name for name, _ in DEN}
        allowed = var_set | den_set
        tree = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=1,
            p_var=0.5,
            rng=rng,
        )
        # Root is an operator
        assert tree.root.arity > 0
        # All children are leaves
        for child in tree.root.children:
            assert child.is_leaf
            assert child.name in allowed

    def test_p_var_one_produces_shallow_trees(self) -> None:
        """p_var=1.0: intermediate positions become variables (early stop)."""
        rng = _make_rng()
        tree = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=4,
            p_var=1.0,
            rng=rng,
        )
        # With p_var=1.0, every non-bottom position should prefer variables,
        # so the tree should be quite shallow (root + var children only).
        assert tree.depth <= 1

    def test_p_var_zero_produces_deeper_trees(self) -> None:
        """p_var=0.0: intermediate nodes always pick operators, deeper."""
        rng = _make_rng()
        tree = random_tree(
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=4,
            p_var=0.0,
            rng=rng,
        )
        # Should have depth > 1 since only bottom layer is vars
        assert tree.depth > 1

    def test_single_var_pool(self) -> None:
        """With only one variable, all non-den leaves must be that variable."""
        rng = _make_rng()
        den_names = {name for name, _ in DEN}
        tree = random_tree(
            vars=["u"],
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        for leaf in _collect_leaves(tree.root):
            assert leaf.name == "u" or leaf.name in den_names


# ===========================================================================
# random_pde
# ===========================================================================


class TestRandomPDE:
    """random_pde initializes a PDE with random terms."""

    @pytest.mark.smoke
    def test_returns_pde_instance(self) -> None:
        config = SGAConfig(width=5, depth=4, p_var=0.5)
        rng = _make_rng()
        pde = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )
        assert isinstance(pde, PDE)

    def test_width_in_valid_range(self) -> None:
        """PDE width must be in [1, config.width]."""
        config = SGAConfig(width=5, depth=4, p_var=0.5)
        for seed in range(50):
            rng = _make_rng(seed)
            pde = random_pde(
                config=config,
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                rng=rng,
            )
            assert 1 <= pde.width <= config.width, (
                f"PDE width {pde.width} not in [1, {config.width}]"
            )

    def test_all_terms_are_valid_trees(self) -> None:
        """Every term in the PDE should be a Tree instance."""
        config = SGAConfig(width=5, depth=4, p_var=0.5)
        rng = _make_rng()
        pde = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )
        for term in pde.terms:
            assert isinstance(term, Tree)

    def test_terms_respect_depth_constraint(self) -> None:
        """All terms must respect config.depth."""
        config = SGAConfig(width=5, depth=3, p_var=0.5)
        rng = _make_rng()
        pde = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )
        for term in pde.terms:
            assert term.depth <= config.depth

    def test_reproducible(self) -> None:
        """Same seed produces the same PDE."""
        config = SGAConfig(width=5, depth=4, p_var=0.5)
        rng1 = _make_rng(99)
        rng2 = _make_rng(99)
        pde1 = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng1,
        )
        pde2 = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng2,
        )
        assert pde1 == pde2


# ===========================================================================
# mutate
# ===========================================================================


class TestMutate:
    """mutate performs node-level replacement while preserving tree structure."""

    @pytest.mark.smoke
    def test_returns_pde_instance(self) -> None:
        pde = PDE(terms=[_make_binary_tree("*", "u", "x")])
        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.5,
            rng=rng,
        )
        assert isinstance(result, PDE)

    def test_does_not_modify_input(self) -> None:
        """Input PDE must remain unchanged after mutation."""
        pde = PDE(terms=[_make_binary_tree("*", "u", "x")])
        original_str = str(pde)
        rng = _make_rng()
        _ = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=1.0,
            rng=rng,
        )
        assert str(pde) == original_str, "Input PDE was modified by mutate"

    def test_returns_new_object(self) -> None:
        """Result must be a different PDE object."""
        pde = PDE(terms=[_make_binary_tree("*", "u", "x")])
        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.5,
            rng=rng,
        )
        assert result is not pde

    def test_arity_preserved_after_mutation(self) -> None:
        """Mutation replaces nodes with same-arity alternatives.
        The arity profile of the entire tree must be unchanged."""
        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])

        original_arities = [n.arity for n in _collect_nodes(pde.terms[0].root)]

        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=1.0,
            rng=rng,
        )

        result_arities = [n.arity for n in _collect_nodes(result.terms[0].root)]
        assert original_arities == result_arities, (
            "Mutation changed tree structure (arities differ)"
        )

    def test_high_p_mute_changes_something(self) -> None:
        """With p_mute=1.0 and enough variety, at least some nodes should change."""
        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])

        changed = False
        for seed in range(20):
            rng = _make_rng(seed)
            result = mutate(
                pde=pde,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=1.0,
                rng=rng,
            )
            if str(result) != str(pde):
                changed = True
                break
        assert changed, "p_mute=1.0 never changed anything over 20 seeds"

    def test_zero_p_mute_preserves_tree(self) -> None:
        """With p_mute=0.0, the output tree should be structurally identical."""
        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])
        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.0,
            rng=rng,
        )
        assert result == pde, "p_mute=0.0 should produce identical PDE"

    def test_leaves_remain_valid(self) -> None:
        """After mutation, all leaves must still be valid variables or den."""
        var_set = set(VARS)
        den_set = {name for name, _ in DEN}
        allowed = var_set | den_set
        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])

        for seed in range(20):
            rng = _make_rng(seed)
            result = mutate(
                pde=pde,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=0.8,
                rng=rng,
            )
            for term in result.terms:
                for leaf in _collect_leaves(term.root):
                    assert leaf.name in allowed

    def test_operators_remain_from_pools(self) -> None:
        """After mutation, unary ops should come from OP1, binary from OP2."""
        op1_names = {name for name, _ in OP1}
        op2_names = {name for name, _ in OP2}

        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])

        for seed in range(20):
            rng = _make_rng(seed)
            result = mutate(
                pde=pde,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=1.0,
                rng=rng,
            )
            for term in result.terms:
                for node in _collect_nodes(term.root):
                    if node.arity == 1:
                        assert node.name in op1_names
                    elif node.arity == 2:
                        assert node.name in op2_names

    def test_width_preserved(self) -> None:
        """Mutation does not add or remove terms."""
        terms = [
            _make_binary_tree("*", "u", "x"),
            _make_unary_tree("^2", "u"),
        ]
        pde = PDE(terms=terms)
        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.5,
            rng=rng,
        )
        assert result.width == pde.width

    def test_no_reference_sharing_with_input(self) -> None:
        """Result terms should not share Node objects with input."""
        tree = _make_binary_tree("*", "u", "x")
        pde = PDE(terms=[tree])
        rng = _make_rng()
        result = mutate(
            pde=pde,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.0,
            rng=rng,
        )
        # Even with no mutation, result nodes should be independent copies
        assert result.terms[0].root is not pde.terms[0].root


# ===========================================================================
# crossover
# ===========================================================================


class TestCrossover:
    """crossover swaps entire terms between two PDEs."""

    @pytest.mark.smoke
    def test_returns_two_pdes(self) -> None:
        pde1 = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        pde2 = PDE(terms=[_make_leaf("t"), _make_leaf("u_x")])
        rng = _make_rng()
        r1, r2 = crossover(pde1, pde2, rng)
        assert isinstance(r1, PDE)
        assert isinstance(r2, PDE)

    def test_does_not_modify_inputs(self) -> None:
        """Input PDEs must remain unchanged after crossover."""
        pde1 = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        pde2 = PDE(terms=[_make_leaf("t"), _make_leaf("u_x")])
        str1, str2 = str(pde1), str(pde2)
        rng = _make_rng()
        _ = crossover(pde1, pde2, rng)
        assert str(pde1) == str1, "pde1 was modified by crossover"
        assert str(pde2) == str2, "pde2 was modified by crossover"

    def test_returns_new_objects(self) -> None:
        """Result PDEs must be different objects from inputs."""
        pde1 = PDE(terms=[_make_leaf("u")])
        pde2 = PDE(terms=[_make_leaf("x")])
        rng = _make_rng()
        r1, r2 = crossover(pde1, pde2, rng)
        assert r1 is not pde1
        assert r2 is not pde2

    def test_width_preserved(self) -> None:
        """Crossover preserves the number of terms in each PDE."""
        pde1 = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        pde2 = PDE(terms=[_make_leaf("t"), _make_leaf("u_x"), _make_leaf("u_t")])
        rng = _make_rng()
        r1, r2 = crossover(pde1, pde2, rng)
        assert r1.width == pde1.width
        assert r2.width == pde2.width

    def test_swaps_entire_terms(self) -> None:
        """After crossover, each result should contain at least one term
        from the other parent (with sufficient diversity)."""
        # Use highly distinct terms
        pde1 = PDE(terms=[_make_binary_tree("*", "u", "x"), _make_leaf("t")])
        pde2 = PDE(terms=[_make_unary_tree("^2", "u_x"), _make_leaf("u_t")])

        # Over many seeds, the cross-pollination should appear at least once
        found_swap = False
        pde1_term_strs = {str(t) for t in pde1.terms}
        pde2_term_strs = {str(t) for t in pde2.terms}

        for seed in range(30):
            rng = _make_rng(seed)
            r1, r2 = crossover(pde1, pde2, rng)
            r1_term_strs = {str(t) for t in r1.terms}
            r2_term_strs = {str(t) for t in r2.terms}

            # Check if r1 got a term from pde2 or r2 got a term from pde1
            if (r1_term_strs & pde2_term_strs) or (r2_term_strs & pde1_term_strs):
                found_swap = True
                break

        assert found_swap, "Crossover never swapped terms over 30 seeds"

    def test_no_reference_sharing_between_outputs_and_inputs(self) -> None:
        """Modifying result terms must not affect the input PDEs."""
        pde1 = PDE(terms=[_make_binary_tree("*", "u", "x")])
        pde2 = PDE(terms=[_make_binary_tree("/", "t", "u_x")])
        rng = _make_rng()
        r1, r2 = crossover(pde1, pde2, rng)

        # Mutate result
        r1.terms[0].root.name = "CORRUPTED"

        # Input must be intact
        assert pde1.terms[0].root.name != "CORRUPTED"
        assert pde2.terms[0].root.name != "CORRUPTED"

    def test_reproducible(self) -> None:
        """Same seed produces the same crossover result."""
        pde1 = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        pde2 = PDE(terms=[_make_leaf("t"), _make_leaf("u_x")])

        rng_a = _make_rng(77)
        rng_b = _make_rng(77)
        r1a, r2a = crossover(pde1, pde2, rng_a)
        r1b, r2b = crossover(pde1, pde2, rng_b)
        assert r1a == r1b
        assert r2a == r2b


# ===========================================================================
# replace
# ===========================================================================


class TestReplace:
    """replace substitutes one random term with a brand new random_tree."""

    @pytest.mark.smoke
    def test_returns_pde_instance(self) -> None:
        pde = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        rng = _make_rng()
        result = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        assert isinstance(result, PDE)

    def test_does_not_modify_input(self) -> None:
        """Input PDE must remain unchanged after replace."""
        pde = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        original_str = str(pde)
        rng = _make_rng()
        _ = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        assert str(pde) == original_str

    def test_returns_new_object(self) -> None:
        pde = PDE(terms=[_make_leaf("u")])
        rng = _make_rng()
        result = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        assert result is not pde

    def test_width_preserved(self) -> None:
        """Replace substitutes one term -- total count stays the same."""
        pde = PDE(terms=[_make_leaf("u"), _make_leaf("x"), _make_leaf("t")])
        rng = _make_rng()
        result = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        assert result.width == pde.width

    def test_exactly_one_term_differs(self) -> None:
        """Replace should change exactly one term (substitute with new tree).

        Note: in rare cases the new random tree might be structurally identical
        to the old term, so we test over many seeds and expect at least one diff.
        """
        pde = PDE(
            terms=[
                _make_binary_tree("*", "u", "x"),
                _make_unary_tree("^2", "t"),
                _make_binary_tree("/", "u_x", "u_t"),
            ]
        )

        found_single_diff = False
        for seed in range(30):
            rng = _make_rng(seed)
            result = replace(
                pde=pde,
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.5,
                rng=rng,
            )
            pairs = zip(pde.terms, result.terms, strict=True)
            diffs = sum(1 for a, b in pairs if str(a) != str(b))
            if diffs == 1:
                found_single_diff = True
                break

        assert found_single_diff, (
            "replace never produced exactly 1 different term over 30 seeds"
        )

    def test_new_term_is_valid_tree(self) -> None:
        """The replacement term should be a well-formed Tree."""
        var_set = set(VARS)
        den_set = {name for name, _ in DEN}
        root_names = {name for name, _ in ROOT}
        allowed_leaves = var_set | den_set
        pde = PDE(terms=[_make_leaf("u")])
        rng = _make_rng()
        result = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        # The single term should be valid
        term = result.terms[0]
        assert isinstance(term, Tree)
        assert term.root.name in root_names | allowed_leaves
        for leaf in _collect_leaves(term.root):
            assert leaf.name in allowed_leaves

    def test_reproducible(self) -> None:
        pde = PDE(terms=[_make_leaf("u"), _make_leaf("x")])
        rng_a = _make_rng(55)
        rng_b = _make_rng(55)
        r1 = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng_a,
        )
        r2 = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng_b,
        )
        assert r1 == r2

    def test_no_reference_sharing(self) -> None:
        """Modifying result must not affect input."""
        pde = PDE(terms=[_make_binary_tree("*", "u", "x")])
        rng = _make_rng()
        result = replace(
            pde=pde,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        result.terms[0].root.name = "CORRUPTED"
        assert pde.terms[0].root.name == "*"


# ===========================================================================
# Property-based invariants (all operators)
# ===========================================================================


class TestGeneticOperatorInvariants:
    """Cross-cutting invariants that all genetic operators must satisfy."""

    def test_mutate_idempotent_width(self) -> None:
        """Multiple mutations preserve term count."""
        pde = PDE(terms=[_make_binary_tree("*", "u", "x"), _make_leaf("t")])
        rng = _make_rng()
        current = pde
        for _ in range(5):
            current = mutate(
                pde=current,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=0.5,
                rng=rng,
            )
            assert current.width == pde.width

    def test_all_operators_produce_finite_trees(self) -> None:
        """Generated trees should have finite depth and size (no cycles)."""
        config = SGAConfig(width=5, depth=4, p_var=0.5)
        rng = _make_rng()

        pde = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )
        for term in pde.terms:
            assert term.depth < 100, "Suspiciously deep tree (possible cycle)"
            assert term.size < 10000, "Suspiciously large tree"

    def test_chained_operations_produce_valid_pdes(self) -> None:
        """A sequence of mutate + replace + crossover should still yield valid PDEs."""
        config = SGAConfig(width=3, depth=3, p_var=0.5)
        rng = _make_rng(0)

        pde1 = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )
        pde2 = random_pde(
            config=config,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            rng=rng,
        )

        # Apply a chain of operations
        pde1 = mutate(
            pde=pde1,
            vars=VARS,
            op1=OP1,
            op2=OP2,
            den=DEN,
            p_mute=0.3,
            rng=rng,
        )
        pde1 = replace(
            pde=pde1,
            vars=VARS,
            ops=OPS,
            root=ROOT,
            den=DEN,
            depth=3,
            p_var=0.5,
            rng=rng,
        )
        pde1, pde2 = crossover(pde1, pde2, rng)

        # Both should still be valid
        var_set = set(VARS)
        den_set = {name for name, _ in DEN}
        allowed = var_set | den_set
        for pde in (pde1, pde2):
            assert isinstance(pde, PDE)
            assert pde.width >= 1
            for term in pde.terms:
                assert isinstance(term, Tree)
                for leaf in _collect_leaves(term.root):
                    assert leaf.name in allowed


# ===========================================================================
# Derivative node constraints in tree generation
# ===========================================================================


def _find_derivative_nodes(node: Node) -> list[tuple[Node, Node, Node]]:
    """Find all d/d^2 nodes and return (deriv_node, left_child, right_child)."""
    results: list[tuple[Node, Node, Node]] = []
    if node.name in {"d", "d^2"} and len(node.children) == 2:
        results.append((node, node.children[0], node.children[1]))
    for child in node.children:
        results.extend(_find_derivative_nodes(child))
    return results


class TestDerivativeNodeGeneration:
    """random_tree must be able to produce derivative nodes with den constraint."""

    def test_derivative_nodes_can_be_generated(self) -> None:
        """Over many seeds, at least one tree should contain a d or d^2 node."""
        found_deriv = False
        for seed in range(200):
            rng = _make_rng(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.3,
                rng=rng,
            )
            derivs = _find_derivative_nodes(tree.root)
            if derivs:
                found_deriv = True
                break
        assert found_deriv, "No derivative nodes generated over 200 seeds"

    def test_derivative_right_child_from_den(self) -> None:
        """The right child of d/d^2 must come from den (non-lhs_axis)."""
        den_names = {name for name, _ in DEN}
        for seed in range(200):
            rng = _make_rng(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.3,
                rng=rng,
            )
            for _, _, right in _find_derivative_nodes(tree.root):
                assert right.name in den_names, (
                    f"Derivative right child '{right.name}' not in den {den_names}"
                )
                assert right.is_leaf, "Derivative right child must be a leaf"

    def test_derivative_right_child_never_lhs_axis(self) -> None:
        """The right child of d/d^2 must never be the lhs_axis (e.g. 't')."""
        lhs_axis = "t" # corresponds to DEN = build_den(["x","t"], "t")
        for seed in range(200):
            rng = _make_rng(seed)
            tree = random_tree(
                vars=VARS,
                ops=OPS,
                root=ROOT,
                den=DEN,
                depth=4,
                p_var=0.3,
                rng=rng,
            )
            for _, _, right in _find_derivative_nodes(tree.root):
                assert right.name != lhs_axis, (
                    f"Derivative denominator is lhs_axis '{lhs_axis}'"
                )


class TestDerivativeNodeMutation:
    """Mutation must respect den constraint for derivative nodes."""

    def test_mutated_derivative_right_child_stays_in_den(self) -> None:
        """When mutating a tree with d(expr, x), the right child must stay in den."""
        den_names = {name for name, _ in DEN}
        # Manually construct a tree: d(u, x)
        d_node = Node(
            name="d",
            arity=2,
            children=[
                Node(name="u", arity=0),
                Node(name="x", arity=0),
            ],
        )
        tree = Tree(root=d_node)
        pde = PDE(terms=[tree])

        for seed in range(50):
            rng = _make_rng(seed)
            result = mutate(
                pde=pde,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=1.0,
                rng=rng,
            )
            for term in result.terms:
                for _, _, right in _find_derivative_nodes(term.root):
                    assert right.name in den_names, (
                        f"After mutation, derivative right child "
                        f"'{right.name}' not in den"
                    )

    def test_mutation_to_d_checks_right_child_compatibility(self) -> None:
        """When mutating a binary op to d/d^2, existing right child must be in den.
        If not, the mutation should be rejected (pick a different operator)."""
        den_names = {name for name, _ in DEN}
        # Tree: *(u, u) -- right child 'u' is NOT in den
        tree = _make_binary_tree("*", "u", "u")
        pde = PDE(terms=[tree])

        for seed in range(50):
            rng = _make_rng(seed)
            result = mutate(
                pde=pde,
                vars=VARS,
                op1=OP1,
                op2=OP2,
                den=DEN,
                p_mute=1.0,
                rng=rng,
            )
            for term in result.terms:
                for _, _, right in _find_derivative_nodes(term.root):
                    assert right.name in den_names, (
                        f"Mutated-to-d node has invalid right child "
                        f"'{right.name}' not in den"
                    )
