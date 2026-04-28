"""SGA genetic operators: random generation, mutation, crossover, replacement.

All operators return NEW objects (immutable-style) -- unlike the predecessor's in-place
modification. Randomness is driven by ``torch.Generator`` for reproducibility.
"""

from __future__ import annotations

import torch

from kd2.search.sga.config import OperatorPool, SGAConfig
from kd2.search.sga.pde import PDE
from kd2.search.sga.tree import Node, Tree

# Helpers


def _rand_choice(pool_size: int, rng: torch.Generator) -> int:
    """Return a random index in [0, pool_size) using the generator."""
    return int(torch.randint(pool_size, (1,), generator=rng).item())


def _rand_float(rng: torch.Generator) -> float:
    """Return a random float in [0, 1) using the generator."""
    return float(torch.rand(1, generator=rng).item())


# Random tree generation


def _random_node(
    pool: OperatorPool,
    rng: torch.Generator,
) -> Node:
    """Create a random Node by sampling from an operator/variable pool."""
    name, arity = pool[_rand_choice(len(pool), rng)]
    return Node(name=name, arity=arity, children=[])


def _is_derivative(name: str) -> bool:
    """Return True when the operator is a derivative node."""
    return name in {"d", "d^2"}


def _child_pool(
    parent: Node,
    child_idx: int,
    var_pool: OperatorPool,
    den: OperatorPool,
) -> OperatorPool:
    """Return the valid leaf pool for a child position."""
    if _is_derivative(parent.name) and child_idx == 1:
        return den
    return var_pool


def _sample_leaf(
    pool: OperatorPool,
    old_name: str,
    rng: torch.Generator,
) -> Node:
    """Sample a replacement leaf, retrying to avoid the same name."""
    new_node = _random_node(pool, rng)
    for _ in range(_MAX_MUTATE_RETRIES):
        if new_node.name != old_name or len(pool) <= 1:
            break
        new_node = _random_node(pool, rng)
    return new_node


def _sample_operator(
    pool: OperatorPool,
    old_name: str,
    old_arity: int,
    rng: torch.Generator,
    right_child: Node | None = None,
    den_names: set[str] | None = None,
) -> tuple[str, int]:
    """Sample an operator replacement, honoring derivative RHS constraints."""
    new_name, new_arity = pool[_rand_choice(len(pool), rng)]
    for _ in range(_MAX_MUTATE_RETRIES):
        if new_name == old_name and len(pool) > 1:
            new_name, new_arity = pool[_rand_choice(len(pool), rng)]
            continue
        if _operator_is_valid(new_name, right_child, den_names):
            return new_name, new_arity
        new_name, new_arity = pool[_rand_choice(len(pool), rng)]
    return old_name, old_arity


def _operator_is_valid(
    name: str,
    right_child: Node | None,
    den_names: set[str] | None,
) -> bool:
    """Return True when a mutated binary operator keeps derivative grammar valid."""
    if not _is_derivative(name):
        return True
    if right_child is None or den_names is None:
        return False
    return right_child.is_leaf and right_child.name in den_names


def random_tree(
    vars: list[str], # noqa: A002
    ops: OperatorPool,
    root: OperatorPool,
    den: OperatorPool,
    depth: int,
    p_var: float,
    rng: torch.Generator,
) -> Tree:
    """Generate a random expression tree.

    Parameters
    ----------
    vars: list[str]
        Available variable names (leaves).
    ops: tuple[tuple[str, int], ...]
        Available operators for internal nodes.
    root: tuple[tuple[str, int], ...]
        Operator pool for the root node (no +/- typically).
    depth: int
        Maximum depth of the tree (root is depth 0, leaves at depth-1).
    p_var: float
        Probability that an internal position becomes a leaf (variable).
    rng: torch.Generator
        Random number generator for reproducibility.

    Returns
    -------
    Tree
        A newly generated random expression tree.
    """
    # Build variable pool as (name, 0) tuples
    var_pool: OperatorPool = tuple((v, 0) for v in vars)

    # Generate root node from ROOT pool
    root_node = _random_node(root, rng)

    # Build tree level by level (BFS-like), matching the predecessor algorithm
    # But store as a real tree structure with children
    _fill_children(root_node, 1, depth, var_pool, ops, den, p_var, rng)

    return Tree(root=root_node)


def _fill_children(
    node: Node,
    current_depth: int,
    max_depth: int,
    var_pool: OperatorPool,
    ops: OperatorPool,
    den: OperatorPool,
    p_var: float,
    rng: torch.Generator,
) -> None:
    """Recursively fill children of a node.

    Rules:
    - Last level (current_depth == max_depth - 1): must be variable (leaf)
    - Other levels: p_var chance of variable, else operator
    """
    if node.arity == 0:
        return

    for j in range(node.arity):
        leaf_pool = _child_pool(node, j, var_pool, den)
        if (
            (_is_derivative(node.name) and j == 1)
            or current_depth >= max_depth - 1
            or _rand_float(rng) <= p_var
        ):
            child = _random_node(leaf_pool, rng)
        else:
            child = _random_node(ops, rng)

        node.children.append(child)
        _fill_children(
            child,
            current_depth + 1,
            max_depth,
            var_pool,
            ops,
            den,
            p_var,
            rng,
        )


# Random PDE generation


def random_pde(
    config: SGAConfig,
    vars: list[str], # noqa: A002
    ops: OperatorPool,
    root: OperatorPool,
    den: OperatorPool,
    rng: torch.Generator,
) -> PDE:
    """Generate a random PDE with 1..config.width terms.

    Parameters
    ----------
    config: SGAConfig
        Configuration (uses width, depth, p_var).
    vars: list[str]
        Available variable names.
    ops: tuple[tuple[str, int], ...]
        Operator pool for internal nodes.
    root: tuple[tuple[str, int], ...]
        Operator pool for root nodes.
    rng: torch.Generator
        Random number generator.

    Returns
    -------
    PDE
        A newly generated random PDE.
    """
    # Random width: 1 to config.width (inclusive), matching the predecessor
    width = int(torch.randint(1, config.width + 1, (1,), generator=rng).item())
    terms: list[Tree] = []
    for _ in range(width):
        tree = random_tree(vars, ops, root, den, config.depth, config.p_var, rng)
        terms.append(tree)
    return PDE(terms=terms)


# Mutation


def mutate(
    pde: PDE,
    vars: list[str], # noqa: A002
    op1: OperatorPool,
    op2: OperatorPool,
    den: OperatorPool,
    p_mute: float,
    rng: torch.Generator,
) -> PDE:
    """Mutate a PDE by randomly replacing nodes with same-arity alternatives.

    Creates a deep copy first, then mutates each node with probability p_mute.
    Leaf nodes are replaced with random variables; operators are replaced with
    same-arity operators (unary with op1, binary with op2).

    Parameters
    ----------
    pde: PDE
        Input PDE (not modified).
    vars: list[str]
        Available variable names.
    op1: tuple[tuple[str, int], ...]
        Unary operator pool for mutation.
    op2: tuple[tuple[str, int], ...]
        Binary operator pool for mutation.
    p_mute: float
        Mutation probability per node.
    rng: torch.Generator
        Random number generator.

    Returns
    -------
    PDE
        A new PDE with mutations applied.
    """
    new_pde = pde.copy()
    var_pool: OperatorPool = tuple((v, 0) for v in vars)
    den_names = {name for name, _ in den}

    for tree in new_pde.terms:
        _mutate_subtree(tree.root, var_pool, op1, op2, den, den_names, p_mute, rng)

    return new_pde


_MAX_MUTATE_RETRIES: int = 10
"""Max retries to avoid same-name mutation (the predecessor uses while loop)."""


def _mutate_subtree(
    node: Node,
    var_pool: OperatorPool,
    op1: OperatorPool,
    op2: OperatorPool,
    den: OperatorPool,
    den_names: set[str],
    p_mute: float,
    rng: torch.Generator,
) -> None:
    """Recursively mutate children of a node (skip root, mutate children).

    Following the predecessor: iteration starts from depth=1 (children of root),
    not the root itself. Mutation replaces with same-arity operators.
    Retry sampling to avoid same-name replacement (the predecessor ``while`` loop).
    """
    for i, child in enumerate(node.children):
        # Decide whether to mutate this child
        if _rand_float(rng) < p_mute:
            if child.arity == 0:
                pool = _child_pool(node, i, var_pool, den)
                node.children[i] = _sample_leaf(pool, child.name, rng)
            elif child.arity == 1:
                new_name, new_arity = _sample_operator(
                    op1,
                    child.name,
                    child.arity,
                    rng,
                )
                node.children[i] = Node(
                    name=new_name,
                    arity=new_arity,
                    children=child.children,
                )
            elif child.arity == 2:
                right_child = child.children[1] if len(child.children) == 2 else None
                new_name, new_arity = _sample_operator(
                    op2,
                    child.name,
                    child.arity,
                    rng,
                    right_child=right_child,
                    den_names=den_names,
                )
                node.children[i] = Node(
                    name=new_name,
                    arity=new_arity,
                    children=child.children,
                )
        _mutate_subtree(
            node.children[i],
            var_pool,
            op1,
            op2,
            den,
            den_names,
            p_mute,
            rng,
        )


# Crossover


def crossover(
    pde1: PDE,
    pde2: PDE,
    rng: torch.Generator,
) -> tuple[PDE, PDE]:
    """Crossover two PDEs by swapping a random term between them.

    Creates deep copies first, then swaps one term at a random position
    from each PDE.

    Parameters
    ----------
    pde1: PDE
        First parent PDE (not modified).
    pde2: PDE
        Second parent PDE (not modified).
    rng: torch.Generator
        Random number generator.

    Returns
    -------
    tuple[PDE, PDE]
        Two new offspring PDEs.
    """
    new1 = pde1.copy()
    new2 = pde2.copy()

    if new1.width == 0 or new2.width == 0:
        return new1, new2

    idx1 = _rand_choice(new1.width, rng)
    idx2 = _rand_choice(new2.width, rng)

    # Swap terms
    new1.terms[idx1], new2.terms[idx2] = new2.terms[idx2], new1.terms[idx1]

    return new1, new2


# Replace


def replace(
    pde: PDE,
    vars: list[str], # noqa: A002
    ops: OperatorPool,
    root: OperatorPool,
    den: OperatorPool,
    depth: int,
    p_var: float,
    rng: torch.Generator,
) -> PDE:
    """Replace a random term in the PDE with a new random tree.

    Creates a deep copy, then replaces one randomly chosen term.

    Parameters
    ----------
    pde: PDE
        Input PDE (not modified).
    vars: list[str]
        Available variable names.
    ops: tuple[tuple[str, int], ...]
        Operator pool for internal nodes.
    root: tuple[tuple[str, int], ...]
        Operator pool for root nodes.
    depth: int
        Maximum tree depth.
    p_var: float
        Probability of variable at each position.
    rng: torch.Generator
        Random number generator.

    Returns
    -------
    PDE
        A new PDE with one term replaced.
    """
    new_pde = pde.copy()

    if new_pde.width == 0:
        return new_pde

    idx = _rand_choice(new_pde.width, rng)
    new_tree = random_tree(vars, ops, root, den, depth, p_var, rng)
    new_pde.terms[idx] = new_tree

    return new_pde
