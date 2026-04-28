"""Phase A integration regression: cross-cutting assertions for T1-T5 bugfixes.

Validates that multiple T1-T5 fixes coexist and work end-to-end:
  - T1: derivative tree ops (d/d^2), den constraint, finite-diff execution, LHS filtering
  - T2: (no pre-eval truncation), (p_cro=0 disables crossover)
  - T3: init population resample for pathological individuals
  - T4: prune_invalid_terms, CandidateResult, genotype sync after evaluation
  - T5: prepare() fail-fast guardrails

Cross-cutting scenarios that exercise multiple fixes simultaneously:
  1. derivative tree + genetic ops + finite AIC
  2. derivative + crossover + genotype-theta alignment
  3. init population all finite + guardrails + Burgers recovery
  4. p_cro=0 full search completion
"""

from __future__ import annotations

import math

import pytest
import torch

from kd2.core.evaluator import Evaluator
from kd2.core.executor.context import ExecutionContext
from kd2.core.expr import FunctionRegistry, PythonExecutor
from kd2.core.linear_solve.least_squares import LeastSquaresSolver
from kd2.data.derivatives.finite_diff import FiniteDiffProvider
from kd2.data.synthetic import generate_burgers_data
from kd2.search.protocol import PlatformComponents
from kd2.search.runner import ExperimentRunner
from kd2.search.sga import SGAConfig, SGAPlugin
from kd2.search.sga.config import OP1, OP2, OPS, ROOT
from kd2.search.sga.evaluate import prune_invalid_terms
from kd2.search.sga.genetic import crossover, mutate, replace
from kd2.search.sga.pde import PDE
from kd2.search.sga.train import evaluate_candidate

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_NX = 128
_NT = 51
_NU = 0.1
_SEED = 42
_POPULATION = 10
_GENERATIONS = 5


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def burgers_components() -> PlatformComponents:
    """Create PlatformComponents for Burgers equation."""
    dataset = generate_burgers_data(nx=_NX, nt=_NT, nu=_NU, seed=_SEED)
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(
        dataset=dataset,
        derivative_provider=provider,
    )
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()
    u_t = provider.get_derivative("u", "t", order=1).flatten()
    evaluator = Evaluator(
        executor=executor,
        solver=solver,
        context=context,
        lhs=u_t,
    )
    return PlatformComponents(
        dataset=dataset,
        executor=executor,
        evaluator=evaluator,
        context=context,
        registry=registry,
    )


@pytest.fixture(scope="module")
def prepared_plugin(burgers_components: PlatformComponents) -> SGAPlugin:
    """A prepared SGAPlugin (population initialized) for reuse in tests."""
    config = SGAConfig(
        num=_POPULATION,
        depth=3,
        width=4,
        p_var=0.5,
        p_mute=0.3,
        p_cro=0.5,
        p_rep=1.0,
        seed=_SEED,
        maxit=5,
        str_iters=5,
        d_tol=0.5,
    )
    plugin = SGAPlugin(config=config)
    plugin.prepare(burgers_components)
    return plugin


# ═══════════════════════════════════════════════════════════════════════════
# 1. Derivative tree + genetic ops + finite AIC
# ═══════════════════════════════════════════════════════════════════════════


class TestDerivativeTreeGeneticOpsFiniteAIC:
    """Derivative nodes survive genetic operations and produce finite AIC."""

    @pytest.mark.integration
    def test_derivative_pde_through_mutation_produces_finite_aic(
        self, prepared_plugin: SGAPlugin
    ) -> None:
        """A PDE with derivative terms, after mutation, can be evaluated
        to produce a finite AIC (T1 execution + T4 pruning)."""
        from kd2.search.sga.tree import Node, Tree

        state = prepared_plugin.state
        data_dict = prepared_plugin._data_dict
        diff_ctx = prepared_plugin._diff_ctx
        default_terms = prepared_plugin._default_terms
        y = prepared_plugin._y
        config = prepared_plugin._config
        vars_list = state["vars"]
        den = prepared_plugin._den

        # Build a PDE with an explicit derivative term: d(u, x)
        d_node = Node(
            name="d",
            arity=2,
            children=[
                Node(name="u", arity=0),
                Node(name="x", arity=0),
            ],
        )
        d_tree = Tree(root=d_node)
        # Second term: simple leaf
        leaf_tree = Tree(root=Node(name="u_x", arity=0))
        pde = PDE(terms=[d_tree, leaf_tree])

        # Apply mutation
        rng = torch.Generator().manual_seed(123)
        mutated = mutate(pde, vars_list, OP1, OP2, den, 0.5, rng)

        # Evaluate the mutated PDE
        cr = evaluate_candidate(
            mutated,
            data_dict,
            default_terms,
            y if y is not None else torch.zeros(1),
            config,
            diff_ctx=diff_ctx,
        )

        # The AIC should be finite (T1 execution + T4 genotype sync)
        assert math.isfinite(cr.aic_score), (
            f"Derivative PDE after mutation has non-finite AIC: {cr.aic_score}"
        )
        # Pruned PDE should have surviving terms aligned with coefficients
        assert len(cr.pruned_pde.terms) >= 1
        # Coefficient vector size must match theta columns (defaults + valid terms)
        n_default = default_terms.shape[1] if default_terms is not None else 0
        n_valid = len(cr.pruned_pde.terms)
        assert cr.coefficients.shape[0] == n_default + n_valid, (
            f"Coeff size {cr.coefficients.shape[0]} != theta cols {n_default + n_valid}"
        )

    @pytest.mark.integration
    def test_derivative_pde_through_replace_produces_valid_result(
        self, prepared_plugin: SGAPlugin
    ) -> None:
        """PDE with derivative term, after replace op, still evaluates."""
        from kd2.search.sga.tree import Node, Tree

        state = prepared_plugin.state
        data_dict = prepared_plugin._data_dict
        diff_ctx = prepared_plugin._diff_ctx
        default_terms = prepared_plugin._default_terms
        y = prepared_plugin._y
        config = prepared_plugin._config
        vars_list = state["vars"]
        den = prepared_plugin._den

        # PDE: d^2(u, x)
        d2_node = Node(
            name="d^2",
            arity=2,
            children=[
                Node(name="u", arity=0),
                Node(name="x", arity=0),
            ],
        )
        pde = PDE(terms=[Tree(root=d2_node)])

        rng = torch.Generator().manual_seed(456)
        replaced = replace(pde, vars_list, OPS, ROOT, den, 3, 0.5, rng)

        cr = evaluate_candidate(
            replaced,
            data_dict,
            default_terms,
            y if y is not None else torch.zeros(1),
            config,
            diff_ctx=diff_ctx,
        )

        # Should not crash and AIC should be a real number (finite or inf, not NaN)
        assert not math.isnan(cr.aic_score)


# ═══════════════════════════════════════════════════════════════════════════
# 2. Derivative + crossover + genotype-theta alignment
# ═══════════════════════════════════════════════════════════════════════════


class TestDerivativeCrossoverGenotypeAlignment:
    """Crossover with derivative PDEs maintains genotype-theta alignment."""

    @pytest.mark.integration
    def test_crossover_derivative_pdes_genotype_sync(
        self, prepared_plugin: SGAPlugin
    ) -> None:
        """After crossover of PDEs containing derivative terms, the
        evaluated genotype (pruned_pde) has terms aligned with theta columns."""
        from kd2.search.sga.tree import Node, Tree

        data_dict = prepared_plugin._data_dict
        diff_ctx = prepared_plugin._diff_ctx
        default_terms = prepared_plugin._default_terms
        y = prepared_plugin._y
        config = prepared_plugin._config

        # PDE 1: [d(u, x), u_x]
        pde1 = PDE(
            terms=[
                Tree(
                    root=Node(
                        name="d",
                        arity=2,
                        children=[
                            Node(name="u", arity=0),
                            Node(name="x", arity=0),
                        ],
                    )
                ),
                Tree(root=Node(name="u_x", arity=0)),
            ]
        )

        # PDE 2: [*(u, x), d(u_x, x)]
        # u_xx is NOT a precomputed terminal; use tree composition
        pde2 = PDE(
            terms=[
                Tree(
                    root=Node(
                        name="*",
                        arity=2,
                        children=[
                            Node(name="u", arity=0),
                            Node(name="x", arity=0),
                        ],
                    )
                ),
                Tree(
                    root=Node(
                        name="d",
                        arity=2,
                        children=[
                            Node(name="u_x", arity=0),
                            Node(name="x", arity=0),
                        ],
                    )
                ),
            ]
        )

        rng = torch.Generator().manual_seed(789)
        c1, c2 = crossover(pde1, pde2, rng)

        # Evaluate both offspring
        for offspring in (c1, c2):
            cr = evaluate_candidate(
                offspring,
                data_dict,
                default_terms,
                y if y is not None else torch.zeros(1),
                config,
                diff_ctx=diff_ctx,
            )

            # Genotype sync: pruned_pde terms count must equal
            # theta columns minus default_terms columns
            n_default = default_terms.shape[1] if default_terms is not None else 0
            n_pde_terms = len(cr.pruned_pde.terms)

            # selected_indices refer to columns in theta (defaults + pde_terms)
            # All selected_indices must be valid column indices
            if cr.selected_indices is not None:
                for idx in cr.selected_indices:
                    assert idx < n_default + n_pde_terms, (
                        f"selected_index {idx} out of range "
                        f"(n_default={n_default}, n_pde_terms={n_pde_terms})"
                    )

            # Coefficients length must match theta column count
            if cr.coefficients is not None and cr.coefficients.numel() > 0:
                assert cr.coefficients.shape[0] == n_default + n_pde_terms, (
                    f"Coefficient size {cr.coefficients.shape[0]} != "
                    f"theta cols {n_default + n_pde_terms}"
                )

    @pytest.mark.integration
    def test_crossover_then_mutation_derivative_alignment(
        self, prepared_plugin: SGAPlugin
    ) -> None:
        """Chain: crossover -> mutation on derivative PDEs keeps alignment."""
        from kd2.search.sga.tree import Node, Tree

        state = prepared_plugin.state
        data_dict = prepared_plugin._data_dict
        diff_ctx = prepared_plugin._diff_ctx
        default_terms = prepared_plugin._default_terms
        y = prepared_plugin._y
        config = prepared_plugin._config
        vars_list = state["vars"]
        den = prepared_plugin._den

        pde1 = PDE(
            terms=[
                Tree(
                    root=Node(
                        name="d^2",
                        arity=2,
                        children=[
                            Node(name="u", arity=0),
                            Node(name="x", arity=0),
                        ],
                    )
                ),
                Tree(root=Node(name="u", arity=0)),
            ]
        )

        pde2 = PDE(
            terms=[
                Tree(root=Node(name="u_x", arity=0)),
                Tree(root=Node(name="x", arity=0)),
            ]
        )

        rng = torch.Generator().manual_seed(101)

        # Crossover
        c1, _ = crossover(pde1, pde2, rng)

        # Mutation
        m1 = mutate(c1, vars_list, OP1, OP2, den, 0.5, rng)

        # Evaluate
        cr = evaluate_candidate(
            m1,
            data_dict,
            default_terms,
            y if y is not None else torch.zeros(1),
            config,
            diff_ctx=diff_ctx,
        )

        # Verify alignment: pruned_pde has only valid terms
        pruned, valid_terms, valid_indices = prune_invalid_terms(
            m1,
            data_dict,
            diff_ctx=diff_ctx,
        )
        assert len(pruned.terms) == len(valid_indices)
        assert len(cr.pruned_pde.terms) == len(cr.valid_term_indices)


# ═══════════════════════════════════════════════════════════════════════════
# 3. Init population all finite + guardrails + Burgers recovery
# ═══════════════════════════════════════════════════════════════════════════


class TestInitPopulationGuardrailsBurgers:
    """Init population produces all-finite AIC, guardrails pass, search works."""

    @pytest.mark.integration
    def test_init_population_all_finite_aic(
        self, burgers_components: PlatformComponents
    ) -> None:
        """After prepare(), every individual in the population must have
        a finite AIC score"""
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.5,
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        scores = plugin.state["scores"]
        assert len(scores) == _POPULATION
        for i, score in enumerate(scores):
            assert math.isfinite(score), (
                f"Population member {i} has non-finite AIC: {score}"
            )

    @pytest.mark.integration
    def test_prepare_guardrails_pass_on_burgers(
        self, burgers_components: PlatformComponents
    ) -> None:
        """T5 guardrails (naming conflict, LHS derivative) should all pass
        cleanly on well-formed Burgers data."""
        config = SGAConfig(
            num=5,
            depth=3,
            width=3,
            p_var=0.5,
            seed=_SEED,
        )
        plugin = SGAPlugin(config=config)
        # Should NOT raise any ValueError from _validate_naming or _extract_lhs_target
        plugin.prepare(burgers_components)

        # Verify LHS target was properly extracted (not zeros)
        assert plugin._y is not None
        assert plugin._y.numel() > 0
        assert torch.isfinite(plugin._y).all()
        # Must not be all-zeros (that would indicate fallback, not real derivative)
        assert plugin._y.abs().sum().item() > 0, (
            "LHS target (_y) is all zeros — derivative extraction may have failed"
        )

    @pytest.mark.integration
    def test_init_population_finite_then_search_recovers_burgers(
        self, burgers_components: PlatformComponents
    ) -> None:
        """End-to-end: finite init pop + guardrails pass + search improves.
        Combines T3 (resample), T5 (guardrails), T1 (derivative ops)."""
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.5,
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)

        # 1. All init scores were finite (validated above)
        # 2. Search completed without crash
        assert result.iterations == _GENERATIONS
        # 3. Best score is finite (search found something)
        assert math.isfinite(result.best_score), (
            f"Best score after {_GENERATIONS} gens is not finite: {result.best_score}"
        )
        # 4. Expression is non-empty
        assert len(result.best_expression) > 0

    @pytest.mark.integration
    def test_vars_exclude_lhs_derivatives_after_prepare(
        self, burgers_components: PlatformComponents
    ) -> None:
        """T1+T5: VARS must exclude lhs_axis and its derivatives.
        This prevents trivial solutions where u_t appears on both sides."""
        config = SGAConfig(num=5, depth=3, width=3, seed=_SEED)
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        vars_list = plugin.state["vars"]
        # lhs_axis = "t", lhs_field = "u" => u_t, u_tt excluded
        assert "t" not in vars_list
        assert "u_t" not in vars_list
        assert "u_tt" not in vars_list
        # But spatial vars and derivatives should be present
        assert "u" in vars_list
        assert "x" in vars_list
        assert "u_x" in vars_list
        # u_xx is NOT a precomputed terminal
        assert "u_xx" not in vars_list


# ═══════════════════════════════════════════════════════════════════════════
# 4. p_cro=0 full search completion
# ═══════════════════════════════════════════════════════════════════════════


class TestPCroZeroFullSearch:
    """With p_cro=0, search completes and no crossover occurs."""

    @pytest.mark.integration
    def test_p_cro_zero_search_completes(
        self, burgers_components: PlatformComponents
    ) -> None:
        """: p_cro=0 config runs full search without error."""
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.0, # <<< Crossover disabled
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)

        # Search completed all iterations
        assert result.iterations == _GENERATIONS
        # Best score is not NaN
        assert not math.isnan(result.best_score)
        # Expression is non-empty
        assert len(result.best_expression) > 0

    @pytest.mark.integration
    def test_p_cro_zero_offspring_count_matches_mutation_only(
        self, burgers_components: PlatformComponents
    ) -> None:
        """With p_cro=0, offspring should come only from mutation/replacement.
        Expected: num-1 offspring per generation (non-elite mutation)."""
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.0,
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        candidates = plugin.propose(config.num)

        # With p_cro=0: no crossover offspring, only mutation offspring
        # mutation loop: range(1, num) => num-1 offspring
        expected_count = config.num - 1
        assert len(candidates) == expected_count, (
            f"With p_cro=0, expected {expected_count} offspring "
            f"(mutation only), got {len(candidates)}"
        )

    @pytest.mark.integration
    def test_p_cro_zero_best_score_improves(
        self, burgers_components: PlatformComponents
    ) -> None:
        """Even without crossover, search should still improve (or not worsen)."""
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.0,
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        initial_score = plugin.best_score

        for _ in range(_GENERATIONS):
            candidates = plugin.propose(config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

        final_score = plugin.best_score
        assert final_score <= initial_score, (
            f"Score worsened: {initial_score} -> {final_score}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 5. offspring not truncated before evaluation
# ═══════════════════════════════════════════════════════════════════════════


class TestDelta007NoPreEvalTruncation:
    """: the full offspring frontier must be evaluated before truncation."""

    @pytest.mark.integration
    def test_propose_returns_full_offspring_frontier(
        self, burgers_components: PlatformComponents
    ) -> None:
        """propose() must return ALL offspring (crossover + mutation), not
        truncated to config.num. With p_cro=0.5 and num=10, crossover
        produces ~5 offspring + mutation produces 9 = total > num."""
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.5,
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        candidates = plugin.propose(config.num)

        # With p_cro=0.5, num=10: crossover uses top 5, producing ~5 offspring
        # (5 individuals paired: 2+2+1=5). Mutation: range(1,10)=9 offspring.
        # Total should be > num (10), proving no truncation occurred.
        assert len(candidates) > config.num, (
            f"Expected offspring count > {config.num} (no pre-eval truncation), "
            f"got {len(candidates)}. may have regressed."
        )

    @pytest.mark.integration
    def test_all_offspring_evaluated_before_selection(
        self, burgers_components: PlatformComponents
    ) -> None:
        """evaluate() must process ALL candidates from propose(), and
        update() performs the truncation to config.num."""
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.5,
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        candidates = plugin.propose(config.num)
        results = plugin.evaluate(candidates)

        # Every candidate must have been evaluated
        assert len(results) == len(candidates), (
            f"evaluate() returned {len(results)} results for "
            f"{len(candidates)} candidates"
        )

        # After update, population is truncated to config.num
        plugin.update(results)
        pop = plugin.state["population"]
        assert len(pop) == config.num, (
            f"After update, population size should be {config.num}, got {len(pop)}"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 6. Plugin-level genotype sync after evaluate()
# ═══════════════════════════════════════════════════════════════════════════


class TestPluginGenotypeSync:
    """Plugin.evaluate() must replace offspring with pruned PDEs."""

    @pytest.mark.integration
    def test_evaluate_syncs_offspring_genotype(
        self, burgers_components: PlatformComponents
    ) -> None:
        """After plugin.evaluate(), internal offspring PDEs should be
        replaced with their pruned versions (genotype sync).

        This ensures that when offspring enter the population via update(),
        their term list matches what was actually scored."""
        config = SGAConfig(
            num=_POPULATION,
            depth=3,
            width=4,
            p_var=0.5,
            p_mute=0.3,
            p_cro=0.5,
            p_rep=1.0,
            seed=_SEED,
            maxit=5,
            str_iters=5,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        candidates = plugin.propose(config.num)

        # Capture offspring before evaluation
        offspring_before = [pde.copy() for pde in (plugin._offspring or [])]
        assert len(offspring_before) > 0

        # Evaluate
        results = plugin.evaluate(candidates)

        # After evaluate, offspring should be synced (pruned)
        offspring_after = plugin._offspring or []
        assert len(offspring_after) == len(offspring_before)

        # At least some offspring should have been pruned (fewer terms)
        # since random trees often produce invalid terms
        pruned_count = sum(
            1
            for before, after in zip(offspring_before, offspring_after, strict=True)
            if after.width < before.width
        )
        # It's acceptable if no pruning happened (all terms valid),
        # but the sync mechanism must not crash
        assert pruned_count >= 0 # Always true; real assertion is no crash

        # Stronger check: for each offspring, verify terms are self-consistent
        # by re-executing and confirming all terms produce finite values
        data_dict = plugin._data_dict
        diff_ctx = plugin._diff_ctx
        for i, pde in enumerate(offspring_after):
            _, valid_terms, valid_indices = prune_invalid_terms(
                pde,
                data_dict,
                diff_ctx=diff_ctx,
            )
            # A synced (pruned) PDE should have ALL its terms valid
            assert len(valid_indices) == pde.width, (
                f"Offspring {i}: synced PDE has {pde.width} terms but only "
                f"{len(valid_indices)} are valid — genotype sync failed"
            )
