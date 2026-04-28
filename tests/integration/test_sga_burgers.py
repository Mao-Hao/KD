"""Integration test: SGA on Burgers equation (end-to-end).

Validates the full SGA pipeline on Burgers data:
  generate_burgers_data → FiniteDiffProvider → PlatformComponents
  → SGAPlugin → ExperimentRunner → RunResult

Two validation layers:
  Layer A: Smoke — completes without crash/NaN, produces a best expression
  Layer B: Scientific sanity — best expression improves over baseline
"""

from __future__ import annotations

import math

import pytest

from kd2.core.evaluator import Evaluator
from kd2.core.executor.context import ExecutionContext
from kd2.core.expr import FunctionRegistry, PythonExecutor
from kd2.core.linear_solve.least_squares import LeastSquaresSolver
from kd2.data.derivatives.finite_diff import FiniteDiffProvider
from kd2.data.synthetic import generate_burgers_data
from kd2.search.protocol import PlatformComponents
from kd2.search.runner import ExperimentRunner
from kd2.search.sga import SGAConfig, SGAPlugin

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_NX = 128
"""Spatial grid points (smaller than default 256 for speed)."""

_NT = 51
"""Time grid points (smaller than default 101 for speed)."""

_NU = 0.1
"""Burgers viscosity coefficient."""

_SEED = 42
"""Fixed seed for reproducibility."""

_SMOKE_GENERATIONS = 5
"""Number of SGA generations for smoke test."""

_SCIENCE_GENERATIONS = 20
"""Number of SGA generations for scientific sanity test."""

_POPULATION = 10
"""Population size (smaller than default 20 for speed)."""


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture(scope="module")
def burgers_components() -> PlatformComponents:
    """Create PlatformComponents for Burgers equation.

    Uses FiniteDiffProvider for derivatives (4th-order central diff).
    Scope=module so the expensive FFT-based data generation runs once.
    """
    dataset = generate_burgers_data(nx=_NX, nt=_NT, nu=_NU, seed=_SEED)
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(
        dataset=dataset,
        derivative_provider=provider,
    )
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()
    # Evaluator needs lhs target (u_t)
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


def _make_sga_config(
    generations: int = _SMOKE_GENERATIONS,
    population: int = _POPULATION,
) -> SGAConfig:
    """Create a fast SGA config for testing."""
    return SGAConfig(
        num=population,
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


# ═══════════════════════════════════════════════════════════════════════════
# Layer A: Smoke Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestSGABurgersSmoke:
    """Layer A: SGA completes on Burgers data without crash/NaN."""

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_plugin_direct_cycle(self, burgers_components: PlatformComponents) -> None:
        """A single prepare → propose → evaluate → update cycle completes."""
        config = _make_sga_config(generations=1)
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        candidates = plugin.propose(config.num)
        assert len(candidates) > 0
        assert all(isinstance(c, str) for c in candidates)

        results = plugin.evaluate(candidates)
        assert len(results) == len(candidates)

        plugin.update(results)
        assert isinstance(plugin.best_score, float)
        assert isinstance(plugin.best_expression, str)

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_runner_completes(self, burgers_components: PlatformComponents) -> None:
        """ExperimentRunner.run() completes all iterations without error."""
        config = _make_sga_config(generations=_SMOKE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_SMOKE_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)

        assert result.iterations == _SMOKE_GENERATIONS
        assert isinstance(result.best_expression, str)
        assert len(result.best_expression) > 0
        assert isinstance(result.best_score, float)
        assert not result.early_stopped

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_no_nan_in_best_score(self, burgers_components: PlatformComponents) -> None:
        """best_score should never be NaN (inf is acceptable for degenerate cases)."""
        config = _make_sga_config(generations=_SMOKE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_SMOKE_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)

        assert not math.isnan(result.best_score)

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_vars_exclude_lhs_axis(
        self, burgers_components: PlatformComponents
    ) -> None:
        """After prepare, VARS should not contain lhs_axis 't' or its derivatives."""
        config = _make_sga_config()
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        vars_list = plugin.state["vars"]
        # "t" is the lhs_axis — excluded from VARS
        assert "t" not in vars_list
        # u_t, u_tt are lhs_axis derivatives — excluded
        assert "u_t" not in vars_list
        assert "u_tt" not in vars_list
        # Spatial variable and derivatives should be present
        assert "u" in vars_list
        assert "x" in vars_list
        assert "u_x" in vars_list
        # u_xx is NOT a precomputed terminal
        assert "u_xx" not in vars_list

    @pytest.mark.smoke
    @pytest.mark.integration
    def test_seed_reproducibility(self, burgers_components: PlatformComponents) -> None:
        """Same seed produces same results."""
        config = _make_sga_config(generations=3)

        plugin1 = SGAPlugin(config=config)
        runner1 = ExperimentRunner(
            algorithm=plugin1, max_iterations=3, batch_size=config.num
        )
        result1 = runner1.run(burgers_components)

        plugin2 = SGAPlugin(config=config)
        runner2 = ExperimentRunner(
            algorithm=plugin2, max_iterations=3, batch_size=config.num
        )
        result2 = runner2.run(burgers_components)

        assert result1.best_expression == result2.best_expression
        assert result1.best_score == result2.best_score


# ═══════════════════════════════════════════════════════════════════════════
# Layer B: Scientific Sanity
# ═══════════════════════════════════════════════════════════════════════════


class TestSGABurgersScience:
    """Layer B: SGA finds scientifically plausible expressions for Burgers."""

    @pytest.mark.integration
    def test_best_score_improves_over_generations(
        self, burgers_components: PlatformComponents
    ) -> None:
        """best_score after N generations should be <= initial population best."""
        config = _make_sga_config(generations=_SCIENCE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        plugin.prepare(burgers_components)

        initial_score = plugin.best_score

        for _ in range(_SCIENCE_GENERATIONS):
            candidates = plugin.propose(config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

        final_score = plugin.best_score
        # AIC should improve (decrease) or at least not worsen
        assert final_score <= initial_score, (
            f"Final score {final_score} > initial {initial_score}"
        )

    @pytest.mark.integration
    def test_best_expression_contains_spatial_variables(
        self, burgers_components: PlatformComponents
    ) -> None:
        """After search, best expression should reference spatial variables.

        For Burgers (u_t = -u*u_x + nu*u_xx), the SGA should discover
        expressions involving u, u_x, u_xx. With the SGA/M1 fix, the
        rendered expression now embeds coefficients (``-1.5*u_x``), so
        we check substring presence rather than whitespace tokens.
        """
        config = _make_sga_config(generations=_SCIENCE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_SCIENCE_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)

        expr = result.best_expression

        # The expression should reference at least one spatial derivative
        spatial_vars = ("u_x", "u_xx", "x", "u")
        assert any(v in expr for v in spatial_vars), (
            f"Best expression '{expr}' does not reference any spatial "
            f"variables from {spatial_vars}"
        )

    @pytest.mark.integration
    def test_best_score_is_finite(self, burgers_components: PlatformComponents) -> None:
        """After sufficient generations, best_score should be finite (not inf).

        With real Burgers data, at least some candidates should produce
        valid AIC scores.
        """
        config = _make_sga_config(generations=_SCIENCE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_SCIENCE_GENERATIONS,
            batch_size=config.num,
        )
        result = runner.run(burgers_components)

        assert math.isfinite(result.best_score), (
            f"best_score is {result.best_score} after {_SCIENCE_GENERATIONS} "
            f"generations — no valid candidate was found"
        )

    @pytest.mark.integration
    def test_population_converges_to_finite_scores(
        self, burgers_components: PlatformComponents
    ) -> None:
        """After search, population scores should be mostly finite (not all inf).

        Note: convergence to a single dominant score is acceptable and
        indicates a strong solution was found.
        """
        config = _make_sga_config(generations=_SCIENCE_GENERATIONS)
        plugin = SGAPlugin(config=config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=_SCIENCE_GENERATIONS,
            batch_size=config.num,
        )
        runner.run(burgers_components)

        state = plugin.state
        scores = state["scores"]
        finite_count = sum(1 for s in scores if math.isfinite(s))
        # At least half the population should have finite scores
        assert finite_count >= len(scores) // 2, (
            f"Only {finite_count}/{len(scores)} population members have "
            f"finite scores — search may have failed"
        )
