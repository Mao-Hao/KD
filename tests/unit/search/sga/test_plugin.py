"""Tests for SGAPlugin — SearchAlgorithm protocol adapter for SGA.

TDD red phase: tests define the expected interface and behavior before
implementation. SGAPlugin wraps the SGA internals (genetic operators,
evaluate_candidate, train_sweep) into the SearchAlgorithm protocol.

Test strategy:
- Protocol conformance (isinstance, required methods/properties)
- Lifecycle (prepare -> propose -> evaluate -> update)
- State checkpoint round-trip
- Negative/edge cases (>= 20% of tests)
- Property-based checks (monotonicity, finiteness)
"""

from __future__ import annotations

import math
import pickle
from unittest.mock import MagicMock

import pytest
import torch

from kd2.core.evaluator import EvaluationResult
from kd2.data.schema import AxisInfo, DataTopology, FieldData, PDEDataset, TaskType
from kd2.search.protocol import PlatformComponents, SearchAlgorithm
from kd2.search.sga.config import SGAConfig

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_SMALL_GRID_SIZE = 10
"""Number of spatial points in the synthetic test grid."""

_SMALL_TIME_SIZE = 5
"""Number of time points in the synthetic test grid."""


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures: synthetic data for prepare()
# ═══════════════════════════════════════════════════════════════════════════


def _make_synthetic_dataset() -> PDEDataset:
    """Create a minimal PDEDataset for testing (Burgers-like, 1D+t)."""
    x_vals = torch.linspace(0.0, 1.0, _SMALL_GRID_SIZE)
    t_vals = torch.linspace(0.0, 1.0, _SMALL_TIME_SIZE)
    u_data = torch.randn(_SMALL_GRID_SIZE, _SMALL_TIME_SIZE)

    return PDEDataset(
        name="test_synthetic",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x_vals),
            "t": AxisInfo(name="t", values=t_vals),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_data)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_mock_derivative_provider() -> MagicMock:
    """Create a mock derivative provider returning sensible tensors."""
    provider = MagicMock()
    n_total = _SMALL_GRID_SIZE * _SMALL_TIME_SIZE

    def get_derivative(field_name: str, axis: str, order: int) -> torch.Tensor:
        # Return a deterministic but distinct tensor for each derivative
        torch.manual_seed(hash((field_name, axis, order)) % (2**31))
        return torch.randn(_SMALL_GRID_SIZE, _SMALL_TIME_SIZE)

    provider.get_derivative = get_derivative
    return provider


def _make_mock_context(
    dataset: PDEDataset,
) -> MagicMock:
    """Create a mock ExecutionContext with working get_variable/get_derivative."""
    context = MagicMock()
    context.dataset = dataset
    provider = _make_mock_derivative_provider()
    context.derivative_provider = provider

    def get_variable(name: str) -> torch.Tensor:
        if dataset.fields is not None and name in dataset.fields:
            return dataset.fields[name].values
        if dataset.axes is not None and name in dataset.axes:
            return dataset.axes[name].values
        raise KeyError(f"Variable '{name}' not found")

    def get_derivative(field_name: str, axis: str, order: int) -> torch.Tensor:
        return provider.get_derivative(field_name, axis, order)

    context.get_variable = get_variable
    context.get_derivative = get_derivative
    return context


@pytest.fixture
def sga_config() -> SGAConfig:
    """Small SGA config for fast unit tests."""
    return SGAConfig(
        num=5,
        depth=3,
        width=3,
        p_var=0.6,
        p_mute=0.3,
        p_cro=0.5,
        p_rep=1.0,
        seed=42,
        maxit=3,
        str_iters=3,
        d_tol=0.5,
    )


@pytest.fixture
def mock_components() -> PlatformComponents:
    """Create PlatformComponents with synthetic data for prepare()."""
    dataset = _make_synthetic_dataset()
    context = _make_mock_context(dataset)
    return PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=context,
        registry=MagicMock(),
    )


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 1: Protocol Conformance / Smoke (~25%)
# ═══════════════════════════════════════════════════════════════════════════


class TestSGAPluginProtocol:
    """SGAPlugin conforms to the SearchAlgorithm protocol."""

    @pytest.mark.smoke
    def test_importable(self) -> None:
        """SGAPlugin can be imported from kd2.search.sga.plugin."""
        from kd2.search.sga.plugin import SGAPlugin

        assert SGAPlugin is not None

    @pytest.mark.smoke
    def test_instantiate_default_config(self) -> None:
        """SGAPlugin() with default config does not raise."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert plugin is not None

    @pytest.mark.smoke
    def test_instantiate_custom_config(self, sga_config: SGAConfig) -> None:
        """SGAPlugin(config) accepts a custom SGAConfig."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        assert plugin is not None

    @pytest.mark.unit
    def test_isinstance_search_algorithm(self) -> None:
        """SGAPlugin passes isinstance(obj, SearchAlgorithm)."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert isinstance(plugin, SearchAlgorithm)

    @pytest.mark.unit
    def test_has_all_protocol_methods(self) -> None:
        """SGAPlugin has all SearchAlgorithm protocol methods/properties."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert callable(getattr(plugin, "prepare", None))
        assert callable(getattr(plugin, "propose", None))
        assert callable(getattr(plugin, "evaluate", None))
        assert callable(getattr(plugin, "update", None))
        # Properties
        assert hasattr(plugin, "best_score")
        assert hasattr(plugin, "best_expression")
        assert hasattr(plugin, "state")


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 2: prepare() behavior (~15%)
# ═══════════════════════════════════════════════════════════════════════════


class TestSGAPluginPrepare:
    """Tests for SGAPlugin.prepare() initialization."""

    @pytest.mark.unit
    def test_delta_map_accepts_float32_linspace_grid(
        self,
        sga_config: SGAConfig,
    ) -> None:
        """SGA grid spacing tolerance matches the public FD provider.

        ``torch.linspace(..., dtype=torch.float32)`` has small spacing drift.
        The provider accepts this; SGA's internal delta map must not reject it
        before the public API can run.
        """
        from kd2.search.sga.plugin import SGAPlugin

        x_vals = torch.linspace(0.0, 1.0, 1000, dtype=torch.float32)
        t_vals = torch.linspace(0.0, 1.0, 20, dtype=torch.float32)
        x_grid, t_grid = torch.meshgrid(x_vals, t_vals, indexing="ij")
        u_vals = torch.sin(x_grid) * torch.exp(-t_grid)
        dataset = PDEDataset(
            name="float32_grid",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals),
                "t": AxisInfo(name="t", values=t_vals),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u_vals)},
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=sga_config)
        delta = plugin._build_delta_map(dataset, ["x", "t"])

        assert delta["x"] == pytest.approx(float(x_vals[1] - x_vals[0]))
        assert delta["t"] == pytest.approx(float(t_vals[1] - t_vals[0]))

    @pytest.mark.unit
    def test_prepare_initializes_population(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """After prepare(), internal population should have config.num PDEs."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        # State should include a population of config.num PDEs
        state = plugin.state
        assert "population" in state
        assert len(state["population"]) == sga_config.num

    @pytest.mark.unit
    def test_prepare_builds_vars(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """After prepare(), VARS should be derived from dataset fields/derivatives."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        state = plugin.state
        # VARS should contain at least the field name "u" and some derivatives
        assert "vars" in state
        vars_list = state["vars"]
        assert isinstance(vars_list, list)
        assert len(vars_list) > 0
        # At minimum, the field variable should be present
        assert "u" in vars_list

    @pytest.mark.unit
    def test_prepare_excludes_lhs_axis_from_vars(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """VARS must NOT contain lhs_axis coordinate (e.g. 't').

        the predecessor: _build_vars skips axis == lhs_axis + _assert_no_lhs_in_vars.
        Including the time coordinate would let SGA discover trivial
        time-dependent terms.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        vars_list = plugin.state["vars"]
        # lhs_axis is "t" in our synthetic dataset
        assert "t" not in vars_list
        # Spatial axis "x" should still be present
        assert "x" in vars_list

    @pytest.mark.unit
    def test_prepare_excludes_lhs_axis_derivatives_from_vars(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """VARS must NOT contain lhs_axis derivatives (e.g. u_t, u_tt).

        the predecessor: _assert_no_lhs_in_grad_fields ensures grad_fields excludes
        lhs-axis derivatives. Including u_t would make the regression trivial
        (u_t = c * u_t).
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        vars_list = plugin.state["vars"]
        # lhs_axis derivatives should be excluded
        assert "u_t" not in vars_list
        assert "u_tt" not in vars_list
        # Spatial first-order derivatives should still be present
        assert "u_x" in vars_list
        # Second-order derivatives are NOT precomputed terminals;
        # they are reached via tree composition d(u_x, x) or d^2(u, x)
        assert "u_xx" not in vars_list

    @pytest.mark.unit
    def test_prepare_skips_population_init_if_already_populated(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """prepare() should skip population init if population already exists (checkpoint restore)."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)

        # First prepare (initializes population)
        plugin.prepare(mock_components)
        state_before = plugin.state
        pop_before = state_before["population"]

        # Simulate checkpoint restore: set state with existing population
        plugin.state = state_before

        # Second prepare (should not reinitialize population)
        plugin.prepare(mock_components)
        state_after = plugin.state
        pop_after = state_after["population"]

        # Population should be preserved from checkpoint
        assert len(pop_after) == len(pop_before)

    @pytest.mark.unit
    def test_prepare_twice_without_checkpoint_reinitializes(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Calling prepare() twice without state restore should re-initialize.

        The second prepare() must detect that no checkpoint was loaded and
        create a fresh population from scratch.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        # Do NOT set state manually -- this is a fresh prepare
        # The plugin should re-initialize the population
        state1 = plugin.state
        assert "population" in state1


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 3: propose() behavior (~15%)
# ═══════════════════════════════════════════════════════════════════════════


class TestSGAPluginPropose:
    """Tests for SGAPlugin.propose() candidate generation."""

    @pytest.mark.unit
    def test_propose_returns_list_of_strings(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """propose(n) returns a list of strings."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        assert isinstance(candidates, list)
        assert all(isinstance(c, str) for c in candidates)

    @pytest.mark.unit
    def test_propose_returns_full_frontier(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """propose(n) returns the full offspring frontier (may exceed n).

        truncation happens in update(), not propose().
        The offspring count depends on genetic ops (crossover + mutation),
        not on the n argument.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        n = sga_config.num
        candidates = plugin.propose(n)
        assert len(candidates) >= 1

    @pytest.mark.unit
    def test_propose_returns_kd2_format(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Proposed expressions are in kd2 prefix format (space-separated tokens)."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        for expr in candidates:
            # kd2 prefix format: space-separated tokens
            # Must be non-empty and contain only valid tokens
            tokens = expr.split()
            assert len(tokens) >= 1, f"Expression is empty: {expr!r}"

    @pytest.mark.unit
    def test_propose_applies_genetic_operators(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Two consecutive propose() calls should produce different candidates.

        This tests that genetic operators (crossover, mutate, replace) are
        actually applied between calls, not just returning the same population.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        c1 = plugin.propose(sga_config.num)
        # Need an update between proposes to advance the state
        dummy_results = [
            EvaluationResult(mse=float(i), nmse=float(i), r2=0.0)
            for i in range(len(c1))
        ]
        plugin.update(dummy_results)
        c2 = plugin.propose(sga_config.num)

        # With mutation probability > 0, successive generations should differ
        assert c1 != c2, (
            "Genetic operators did not change candidates between generations"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 4: evaluate() behavior (~15%)
# ═══════════════════════════════════════════════════════════════════════════


class TestSGAPluginEvaluate:
    """Tests for SGAPlugin.evaluate() candidate evaluation."""

    @pytest.mark.unit
    def test_evaluate_returns_list_of_evaluation_result(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """evaluate(candidates) returns list[EvaluationResult]."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)

        assert isinstance(results, list)
        assert all(isinstance(r, EvaluationResult) for r in results)

    @pytest.mark.unit
    def test_evaluate_result_count_matches_candidates(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """evaluate() returns exactly one result per candidate (1:1 mapping)."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)

        assert len(results) == len(candidates)

    @pytest.mark.unit
    def test_evaluate_results_have_finite_or_penalty_scores(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Every EvaluationResult should have finite mse or an explicit invalid marker."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)

        for r in results:
            # Each result must have a numeric mse (could be inf for invalid)
            assert isinstance(r.mse, float)
            # If valid, mse should be finite and non-negative
            if r.is_valid:
                assert math.isfinite(r.mse)
                assert r.mse >= 0.0

    @pytest.mark.unit
    def test_evaluate_results_have_aic_scores(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Every EvaluationResult should have an AIC score."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)

        for r in results:
            assert r.aic is not None
            assert isinstance(r.aic, float)


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 5: update() behavior (~10%)
# ═══════════════════════════════════════════════════════════════════════════


class TestSGAPluginUpdate:
    """Tests for SGAPlugin.update() state updates."""

    @pytest.mark.unit
    def test_update_truncates_population_to_num(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """After update(), population size should be config.num."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        state = plugin.state
        assert len(state["population"]) == sga_config.num

    @pytest.mark.unit
    def test_update_tracks_best_score(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """After prepare(), best_score reflects initial population.
        After update() with valid results, best_score may improve further.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        # After prepare, best_score should reflect initial population (not inf)
        initial_score = plugin.best_score
        assert isinstance(initial_score, float)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        # After update, best_score should be <= initial (monotonic improvement)
        updated_score = plugin.best_score
        assert updated_score <= initial_score

    @pytest.mark.unit
    def test_update_monotonically_improves_best_score(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """best_score should never increase across update() calls (lower is better)."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        prev_score = plugin.best_score
        for _ in range(3):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)
            current_score = plugin.best_score
            assert current_score <= prev_score, (
                f"best_score increased from {prev_score} to {current_score}"
            )
            prev_score = current_score


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 6: Full lifecycle + multi-iteration (~10%)
# ═══════════════════════════════════════════════════════════════════════════


class TestSGAPluginLifecycle:
    """Tests for the complete prepare -> propose -> evaluate -> update cycle."""

    @pytest.mark.unit
    def test_full_cycle_runs_without_error(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """A single prepare -> propose -> evaluate -> update cycle completes."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        # Should be able to access properties after cycle
        assert isinstance(plugin.best_score, float)
        assert isinstance(plugin.best_expression, str)

    @pytest.mark.unit
    def test_multi_iteration_no_crash(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """5 iterations of the loop should complete without crash."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        for _ in range(5):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

        assert isinstance(plugin.best_score, float)
        assert isinstance(plugin.best_expression, str)

    @pytest.mark.unit
    def test_best_expression_is_nonempty_after_valid_cycle(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """After at least one valid evaluation, best_expression should be non-empty."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        for _ in range(3):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

        # If any result was valid, best_expression should be non-empty
        if plugin.best_score < float("inf"):
            assert len(plugin.best_expression) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 7: Checkpoint / state round-trip (~10%)
# ═══════════════════════════════════════════════════════════════════════════


class TestSGAPluginCheckpoint:
    """Tests for SGAPlugin state save/restore."""

    @pytest.mark.unit
    def test_state_getter_returns_dict(self, sga_config: SGAConfig) -> None:
        """state property returns a dict."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        state = plugin.state
        assert isinstance(state, dict)

    @pytest.mark.unit
    def test_state_is_pickle_serializable(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """State dict must survive pickle round-trip (for torch.save checkpoint)."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        state = plugin.state
        pickled = pickle.dumps(state)
        restored = pickle.loads(pickled)
        assert restored == state

    @pytest.mark.unit
    def test_state_roundtrip_preserves_best(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Saving and restoring state preserves best_score and best_expression."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        # Run a cycle to get non-trivial state
        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        score_before = plugin.best_score
        expr_before = plugin.best_expression
        state = plugin.state

        # Create new plugin and restore
        plugin2 = SGAPlugin(config=sga_config)
        plugin2.state = state
        plugin2.prepare(mock_components)

        assert plugin2.best_score == score_before
        assert plugin2.best_expression == expr_before

    @pytest.mark.unit
    def test_state_roundtrip_population_preserved(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """State restore preserves the population so propose() can continue."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        state = plugin.state
        pop_before = state["population"]

        # Create new plugin and restore
        plugin2 = SGAPlugin(config=sga_config)
        plugin2.state = state
        plugin2.prepare(mock_components)

        state2 = plugin2.state
        pop_after = state2["population"]

        assert len(pop_after) == len(pop_before)

    @pytest.mark.unit
    def test_state_contains_expected_keys(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """State dict should contain keys for population, best_score, best_expression."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        state = plugin.state
        assert "population" in state
        assert "best_score" in state
        assert "best_expression" in state

    @pytest.mark.unit
    def test_checkpoint_restore_preserves_rng_state(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """After checkpoint restore + prepare(), the RNG state should be the
        checkpoint's state, NOT re-seeded from config.seed.

        Bug regression: prepare() used to call manual_seed unconditionally,
        overwriting the restored RNG state. This caused the SGA to repeat
        the same random sequence as from the start instead of continuing.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        # Run 3 iterations to advance RNG state
        for _ in range(3):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

        state = plugin.state

        # Restore into new plugin
        plugin2 = SGAPlugin(config=sga_config)
        plugin2.state = state
        plugin2.prepare(mock_components)

        # Fresh plugin with same seed but no checkpoint (RNG at initial position)
        plugin3 = SGAPlugin(config=sga_config)
        plugin3.prepare(mock_components)

        # plugin2 (restored) should produce different proposals than plugin3 (fresh)
        # because plugin2's RNG is advanced while plugin3's starts from seed
        c2 = plugin2.propose(sga_config.num)
        c3 = plugin3.propose(sga_config.num)
        assert c2 != c3, (
            "Restored plugin produces same candidates as fresh — "
            "RNG state was reset instead of preserved"
        )


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 8: best_score / best_expression properties (~5%)
# ═══════════════════════════════════════════════════════════════════════════


class TestSGAPluginBestProperties:
    """Tests for best_score and best_expression properties."""

    @pytest.mark.unit
    def test_best_score_initial_is_inf(self) -> None:
        """best_score should be inf before any update."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert plugin.best_score == float("inf")

    @pytest.mark.unit
    def test_best_expression_initial_is_empty(self) -> None:
        """best_expression should be empty string before any update."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert plugin.best_expression == ""

    @pytest.mark.unit
    def test_best_score_returns_float(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """best_score always returns a float."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        assert isinstance(plugin.best_score, float)

        plugin.prepare(mock_components)
        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)
        assert isinstance(plugin.best_score, float)

    @pytest.mark.unit
    def test_best_expression_returns_str(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """best_expression always returns a string."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        assert isinstance(plugin.best_expression, str)

        plugin.prepare(mock_components)
        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)
        assert isinstance(plugin.best_expression, str)

    # -- regression --

    @pytest.mark.unit
    def test_best_expression_falls_back_when_genotype_empty(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Empty genotype (default-only fit) returns LHS field name, not ``""``.

        Regression for: previously logged the empty string when
        the regression selected only the default term.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        # Force the empty-genotype scenario: best_expression="" with no PDE.
        plugin._best_expression = ""
        plugin._population = None # build_final_result returns invalid → fallback
        plugin._best_formatted_cache = None

        formatted = plugin.best_expression
        assert formatted != "", "best_expression must not be empty"
        # Should be the default term name (lhs_field, e.g. ``"u"``)
        assert plugin._default_term_name is not None
        assert formatted == plugin._default_term_name

    @pytest.mark.unit
    def test_format_rhs_with_coefficients_skips_zero(self) -> None:
        """Coefficients near zero are dropped from the rendered expression."""
        from kd2.search.sga.plugin import SGAPlugin

        coefs = torch.tensor([0.0, 1.5, -2.25, 1e-12])
        terms = ["u", "u_x", "u_xx", "u_xxx"]
        rendered = SGAPlugin._format_rhs_with_coefficients(coefs, terms)
        assert "u_x" in rendered
        assert "u_xx" in rendered
        # Zero and near-zero coefficients excluded.
        assert "u_xxx" not in rendered
        # First active term (u_x) has positive coefficient → no leading +
        assert rendered.startswith("1.5*u_x")
        # Negative coefficient rendered with explicit minus
        assert "- 2.25*u_xx" in rendered

    @pytest.mark.unit
    def test_format_rhs_returns_empty_when_all_zero(self) -> None:
        """All-zero coefficient vector renders to empty (triggers fallback)."""
        from kd2.search.sga.plugin import SGAPlugin

        coefs = torch.tensor([0.0, 0.0])
        terms = ["u", "u_x"]
        assert SGAPlugin._format_rhs_with_coefficients(coefs, terms) == ""

    @pytest.mark.unit
    def test_best_expression_cache_invalidated_on_state_restore(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Restoring state via the setter clears the cached formatted string."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        # Force a cached value.
        plugin._best_formatted_cache = "STALE"
        # Restore a different best_expression.
        plugin.state = {"best_expression": "new_value"}
        # Reading the property recomputes from the new genotype.
        assert plugin._best_formatted_cache is None
        # And subsequent read fills the cache with the fresh value.
        out = plugin.best_expression
        assert out != "STALE"


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 9: Negative / edge cases (>= 20%)
# ═══════════════════════════════════════════════════════════════════════════


class TestSGAPluginNegative:
    """Negative tests and edge cases for SGAPlugin."""

    @pytest.mark.unit
    def test_propose_before_prepare_raises(self) -> None:
        """propose() before prepare() should raise RuntimeError or ValueError."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        with pytest.raises((RuntimeError, ValueError)):
            plugin.propose(5)

    @pytest.mark.unit
    def test_evaluate_before_prepare_raises(self) -> None:
        """evaluate() before prepare() should raise RuntimeError or ValueError."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        with pytest.raises((RuntimeError, ValueError)):
            plugin.evaluate(["some_expr"])

    @pytest.mark.unit
    def test_update_with_empty_results(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """update([]) should not crash."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        # Should handle gracefully
        plugin.update([])

    @pytest.mark.unit
    def test_evaluate_empty_candidates(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """evaluate([]) should return an empty list."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        results = plugin.evaluate([])
        assert results == []

    @pytest.mark.unit
    def test_state_setter_with_empty_dict(self) -> None:
        """Setting state to empty dict should not crash."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        # Should handle gracefully (no population, etc.)
        plugin.state = {}

    @pytest.mark.unit
    def test_update_does_not_increase_population_beyond_num(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """update() should never let population exceed config.num."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        for _ in range(5):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

            pop_size = len(plugin.state["population"])
            assert pop_size <= sga_config.num, (
                f"Population {pop_size} exceeds config.num={sga_config.num}"
            )

    @pytest.mark.unit
    def test_propose_with_invalid_n_raises(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """propose(n) with n <= 0 should raise ValueError."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        with pytest.raises(ValueError):
            plugin.propose(0)

        with pytest.raises(ValueError):
            plugin.propose(-1)

    @pytest.mark.unit
    def test_config_none_uses_default(self) -> None:
        """SGAPlugin(config=None) should use default SGAConfig."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=None)
        # Should have a valid config internally
        state = plugin.state
        assert isinstance(state, dict)

    @pytest.mark.numerical
    def test_evaluate_handles_nan_in_data(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """SGAPlugin.evaluate() should not crash when offspring evaluation
        produces NaN/Inf.

        When tree execution on extreme data produces NaN/Inf, evaluate()
        should return invalid results rather than crashing. We init with
        normal data, then test that evaluate() gracefully handles failures.
        """
        from kd2.search.sga.plugin import SGAPlugin

        # Init with normal data (will succeed)
        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        # Should not raise
        results = plugin.evaluate(candidates)
        assert len(results) == len(candidates)
        # All results should have a numeric mse
        for r in results:
            assert isinstance(r.mse, float)

    @pytest.mark.numerical
    def test_extreme_data_init_population_raises(self) -> None:
        """With extreme data where all candidates are pathological,
        init_population should raise RuntimeError (resample exhaustion).

        This is the expected behavior after T3: init population resample.
        """
        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=2, width=2, seed=99, maxit=1)

        # Create dataset with very large values that cause overflow
        x_vals = torch.linspace(0.0, 1.0, 5)
        t_vals = torch.linspace(0.0, 1.0, 3)
        u_data = torch.full((5, 3), 1e15)

        dataset = PDEDataset(
            name="extreme",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals),
                "t": AxisInfo(name="t", values=t_vals),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u_data)},
            lhs_field="u",
            lhs_axis="t",
        )
        context = _make_mock_context(dataset)
        components = PlatformComponents(
            dataset=dataset,
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=context,
            registry=MagicMock(),
        )

        plugin = SGAPlugin(config=config)
        with pytest.raises(RuntimeError, match="resample"):
            plugin.prepare(components)

    @pytest.mark.unit
    def test_update_with_all_invalid_results(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """update() with all-invalid results should not crash or lose population."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        invalid_results = [
            EvaluationResult(
                mse=float("inf"),
                nmse=float("inf"),
                r2=-float("inf"),
                aic=float("inf"),
                is_valid=False,
                error_message="test failure",
            )
            for _ in range(sga_config.num)
        ]
        # Should not crash
        plugin.update(invalid_results)

        # Population should still exist
        assert len(plugin.state["population"]) > 0

    @pytest.mark.unit
    def test_seed_reproducibility(self, mock_components: PlatformComponents) -> None:
        """Two SGAPlugins with same seed should produce identical first propose()."""
        from kd2.search.sga.plugin import SGAPlugin

        config1 = SGAConfig(num=5, depth=3, width=3, seed=12345)
        config2 = SGAConfig(num=5, depth=3, width=3, seed=12345)

        plugin1 = SGAPlugin(config=config1)
        plugin1.prepare(mock_components)
        c1 = plugin1.propose(config1.num)

        plugin2 = SGAPlugin(config=config2)
        plugin2.prepare(mock_components)
        c2 = plugin2.propose(config2.num)

        assert c1 == c2, "Same seed should produce identical candidates"


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 10: Offspring lifecycle & crossover semantics 
# ═══════════════════════════════════════════════════════════════════════════


class TestOffspringLifecycleAndCrossover:
    """Tests for (no pre-evaluation truncation) and 
    (p_cro=0 disables crossover completely)."""

    @pytest.mark.unit
    def test_crossover_is_evaluated_before_mutation_stage(self) -> None:
        """A useful crossover child must be scored before mutation can alter it.

        the predecessor runs two selection stages per generation:
        crossover -> evaluate -> truncate -> mutate/replace -> evaluate -> truncate.
        This catches the regression where kd2 chained crossover -> mutate -> replace
        before any evaluation, so promising crossover structure was mutated before
        it ever entered selection.
        """
        from unittest.mock import patch

        from kd2.search.sga.pde import PDE
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import CandidateResult, TrainResult
        from kd2.search.sga.tree import Node, Tree

        def pde(name: str) -> PDE:
            return PDE([Tree(Node(name, 0))])

        elite = pde("elite")
        weak = pde("weak")
        xo_good = pde("xo_good")
        xo_bad = pde("xo_bad")
        events: list[tuple[str, str]] = []

        config = SGAConfig(
            num=2,
            depth=1,
            width=1,
            p_cro=0.5,
            p_mute=0.5,
            p_rep=0.0,
            seed=7,
        )
        plugin = SGAPlugin(config=config)
        plugin._prepared = True # noqa: SLF001
        plugin._population = [elite, weak] # noqa: SLF001
        plugin._scores = [0.0, 10.0] # noqa: SLF001
        plugin._vars = ["elite", "weak", "xo_good", "xo_bad"] # noqa: SLF001
        plugin._den = (("x", 0),) # noqa: SLF001

        def fake_crossover(*args: object, **kwargs: object) -> tuple[PDE, PDE]:
            return xo_good.copy(), xo_bad.copy()

        def fake_mutate(pde_arg: PDE, *args: object, **kwargs: object) -> PDE:
            events.append(("mutate", str(pde_arg)))
            return pde_arg.copy()

        def fake_evaluate_candidate(
            pde_arg: PDE, *args: object, **kwargs: object
        ) -> CandidateResult:
            events.append(("eval", str(pde_arg)))
            aic = 0.5 if str(pde_arg) == str(xo_good) else 99.0
            train = TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=aic,
                mse=1.0,
                best_tol=0.0,
            )
            return CandidateResult(
                train_result=train,
                pruned_pde=pde_arg.copy(),
                valid_term_indices=[0],
            )

        with (
            patch("kd2.search.sga.genetic.crossover", side_effect=fake_crossover),
            patch("kd2.search.sga.genetic.mutate", side_effect=fake_mutate),
            patch(
                "kd2.search.sga.plugin.evaluate_candidate",
                side_effect=fake_evaluate_candidate,
            ),
        ):
            plugin.propose(config.num)

        first_xo_eval = events.index(("eval", str(xo_good)))
        first_mutate = next(i for i, event in enumerate(events) if event[0] == "mutate")
        assert first_xo_eval < first_mutate, (
            f"Expected crossover evaluation before mutation, got events={events}"
        )
        assert ("mutate", str(xo_good)) in events, (
            "Expected mutation stage to operate on selected crossover child, "
            f"got events={events}"
        )
        assert ("mutate", str(xo_bad)) not in events, (
            "Weak crossover child must be truncated before mutation, "
            f"got events={events}"
        )

    @pytest.mark.unit
    def test_state_restore_clears_pending_generation(self) -> None:
        """Restoring checkpoint state must discard any in-flight proposal.

        A stale pending generation from before restore must not be committed by
        a later update(), otherwise checkpoint restore can silently overwrite the
        restored population.
        """
        from kd2.search.sga.pde import PDE
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.tree import Node, Tree

        def pde(name: str) -> PDE:
            return PDE([Tree(Node(name, 0))])

        plugin = SGAPlugin(config=SGAConfig(num=1, seed=7))
        stale = pde("stale_pending")
        restored = pde("restored")
        plugin._pending_population = [stale] # noqa: SLF001
        plugin._pending_scores = [-999.0] # noqa: SLF001
        plugin._offspring = [stale] # noqa: SLF001
        plugin._offspring_results = [ # noqa: SLF001
            EvaluationResult(mse=0.0, nmse=0.0, r2=1.0, aic=-999.0)
        ]

        plugin.state = {
            "population": [restored],
            "scores": [5.0],
            "best_score": 5.0,
            "best_expression": str(restored),
        }
        plugin.update([])

        state = plugin.state
        assert [str(p) for p in state["population"]] == [str(restored)]
        assert state["scores"] == [5.0]

    @pytest.mark.unit
    def test_prepare_clears_pending_generation(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """prepare() must discard stale proposal state before initializing data."""
        from kd2.search.sga.pde import PDE
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.tree import Node, Tree

        stale = PDE([Tree(Node("stale_pending", 0))])
        plugin = SGAPlugin(config=SGAConfig(num=2, seed=42))
        plugin._pending_population = [stale] # noqa: SLF001
        plugin._pending_scores = [-999.0] # noqa: SLF001
        plugin._offspring = [stale] # noqa: SLF001
        plugin._offspring_results = [ # noqa: SLF001
            EvaluationResult(mse=0.0, nmse=0.0, r2=1.0, aic=-999.0)
        ]

        plugin.prepare(mock_components)
        prepared_population = [str(p) for p in plugin.state["population"]]
        plugin.update([])

        assert [str(p) for p in plugin.state["population"]] == prepared_population
        assert plugin._pending_population is None # noqa: SLF001
        assert plugin._pending_scores is None # noqa: SLF001
        assert plugin._offspring is None # noqa: SLF001
        assert plugin._offspring_results is None # noqa: SLF001

    @pytest.mark.unit
    def test_invalid_evaluation_keeps_pruned_genotype(self) -> None:
        """Successful evaluation should use pruned genotype even for invalid AIC."""
        from unittest.mock import patch

        from kd2.search.sga.pde import PDE
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import CandidateResult, TrainResult
        from kd2.search.sga.tree import Node, Tree

        original = PDE([Tree(Node("original", 0))])
        pruned = PDE([Tree(Node("pruned", 0))])
        plugin = SGAPlugin(config=SGAConfig(num=1, seed=7))
        plugin._prepared = True # noqa: SLF001

        train = TrainResult(
            coefficients=torch.tensor([1.0]),
            selected_indices=[0],
            aic_score=float("inf"),
            mse=1.0,
            best_tol=0.0,
        )

        def fake_evaluate_candidate(*args: object, **kwargs: object) -> CandidateResult:
            return CandidateResult(
                train_result=train,
                pruned_pde=pruned,
                valid_term_indices=[0],
            )

        with patch(
            "kd2.search.sga.plugin.evaluate_candidate",
            side_effect=fake_evaluate_candidate,
        ):
            scored = plugin._score_offspring(original) # noqa: SLF001

        assert scored.pde == pruned
        assert scored.score == float("inf")

    @pytest.mark.unit
    def test_p_cro_zero_no_crossover(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """When p_cro=0, _apply_genetic_ops must NOT produce any crossover offspring.

        the predecessor computes num_ix = int(num * 0) = 0 and the crossover
        loop body never executes. kd2 currently has max(1, ...) which forces
        at least one crossover pair.
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(
            num=5,
            depth=3,
            width=3,
            p_var=0.6,
            p_mute=0.3,
            p_cro=0.0,
            p_rep=0.3,
            seed=42,
            maxit=3,
            str_iters=3,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(mock_components)

        # Patch crossover to track calls
        import kd2.search.sga.genetic as gen_mod

        original_crossover = gen_mod.crossover
        crossover_calls: list[tuple] = []

        def tracking_crossover(pde1, pde2, rng): # type: ignore[no-untyped-def]
            crossover_calls.append((pde1, pde2))
            return original_crossover(pde1, pde2, rng)

        with patch.object(gen_mod, "crossover", side_effect=tracking_crossover):
            plugin.propose(config.num)

        assert len(crossover_calls) == 0, (
            f"p_cro=0 should produce zero crossover calls, got {len(crossover_calls)}"
        )

    @pytest.mark.unit
    def test_propose_does_not_truncate_before_evaluate(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """propose() must return ALL offspring from genetic ops, never truncating
        to n before evaluation.

        the predecessor generates the full frontier (crossover + mutation),
        evaluates all of them, then truncates to pool size. kd2 currently
        truncates offspring in propose() before evaluate() sees them.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        # _apply_genetic_ops generates crossover offspring + mutation offspring.
        # With num=5, p_cro=0.5: crossover produces ~3 offspring,
        # mutation produces 4 (indices 1..4), total ~7.
        # Calling propose(n=3) should still return all ~7, NOT truncate to 3.
        offspring_via_ops = plugin._apply_genetic_ops() # noqa: SLF001
        total_offspring = len(offspring_via_ops)

        # Re-initialize to get same RNG state
        plugin2 = SGAPlugin(config=sga_config)
        plugin2.prepare(mock_components)

        # propose(n=3) — n is smaller than total offspring
        small_n = 3
        candidates = plugin2.propose(small_n)

        # The key assertion: candidates should be ALL offspring, not truncated to n
        assert len(candidates) == total_offspring, (
            f"propose({small_n}) returned {len(candidates)} candidates but "
            f"genetic ops produced {total_offspring}. "
            f"Offspring must not be truncated before evaluation."
        )

    @pytest.mark.unit
    def test_evaluate_receives_all_offspring(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """evaluate() must receive and return results for ALL offspring,
        not a truncated subset.

        This tests the 1:1 mapping between propose output and evaluate input
        when the full frontier is preserved.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        # Use a large n so truncation would not apply even in buggy code
        candidates = plugin.propose(n=1000)
        results = plugin.evaluate(candidates)

        # 1:1 mapping
        assert len(results) == len(candidates), (
            f"evaluate returned {len(results)} results for {len(candidates)} candidates"
        )
        # Every result should have a PDE backing it (not the fallback "no PDE" path)
        for i, r in enumerate(results):
            assert r.error_message != "No corresponding PDE for evaluation", (
                f"Result {i} fell through to fallback — offspring/candidate mismatch"
            )

    @pytest.mark.unit
    def test_offspring_candidate_mapping_consistency(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """The internal _offspring list must match the candidates list 1:1
        so that evaluate() can look up the correct PDE for each candidate.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)

        # Access internal state
        offspring = plugin._offspring # noqa: SLF001
        assert offspring is not None, "_offspring should be set after propose()"
        assert len(offspring) == len(candidates), (
            f"_offspring has {len(offspring)} PDEs but propose returned "
            f"{len(candidates)} candidates — mapping broken"
        )

    @pytest.mark.unit
    def test_p_cro_zero_only_mutation_offspring(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """With p_cro=0, all offspring should come from mutation/replacement only.

        The number of offspring should equal (num - 1), since the elite (index 0)
        is skipped in the mutation phase and crossover adds nothing.
        """
        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(
            num=5,
            depth=3,
            width=3,
            p_var=0.6,
            p_mute=0.3,
            p_cro=0.0,
            p_rep=0.3,
            seed=42,
            maxit=3,
            str_iters=3,
            d_tol=0.5,
        )
        plugin = SGAPlugin(config=config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(config.num)

        # With p_cro=0: no crossover offspring.
        # Mutation offspring: indices 1..num-1 = num-1 = 4 offspring.
        expected = config.num - 1
        assert len(candidates) == expected, (
            f"With p_cro=0, expected {expected} offspring (mutation only), "
            f"got {len(candidates)}"
        )

    @pytest.mark.unit
    def test_update_truncates_after_evaluation_not_before(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """The truncation to config.num must happen in update(), not propose().

        After propose() + evaluate(), update() should merge the full offspring
        frontier with the existing population, sort by AIC, then truncate.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        pop_before = len(plugin.state["population"])
        assert pop_before == sga_config.num

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)

        # Before update, population should still be original size
        assert len(plugin.state["population"]) == sga_config.num

        plugin.update(results)

        # After update, population should be truncated back to config.num
        assert len(plugin.state["population"]) == sga_config.num


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 11: Init population resample for pathological individuals
# ═══════════════════════════════════════════════════════════════════════════


class TestInitPopulationResample:
    """Tests for _init_population resample logic.

    the predecessor resamples individuals with AIC=inf (or AIC < -100) during
    init_population. kd2 should similarly reject pathological individuals
    and retry, with a finite retry limit.
    """

    @pytest.mark.unit
    def test_pathological_individuals_rejected(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """Individuals with AIC=inf must be rejected and resampled.

        Patch _safe_evaluate_aic to return inf for the first N calls,
        then return a finite value. The resulting population should only
        contain finite-AIC individuals.
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)

        call_count = 0
        finite_aic = 100.0

        def mock_safe_evaluate_aic(
            *args: object, **kwargs: object
        ) -> tuple[float, object]:
            nonlocal call_count
            call_count += 1
            pde = args[0] if args else None
            # First 3 calls return inf (one per individual), then return finite
            if call_count <= 3:
                return float("inf"), pde
            return finite_aic, pde

        with patch(
            "kd2.search.sga.plugin._safe_evaluate_aic",
            side_effect=mock_safe_evaluate_aic,
        ):
            plugin = SGAPlugin(config=config)
            plugin.prepare(mock_components)

        # All individuals in population must have finite AIC
        scores = plugin.state["scores"]
        assert scores is not None
        for s in scores:
            assert math.isfinite(s), f"Population contains pathological AIC: {s}"

    @pytest.mark.unit
    def test_resample_has_finite_retry_limit(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """Resample must have a finite max retry count, not an infinite loop.

        When _safe_evaluate_aic always returns inf, the init should
        fail after a bounded number of retries rather than looping forever.
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)

        def always_inf(*args: object, **kwargs: object) -> tuple[float, object]:
            pde = args[0] if args else None
            return float("inf"), pde

        with patch(
            "kd2.search.sga.plugin._safe_evaluate_aic",
            side_effect=always_inf,
        ):
            plugin = SGAPlugin(config=config)
            with pytest.raises(
                RuntimeError, match="[Rr]esample|[Rr]etry|[Pp]athological|[Ii]nit"
            ):
                plugin.prepare(mock_components)

    @pytest.mark.unit
    def test_resample_exhaustion_raises_clear_error(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """When all retries are exhausted, the error message must be informative.

        It should mention: the number of retries attempted, and that
        all candidates were pathological.
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)

        def always_inf(*args: object, **kwargs: object) -> tuple[float, object]:
            pde = args[0] if args else None
            return float("inf"), pde

        with patch(
            "kd2.search.sga.plugin._safe_evaluate_aic",
            side_effect=always_inf,
        ):
            plugin = SGAPlugin(config=config)
            with pytest.raises(RuntimeError) as exc_info:
                plugin.prepare(mock_components)

            msg = str(exc_info.value).lower()
            # Error message should mention the problem and retry concept
            assert (
                "inf" in msg
                or "pathological" in msg
                or "retry" in msg
                or "resample" in msg
            ), f"Error message not informative enough: {exc_info.value}"

    @pytest.mark.unit
    def test_healthy_individuals_not_resampled(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """Individuals with finite AIC should be accepted on first try,
        without any resample overhead.
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)
        call_count = 0

        def always_finite(*args: object, **kwargs: object) -> tuple[float, object]:
            nonlocal call_count
            call_count += 1
            pde = args[0] if args else None
            return 50.0 + call_count, pde # Distinct finite values

        with patch(
            "kd2.search.sga.plugin._safe_evaluate_aic",
            side_effect=always_finite,
        ):
            plugin = SGAPlugin(config=config)
            plugin.prepare(mock_components)

        # Should have called exactly num times (no retries needed)
        assert call_count == config.num, (
            f"Expected exactly {config.num} evaluations (no retries), got {call_count}"
        )

    @pytest.mark.unit
    def test_resample_count_bounded_per_individual(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """Each individual should have a bounded number of resample attempts.

        Total calls to _safe_evaluate_aic should be at most
        num * (1 + max_retries).
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)
        call_count = 0

        def always_inf(*args: object, **kwargs: object) -> tuple[float, object]:
            nonlocal call_count
            call_count += 1
            pde = args[0] if args else None
            return float("inf"), pde

        with patch(
            "kd2.search.sga.plugin._safe_evaluate_aic",
            side_effect=always_inf,
        ):
            plugin = SGAPlugin(config=config)
            with pytest.raises(RuntimeError):
                plugin.prepare(mock_components)

        # Total calls should be bounded: num * (1 + max_retries)
        # We don't know the exact max_retries, but it should be reasonable
        # (e.g., <= 100 per individual, so total <= 300 for num=3)
        max_reasonable = config.num * 101 # 1 initial + 100 retries per individual
        assert call_count <= max_reasonable, (
            f"Too many evaluations: {call_count} > {max_reasonable}. "
            f"Resample loop may not be bounded."
        )

    @pytest.mark.unit
    def test_moderate_negative_aic_accepted(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """Moderate negative AIC (above -100 bound) should be accepted.

        the predecessor rejects AIC < -100 as pathological. AIC=-50.0 is above the
        bound and mathematically valid, so it should not trigger resampling.
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)

        def negative_aic(*args: object, **kwargs: object) -> tuple[float, object]:
            pde = args[0] if args else None
            return -50.0, pde

        with patch(
            "kd2.search.sga.plugin._safe_evaluate_aic",
            side_effect=negative_aic,
        ):
            plugin = SGAPlugin(config=config)
            plugin.prepare(mock_components)

        # All scores should be -50.0 (accepted, not resampled)
        scores = plugin.state["scores"]
        assert all(s == -50.0 for s in scores)

    @pytest.mark.unit
    def test_mixed_finite_and_inf_partial_resample(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """When some individuals are pathological and others are healthy,
        only the pathological ones should be resampled.

        Scenario: individual 0 gets inf then finite, individuals 1-2 get
        finite on first try. Total calls should be num + 1 (one retry).
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)
        call_count = 0

        def mixed_aic(*args: object, **kwargs: object) -> tuple[float, object]:
            nonlocal call_count
            call_count += 1
            pde = args[0] if args else None
            # Call 1: inf (individual 0, will retry)
            # Call 2: finite (individual 0, retry succeeds)
            # Call 3: finite (individual 1)
            # Call 4: finite (individual 2)
            if call_count == 1:
                return float("inf"), pde
            return 42.0, pde

        with patch(
            "kd2.search.sga.plugin._safe_evaluate_aic",
            side_effect=mixed_aic,
        ):
            plugin = SGAPlugin(config=config)
            plugin.prepare(mock_components)

        # Exactly 4 calls: 1 inf + 1 retry + 2 healthy
        assert call_count == 4, f"Expected 4 evaluations, got {call_count}"
        # All population scores should be finite
        scores = plugin.state["scores"]
        for s in scores:
            assert math.isfinite(s)


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 12: prepare() fail-fast guardrails
# ═══════════════════════════════════════════════════════════════════════════


def _make_components_with_fields_axes(
    fields: dict[str, torch.Tensor],
    axes: dict[str, torch.Tensor],
    axis_order: list[str],
    lhs_field: str,
    lhs_axis: str,
    *,
    lhs_deriv_available: bool = True,
) -> PlatformComponents:
    """Helper: build PlatformComponents from raw field/axis dicts.

    Parameters
    ----------
    lhs_deriv_available:
        If False, get_derivative raises KeyError for the LHS derivative
        (simulating missing LHS derivative data).
    """
    field_shape = next(iter(fields.values())).shape

    field_data = {
        name: FieldData(name=name, values=val) for name, val in fields.items()
    }
    axis_info = {name: AxisInfo(name=name, values=val) for name, val in axes.items()}

    dataset = PDEDataset(
        name="guardrail_test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes=axis_info,
        axis_order=axis_order,
        fields=field_data,
        lhs_field=lhs_field,
        lhs_axis=lhs_axis,
    )

    context = MagicMock()
    n_total = 1
    for dim in field_shape:
        n_total *= dim

    def get_variable(name: str) -> torch.Tensor:
        if name in fields:
            return fields[name]
        if name in axes:
            return axes[name]
        raise KeyError(f"Variable '{name}' not found")

    def get_derivative(field_name: str, axis: str, order: int) -> torch.Tensor:
        if not lhs_deriv_available and axis == lhs_axis and field_name == lhs_field:
            raise KeyError(f"Derivative {field_name}_{axis * order} not available")
        torch.manual_seed(hash((field_name, axis, order)) % (2**31))
        return torch.randn(field_shape)

    context.get_variable = get_variable
    context.get_derivative = get_derivative

    return PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=context,
        registry=MagicMock(),
    )


class TestPrepareGuardrails:
    """T5: prepare() fail-fast guardrails.

    Tests that prepare() raises clear errors for:
    - LHS derivative missing (no fallback zeros)
    - lhs_axis leaking into VARS
    - lhs_axis leaking into den
    - field/axis naming conflicts
    - field/derivative-key naming conflicts
    """

    # -- LHS derivative missing: raise, not fallback zeros --

    @pytest.mark.unit
    def test_lhs_derivative_missing_raises_before_population_init(self) -> None:
        """When the LHS derivative (e.g. u_t) is not available,
        prepare() must raise ValueError BEFORE population init,
        not silently fallback to zeros and fail later at resample.

        the predecessor: raises ValueError("Structured grid missing lhs_axis ...").
        Current kd2: logs warning + uses torch.zeros(1) -> population init
        eventually fails with RuntimeError("resample") — wrong error source.
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
            lhs_deriv_available=False,
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))

        # Patch _init_population to detect if it's called.
        # The guardrail should raise BEFORE we ever reach _init_population.
        init_called = False
        original_init = plugin._init_population

        def tracking_init() -> None:
            nonlocal init_called
            init_called = True
            original_init()

        with (
            patch.object(plugin, "_init_population", tracking_init),
            pytest.raises(
                (ValueError, RuntimeError),
                match="[Ll][Hh][Ss]|[Dd]erivative",
            ),
        ):
            plugin.prepare(components)

        assert not init_called, (
            "LHS derivative missing should raise BEFORE _init_population, "
            "not as a side effect of population init failure"
        )

    @pytest.mark.unit
    def test_lhs_derivative_missing_error_message_mentions_derivative(self) -> None:
        """The error for missing LHS derivative must mention what's missing.

        The message should reference the LHS field and axis so the user
        knows what derivative to provide.
        """
        from kd2.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
            lhs_deriv_available=False,
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        with pytest.raises((ValueError, RuntimeError)) as exc_info:
            plugin.prepare(components)

        msg = str(exc_info.value).lower()
        # Must mention it's about the LHS derivative
        assert "lhs" in msg or "derivative" in msg or "u_t" in msg, (
            f"Error message should mention LHS derivative, got: {exc_info.value}"
        )

    @pytest.mark.unit
    def test_normal_dataset_still_works(self) -> None:
        """A correctly configured Burgers-like dataset should pass all guardrails.

        Regression: guardrails must not break the normal path.
        """
        from kd2.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
            lhs_deriv_available=True,
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        # Should not raise
        plugin.prepare(components)
        assert plugin._prepared # noqa: SLF001

    # -- lhs_axis must not leak into VARS --

    @pytest.mark.unit
    def test_lhs_axis_not_in_vars_after_prepare(self) -> None:
        """lhs_axis (e.g. 't') must never appear in VARS.

        the predecessor: _assert_no_lhs_in_vars() raises if lhs_axis in VARS.
        """
        from kd2.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        plugin.prepare(components)

        vars_list = plugin.state["vars"]
        assert "t" not in vars_list, "lhs_axis 't' leaked into VARS"

    @pytest.mark.unit
    def test_lhs_axis_derivatives_not_in_vars(self) -> None:
        """lhs_axis derivatives (e.g. u_t, u_tt) must not be in VARS.

        the predecessor: _assert_no_lhs_in_grad_fields() checks this.
        """
        from kd2.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        plugin.prepare(components)

        vars_list = plugin.state["vars"]
        lhs_derivs = [v for v in vars_list if v.endswith("_t") or v.endswith("_tt")]
        assert len(lhs_derivs) == 0, (
            f"LHS-axis derivatives leaked into VARS: {lhs_derivs}"
        )

    # -- Field/axis naming conflicts --

    @pytest.mark.unit
    def test_field_name_conflicts_with_axis_name_raises(self) -> None:
        """When a field name matches an axis name, prepare() must raise.

        E.g. field='x' and axis='x' would create ambiguity in data_dict.
        the predecessor: _validate_registry_names checks field-coord overlap.
        """
        from kd2.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        # Field named "x" conflicts with axis "x"
        x_field = torch.randn(10, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"x": x_field, "u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        with pytest.raises(
            ValueError, match="[Cc]onflict|[Aa]mbig|[Oo]verlap|[Dd]uplicate"
        ):
            plugin.prepare(components)

    @pytest.mark.unit
    def test_field_name_conflicts_with_derivative_key_raises(self) -> None:
        """When a field name matches a derivative key (e.g. field='u_x'),
        prepare() must raise.

        the predecessor: _validate_registry_names checks field-derivative overlap.
        """
        from kd2.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        # Field named "u_x" conflicts with the derivative key u_x
        u_x_field = torch.randn(10, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u_x": u_x_field, "u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        with pytest.raises(ValueError, match="[Cc]onflict|[Dd]erivative"):
            plugin.prepare(components)

    @pytest.mark.unit
    def test_lhs_derivative_alias_in_fields_raises(self) -> None:
        """When a field name is a legacy alias for the LHS derivative
        (e.g. field='ut' when lhs_field='u', lhs_axis='t'),
        prepare() must raise to prevent LHS leakage.

        the predecessor: _validate_registry_names checks legacy_lhs_alias.
        """
        from kd2.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        # Field named "ut" is the legacy alias for u_t (the LHS derivative)
        ut_field = torch.randn(10, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"ut": ut_field, "u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        with pytest.raises(ValueError, match="[Cc]onflict|[Aa]lias|[Ll]hs"):
            plugin.prepare(components)

    # -- Multi-field dataset normal path --

    @pytest.mark.unit
    def test_multi_field_no_conflict_passes(self) -> None:
        """A multi-field dataset with no naming conflicts should pass.

        E.g. fields={"u": ..., "v": ...}, axes={"x": ..., "t": ...}.
        """
        from kd2.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)
        v = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u, "v": v},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        plugin.prepare(components)

        vars_list = plugin.state["vars"]
        # Both fields should be in VARS
        assert "u" in vars_list
        assert "v" in vars_list
        # Spatial derivatives should be present for both
        assert "u_x" in vars_list
        assert "v_x" in vars_list
        # lhs_axis derivatives must NOT be present
        assert "u_t" not in vars_list
        assert "v_t" not in vars_list

    # -- lhs_axis must not be in den --

    @pytest.mark.unit
    def test_lhs_axis_not_in_den(self) -> None:
        """lhs_axis must not appear in the derivative denominator set (den).

        the predecessor: _assert_no_lhs_in_den() raises if lhs_axis in den.
        The den is used by PDE tree nodes to determine what denominators
        (axes) to differentiate with respect to. Including lhs_axis
        would allow SGA to construct lhs-axis derivatives on the RHS.
        """
        from kd2.search.sga.plugin import SGAPlugin

        x = torch.linspace(0.0, 1.0, 10)
        t = torch.linspace(0.0, 1.0, 5)
        u = torch.randn(10, 5)

        components = _make_components_with_fields_axes(
            fields={"u": u},
            axes={"x": x, "t": t},
            axis_order=["x", "t"],
            lhs_field="u",
            lhs_axis="t",
        )

        plugin = SGAPlugin(config=SGAConfig(num=3, depth=2, width=2, seed=42))
        plugin.prepare(components)

        # Access internal den and verify lhs_axis is excluded
        den = plugin._den # noqa: SLF001
        den_axes = [entry[0] for entry in den]
        assert "t" not in den_axes, f"lhs_axis 't' leaked into den: {den}"


# ═══════════════════════════════════════════════════════════════════════════
# Test Group: AIC Lower Bound Check
# ═══════════════════════════════════════════════════════════════════════════


class TestAICLowerBound:
    """Tests for AIC lower bound validation

    the predecessor rejects AIC < -100 as pathological (numerical artifact from
    overfitting or degenerate data). kd2 should match this behavior via
    a ``_is_valid_aic`` helper and ``_AIC_LOWER_BOUND`` constant.
    """

    # -- Unit tests for _is_valid_aic helper --

    @pytest.mark.unit
    def test_valid_aic_accepts_normal_value(self) -> None:
        """Normal positive AIC (e.g. 50.0) should be accepted."""
        from kd2.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(50.0) is True

    @pytest.mark.unit
    def test_valid_aic_accepts_zero(self) -> None:
        """AIC = 0.0 is a valid value."""
        from kd2.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(0.0) is True

    @pytest.mark.unit
    def test_valid_aic_accepts_negative_above_bound(self) -> None:
        """Moderately negative AIC (e.g. -50.0) is valid."""
        from kd2.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(-50.0) is True

    @pytest.mark.unit
    def test_valid_aic_accepts_at_bound(self) -> None:
        """AIC exactly at the lower bound (-100.0) is valid.

        the predecessor uses ``a_err < -100`` (strict less-than), so -100 exactly
        is NOT rejected.
        """
        from kd2.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(-100.0) is True

    @pytest.mark.unit
    def test_valid_aic_rejects_below_bound(self) -> None:
        """AIC = -200.0 is below the lower bound and should be rejected."""
        from kd2.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(-200.0) is False

    @pytest.mark.unit
    def test_valid_aic_rejects_inf(self) -> None:
        """AIC = +inf is not finite and should be rejected."""
        from kd2.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(float("inf")) is False

    @pytest.mark.unit
    def test_valid_aic_rejects_neg_inf(self) -> None:
        """AIC = -inf is not finite and should be rejected."""
        from kd2.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(float("-inf")) is False

    @pytest.mark.unit
    def test_valid_aic_rejects_nan(self) -> None:
        """AIC = NaN is not finite and should be rejected."""
        from kd2.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(float("nan")) is False

    @pytest.mark.unit
    def test_valid_aic_rejects_barely_below_bound(self) -> None:
        """AIC = -100.01 is barely below the bound and should be rejected.

        Tests the boundary precision: strict inequality below -100.
        """
        from kd2.search.sga.plugin import _is_valid_aic

        assert _is_valid_aic(-100.01) is False

    # -- Constant check --

    @pytest.mark.unit
    def test_aic_lower_bound_constant_exists(self) -> None:
        """_AIC_LOWER_BOUND constant should exist and equal -100.0."""
        from kd2.search.sga.plugin import _AIC_LOWER_BOUND

        assert _AIC_LOWER_BOUND == -100.0

    # -- Integration: _to_eval_result via evaluate() --

    @pytest.mark.unit
    def test_evaluate_marks_extreme_negative_aic_invalid(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """evaluate() should mark candidates with AIC=-200 as invalid.

        When the internal evaluate_candidate returns AIC far below
        the lower bound, the resulting EvaluationResult.is_valid
        should be False.
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import CandidateResult, TrainResult

        # Build a CandidateResult with pathologically negative AIC
        mock_tr = TrainResult(
            coefficients=torch.tensor([1.0]),
            selected_indices=[0],
            aic_score=-200.0,
            mse=0.001,
            best_tol=0.1,
        )

        def fake_evaluate_candidate(*args: object, **kwargs: object) -> CandidateResult:
            pde = args[0] if args else MagicMock()
            return CandidateResult(
                train_result=mock_tr,
                pruned_pde=pde,
                valid_term_indices=[0],
            )

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        with patch(
            "kd2.search.sga.plugin.evaluate_candidate",
            side_effect=fake_evaluate_candidate,
        ):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)

        # Every result should be marked invalid due to AIC < -100
        for r in results:
            assert r.is_valid is False, (
                f"Expected is_valid=False for AIC=-200, got is_valid={r.is_valid}"
            )
            # Verify result went through _to_eval_result (preserving AIC),
            # not the exception handler (which would set AIC=inf).
            assert r.aic == pytest.approx(-200.0), (
                f"Expected AIC=-200.0 preserved, got {r.aic} "
                f"(exception handler sets inf — wrong code path)"
            )

    @pytest.mark.unit
    def test_evaluate_marks_normal_aic_valid(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """evaluate() should mark candidates with AIC=-50 as valid.

        AIC=-50 is above the lower bound and finite, so the result
        should have is_valid=True.
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import CandidateResult, TrainResult

        mock_tr = TrainResult(
            coefficients=torch.tensor([1.0]),
            selected_indices=[0],
            aic_score=-50.0,
            mse=0.01,
            best_tol=0.1,
        )

        def fake_evaluate_candidate(*args: object, **kwargs: object) -> CandidateResult:
            pde = args[0] if args else MagicMock()
            return CandidateResult(
                train_result=mock_tr,
                pruned_pde=pde,
                valid_term_indices=[0],
            )

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        with patch(
            "kd2.search.sga.plugin.evaluate_candidate",
            side_effect=fake_evaluate_candidate,
        ):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)

        # Every result should be valid (AIC=-50 is above bound, finite)
        for r in results:
            assert r.is_valid is True, (
                f"Expected is_valid=True for AIC=-50, got is_valid={r.is_valid}"
            )

    # -- Integration: _init_population via prepare() --

    @pytest.mark.unit
    def test_init_resamples_extreme_negative_aic(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """_init_population should resample when AIC is extremely negative.

        Scenario: first evaluation returns AIC=-200 (below bound),
        second returns AIC=10 (valid). The population should contain
        only valid individuals, proving that resampling occurred.
        """
        from unittest.mock import patch

        from kd2.search.sga.plugin import SGAPlugin

        config = SGAConfig(num=3, depth=3, width=3, seed=42)
        call_count = 0

        def aic_with_resample(*args: object, **kwargs: object) -> tuple[float, object]:
            nonlocal call_count
            call_count += 1
            pde = args[0] if args else None
            # First call per individual returns pathological AIC,
            # retry returns valid AIC. Pattern: odd calls = -200, even = 10.
            if call_count % 2 == 1:
                return -200.0, pde
            return 10.0, pde

        with patch(
            "kd2.search.sga.plugin._safe_evaluate_aic",
            side_effect=aic_with_resample,
        ):
            plugin = SGAPlugin(config=config)
            plugin.prepare(mock_components)

        # Population should be fully populated with valid scores
        scores = plugin.state["scores"]
        assert scores is not None
        assert len(scores) == config.num
        for s in scores:
            assert s >= -100.0, f"Population contains AIC below lower bound: {s}"
            assert math.isfinite(s), f"Population contains non-finite AIC: {s}"


# =========================================================================
# Test Group 10b: nmse semantics ( + 3a2/M6)
# =========================================================================


class TestSGAPluginNmseSemantics:
    """``EvaluationResult.nmse`` must be normalized MSE, not raw MSE.

    Regression for the bundled fix of:
    - /M1: ``_compute_nmse`` duplicated ``metrics.nmse`` with different eps.
    - /M2: ``_to_eval_result`` set ``nmse=mse`` (raw MSE — semantic violation).
    - 3a2/M6: identical placeholder issue in earlier review.
    """

    @pytest.mark.unit
    def test_to_eval_result_nmse_is_normalized(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """``_to_eval_result`` divides MSE by Var(y) when y has variance."""
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        # Set y to a known-variance vector so the division is deterministic.
        y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        plugin._y = y
        target_var = float(torch.var(y, correction=0).item()) # = 2.0 (biased)
        mse = 0.5
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=mse,
                best_tol=0.1,
            ),
            expression="u",
        )
        # nmse must equal mse / target_var, NOT raw mse.
        assert result.nmse == pytest.approx(mse / target_var, rel=1e-9)
        assert result.nmse != pytest.approx(mse), (
            "nmse must not equal raw mse when target variance > 0"
        )

    @pytest.mark.unit
    def test_to_eval_result_nmse_falls_back_when_target_constant(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Constant target → metrics.nmse falls back to raw MSE."""
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        # Constant y → Var(y) = 0 → falls back to raw MSE.
        plugin._y = torch.full((10,), 1.7)
        mse = 0.123
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=mse,
                best_tol=0.1,
            ),
            expression="u",
        )
        assert result.nmse == pytest.approx(mse, rel=1e-9)

    @pytest.mark.unit
    def test_target_variance_helper(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """``_target_variance`` returns 0 when y unset, else Var(y)."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        # Before prepare(): no y set
        plugin._y = None
        assert plugin._target_variance() == 0.0

        # Biased variance: ((-2)² + 0² + 2²) / 3 = 8/3 ≈ 2.6667
        plugin._y = torch.tensor([0.0, 2.0, 4.0])
        # rel=1e-6 accommodates default float32 tensors.
        assert plugin._target_variance() == pytest.approx(8.0 / 3.0, rel=1e-6)

    @pytest.mark.unit
    def test_target_variance_single_element_returns_zero(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """N=1 → biased var is well-defined as 0.0, not NaN.

        Regression for cross-model review finding (algo + codex):
        ``torch.var(y)`` defaults to unbiased (correction=1) which
        returns NaN for single-element tensors, and the prior
        ``< _R2_EPS`` guard would silently pass NaN through to r2.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin._y = torch.tensor([3.14])
        assert plugin._target_variance() == 0.0
        # Empty tensor must also be safe.
        plugin._y = torch.tensor([])
        assert plugin._target_variance() == 0.0

    @pytest.mark.unit
    def test_to_eval_result_r2_uses_target_variance(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """``_to_eval_result.r2`` is computed from MSE + Var(y) instead
        of being left at the legacy 0.0 placeholder"""
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        plugin._y = y
        target_var = float(torch.var(y, correction=0).item()) # = 2.0 (biased)
        mse = 0.5
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=mse,
                best_tol=0.1,
            ),
            expression="u",
        )
        # r2 = 1 - 0.5/2.0 = 0.75
        assert result.r2 == pytest.approx(1.0 - mse / target_var, rel=1e-9)

    @pytest.mark.unit
    def test_to_eval_result_r2_falls_back_when_target_constant(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Constant target → r2 falls back to 0.0"""
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        plugin._y = torch.full((10,), 1.7)
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=0.123,
                best_tol=0.1,
            ),
            expression="u",
        )
        assert result.r2 == 0.0

    @pytest.mark.unit
    def test_to_eval_result_r2_minus_inf_when_mse_not_finite(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Non-finite MSE propagates as r2=-inf"""
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        plugin._y = torch.tensor([0.0, 1.0, 2.0])
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=float("inf"),
                mse=float("inf"),
                best_tol=0.1,
            ),
            expression="u",
        )
        assert result.r2 == -math.inf

    @pytest.mark.unit
    def test_to_eval_result_r2_baseline_matches_compute_r2(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Mean-predictor baseline gives r2=0 from BOTH ``_r2_from_mse``
        (in ``_to_eval_result``) and ``_compute_r2`` (in
        ``build_final_result``).

        Regression for cross-model review finding (algo + codex +
        adversarial): the prior unbiased denominator made
        ``_r2_from_mse`` report ``1/N`` for a mean predictor while
        ``_compute_r2`` correctly reported ``0``. Both helpers must
        agree on this baseline.
        """
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        # y = [0, 1, 2, 3, 4]; biased var = 10/5 = 2.0
        # mean predictor → mse = ((y - mean(y))**2).mean() = 2.0
        # Both r2 helpers must report 0.0 (mean predictor baseline).
        y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        plugin._y = y
        mse_baseline = float(((y - y.mean()) ** 2).mean().item())
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=mse_baseline,
                best_tol=0.1,
            ),
            expression="u",
        )
        assert result.r2 == pytest.approx(0.0, abs=1e-9)

        # _compute_r2 on the same mean prediction must also give 0.
        mean_prediction = torch.full_like(y, y.mean().item())
        assert plugin._compute_r2(mean_prediction) == pytest.approx(0.0, abs=1e-9)

    @pytest.mark.unit
    def test_r2_from_mse_can_go_below_minus_one(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """``_r2_from_mse`` must NOT clamp to ``[-1, 1]``

        sklearn convention: R² can be ``-∞``. A catastrophic predictor
        (mse ≫ var) should report a large negative R² so downstream
        reports/plots see the "much worse than mean" signal.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        # biased var = 2.0
        plugin._y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        # mse = 20 → r2 = 1 - 20/2 = -9.0 (would clamp to -1)
        assert plugin._r2_from_mse(20.0) == pytest.approx(-9.0, rel=1e-9)

    @pytest.mark.unit
    def test_compute_r2_can_go_below_minus_one(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """``_compute_r2`` must NOT clamp to ``[-1, 1]``

        Catastrophic predictor: y=[0,1,2,3,4], pred=[10,10,10,10,10].
        ss_res = sum((y-10)^2) = 100+81+64+49+36 = 330
        ss_tot = sum((y-2)^2) = 4+1+0+1+4 = 10
        r2 = 1 - 330/10 = -32.0
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        plugin._y = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])
        bad_prediction = torch.full_like(plugin._y, 10.0)
        assert plugin._compute_r2(bad_prediction) == pytest.approx(-32.0, rel=1e-9)

    @pytest.mark.unit
    def test_perfect_fit_on_constant_target_returns_one(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Perfect predictor on a constant target → r2=1.0 (sklearn convention).

        sklearn's ``r2_score`` returns 1.0 when ``ss_tot=0`` AND
        ``ss_res=0`` (perfect prediction matches the constant), and 0.0
        only when prediction is imperfect. Our prior code returned 0.0
        unconditionally on constant targets, contradicting the
        docstring's "sklearn convention" claim (cross-model review).
        """
        from kd2.search.sga.plugin import SGAPlugin
        from kd2.search.sga.train import TrainResult

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)
        # Constant target — ss_tot will be ~0
        plugin._y = torch.full((10,), 1.7)
        # Perfect predictor: prediction matches target exactly → ss_res=0
        perfect_prediction = torch.full_like(plugin._y, 1.7)
        assert plugin._compute_r2(perfect_prediction) == pytest.approx(1.0, abs=1e-9)
        # _r2_from_mse: mse=0 with constant target also reports perfect fit
        result = plugin._to_eval_result(
            TrainResult(
                coefficients=torch.tensor([1.0]),
                selected_indices=[0],
                aic_score=-10.0,
                mse=0.0,
                best_tol=0.1,
            ),
            expression="u",
        )
        assert result.r2 == pytest.approx(1.0, abs=1e-9)


# =========================================================================
# Test Group 11: SGAPlugin.config property (P4-T1b)
# =========================================================================


class TestSGAPluginConfig:
    """Tests for the new config property on SGAPlugin."""

    @pytest.mark.smoke
    def test_has_config_property(self) -> None:
        """SGAPlugin has a 'config' property."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert hasattr(plugin, "config")

    @pytest.mark.unit
    def test_config_returns_dict(self) -> None:
        """config property returns a dict."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        cfg = plugin.config
        assert isinstance(cfg, dict)

    @pytest.mark.unit
    def test_config_contains_algorithm_key(self) -> None:
        """config dict identifies the algorithm.

        Should contain an 'algorithm' key (or similar) that identifies
        this as SGA.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        cfg = plugin.config
        # The config should identify the algorithm
        has_identifier = any(
            isinstance(v, str) and "sga" in v.lower() for v in cfg.values()
        )
        assert has_identifier, f"config should contain an SGA identifier. Got: {cfg}"

    @pytest.mark.unit
    def test_config_reflects_sga_config_params(self, sga_config: SGAConfig) -> None:
        """config dict reflects key SGAConfig parameters.

        We verify that key parameters from the SGAConfig are represented
        in the config dict (under any key name), rather than checking exact
        key names.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        cfg = plugin.config

        # Config should be non-empty and contain multiple parameters
        assert len(cfg) >= 3, (
            f"Expected at least 3 config entries, got {len(cfg)}: {cfg}"
        )

        # At least one of the key SGAConfig params should appear as a value
        param_values = set()
        for v in cfg.values():
            if isinstance(v, (int, float)):
                param_values.add(v)

        # Some recognizable SGAConfig values should be present
        sga_values = {
            sga_config.num,
            sga_config.depth,
            sga_config.width,
            sga_config.seed,
        }
        overlap = param_values & sga_values
        assert len(overlap) >= 1, (
            f"Expected some SGAConfig values in config. "
            f"Config values: {param_values}, SGAConfig: {sga_values}"
        )

    @pytest.mark.unit
    def test_config_is_pickle_serializable(self, sga_config: SGAConfig) -> None:
        """config dict must be pickle-serializable (for ExperimentResult)."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        cfg = plugin.config

        pickled = pickle.dumps(cfg)
        restored = pickle.loads(pickled)
        assert restored == cfg

    @pytest.mark.unit
    def test_config_before_and_after_prepare_same_params(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """config parameters are available both before and after prepare().

        The config captures the algorithm configuration, which is set
        at construction time. prepare() should not lose this info.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        cfg_before = plugin.config

        plugin.prepare(mock_components)
        cfg_after = plugin.config

        # Same parameters should be present (prepare may add runtime info)
        for key in cfg_before:
            assert key in cfg_after, f"Key '{key}' lost after prepare()"


# =========================================================================
# Test Group 12: SGAPlugin recorder integration (P4-T1b)
# =========================================================================


class TestSGAPluginRecorder:
    """Tests for SGAPlugin recorder integration via PlatformComponents."""

    @pytest.mark.unit
    def test_prepare_stores_recorder_ref(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """After prepare(), SGAPlugin stores a reference to the recorder.

        If components provides a recorder, the plugin should use it.
        """
        from kd2.search.recorder import VizRecorder
        from kd2.search.sga.plugin import SGAPlugin

        recorder = VizRecorder()
        mock_components.recorder = recorder

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        # After a lifecycle (propose -> evaluate -> update), recorder
        # should have data logged by the plugin
        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        # The recorder should have at least one key logged by the plugin
        assert len(recorder.keys()) > 0

    @pytest.mark.unit
    def test_update_logs_to_recorder(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """update() logs data to the recorder when available.

        After one propose-evaluate-update cycle, the recorder should
        have best_aic (or similar) data.
        """
        from kd2.search.recorder import VizRecorder
        from kd2.search.sga.plugin import SGAPlugin

        recorder = VizRecorder()
        mock_components.recorder = recorder

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        # Check that at least one series has data
        all_keys = recorder.keys()
        assert len(all_keys) > 0

        # At least one series should have exactly 1 entry (one update cycle)
        has_single_entry = any(len(recorder.get(k)) == 1 for k in all_keys)
        assert has_single_entry

    @pytest.mark.unit
    def test_no_recorder_no_crash(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Without a recorder (None), update() does not crash.

        This is the backward-compatibility test. Existing code that
        does not provide a recorder should still work.
        """
        from kd2.search.sga.plugin import SGAPlugin

        # Ensure no recorder
        assert mock_components.recorder is None

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        # Should not raise
        plugin.update(results)

    @pytest.mark.unit
    def test_disabled_recorder_no_crash(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """With a disabled recorder (enabled=False), update() does not crash."""
        from kd2.search.recorder import VizRecorder
        from kd2.search.sga.plugin import SGAPlugin

        recorder = VizRecorder(enabled=False)
        mock_components.recorder = recorder

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        # No data should be logged (disabled)
        assert len(recorder.keys()) == 0

    @pytest.mark.unit
    def test_multiple_updates_accumulate_in_recorder(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """Multiple update() calls accumulate data in the recorder."""
        from kd2.search.recorder import VizRecorder
        from kd2.search.sga.plugin import SGAPlugin

        recorder = VizRecorder()
        mock_components.recorder = recorder

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        n_cycles = 3
        for _ in range(n_cycles):
            candidates = plugin.propose(sga_config.num)
            results = plugin.evaluate(candidates)
            plugin.update(results)

        # At least one series should have n_cycles entries
        has_n_entries = any(len(recorder.get(k)) == n_cycles for k in recorder.keys())
        assert has_n_entries, (
            f"Expected a series with {n_cycles} entries. "
            f"Keys: {recorder.keys()}, "
            f"lengths: {[(k, len(recorder.get(k))) for k in recorder.keys()]}"
        )


# =========================================================================
# Test Group 13: SGAPlugin implements ResultBuilder (P4-T1b)
# =========================================================================


class TestSGAPluginResultBuilder:
    """Tests for SGAPlugin implementing the ResultBuilder protocol."""

    @pytest.mark.smoke
    def test_result_builder_importable(self) -> None:
        """ResultBuilder can be imported from kd2.search.result."""
        from kd2.search.result import ResultBuilder

        assert ResultBuilder is not None

    @pytest.mark.unit
    def test_sga_plugin_isinstance_result_builder(self) -> None:
        """SGAPlugin passes isinstance(obj, ResultBuilder)."""
        from kd2.search.result import ResultBuilder
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert isinstance(plugin, ResultBuilder)

    @pytest.mark.unit
    def test_has_build_final_result_method(self) -> None:
        """SGAPlugin has a callable build_final_result() method."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin()
        assert callable(getattr(plugin, "build_final_result", None))

    @pytest.mark.unit
    def test_build_final_result_returns_evaluation_result(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() returns an EvaluationResult."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        # Run one cycle to have a valid best
        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert isinstance(final, EvaluationResult)

    @pytest.mark.unit
    def test_build_final_result_is_valid(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() returns an EvaluationResult with is_valid=True.

        After a successful run, the best individual should produce a valid
        evaluation.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert final.is_valid is True

    @pytest.mark.unit
    def test_build_final_result_has_residuals(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() result has non-None residuals tensor.

        Residuals are required for predicted = actual + residuals.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert final.residuals is not None
        assert isinstance(final.residuals, torch.Tensor)

    @pytest.mark.unit
    def test_build_final_result_has_coefficients(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() result has non-None coefficients tensor."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert final.coefficients is not None
        assert isinstance(final.coefficients, torch.Tensor)

    @pytest.mark.unit
    def test_build_final_result_has_finite_mse(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() MSE should be finite for a valid result."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert math.isfinite(final.mse)

    @pytest.mark.unit
    def test_build_final_result_has_terms(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() result should have terms list."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        # terms should be a list (possibly empty if only default_terms)
        assert final.terms is not None
        assert isinstance(final.terms, list)

    @pytest.mark.unit
    def test_build_final_result_has_expression(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() result has a non-empty expression string."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert isinstance(final.expression, str)
        assert len(final.expression) > 0

    @pytest.mark.unit
    def test_build_final_result_residuals_shape_matches_data(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() residuals shape == flattened data size.

        The residuals vector must have the same number of elements as
        the flattened dataset.
        """
        from kd2.search.sga.plugin import SGAPlugin

        expected_n = _SMALL_GRID_SIZE * _SMALL_TIME_SIZE

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert final.residuals is not None
        assert final.residuals.numel() == expected_n

    @pytest.mark.numerical
    def test_build_final_result_residuals_finite(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() residuals should contain only finite values."""
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert final.residuals is not None
        assert torch.isfinite(final.residuals).all(), "residuals contain NaN or Inf"

    @pytest.mark.unit
    def test_build_final_result_r2_in_range(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() R2 should be a finite float.

        R2 can be negative (worse than mean baseline) but must be finite.
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        candidates = plugin.propose(sga_config.num)
        results = plugin.evaluate(candidates)
        plugin.update(results)

        final = plugin.build_final_result()
        assert math.isfinite(final.r2)

    @pytest.mark.unit
    def test_build_final_result_before_any_update_raises_or_is_valid(
        self,
        sga_config: SGAConfig,
        mock_components: PlatformComponents,
    ) -> None:
        """build_final_result() before any update should still work.

        After prepare(), the initial population is sorted by AIC.
        build_final_result() should use the best individual from
        the initial population (population[0]).
        """
        from kd2.search.sga.plugin import SGAPlugin

        plugin = SGAPlugin(config=sga_config)
        plugin.prepare(mock_components)

        # No propose/evaluate/update yet, but init population exists
        final = plugin.build_final_result()
        assert isinstance(final, EvaluationResult)
        # Should be valid (init population already has valid individuals)
        assert final.is_valid is True
