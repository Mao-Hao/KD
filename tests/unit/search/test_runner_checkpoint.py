"""Tests for ExperimentRunner checkpoint and advanced lifecycle behavior.

Tests:
- Checkpoint save/load round-trip
- Checkpoint dict structure (version field, keys)
- load_checkpoint restores algorithm state and iteration
- save_checkpoint creates parent directories
- Integration with try/finally and callbacks

Written in TDD red phase -- all tests fail until runner.py is implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

from kd2.search.protocol import PlatformComponents
from kd2.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import (
    StatefulAlgorithm,
    StateVerifyingAlgorithm,
)

# ============================================================================
# Test Group 1: Checkpoint save structure
# ============================================================================


class TestCheckpointSave:
    """Tests for save_checkpoint dict structure."""

    @pytest.mark.unit
    def test_save_creates_file(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """save_checkpoint creates a file at the given path."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=2)
        runner.run(mock_components)

        ckpt_path = tmp_path / "checkpoint.pt"
        runner.save_checkpoint(ckpt_path)
        assert ckpt_path.exists()

    @pytest.mark.unit
    def test_save_creates_parent_dirs(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """save_checkpoint creates parent directories if they don't exist."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        nested = tmp_path / "deep" / "nested" / "dir" / "ckpt.pt"
        runner.save_checkpoint(nested)
        assert nested.exists()

    @pytest.mark.unit
    def test_save_contains_version(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Checkpoint dict must contain "version": 1."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert "version" in data
        assert data["version"] == 1

    @pytest.mark.unit
    def test_save_contains_required_keys(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Checkpoint dict must contain all required keys."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=3)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)

        required_keys = {
            "version",
            "iteration",
            "algorithm_state",
            "best_score",
            "best_expression",
        }
        assert required_keys.issubset(set(data.keys())), (
            f"Missing keys: {required_keys - set(data.keys())}"
        )

    @pytest.mark.unit
    def test_save_iteration_matches_run(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Checkpoint iteration matches the number of completed iterations."""
        max_iter = 5
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=max_iter)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert data["iteration"] == max_iter

    @pytest.mark.unit
    def test_save_algorithm_state_is_dict(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Checkpoint algorithm_state is a dict (from algorithm.state)."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert isinstance(data["algorithm_state"], dict)

    @pytest.mark.unit
    def test_save_best_score_from_algorithm(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Checkpoint best_score matches algorithm.best_score."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=3)
        runner.run(mock_components)

        expected_score = stateful_algorithm.best_score
        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert data["best_score"] == expected_score

    @pytest.mark.unit
    def test_save_best_expression_from_algorithm(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Checkpoint best_expression matches algorithm.best_expression."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=3)
        runner.run(mock_components)

        expected_expr = stateful_algorithm.best_expression
        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert data["best_expression"] == expected_expr


# ============================================================================
# Test Group 2: Checkpoint load and round-trip
# ============================================================================


class TestCheckpointLoad:
    """Tests for load_checkpoint restoring state."""

    @pytest.mark.unit
    def test_load_restores_algorithm_state(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """load_checkpoint restores algorithm.state from file."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=3)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        # Get the saved state for comparison
        saved_state = torch.load(ckpt_path, weights_only=False)["algorithm_state"]

        # Create fresh algorithm and runner, then load
        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo, max_iterations=10)
        fresh_runner.load_checkpoint(ckpt_path)

        # Algorithm state should match what was saved
        assert fresh_algo.state == saved_state

    @pytest.mark.unit
    def test_load_sets_algorithm_state_setter(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """load_checkpoint calls algorithm.state = ... (not just reads)."""
        algo = StateVerifyingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        # Load into fresh instance
        fresh_algo = StateVerifyingAlgorithm()
        assert fresh_algo.state.get("restored") is False

        fresh_runner = ExperimentRunner(algorithm=fresh_algo, max_iterations=1)
        fresh_runner.load_checkpoint(ckpt_path)

        # The setter adds "restored": True
        assert fresh_algo.state.get("restored") is True

    @pytest.mark.unit
    def test_round_trip_preserves_generation(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Save then load preserves the generation counter in state."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=4)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        gen_before = stateful_algorithm.state["generation"]

        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo, max_iterations=1)
        fresh_runner.load_checkpoint(ckpt_path)

        assert fresh_algo.state["generation"] == gen_before

    @pytest.mark.unit
    def test_round_trip_preserves_population(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Save then load preserves the population list in state."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=2)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        pop_before = stateful_algorithm.state["population"]

        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo, max_iterations=1)
        fresh_runner.load_checkpoint(ckpt_path)

        assert fresh_algo.state["population"] == pop_before

    @pytest.mark.unit
    def test_load_nonexistent_file_raises(
        self,
        stateful_algorithm: StatefulAlgorithm,
        tmp_path: Path,
    ) -> None:
        """load_checkpoint with non-existent path raises an error."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        bogus_path = tmp_path / "does_not_exist.pt"

        with pytest.raises((FileNotFoundError, OSError)):
            runner.load_checkpoint(bogus_path)


# ============================================================================
# Test Group 3: Checkpoint + iteration resumption
# ============================================================================


class TestCheckpointResumption:
    """Tests that checkpoint iteration is preserved for resumption."""

    @pytest.mark.unit
    def test_checkpoint_iteration_stored(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Checkpoint file stores the iteration count."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=7)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        data = torch.load(ckpt_path, weights_only=False)
        assert data["iteration"] == 7

    @pytest.mark.unit
    def test_load_restores_internal_iteration(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """After load, the runner's internal iteration counter is restored.

        Verified indirectly: saving again after load should produce
        same iteration count (without running).
        """
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=5)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        # Load into fresh runner
        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo, max_iterations=100)
        fresh_runner.load_checkpoint(ckpt_path)

        # Save again from fresh runner
        ckpt2_path = tmp_path / "ckpt2.pt"
        fresh_runner.save_checkpoint(ckpt2_path)

        data2 = torch.load(ckpt2_path, weights_only=False)
        assert data2["iteration"] == 5


# ============================================================================
# Test Group 4: Checkpoint edge cases
# ============================================================================


class TestCheckpointEdgeCases:
    """Edge cases for checkpoint behavior."""

    @pytest.mark.unit
    def test_save_before_run(
        self,
        stateful_algorithm: StatefulAlgorithm,
        tmp_path: Path,
    ) -> None:
        """save_checkpoint before run() should save iteration=0 state."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=10)
        # Don't run, just save
        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        data = torch.load(ckpt_path, weights_only=False)
        assert data["version"] == 1
        # Iteration should be 0 (nothing ran)
        assert data["iteration"] == 0

    @pytest.mark.unit
    def test_save_after_early_stop(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """save_checkpoint after early stop saves actual iteration count."""

        class _StopCallback:
            def __init__(self) -> None:
                self._should_stop = False

            @property
            def should_stop(self) -> bool:
                return self._should_stop

            def on_experiment_start(self, algorithm: Any) -> None:
                pass

            def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
                pass

            def on_iteration_end(
                self,
                iteration: int,
                algorithm: Any,
                candidates: list[str],
                results: list[Any],
            ) -> None:
                if iteration >= 2:
                    self._should_stop = True

            def on_experiment_end(self, algorithm: Any) -> None:
                pass

        algo = StatefulAlgorithm()
        cb = _StopCallback()
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=100,
            callbacks=[cb],
        )
        result = runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        data = torch.load(ckpt_path, weights_only=False)
        # Should be 3 iterations (0, 1, 2 completed before stop)
        assert data["iteration"] == result.iterations

    @pytest.mark.unit
    def test_overwrite_existing_checkpoint(
        self,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """save_checkpoint overwrites an existing file at same path.

        H1 fix: Uses two independent algorithm instances to guarantee
        different iteration counts (2 vs 5), then asserts the second
        save's iteration differs from the first.
        """
        ckpt_path = tmp_path / "ckpt.pt"

        # First run: 2 iterations with a fresh algorithm
        algo1 = StatefulAlgorithm()
        runner1 = ExperimentRunner(algorithm=algo1, max_iterations=2)
        runner1.run(mock_components)
        runner1.save_checkpoint(ckpt_path)
        data1 = torch.load(ckpt_path, weights_only=False)

        # Second run: 5 iterations with a fresh algorithm
        algo2 = StatefulAlgorithm()
        runner2 = ExperimentRunner(algorithm=algo2, max_iterations=5)
        runner2.run(mock_components)
        runner2.save_checkpoint(ckpt_path)
        data2 = torch.load(ckpt_path, weights_only=False)

        # The overwrite must have happened: iteration counts differ
        assert data1["iteration"] == 2
        assert data2["iteration"] == 5
        assert data2["iteration"] != data1["iteration"]

    @pytest.mark.unit
    def test_checkpoint_uses_torch_save(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Checkpoint is loadable via torch.load (format verification)."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)

        # torch.load should not raise
        data = torch.load(ckpt_path, weights_only=False)
        assert isinstance(data, dict)


# ============================================================================
# Test Group 5: Checkpoint version field
# ============================================================================


class TestCheckpointVersion:
    """Tests specifically for the version field in checkpoints."""

    @pytest.mark.unit
    def test_version_is_integer(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Version field is an integer."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert isinstance(data["version"], int)

    @pytest.mark.unit
    def test_version_is_one(
        self,
        stateful_algorithm: StatefulAlgorithm,
        mock_components: PlatformComponents,
        tmp_path: Path,
    ) -> None:
        """Initial checkpoint version is 1."""
        runner = ExperimentRunner(algorithm=stateful_algorithm, max_iterations=1)
        runner.run(mock_components)

        ckpt_path = tmp_path / "ckpt.pt"
        runner.save_checkpoint(ckpt_path)
        data = torch.load(ckpt_path, weights_only=False)
        assert data["version"] == 1
