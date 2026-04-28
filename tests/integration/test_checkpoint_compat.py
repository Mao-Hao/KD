"""Integration tests: CheckpointCallback <-> ExperimentRunner.load_checkpoint.

Verifies that checkpoint files produced by CheckpointCallback are
loadable by ExperimentRunner.load_checkpoint and contain the complete
schema expected by the runner.

The authoritative checkpoint schema (from Runner.save_checkpoint):
    {
        "version": 1,
        "iteration": int,
        "algorithm_state": dict,
        "best_score": float,
        "best_expression": str,
    }

These tests should FAIL (RED) until CheckpointCallback is fixed to
emit the complete schema.

RED tests added for reviewer feedback:
- High #1: final checkpoint iteration=-1 when every_n > max_iterations
- High #2: round-trip of best_score/best_expression through checkpoint
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch

from kd2.search.callbacks import CheckpointCallback
from kd2.search.protocol import PlatformComponents
from kd2.search.runner import ExperimentRunner
from tests.unit.search._runner_mocks import (
    RecordingAlgorithm,
    StatefulAlgorithm,
)

# Constants

AUTHORITATIVE_KEYS = frozenset(
    {
        "version",
        "iteration",
        "algorithm_state",
        "best_score",
        "best_expression",
    }
)


# Fixtures


@pytest.fixture
def mock_components() -> PlatformComponents:
    """PlatformComponents with MagicMock fields.

    ``evaluator.lhs_target`` is set to an empty Tensor so the Runner's
    strict ``_actual`` contract (round-2 fix) is satisfied without
    relying on a silent zero-fallback that would mask plugin bugs.
    """
    components = PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
    )
    components.evaluator.lhs_target = torch.zeros(0)
    return components


# Helpers


def _run_experiment_with_checkpoint(
    algorithm: RecordingAlgorithm | StatefulAlgorithm,
    components: PlatformComponents,
    checkpoint_dir: Path,
    max_iterations: int = 5,
    every_n: int = 2,
    batch_size: int = 3,
) -> ExperimentRunner:
    """Run an experiment with a CheckpointCallback attached.

    Returns the runner so callers can use load_checkpoint on it.
    """
    cb = CheckpointCallback(directory=checkpoint_dir, every_n=every_n)
    runner = ExperimentRunner(
        algorithm=algorithm,
        max_iterations=max_iterations,
        batch_size=batch_size,
        callbacks=[cb],
    )
    runner.run(components)
    return runner


def _load_checkpoint_data(path: Path) -> dict:
    """Load raw checkpoint dict from disk."""
    return torch.load(path, weights_only=False)


# Schema completeness tests


@pytest.mark.integration
class TestCheckpointSchemaCompleteness:
    """Verify that CheckpointCallback writes the full authoritative schema."""

    def test_iteration_checkpoint_has_all_keys(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """on_iteration_end checkpoint must contain all 5 schema keys."""
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=1,
        )

        # Iteration 0 checkpoint should exist
        ckpt_path = tmp_path / "checkpoint_000000.pt"
        assert ckpt_path.exists(), "Expected iteration checkpoint at iteration 0"

        data = _load_checkpoint_data(ckpt_path)
        missing = AUTHORITATIVE_KEYS - set(data.keys())
        assert not missing, (
            f"Iteration checkpoint missing keys: {missing}. "
            f"Got keys: {set(data.keys())}"
        )

    def test_final_checkpoint_has_all_keys(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """on_experiment_end checkpoint must contain all 5 schema keys."""
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=10, # no iteration checkpoints
        )

        ckpt_path = tmp_path / "checkpoint_final.pt"
        assert ckpt_path.exists(), "Expected final checkpoint"

        data = _load_checkpoint_data(ckpt_path)
        missing = AUTHORITATIVE_KEYS - set(data.keys())
        assert not missing, (
            f"Final checkpoint missing keys: {missing}. Got keys: {set(data.keys())}"
        )

    def test_version_field_value(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """Checkpoint version must be an integer >= 1."""
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=2,
            every_n=1,
        )

        data = _load_checkpoint_data(tmp_path / "checkpoint_000000.pt")
        assert "version" in data, "Checkpoint must contain 'version' key"
        assert isinstance(data["version"], int), "version must be int"
        assert data["version"] >= 1, "version must be >= 1"


# Load compatibility tests


@pytest.mark.integration
class TestCheckpointLoadCompat:
    """Verify Runner.load_checkpoint succeeds on CheckpointCallback output."""

    def test_load_iteration_checkpoint_no_error(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """Runner.load_checkpoint must not raise on iteration checkpoint."""
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=1,
        )

        # Create a fresh runner and try loading iteration 0 checkpoint
        fresh_algo = RecordingAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo)
        ckpt_path = tmp_path / "checkpoint_000000.pt"
        # This should not raise KeyError
        fresh_runner.load_checkpoint(ckpt_path)

    def test_load_final_checkpoint_no_error(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """Runner.load_checkpoint must not raise on final checkpoint."""
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=10,
        )

        fresh_algo = RecordingAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo)
        ckpt_path = tmp_path / "checkpoint_final.pt"
        fresh_runner.load_checkpoint(ckpt_path)


# Field value correctness tests


@pytest.mark.integration
class TestCheckpointFieldValues:
    """Verify checkpoint field values reflect algorithm state at save time."""

    def test_iteration_field_matches_save_time(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """Iteration field in checkpoint must match the iteration at save time."""
        algo = RecordingAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=5,
            every_n=2,
        )

        # every_n=2 -> saves at iteration 0, 2, 4
        for expected_iter in [0, 2, 4]:
            ckpt = tmp_path / f"checkpoint_{expected_iter:06d}.pt"
            assert ckpt.exists(), f"Expected checkpoint at iteration {expected_iter}"
            data = _load_checkpoint_data(ckpt)
            assert data["iteration"] == expected_iter, (
                f"Expected iteration={expected_iter}, got {data['iteration']}"
            )

    def test_algorithm_state_is_dict(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """algorithm_state must be a dict (serialized algorithm state)."""
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=1,
        )

        data = _load_checkpoint_data(tmp_path / "checkpoint_000000.pt")
        assert isinstance(data["algorithm_state"], dict)

    def test_best_score_is_numeric(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """best_score field must be a numeric value at iteration checkpoint."""
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=1,
        )

        # After iteration 0, StatefulAlgorithm updates best_score
        data = _load_checkpoint_data(tmp_path / "checkpoint_000000.pt")
        assert "best_score" in data, "Checkpoint must contain 'best_score'"
        score = data["best_score"]
        assert isinstance(score, (int, float)), (
            f"best_score must be numeric, got {type(score)}"
        )

    def test_best_expression_is_str(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """best_expression field must be a string."""
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=3,
            every_n=1,
        )

        data = _load_checkpoint_data(tmp_path / "checkpoint_000000.pt")
        assert "best_expression" in data, "Checkpoint must contain 'best_expression'"
        assert isinstance(data["best_expression"], str)


# Round-trip state restoration tests


@pytest.mark.integration
class TestCheckpointRoundTrip:
    """Verify save-via-callback -> load -> algorithm state fully restored."""

    def test_stateful_algorithm_round_trip(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """StatefulAlgorithm state must be fully restorable from checkpoint.

        The generation counter and population list should survive
        a save-load cycle through CheckpointCallback.
        """
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=5,
            every_n=1,
            batch_size=3,
        )

        # Capture the state after iteration 2 (checkpoint_000002.pt)
        ckpt_path = tmp_path / "checkpoint_000002.pt"
        assert ckpt_path.exists()
        data = _load_checkpoint_data(ckpt_path)

        # Load into a fresh algorithm
        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo)
        fresh_runner.load_checkpoint(ckpt_path)

        # Verify algorithm state was restored
        assert fresh_algo.state["generation"] == data["algorithm_state"]["generation"]
        assert fresh_algo.state["population"] == data["algorithm_state"]["population"]

    def test_checkpoint_preserves_best_score_and_expression(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """Checkpoint must preserve best_score and best_expression values.

        After a save-load cycle, the checkpoint file must contain
        best_score and best_expression that match the algorithm's state
        at the time of saving. This is distinct from algorithm_state
        round-trip: it verifies the top-level checkpoint schema fields.

        RED: CheckpointCallback currently does not write best_score or
        best_expression to checkpoint files, so these keys will be missing.
        """
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=5,
            every_n=2,
            batch_size=3,
        )

        # Check iteration checkpoint at iteration 2
        ckpt_path = tmp_path / "checkpoint_000002.pt"
        assert ckpt_path.exists()
        data = _load_checkpoint_data(ckpt_path)

        # The checkpoint must contain these top-level fields with correct values.
        # We verify against the algorithm's own state dict (which tracks the same
        # values internally) rather than hardcoding expected numbers.
        algo_internal = data["algorithm_state"]
        assert "best_score" in data, "Checkpoint missing 'best_score' top-level field"
        assert "best_expression" in data, (
            "Checkpoint missing 'best_expression' top-level field"
        )
        # The top-level best_score/expression must match what was in algorithm
        # state at save time (StatefulAlgorithm stores these in its state dict).
        assert data["best_score"] == algo_internal["best_score"], (
            f"best_score mismatch: checkpoint top-level has {data['best_score']}, "
            f"but algorithm_state has {algo_internal['best_score']}"
        )
        assert data["best_expression"] == algo_internal["best_expression"], (
            f"best_expression mismatch: checkpoint top-level has "
            f"{data['best_expression']!r}, but algorithm_state has "
            f"{algo_internal['best_expression']!r}"
        )

    def test_final_checkpoint_round_trip(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """Final checkpoint must restore algorithm state correctly."""
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=4,
            every_n=10,
            batch_size=2,
        )

        # Capture original algorithm state at experiment end
        original_state = algo.state.copy()

        ckpt_path = tmp_path / "checkpoint_final.pt"
        assert ckpt_path.exists()

        fresh_algo = StatefulAlgorithm()
        fresh_runner = ExperimentRunner(algorithm=fresh_algo)
        fresh_runner.load_checkpoint(ckpt_path)

        # The restored state should match what the original algorithm had
        assert fresh_algo.state["generation"] == original_state["generation"]
        assert fresh_algo.state["population"] == original_state["population"]


# Boundary condition tests


@pytest.mark.integration
class TestCheckpointBoundary:
    """Boundary conditions and edge cases."""

    def test_first_iteration_checkpoint(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """Checkpoint at iteration 0 must have iteration=0 and valid state."""
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=1,
            every_n=1,
        )

        ckpt_path = tmp_path / "checkpoint_000000.pt"
        assert ckpt_path.exists()

        data = _load_checkpoint_data(ckpt_path)
        assert data["iteration"] == 0
        assert isinstance(data["algorithm_state"], dict)

    def test_final_checkpoint_iteration_not_negative_when_zero_iterations(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """Final checkpoint must not have iteration=-1 when no iterations run.

        When max_iterations=0 the search loop body never executes, so
        on_iteration_end is never called and CheckpointCallback._last_iteration
        stays at its initial sentinel (-1). The final checkpoint should record
        iteration=0 (or another non-negative value), not leak the sentinel.

        RED: CheckpointCallback currently stores _last_iteration=-1 in this case.
        """
        algo = StatefulAlgorithm()
        _run_experiment_with_checkpoint(
            algo,
            mock_components,
            tmp_path,
            max_iterations=0,
            every_n=1,
        )

        ckpt_path = tmp_path / "checkpoint_final.pt"
        assert ckpt_path.exists(), "Expected final checkpoint"

        data = _load_checkpoint_data(ckpt_path)
        assert data["iteration"] >= 0, (
            f"Final checkpoint iteration must be >= 0, got {data['iteration']}. "
            "CheckpointCallback leaked its internal sentinel (-1) because "
            "no iterations ran and on_iteration_end was never called."
        )

    def test_callback_and_runner_save_same_keys(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """CheckpointCallback and Runner.save_checkpoint must produce
        checkpoints with identical key sets.

        This is the core schema alignment contract.
        """
        algo = StatefulAlgorithm()
        cb = CheckpointCallback(directory=tmp_path / "cb_dir", every_n=1)
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=3,
            batch_size=2,
            callbacks=[cb],
        )
        runner.run(mock_components)

        # Runner saves its own checkpoint
        runner_ckpt = tmp_path / "runner_ckpt.pt"
        runner.save_checkpoint(runner_ckpt)

        # Load both checkpoints
        runner_data = _load_checkpoint_data(runner_ckpt)
        cb_data = _load_checkpoint_data(tmp_path / "cb_dir" / "checkpoint_final.pt")

        runner_keys = set(runner_data.keys())
        cb_keys = set(cb_data.keys())

        assert runner_keys == cb_keys, (
            f"Key mismatch! Runner has {runner_keys - cb_keys} extra, "
            f"Callback has {cb_keys - runner_keys} extra. "
            f"Runner keys: {runner_keys}, Callback keys: {cb_keys}"
        )

    def test_callback_iteration_ckpt_and_runner_save_same_keys(
        self,
        tmp_path: Path,
        mock_components: PlatformComponents,
    ) -> None:
        """Iteration checkpoints (not just final) must also match runner schema."""
        algo = RecordingAlgorithm()
        cb = CheckpointCallback(directory=tmp_path / "cb_dir", every_n=1)
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=2,
            batch_size=2,
            callbacks=[cb],
        )
        runner.run(mock_components)

        runner_ckpt = tmp_path / "runner_ckpt.pt"
        runner.save_checkpoint(runner_ckpt)

        runner_keys = set(_load_checkpoint_data(runner_ckpt).keys())
        iter_keys = set(
            _load_checkpoint_data(tmp_path / "cb_dir" / "checkpoint_000000.pt").keys()
        )

        assert runner_keys == iter_keys, (
            f"Iteration checkpoint keys {iter_keys} differ from "
            f"runner checkpoint keys {runner_keys}"
        )
