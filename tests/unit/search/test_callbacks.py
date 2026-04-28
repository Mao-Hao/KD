"""Contract tests for RunnerCallback protocol and implementations.

Tests the interface contract for:
- RunnerCallback Protocol
- LoggingCallback
- EarlyStoppingCallback
- CheckpointCallback

Written in TDD red phase -- these define expected API before implementation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from kd2.core.evaluator import EvaluationResult

# -- Imports under test (will fail until callbacks.py is implemented) --------
from kd2.search.callbacks import (
    CheckpointCallback,
    EarlyStoppingCallback,
    LoggingCallback,
    RunnerCallback,
)

# ============================================================================
# Helpers: Mock algorithm for callback testing
# ============================================================================


class _MockAlgorithm:
    """A minimal mock satisfying SearchAlgorithm for callback tests.

    Provides settable best_score, best_expression, and state.
    """

    def __init__(
        self,
        best_score: float = float("inf"),
        best_expression: str = "",
    ) -> None:
        self._best_score = best_score
        self._best_expression = best_expression
        self._state: dict[str, Any] = {}

    @property
    def best_score(self) -> float:
        return self._best_score

    @best_score.setter
    def best_score(self, value: float) -> None:
        self._best_score = value

    @property
    def best_expression(self) -> str:
        return self._best_expression

    @best_expression.setter
    def best_expression(self, value: str) -> None:
        self._best_expression = value

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = value


def _make_eval_result(mse: float = 0.1) -> EvaluationResult:
    """Create a minimal valid EvaluationResult for testing."""
    return EvaluationResult(mse=mse, nmse=mse, r2=1.0 - mse)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def mock_algo() -> _MockAlgorithm:
    """Return a mock algorithm with default values."""
    return _MockAlgorithm(best_score=1.0, best_expression="u_x")


@pytest.fixture
def sample_candidates() -> list[str]:
    """Return sample candidate expressions."""
    return ["u_x", "mul(u, u_x)", "u_xx"]


@pytest.fixture
def sample_results() -> list[EvaluationResult]:
    """Return sample evaluation results matching sample_candidates."""
    return [
        _make_eval_result(0.1),
        _make_eval_result(0.05),
        _make_eval_result(0.2),
    ]


# ============================================================================
# Test Group 1: Protocol Conformance
# ============================================================================


class TestRunnerCallbackProtocol:
    """Tests that callback classes satisfy the RunnerCallback protocol."""

    @pytest.mark.smoke
    def test_protocol_is_importable(self) -> None:
        """RunnerCallback can be imported from kd2.search.callbacks."""
        assert RunnerCallback is not None

    @pytest.mark.smoke
    def test_protocol_is_runtime_checkable(self) -> None:
        """RunnerCallback supports isinstance() checks."""
        # If not runtime_checkable, isinstance raises TypeError
        obj = LoggingCallback()
        result = isinstance(obj, RunnerCallback)
        assert isinstance(result, bool)

    def test_logging_callback_is_runner_callback(self) -> None:
        """LoggingCallback satisfies RunnerCallback protocol."""
        cb = LoggingCallback()
        assert isinstance(cb, RunnerCallback)

    def test_early_stopping_is_runner_callback(self) -> None:
        """EarlyStoppingCallback satisfies RunnerCallback protocol."""
        cb = EarlyStoppingCallback()
        assert isinstance(cb, RunnerCallback)

    def test_checkpoint_is_runner_callback(self, tmp_path: Path) -> None:
        """CheckpointCallback satisfies RunnerCallback protocol."""
        cb = CheckpointCallback(directory=tmp_path)
        assert isinstance(cb, RunnerCallback)

    def test_protocol_defines_on_experiment_start(self) -> None:
        """Protocol has on_experiment_start method."""
        assert hasattr(RunnerCallback, "on_experiment_start")

    def test_protocol_defines_on_iteration_start(self) -> None:
        """Protocol has on_iteration_start method."""
        assert hasattr(RunnerCallback, "on_iteration_start")

    def test_protocol_defines_on_iteration_end(self) -> None:
        """Protocol has on_iteration_end method."""
        assert hasattr(RunnerCallback, "on_iteration_end")

    def test_protocol_defines_on_experiment_end(self) -> None:
        """Protocol has on_experiment_end method."""
        assert hasattr(RunnerCallback, "on_experiment_end")

    def test_protocol_defines_should_stop(self) -> None:
        """Protocol has should_stop property."""
        assert hasattr(RunnerCallback, "should_stop")


# ============================================================================
# Test Group 2: LoggingCallback
# ============================================================================


class TestLoggingCallback:
    """Tests for LoggingCallback contract."""

    def test_default_every_n(self) -> None:
        """Default every_n is 1."""
        cb = LoggingCallback()
        # Should not raise -- default is valid
        assert isinstance(cb, RunnerCallback)

    def test_every_n_custom(self) -> None:
        """Custom every_n is accepted."""
        cb = LoggingCallback(every_n=5)
        assert isinstance(cb, RunnerCallback)

    def test_every_n_less_than_one_raises(self) -> None:
        """every_n < 1 raises ValueError."""
        with pytest.raises(ValueError):
            LoggingCallback(every_n=0)

    def test_every_n_negative_raises(self) -> None:
        """Negative every_n raises ValueError."""
        with pytest.raises(ValueError):
            LoggingCallback(every_n=-1)

    def test_should_stop_always_false(self, mock_algo: _MockAlgorithm) -> None:
        """should_stop is always False for LoggingCallback."""
        cb = LoggingCallback()
        assert cb.should_stop is False

        # Still False after experiment lifecycle
        cb.on_experiment_start(mock_algo)
        assert cb.should_stop is False

        cb.on_iteration_end(0, mock_algo, ["u_x"], [_make_eval_result()])
        assert cb.should_stop is False

        cb.on_experiment_end(mock_algo)
        assert cb.should_stop is False

    def test_logs_at_iteration_zero(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Logs at iteration 0 (first iteration always logged)."""
        cb = LoggingCallback(every_n=5)
        cb.on_experiment_start(mock_algo)

        with caplog.at_level(logging.DEBUG):
            cb.on_iteration_end(0, mock_algo, ["u_x"], [_make_eval_result()])

        # Should have logged something
        assert len(caplog.records) > 0

    def test_logs_at_correct_intervals(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Logs at iterations 0, N, 2N for every_n=N."""
        every_n = 3
        cb = LoggingCallback(every_n=every_n)
        cb.on_experiment_start(mock_algo)

        logged_iterations: list[int] = []

        for i in range(10):
            with caplog.at_level(logging.DEBUG):
                caplog.clear()
                cb.on_iteration_end(i, mock_algo, ["u_x"], [_make_eval_result()])

                if len(caplog.records) > 0:
                    logged_iterations.append(i)

        # Should log at 0, 3, 6, 9
        assert logged_iterations == [0, 3, 6, 9]

    def test_does_not_log_at_non_matching_iterations(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Does NOT log at iterations that are not multiples of every_n."""
        cb = LoggingCallback(every_n=5)
        cb.on_experiment_start(mock_algo)

        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            cb.on_iteration_end(1, mock_algo, ["u_x"], [_make_eval_result()])

        assert len(caplog.records) == 0

    def test_logs_best_score_and_expression(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Log output includes best_score and best_expression."""
        algo = _MockAlgorithm(best_score=0.42, best_expression="u_xx")
        cb = LoggingCallback(every_n=1)
        cb.on_experiment_start(algo)

        with caplog.at_level(logging.DEBUG):
            cb.on_iteration_end(0, algo, ["u_xx"], [_make_eval_result()])

        log_text = " ".join(r.message for r in caplog.records)
        # Should mention both the score and expression
        assert "0.42" in log_text
        assert "u_xx" in log_text

    def test_logs_on_experiment_start(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Logs a message at experiment start."""
        cb = LoggingCallback()
        with caplog.at_level(logging.DEBUG):
            cb.on_experiment_start(mock_algo)

        assert len(caplog.records) > 0

    def test_logs_on_experiment_end(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """Logs a message at experiment end."""
        cb = LoggingCallback()
        cb.on_experiment_start(mock_algo)

        with caplog.at_level(logging.DEBUG):
            caplog.clear()
            cb.on_experiment_end(mock_algo)

        assert len(caplog.records) > 0

    def test_every_n_one_logs_every_iteration(
        self,
        mock_algo: _MockAlgorithm,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        """every_n=1 logs at every iteration."""
        cb = LoggingCallback(every_n=1)
        cb.on_experiment_start(mock_algo)

        logged_count = 0
        for i in range(5):
            with caplog.at_level(logging.DEBUG):
                caplog.clear()
                cb.on_iteration_end(i, mock_algo, ["u_x"], [_make_eval_result()])
                if len(caplog.records) > 0:
                    logged_count += 1

        assert logged_count == 5


# ============================================================================
# Test Group 3: EarlyStoppingCallback
# ============================================================================


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback contract."""

    def test_default_parameters(self) -> None:
        """Default parameters: patience=10, min_delta=1e-6, mode='min'."""
        cb = EarlyStoppingCallback()
        assert isinstance(cb, RunnerCallback)

    def test_negative_patience_raises(self) -> None:
        """Negative patience raises ValueError."""
        with pytest.raises(ValueError, match="patience must be >= 0"):
            EarlyStoppingCallback(patience=-1)

    def test_negative_min_delta_raises(self) -> None:
        """Negative min_delta raises ValueError."""
        with pytest.raises(ValueError, match="min_delta must be >= 0"):
            EarlyStoppingCallback(min_delta=-0.1)

    def test_invalid_mode_raises(self) -> None:
        """Invalid mode string raises ValueError."""
        with pytest.raises(ValueError, match="mode must be"):
            EarlyStoppingCallback(mode="minimize") # type: ignore[arg-type]

    def test_should_stop_starts_false(self) -> None:
        """should_stop is False initially."""
        cb = EarlyStoppingCallback(patience=5)
        assert cb.should_stop is False

    def test_should_stop_false_after_experiment_start(
        self, mock_algo: _MockAlgorithm
    ) -> None:
        """should_stop is False after on_experiment_start."""
        cb = EarlyStoppingCallback(patience=5)
        cb.on_experiment_start(mock_algo)
        assert cb.should_stop is False

    def test_stops_after_patience_stale_iterations_min_mode(self) -> None:
        """After patience stale iterations, should_stop becomes True (min mode)."""
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="min")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)

        # First iteration: always counts as improvement (baseline)
        cb.on_iteration_end(0, algo, [], [])

        # Next patience iterations with no improvement
        for i in range(1, patience + 1):
            assert cb.should_stop is False, f"Stopped too early at iteration {i}"
            cb.on_iteration_end(i, algo, [], [])

        assert cb.should_stop is True

    def test_stops_after_patience_stale_iterations_max_mode(self) -> None:
        """After patience stale iterations, should_stop becomes True (max mode)."""
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="max")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)

        # First iteration: baseline
        cb.on_iteration_end(0, algo, [], [])

        # Next patience iterations with no improvement
        for i in range(1, patience + 1):
            assert cb.should_stop is False
            cb.on_iteration_end(i, algo, [], [])

        assert cb.should_stop is True

    def test_improvement_resets_counter_min_mode(self) -> None:
        """In min mode, lower score resets patience counter."""
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="min")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)

        # Iteration 0: baseline
        cb.on_iteration_end(0, algo, [], [])

        # 2 stale iterations
        cb.on_iteration_end(1, algo, [], [])
        cb.on_iteration_end(2, algo, [], [])
        assert cb.should_stop is False

        # Improvement: score drops
        algo.best_score = 0.5
        cb.on_iteration_end(3, algo, [], [])
        assert cb.should_stop is False

        # 2 more stale iterations -- should NOT trigger yet (counter reset)
        cb.on_iteration_end(4, algo, [], [])
        cb.on_iteration_end(5, algo, [], [])
        assert cb.should_stop is False

    def test_improvement_resets_counter_max_mode(self) -> None:
        """In max mode, higher score resets patience counter."""
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="max")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)

        # Iteration 0: baseline
        cb.on_iteration_end(0, algo, [], [])

        # 2 stale iterations
        cb.on_iteration_end(1, algo, [], [])
        cb.on_iteration_end(2, algo, [], [])
        assert cb.should_stop is False

        # Improvement: score increases
        algo.best_score = 2.0
        cb.on_iteration_end(3, algo, [], [])
        assert cb.should_stop is False

        # 2 more stale -- still within new patience window
        cb.on_iteration_end(4, algo, [], [])
        cb.on_iteration_end(5, algo, [], [])
        assert cb.should_stop is False

    def test_min_mode_lower_is_improvement(self) -> None:
        """mode='min': lower score counts as improvement."""
        cb = EarlyStoppingCallback(patience=2, min_delta=0.01, mode="min")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)

        # Baseline
        cb.on_iteration_end(0, algo, [], [])

        # Sufficient improvement: drop by more than min_delta
        algo.best_score = 0.5
        cb.on_iteration_end(1, algo, [], [])

        # 2 stale iterations (no further improvement)
        cb.on_iteration_end(2, algo, [], [])
        cb.on_iteration_end(3, algo, [], [])

        # patience=2 stale iterations reached
        assert cb.should_stop is True

    def test_max_mode_higher_is_improvement(self) -> None:
        """mode='max': higher score counts as improvement."""
        cb = EarlyStoppingCallback(patience=2, min_delta=0.01, mode="max")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)

        # Baseline
        cb.on_iteration_end(0, algo, [], [])

        # Sufficient improvement: increase by more than min_delta
        algo.best_score = 2.0
        cb.on_iteration_end(1, algo, [], [])

        # 2 stale iterations
        cb.on_iteration_end(2, algo, [], [])
        cb.on_iteration_end(3, algo, [], [])

        assert cb.should_stop is True

    def test_min_delta_threshold(self) -> None:
        """Improvement smaller than min_delta does not count."""
        min_delta = 0.1
        cb = EarlyStoppingCallback(patience=2, min_delta=min_delta, mode="min")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)

        # Baseline
        cb.on_iteration_end(0, algo, [], [])

        # Tiny improvement (less than min_delta) -- should NOT reset counter
        algo.best_score = 1.0 - 0.05 # only 0.05 improvement < 0.1 threshold
        cb.on_iteration_end(1, algo, [], [])

        algo.best_score = 1.0 - 0.05 # still tiny
        cb.on_iteration_end(2, algo, [], [])

        # After 2 non-improving iterations, should stop
        assert cb.should_stop is True

    def test_patience_zero_stops_after_first_non_improving(self) -> None:
        """patience=0 stops immediately after first non-improving iteration.

        Improvement iterations must NOT trigger stop even with patience=0.
        """
        cb = EarlyStoppingCallback(patience=0, mode="min")
        algo = _MockAlgorithm(best_score=1.0)
        cb.on_experiment_start(algo)

        # First iteration: improvement (from inf sentinel) — should NOT stop
        cb.on_iteration_end(0, algo, [], [])
        assert cb.should_stop is False

        # Second iteration: same score, no improvement — should stop
        cb.on_iteration_end(1, algo, [], [])
        assert cb.should_stop is True

    def test_reusable_across_experiments(self) -> None:
        """on_experiment_start resets internal state for reuse."""
        patience = 2
        cb = EarlyStoppingCallback(patience=patience, mode="min")
        algo = _MockAlgorithm(best_score=1.0)

        # --- First experiment: exhaust patience ---
        cb.on_experiment_start(algo)
        cb.on_iteration_end(0, algo, [], [])
        cb.on_iteration_end(1, algo, [], [])
        cb.on_iteration_end(2, algo, [], [])
        assert cb.should_stop is True

        # --- Second experiment: should reset ---
        algo.best_score = 5.0 # fresh start
        cb.on_experiment_start(algo)
        assert cb.should_stop is False

        cb.on_iteration_end(0, algo, [], [])
        assert cb.should_stop is False

    def test_continuous_improvement_never_stops(self) -> None:
        """Continuous improvement means should_stop stays False."""
        cb = EarlyStoppingCallback(patience=2, mode="min")
        algo = _MockAlgorithm(best_score=10.0)
        cb.on_experiment_start(algo)

        for i in range(20):
            algo.best_score = 10.0 - i * 0.5
            cb.on_iteration_end(i, algo, [], [])
            assert cb.should_stop is False

    def test_on_iteration_start_does_not_affect_stopping(
        self, mock_algo: _MockAlgorithm
    ) -> None:
        """on_iteration_start should not change should_stop state."""
        cb = EarlyStoppingCallback(patience=5)
        cb.on_experiment_start(mock_algo)
        cb.on_iteration_start(0, mock_algo)
        assert cb.should_stop is False


# ============================================================================
# Test Group 4: CheckpointCallback
# ============================================================================


class TestCheckpointCallback:
    """Tests for CheckpointCallback contract."""

    def test_default_every_n(self, tmp_path: Path) -> None:
        """Default every_n is 10."""
        cb = CheckpointCallback(directory=tmp_path)
        assert isinstance(cb, RunnerCallback)

    def test_every_n_custom(self, tmp_path: Path) -> None:
        """Custom every_n is accepted."""
        cb = CheckpointCallback(directory=tmp_path, every_n=5)
        assert isinstance(cb, RunnerCallback)

    def test_every_n_less_than_one_raises(self, tmp_path: Path) -> None:
        """every_n < 1 raises ValueError."""
        with pytest.raises(ValueError):
            CheckpointCallback(directory=tmp_path, every_n=0)

    def test_every_n_negative_raises(self, tmp_path: Path) -> None:
        """Negative every_n raises ValueError."""
        with pytest.raises(ValueError):
            CheckpointCallback(directory=tmp_path, every_n=-1)

    def test_should_stop_always_false(self, tmp_path: Path) -> None:
        """should_stop is always False for CheckpointCallback."""
        cb = CheckpointCallback(directory=tmp_path)
        assert cb.should_stop is False

        algo = _MockAlgorithm()
        cb.on_experiment_start(algo)
        assert cb.should_stop is False

        cb.on_iteration_end(0, algo, [], [])
        assert cb.should_stop is False

        cb.on_experiment_end(algo)
        assert cb.should_stop is False

    def test_creates_directory_on_experiment_start(self, tmp_path: Path) -> None:
        """Creates output directory on experiment start."""
        ckpt_dir = tmp_path / "checkpoints"
        assert not ckpt_dir.exists()

        cb = CheckpointCallback(directory=ckpt_dir)
        cb.on_experiment_start(_MockAlgorithm())

        assert ckpt_dir.exists()
        assert ckpt_dir.is_dir()

    def test_saves_at_correct_iterations(self, tmp_path: Path) -> None:
        """Saves checkpoint at iterations 0, N, 2N for every_n=N."""
        every_n = 3
        cb = CheckpointCallback(directory=tmp_path, every_n=every_n)
        algo = _MockAlgorithm()
        algo.state = {"gen": 0}
        cb.on_experiment_start(algo)

        for i in range(10):
            algo.state = {"gen": i}
            cb.on_iteration_end(i, algo, [], [])

        # Should have checkpoints at 0, 3, 6, 9
        expected = [
            tmp_path / "checkpoint_000000.pt",
            tmp_path / "checkpoint_000003.pt",
            tmp_path / "checkpoint_000006.pt",
            tmp_path / "checkpoint_000009.pt",
        ]
        for path in expected:
            assert path.exists(), f"Expected checkpoint at {path}"

    def test_does_not_save_at_non_matching_iterations(self, tmp_path: Path) -> None:
        """Does NOT save at iterations that are not multiples of every_n."""
        cb = CheckpointCallback(directory=tmp_path, every_n=5)
        algo = _MockAlgorithm()
        cb.on_experiment_start(algo)

        # Only iteration 1 -- should NOT produce a checkpoint
        cb.on_iteration_end(1, algo, [], [])

        unexpected = tmp_path / "checkpoint_000001.pt"
        assert not unexpected.exists()

    def test_file_naming_pattern(self, tmp_path: Path) -> None:
        """File names follow checkpoint_NNNNNN.pt (6-digit zero-padded)."""
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        algo = _MockAlgorithm()
        cb.on_experiment_start(algo)

        cb.on_iteration_end(0, algo, [], [])
        cb.on_iteration_end(42, algo, [], [])

        assert (tmp_path / "checkpoint_000000.pt").exists()
        assert (tmp_path / "checkpoint_000042.pt").exists()

    def test_saved_dict_has_required_keys(self, tmp_path: Path) -> None:
        """Saved checkpoint dict has 'iteration' and 'algorithm_state' keys."""
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        algo = _MockAlgorithm()
        algo.state = {"population": ["u_x", "u_xx"], "gen": 5}
        cb.on_experiment_start(algo)

        cb.on_iteration_end(0, algo, [], [])

        ckpt_path = tmp_path / "checkpoint_000000.pt"
        data = torch.load(ckpt_path, weights_only=False)

        assert "iteration" in data
        assert "algorithm_state" in data
        assert data["iteration"] == 0
        assert data["algorithm_state"] == algo.state

    def test_saves_final_checkpoint_on_experiment_end(self, tmp_path: Path) -> None:
        """Saves checkpoint_final.pt on experiment end with same schema."""
        cb = CheckpointCallback(directory=tmp_path, every_n=10)
        algo = _MockAlgorithm()
        algo.state = {"final": True}
        cb.on_experiment_start(algo)

        # Run some iterations so last_iteration is set
        cb.on_iteration_end(0, algo, [], [])

        cb.on_experiment_end(algo)

        final_path = tmp_path / "checkpoint_final.pt"
        assert final_path.exists()

        data = torch.load(final_path, weights_only=False)
        assert "algorithm_state" in data
        assert "iteration" in data
        assert data["iteration"] == 0

    def test_directory_already_exists(self, tmp_path: Path) -> None:
        """Does not raise if directory already exists."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        assert ckpt_dir.exists()

        cb = CheckpointCallback(directory=ckpt_dir)
        # Should not raise
        cb.on_experiment_start(_MockAlgorithm())
        assert ckpt_dir.exists()

    def test_nested_directory_creation(self, tmp_path: Path) -> None:
        """Creates nested directories if needed."""
        nested_dir = tmp_path / "a" / "b" / "c"
        cb = CheckpointCallback(directory=nested_dir)
        cb.on_experiment_start(_MockAlgorithm())
        assert nested_dir.exists()


# ============================================================================
# Test Group 5: Edge Cases
# ============================================================================


class TestCallbackEdgeCases:
    """Edge case tests for callback system."""

    def test_empty_candidates_and_results(self, mock_algo: _MockAlgorithm) -> None:
        """Callbacks handle empty candidates/results without error."""
        log_cb = LoggingCallback(every_n=1)
        es_cb = EarlyStoppingCallback(patience=5)

        log_cb.on_experiment_start(mock_algo)
        es_cb.on_experiment_start(mock_algo)

        # Empty lists should not raise
        log_cb.on_iteration_end(0, mock_algo, [], [])
        es_cb.on_iteration_end(0, mock_algo, [], [])

    def test_multiple_callbacks_coexist(self, tmp_path: Path) -> None:
        """Multiple callbacks can coexist: logging + early stopping."""
        log_cb = LoggingCallback(every_n=1)
        es_cb = EarlyStoppingCallback(patience=2, mode="min")

        algo = _MockAlgorithm(best_score=1.0)

        log_cb.on_experiment_start(algo)
        es_cb.on_experiment_start(algo)

        # Run iterations
        for i in range(5):
            log_cb.on_iteration_end(i, algo, ["u_x"], [_make_eval_result()])
            es_cb.on_iteration_end(i, algo, ["u_x"], [_make_eval_result()])

        # LoggingCallback never stops
        assert log_cb.should_stop is False

        # EarlyStoppingCallback should have triggered (no improvement)
        assert es_cb.should_stop is True

    def test_on_iteration_start_is_callable(
        self, mock_algo: _MockAlgorithm, tmp_path: Path
    ) -> None:
        """All callbacks accept on_iteration_start without error."""
        callbacks = [
            LoggingCallback(),
            EarlyStoppingCallback(),
            CheckpointCallback(directory=tmp_path),
        ]

        for cb in callbacks:
            cb.on_experiment_start(mock_algo)
            cb.on_iteration_start(0, mock_algo) # Should not raise

    def test_checkpoint_and_early_stopping_together(self, tmp_path: Path) -> None:
        """CheckpointCallback saves even when EarlyStopping wants to stop."""
        ckpt_cb = CheckpointCallback(directory=tmp_path, every_n=1)
        es_cb = EarlyStoppingCallback(patience=1, mode="min")

        algo = _MockAlgorithm(best_score=1.0)

        ckpt_cb.on_experiment_start(algo)
        es_cb.on_experiment_start(algo)

        # Iteration 0: baseline
        ckpt_cb.on_iteration_end(0, algo, [], [])
        es_cb.on_iteration_end(0, algo, [], [])

        # Iteration 1: no improvement
        ckpt_cb.on_iteration_end(1, algo, [], [])
        es_cb.on_iteration_end(1, algo, [], [])

        # ES wants to stop, but checkpoint should have saved both
        assert es_cb.should_stop is True
        assert ckpt_cb.should_stop is False
        assert (tmp_path / "checkpoint_000000.pt").exists()
        assert (tmp_path / "checkpoint_000001.pt").exists()


# ============================================================================
# Layer 2: Design Verification Tests
# Protect specific design decisions from the T7 Callback System design brief.
# ============================================================================


def _fire_iteration_end(
    callback: Any,
    iteration: int,
    algorithm: _MockAlgorithm,
) -> None:
    """Call on_iteration_end with empty candidates/results."""
    callback.on_iteration_end(
        iteration=iteration,
        algorithm=algorithm,
        candidates=[],
        results=[],
    )


# ============================================================================
# D1, D2, D3: EarlyStopping Numerical Behavior
# ============================================================================


class TestEarlyStoppingNumerical:
    """Verify NaN handling (D1), Inf handling (D2), and min_delta edge cases (D3)."""

    # -- D1: NaN is handled explicitly with math.isnan() -----------------

    @pytest.mark.numerical
    def test_nan_score_triggers_stop_after_patience(self) -> None:
        """D1: NaN score must never count as improvement.

        Feed NaN for `patience` iterations -> should_stop becomes True.
        """
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="min")
        algo = _MockAlgorithm(best_score=float("nan"))

        for i in range(patience):
            _fire_iteration_end(cb, iteration=i, algorithm=algo)

        assert cb.should_stop is True

    @pytest.mark.numerical
    def test_nan_score_does_not_corrupt_best(self) -> None:
        """D1: NaN score must not become the new best.

        After seeing NaN, a subsequent finite improvement should still
        compare against the original best, not NaN.
        """
        cb = EarlyStoppingCallback(patience=5, mode="min")

        # First: a valid score establishes the baseline
        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))

        # Then: NaN should not corrupt _best
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=float("nan")))

        # Then: a genuine improvement resets counter
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=0.5))

        assert cb.should_stop is False

    @pytest.mark.numerical
    def test_nan_interspersed_with_valid_scores(self) -> None:
        """D1: patience=2, scores=[valid, nan, nan] -> should_stop=True."""
        cb = EarlyStoppingCallback(patience=2, mode="min")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=float("nan")))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("nan")))

        assert cb.should_stop is True

    @pytest.mark.numerical
    def test_nan_in_max_mode_increments_counter(self) -> None:
        """D1: NaN in max mode also never counts as improvement."""
        patience = 2
        cb = EarlyStoppingCallback(patience=patience, mode="max")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=float("nan")))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("nan")))

        assert cb.should_stop is True

    # -- D2: +-Inf handling in both modes --------------------------------

    @pytest.mark.numerical
    def test_min_mode_neg_inf_is_ultimate_improvement(self) -> None:
        """D2: In min mode, -inf is the ultimate improvement (resets counter)."""
        cb = EarlyStoppingCallback(patience=2, mode="min")

        # Stale iteration
        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=1.0))
        # -inf is improvement, should reset counter
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("-inf")))

        assert cb.should_stop is False

    @pytest.mark.numerical
    def test_min_mode_pos_inf_is_no_improvement(self) -> None:
        """D2: In min mode, +inf is no improvement (increments counter)."""
        cb = EarlyStoppingCallback(patience=2, mode="min")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=float("inf")))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("inf")))

        assert cb.should_stop is True

    @pytest.mark.numerical
    def test_max_mode_pos_inf_is_ultimate_improvement(self) -> None:
        """D2: In max mode, +inf is the ultimate improvement (resets counter)."""
        cb = EarlyStoppingCallback(patience=2, mode="max")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("inf")))

        assert cb.should_stop is False

    @pytest.mark.numerical
    def test_max_mode_neg_inf_is_no_improvement(self) -> None:
        """D2: In max mode, -inf is no improvement (increments counter)."""
        cb = EarlyStoppingCallback(patience=2, mode="max")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=float("-inf")))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=float("-inf")))

        assert cb.should_stop is True

    @pytest.mark.numerical
    def test_min_mode_inf_score_against_inf_best_no_nan(self) -> None:
        """D2: inf - inf must not produce NaN intermediate; score is not improvement."""
        cb = EarlyStoppingCallback(patience=1, mode="min")

        # First iteration: best is inf (sentinel), score is inf → no improvement
        # (With old code, inf - inf would produce NaN intermediate)
        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=float("inf")))

        assert cb.should_stop is True

    @pytest.mark.numerical
    def test_max_mode_neg_inf_score_against_neg_inf_best_no_nan(self) -> None:
        """D2: -inf - (-inf) must not produce NaN intermediate; score is not improvement."""
        cb = EarlyStoppingCallback(patience=1, mode="max")

        # First iteration: best is -inf (sentinel), score is -inf → no improvement
        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=float("-inf")))

        assert cb.should_stop is True

    # -- D3: min_delta threshold edge cases ------------------------------

    @pytest.mark.unit
    def test_min_delta_rejects_micro_improvement_min_mode(self) -> None:
        """D3: In min mode, delta < min_delta does NOT count as improvement.

        score goes from 1.0 to 0.95 (delta=0.05 < min_delta=0.1) -> stale.
        """
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1, mode="min")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        # delta = 1.0 - 0.95 = 0.05 < 0.1 -> not an improvement
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=0.95))

        assert cb.should_stop is True

    @pytest.mark.unit
    def test_min_delta_accepts_sufficient_improvement_min_mode(self) -> None:
        """D3: In min mode, delta > min_delta counts as improvement.

        score goes from 1.0 to 0.85 (delta=0.15 > min_delta=0.1) -> improvement.
        """
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1, mode="min")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        # delta = 1.0 - 0.85 = 0.15 > 0.1 -> improvement
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=0.85))

        assert cb.should_stop is False

    @pytest.mark.unit
    def test_min_delta_rejects_micro_improvement_max_mode(self) -> None:
        """D3: In max mode, delta < min_delta does NOT count as improvement."""
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1, mode="max")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        # delta = 1.05 - 1.0 = 0.05 < 0.1 -> not improvement
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=1.05))

        assert cb.should_stop is True

    @pytest.mark.unit
    def test_min_delta_accepts_sufficient_improvement_max_mode(self) -> None:
        """D3: In max mode, delta > min_delta counts as improvement."""
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1, mode="max")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        # delta = 1.15 - 1.0 = 0.15 > 0.1 -> improvement
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=1.15))

        assert cb.should_stop is False

    @pytest.mark.unit
    def test_min_delta_exact_boundary_is_not_improvement(self) -> None:
        """D3: delta == min_delta is NOT improvement (must be strictly greater).

        min mode: 1.0 -> 0.9, delta = 0.1, min_delta = 0.1 -> NOT improvement.
        Uses strict inequality: best - score > min_delta.
        """
        cb = EarlyStoppingCallback(patience=1, min_delta=0.1, mode="min")

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        # delta = 1.0 - 0.9 = 0.1 == min_delta -> NOT improvement (strictly >)
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=0.9))

        assert cb.should_stop is True


# ============================================================================
# D4: on_experiment_start resets state for reuse
# ============================================================================


class TestEarlyStoppingStateReset:
    """Verify on_experiment_start resets EarlyStoppingCallback for reuse (D4)."""

    @pytest.mark.unit
    def test_reset_clears_should_stop(self) -> None:
        """D4: After triggering stop, on_experiment_start resets should_stop."""
        cb = EarlyStoppingCallback(patience=1, mode="min")
        algo = _MockAlgorithm(best_score=1.0)

        # Trigger stop
        _fire_iteration_end(cb, 0, algo)
        _fire_iteration_end(cb, 1, algo)
        assert cb.should_stop is True

        # Reset
        cb.on_experiment_start(algo)
        assert cb.should_stop is False

    @pytest.mark.unit
    def test_reset_clears_counter(self) -> None:
        """D4: on_experiment_start resets counter to 0.

        After reset, _best is back to sentinel (inf for min mode), so the
        first iteration is an improvement. Then need `patience` fully stale
        iterations to trigger stop again.
        """
        patience = 3
        cb = EarlyStoppingCallback(patience=patience, mode="min")
        algo_stale = _MockAlgorithm(best_score=1.0)

        # Accumulate some stale iterations (but don't trigger stop)
        _fire_iteration_end(cb, 0, algo_stale)
        _fire_iteration_end(cb, 1, algo_stale)

        # Reset — _best goes back to inf, counter=0
        cb.on_experiment_start(algo_stale)

        # First iteration post-reset: improvement (inf → 1.0), counter stays 0
        _fire_iteration_end(cb, 0, algo_stale)
        # Then patience stale iterations needed
        for i in range(1, patience):
            _fire_iteration_end(cb, i, algo_stale)
        assert cb.should_stop is False # counter = patience-1

        _fire_iteration_end(cb, patience, algo_stale)
        assert cb.should_stop is True # counter = patience

    @pytest.mark.unit
    def test_reset_restores_sentinel_best_min_mode(self) -> None:
        """D4: on_experiment_start resets _best to inf for min mode.

        After reset, any finite score should count as improvement.
        """
        cb = EarlyStoppingCallback(patience=5, mode="min")

        # Establish a good best
        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=0.5))

        # Reset
        cb.on_experiment_start(_MockAlgorithm())

        # After reset, even a very high score should be an improvement
        # (because best is back to inf, and 100.0 < inf)
        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=100.0))
        # If best was still 0.5, score=100.0 would be stale
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=100.0))

        assert cb.should_stop is False

    @pytest.mark.unit
    def test_reset_restores_sentinel_best_max_mode(self) -> None:
        """D4: on_experiment_start resets _best to -inf for max mode.

        After reset, even a very low score should count as improvement.
        """
        cb = EarlyStoppingCallback(patience=5, mode="max")

        # Establish a high best
        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=100.0))

        # Reset
        cb.on_experiment_start(_MockAlgorithm())

        # After reset, even a very negative score is improvement (> -inf)
        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=-100.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=-100.0))

        assert cb.should_stop is False

    @pytest.mark.unit
    def test_full_reuse_two_experiments(self) -> None:
        """D4: A single callback instance behaves identically across experiments."""
        cb = EarlyStoppingCallback(patience=2, mode="min")

        # Experiment 1: trigger stop
        cb.on_experiment_start(_MockAlgorithm())
        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=1.0))
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=1.0))
        assert cb.should_stop is True

        # Experiment 2: reset and verify clean start
        cb.on_experiment_start(_MockAlgorithm())
        assert cb.should_stop is False

        _fire_iteration_end(cb, 0, _MockAlgorithm(best_score=0.5))
        assert cb.should_stop is False
        _fire_iteration_end(cb, 1, _MockAlgorithm(best_score=0.3))
        assert cb.should_stop is False
        _fire_iteration_end(cb, 2, _MockAlgorithm(best_score=0.1))
        assert cb.should_stop is False


# ============================================================================
# D5, D6: Checkpoint Design Decisions
# ============================================================================


class TestCheckpointDesign:
    """Verify checkpoint file format (D5) and final checkpoint (D6)."""

    # -- D5: Checkpoint is NOT raw state, it is wrapped with metadata ----

    @pytest.mark.unit
    def test_checkpoint_is_not_raw_state(self, tmp_path: Path) -> None:
        """D5: Checkpoint must NOT be raw algorithm.state -- must be wrapped.

        The saved dict must have exactly {iteration, algorithm_state},
        not just the raw state dict dumped directly.
        """
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        algo = _MockAlgorithm()
        algo.state = {"weights": [1.0]}
        cb.on_experiment_start(algo)

        _fire_iteration_end(cb, iteration=0, algorithm=algo)

        pt_files = list(tmp_path.glob("checkpoint_0*.pt"))
        assert len(pt_files) == 1

        data = torch.load(pt_files[0], weights_only=False)

        # Must have the wrapper keys, not be the raw state
        assert set(data.keys()) == {
            "version",
            "iteration",
            "algorithm_state",
            "best_score",
            "best_expression",
        }

    @pytest.mark.unit
    def test_checkpoint_iteration_value_matches(self, tmp_path: Path) -> None:
        """D5: iteration value matches the iteration number passed in."""
        cb = CheckpointCallback(directory=tmp_path, every_n=5)
        algo = _MockAlgorithm()
        algo.state = {"gen": 7}
        cb.on_experiment_start(algo)

        _fire_iteration_end(cb, iteration=5, algorithm=algo)

        pt_files = list(tmp_path.glob("checkpoint_0*.pt"))
        assert len(pt_files) == 1

        data = torch.load(pt_files[0], weights_only=False)
        assert data["iteration"] == 5

    @pytest.mark.unit
    def test_checkpoint_algorithm_state_matches(self, tmp_path: Path) -> None:
        """D5: algorithm_state contains the actual algorithm.state dict."""
        cb = CheckpointCallback(directory=tmp_path, every_n=1)
        state = {"generation": 42, "population": ["a", "b"]}
        algo = _MockAlgorithm()
        algo.state = state
        cb.on_experiment_start(algo)

        _fire_iteration_end(cb, iteration=0, algorithm=algo)

        pt_files = list(tmp_path.glob("checkpoint_0*.pt"))
        data = torch.load(pt_files[0], weights_only=False)
        assert data["algorithm_state"] == state

    # -- D6: checkpoint_final.pt saved on experiment_end -----------------

    @pytest.mark.unit
    def test_final_checkpoint_exact_filename(self, tmp_path: Path) -> None:
        """D6: on_experiment_end saves exactly 'checkpoint_final.pt'."""
        cb = CheckpointCallback(directory=tmp_path, every_n=10)
        algo = _MockAlgorithm()
        algo.state = {}
        cb.on_experiment_start(algo)

        cb.on_experiment_end(algo)

        files = [f.name for f in tmp_path.iterdir()]
        assert "checkpoint_final.pt" in files

    @pytest.mark.unit
    def test_final_checkpoint_contains_state(self, tmp_path: Path) -> None:
        """D6: checkpoint_final.pt contains algorithm state."""
        cb = CheckpointCallback(directory=tmp_path, every_n=10)
        state = {"generation": 100, "converged": True}
        algo = _MockAlgorithm()
        algo.state = state
        cb.on_experiment_start(algo)

        cb.on_experiment_end(algo)

        final_path = tmp_path / "checkpoint_final.pt"
        data = torch.load(final_path, weights_only=False)
        assert isinstance(data, dict)
        assert "algorithm_state" in data
        assert data["algorithm_state"] == state


# ============================================================================
# D7, D8: Iteration Indexing and Logging Content
# ============================================================================


class TestIterationIndexing:
    """Verify 0-indexed alignment (D7) and logging content (D8)."""

    # -- D7: 0-indexed iteration alignment (regression guard) ------------

    @pytest.mark.unit
    def test_logging_not_one_indexed(self, caplog: pytest.LogCaptureFixture) -> None:
        """D7: LoggingCallback with every_n=5 must NOT log at 1, 6, 11.

        Guards against the off-by-one bug: (iteration+1) % N == 0.
        """
        cb = LoggingCallback(every_n=5)
        algo = _MockAlgorithm(best_score=0.1, best_expression="u_x")
        cb.on_experiment_start(algo)

        for i in [1, 6, 11]:
            caplog.clear()
            with caplog.at_level(logging.INFO):
                _fire_iteration_end(cb, iteration=i, algorithm=algo)
            assert len(caplog.records) == 0, (
                f"Should not log at 1-indexed iteration {i}"
            )

    @pytest.mark.unit
    def test_logging_zero_indexed_full_sequence(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """D7: LoggingCallback with every_n=5 logs at 0, 5, 10 exactly."""
        cb = LoggingCallback(every_n=5)
        algo = _MockAlgorithm(best_score=0.1, best_expression="u_x")
        cb.on_experiment_start(algo)

        logged_iterations: list[int] = []
        for i in range(15):
            caplog.clear()
            with caplog.at_level(logging.INFO):
                _fire_iteration_end(cb, iteration=i, algorithm=algo)
            if caplog.records:
                logged_iterations.append(i)

        assert logged_iterations == [0, 5, 10]

    @pytest.mark.unit
    def test_checkpoint_zero_indexed_full_sequence(self, tmp_path: Path) -> None:
        """D7: CheckpointCallback with every_n=5 saves at 0, 5, 10."""
        cb = CheckpointCallback(directory=tmp_path, every_n=5)
        algo = _MockAlgorithm()
        algo.state = {"gen": 0}
        cb.on_experiment_start(algo)

        for i in range(15):
            _fire_iteration_end(cb, iteration=i, algorithm=algo)

        # Filter out checkpoint_final.pt if present
        pt_files = sorted(
            f for f in tmp_path.glob("checkpoint_*.pt") if "final" not in f.name
        )
        assert len(pt_files) == 3 # iterations 0, 5, 10

    @pytest.mark.unit
    def test_checkpoint_not_one_indexed(self, tmp_path: Path) -> None:
        """D7: CheckpointCallback with every_n=5 must NOT save at iteration 1."""
        cb = CheckpointCallback(directory=tmp_path, every_n=5)
        algo = _MockAlgorithm()
        algo.state = {"gen": 1}
        cb.on_experiment_start(algo)

        _fire_iteration_end(cb, iteration=1, algorithm=algo)

        pt_files = [
            f for f in tmp_path.glob("checkpoint_*.pt") if "final" not in f.name
        ]
        assert len(pt_files) == 0

    # -- D8: LoggingCallback logs experiment start/end content -----------

    @pytest.mark.unit
    def test_experiment_start_log_contains_start(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """D8: on_experiment_start logs a message containing 'start'."""
        cb = LoggingCallback(every_n=1)
        algo = _MockAlgorithm()

        with caplog.at_level(logging.INFO):
            cb.on_experiment_start(algo)

        assert len(caplog.records) > 0
        combined = " ".join(r.message.lower() for r in caplog.records)
        assert "start" in combined

    @pytest.mark.unit
    def test_experiment_end_log_contains_best_score(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """D8: on_experiment_end logs the final best_score value."""
        cb = LoggingCallback(every_n=1)
        algo = _MockAlgorithm(best_score=0.001, best_expression="mul(u, u_x)")

        with caplog.at_level(logging.INFO):
            cb.on_experiment_end(algo)

        assert len(caplog.records) > 0
        combined = " ".join(r.message for r in caplog.records)
        # Score should appear in some numeric format
        assert "0.001" in combined or "1e-03" in combined

    @pytest.mark.unit
    def test_experiment_end_log_contains_best_expression(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """D8: on_experiment_end logs the final best_expression."""
        cb = LoggingCallback(every_n=1)
        algo = _MockAlgorithm(best_score=0.001, best_expression="mul(u, u_x)")

        with caplog.at_level(logging.INFO):
            cb.on_experiment_end(algo)

        assert len(caplog.records) > 0
        combined = " ".join(r.message for r in caplog.records)
        assert "mul(u, u_x)" in combined


# ============================================================================
# VizDataCollector Tests (P4-T1b)
# ============================================================================


class TestVizDataCollectorSmoke:
    """Smoke tests for VizDataCollector importability and construction."""

    @pytest.mark.smoke
    def test_importable(self) -> None:
        """VizDataCollector can be imported from kd2.search.callbacks."""
        from kd2.search.callbacks import VizDataCollector

        assert VizDataCollector is not None

    @pytest.mark.smoke
    def test_construction_with_recorder(self) -> None:
        """VizDataCollector can be constructed with a VizRecorder."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        assert collector is not None

    @pytest.mark.smoke
    def test_satisfies_runner_callback_protocol(self) -> None:
        """VizDataCollector satisfies the RunnerCallback protocol."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        collector = VizDataCollector(VizRecorder())
        assert isinstance(collector, RunnerCallback)


class TestVizDataCollectorBehavior:
    """Behavioral tests for VizDataCollector callback."""

    @pytest.mark.unit
    def test_should_stop_always_false(self) -> None:
        """VizDataCollector.should_stop always returns False.

        VizDataCollector is a passive data collector -- it must never
        request early termination.
        """
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        collector = VizDataCollector(VizRecorder())
        # Before any events
        assert collector.should_stop is False

    @pytest.mark.unit
    def test_should_stop_false_after_many_iterations(self) -> None:
        """should_stop remains False even after many iteration_end events."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=1.0, best_expression="u_x")

        for i in range(50):
            collector.on_iteration_end(i, algo, ["u_x"], [_make_eval_result()])

        assert collector.should_stop is False

    @pytest.mark.unit
    def test_on_iteration_end_logs_best_score(self) -> None:
        """on_iteration_end logs _best_score from algorithm.best_score."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=0.42, best_expression="u_x")

        collector.on_iteration_end(0, algo, ["u_x"], [_make_eval_result()])

        scores = recorder.get("_best_score")
        assert len(scores) == 1
        assert scores[0] == 0.42

    @pytest.mark.unit
    def test_on_iteration_end_logs_best_expr(self) -> None:
        """on_iteration_end logs _best_expr from algorithm.best_expression."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=1.0, best_expression="mul(u, u_x)")

        collector.on_iteration_end(0, algo, ["u_x"], [_make_eval_result()])

        exprs = recorder.get("_best_expr")
        assert len(exprs) == 1
        assert exprs[0] == "mul(u, u_x)"

    @pytest.mark.unit
    def test_on_iteration_end_logs_n_candidates(self) -> None:
        """on_iteration_end logs _n_candidates as len(candidates)."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=1.0, best_expression="u_x")

        candidates = ["u_x", "u_xx", "mul(u, u_x)"]
        results = [_make_eval_result() for _ in candidates]
        collector.on_iteration_end(0, algo, candidates, results)

        n_cands = recorder.get("_n_candidates")
        assert len(n_cands) == 1
        assert n_cands[0] == 3

    @pytest.mark.unit
    def test_multiple_iterations_accumulate(self) -> None:
        """Multiple on_iteration_end calls accumulate data in recorder."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        n_iterations = 5

        for i in range(n_iterations):
            score = 1.0 / (i + 1)
            algo = _MockAlgorithm(best_score=score, best_expression=f"expr_{i}")
            collector.on_iteration_end(i, algo, ["c"], [_make_eval_result()])

        scores = recorder.get("_best_score")
        exprs = recorder.get("_best_expr")
        n_cands = recorder.get("_n_candidates")

        # All three series have n_iterations entries
        assert len(scores) == n_iterations
        assert len(exprs) == n_iterations
        assert len(n_cands) == n_iterations

        # Scores should be monotonically decreasing (1/1, 1/2, ..., 1/5)
        for j in range(1, n_iterations):
            assert scores[j] < scores[j - 1]

    @pytest.mark.unit
    def test_on_experiment_start_is_noop(self) -> None:
        """on_experiment_start does not record anything."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm()

        collector.on_experiment_start(algo)

        assert len(recorder.keys()) == 0

    @pytest.mark.unit
    def test_on_experiment_end_is_noop(self) -> None:
        """on_experiment_end does not record anything."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm()

        collector.on_experiment_end(algo)

        assert len(recorder.keys()) == 0

    @pytest.mark.unit
    def test_on_iteration_start_is_noop(self) -> None:
        """on_iteration_start does not record anything."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm()

        collector.on_iteration_start(0, algo)

        assert len(recorder.keys()) == 0


class TestVizDataCollectorEdgeCases:
    """Edge cases and negative tests for VizDataCollector."""

    @pytest.mark.unit
    def test_disabled_recorder_no_crash(self) -> None:
        """VizDataCollector with enabled=False recorder does not crash.

        The recorder silently discards logs when disabled, so
        VizDataCollector must not break.
        """
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder(enabled=False)
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=1.0, best_expression="u_x")

        # Should not raise
        collector.on_iteration_end(0, algo, ["u_x"], [_make_eval_result()])

        # Recorder should have no data (disabled)
        assert len(recorder.keys()) == 0

    @pytest.mark.unit
    def test_empty_candidates_list(self) -> None:
        """on_iteration_end with empty candidates logs _n_candidates=0."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=1.0, best_expression="u_x")

        collector.on_iteration_end(0, algo, [], [])

        n_cands = recorder.get("_n_candidates")
        assert len(n_cands) == 1
        assert n_cands[0] == 0

    @pytest.mark.numerical
    def test_inf_best_score_logged(self) -> None:
        """VizDataCollector logs inf best_score without crash."""
        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=float("inf"), best_expression="")

        collector.on_iteration_end(0, algo, ["c"], [_make_eval_result()])

        scores = recorder.get("_best_score")
        assert len(scores) == 1
        assert scores[0] == float("inf")

    @pytest.mark.numerical
    def test_nan_best_score_logged(self) -> None:
        """VizDataCollector logs NaN best_score without crash.

        NaN should be recorded as-is -- it is the recorder/serializer's
        job to sanitize NaN, not VizDataCollector's.
        """
        import math

        from kd2.search.callbacks import VizDataCollector
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        collector = VizDataCollector(recorder)
        algo = _MockAlgorithm(best_score=float("nan"), best_expression="bad")

        collector.on_iteration_end(0, algo, ["c"], [_make_eval_result()])

        scores = recorder.get("_best_score")
        assert len(scores) == 1
        assert math.isnan(scores[0])
