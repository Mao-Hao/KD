"""Runner callbacks for experiment orchestration.

Provides the RunnerCallback protocol and built-in implementations:
- LoggingCallback: logs best score/expression every N iterations
- EarlyStoppingCallback: patience-based early stopping with NaN safety
- CheckpointCallback: periodic checkpoint saving via torch.save
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

import torch

if TYPE_CHECKING:
    from kd2.core.evaluator import EvaluationResult
    from kd2.search.protocol import SearchAlgorithm
    from kd2.search.recorder import VizRecorder

logger = logging.getLogger(__name__)

__all__ = [
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LoggingCallback",
    "RunnerCallback",
    "VizDataCollector",
]

# Constants

_MIN_EVERY_N = 1


# Protocol


@runtime_checkable
class RunnerCallback(Protocol):
    """Protocol for runner callbacks.

    Callbacks are invoked by the Runner at key lifecycle points:
    - on_experiment_start: before the search loop begins
    - on_iteration_start: before each iteration
    - on_iteration_end: after each iteration (with candidates and results)
    - on_experiment_end: after the search loop finishes

    The should_stop property allows callbacks to signal early termination.
    """

    def on_experiment_start(self, algorithm: SearchAlgorithm) -> None:
        """Called before the search loop begins."""
        ...

    def on_iteration_start(self, iteration: int, algorithm: SearchAlgorithm) -> None:
        """Called before each iteration."""
        ...

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: SearchAlgorithm,
        candidates: list[str],
        results: list[EvaluationResult],
    ) -> None:
        """Called after each iteration with candidates and results."""
        ...

    def on_experiment_end(self, algorithm: SearchAlgorithm) -> None:
        """Called after the search loop finishes."""
        ...

    @property
    def should_stop(self) -> bool:
        """Whether the callback requests early termination."""
        ...


# LoggingCallback


class LoggingCallback:
    """Logs best score and expression at experiment lifecycle points.

    Args:
        every_n: Log iteration results every N iterations
            (0-indexed: logs at 0, N, 2N, ...). Must be >= 1.
    """

    def __init__(self, every_n: int = 1) -> None:
        if every_n < _MIN_EVERY_N:
            raise ValueError(f"every_n must be >= 1, got {every_n}")
        self._every_n = every_n

    @property
    def should_stop(self) -> bool:
        """LoggingCallback never requests stopping."""
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        """Log experiment start."""
        logger.info("Experiment started")

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        """No-op."""

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        """Log best score and expression if iteration aligns with every_n."""
        if iteration % self._every_n == 0:
            logger.info(
                "Iteration %d: best_score=%.6g, best_expression='%s'",
                iteration,
                algorithm.best_score,
                algorithm.best_expression,
            )

    def on_experiment_end(self, algorithm: Any) -> None:
        """Log final best score and expression."""
        logger.info(
            "Experiment ended: best_score=%.6g, best_expression='%s'",
            algorithm.best_score,
            algorithm.best_expression,
        )


class VizDataCollector:
    """Record per-iteration summary data for visualization."""

    def __init__(self, recorder: VizRecorder) -> None:
        self._recorder = recorder

    @property
    def recorder(self) -> VizRecorder:
        """Return the recorder used by this callback."""
        return self._recorder

    @property
    def should_stop(self) -> bool:
        """VizDataCollector never requests stopping."""
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        """No-op."""

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        """No-op."""

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        """Record best-so-far metrics for the current iteration."""
        self._recorder.log("_best_score", algorithm.best_score)
        self._recorder.log("_best_expr", algorithm.best_expression)
        self._recorder.log("_n_candidates", len(candidates))

    def on_experiment_end(self, algorithm: Any) -> None:
        """No-op."""


# EarlyStoppingCallback


def _initial_best(mode: Literal["min", "max"]) -> float:
    """Return the initial sentinel best value for a comparison mode."""
    if mode == "min":
        return float("inf")
    return float("-inf")


class EarlyStoppingCallback:
    """Patience-based early stopping.

    Monitors algorithm.best_score and stops if no improvement
    exceeding min_delta is observed for ``patience`` consecutive iterations.

    Args:
        patience: Number of stale iterations before stopping.
            patience=0 means stop immediately after any non-improving iteration.
        min_delta: Minimum improvement to reset the patience counter.
            Uses strict inequality (delta must be > min_delta).
        mode: "min" (lower is better) or "max" (higher is better).
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-6,
        mode: Literal["min", "max"] = "min",
    ) -> None:
        if patience < 0:
            raise ValueError(f"patience must be >= 0, got {patience}")
        if min_delta < 0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self._patience = patience
        self._min_delta = min_delta
        self._mode = mode
        self._best: float = _initial_best(mode)
        self._counter: int = 0
        self._should_stop: bool = False

    @property
    def should_stop(self) -> bool:
        """Whether patience has been exhausted."""
        return self._should_stop

    def on_experiment_start(self, algorithm: Any) -> None:
        """Reset all state for callback reuse across experiments."""
        self._best = _initial_best(self._mode)
        self._counter = 0
        self._should_stop = False

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        """No-op."""

    def on_experiment_end(self, algorithm: Any) -> None:
        """No-op."""

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        """Check for improvement and update patience counter."""
        score = algorithm.best_score

        # NaN never counts as improvement (explicit check per LEARNINGS rule)
        if math.isnan(score):
            self._counter += 1
        elif self._is_improvement(score):
            self._best = score
            self._counter = 0
        else:
            self._counter += 1

        # Only trigger stop when counter was incremented (counter > 0 guards
        # patience=0 from firing on improvement). Log only on transition.
        if (
            self._counter > 0
            and self._counter >= self._patience
            and not self._should_stop
        ):
            self._should_stop = True
            logger.info(
                "Early stopping at iteration %d (patience=%d)",
                iteration,
                self._patience,
            )

    def _is_improvement(self, score: float) -> bool:
        """Check if score improves on current best by more than min_delta.

        Uses strict inequality: improvement = (delta > min_delta).
        Rearranged algebraically to avoid inf - inf → NaN.

        Args:
            score: The new score to compare against self._best.

        Returns:
            True if score improves on self._best by more than min_delta.
        """
        if self._mode == "min":
            return score < self._best - self._min_delta
        # max mode
        return score > self._best + self._min_delta


# CheckpointCallback

_CHECKPOINT_VERSION = 1
_CHECKPOINT_PATTERN = "checkpoint_{iteration:06d}.pt"
_CHECKPOINT_FINAL = "checkpoint_final.pt"


class CheckpointCallback:
    """Periodic checkpoint saving via torch.save.

    Saves algorithm state to disk at regular intervals and at experiment end.

    Args:
        directory: Directory to save checkpoint files.
        every_n: Save every N iterations (0-indexed: saves at 0, N, 2N, ...).
            Must be >= 1.
    """

    def __init__(self, directory: Path, every_n: int = 10) -> None:
        if every_n < _MIN_EVERY_N:
            raise ValueError(f"every_n must be >= 1, got {every_n}")
        self._directory = Path(directory)
        self._every_n = every_n
        self._last_iteration: int = -1

    @property
    def should_stop(self) -> bool:
        """CheckpointCallback never requests stopping."""
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        """Create output directory and reset iteration counter."""
        self._directory.mkdir(parents=True, exist_ok=True)
        self._last_iteration = -1

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        """No-op."""

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        """Save checkpoint if iteration aligns with every_n."""
        self._last_iteration = iteration
        if iteration % self._every_n == 0:
            path = self._directory / _CHECKPOINT_PATTERN.format(iteration=iteration)
            torch.save(
                {
                    "version": _CHECKPOINT_VERSION,
                    "iteration": iteration,
                    "algorithm_state": algorithm.state,
                    "best_score": algorithm.best_score,
                    "best_expression": algorithm.best_expression,
                },
                path,
            )
            logger.debug("Saved checkpoint to %s", path)

    def on_experiment_end(self, algorithm: Any) -> None:
        """Save final checkpoint with algorithm state."""
        path = self._directory / _CHECKPOINT_FINAL
        iteration = max(self._last_iteration, 0)
        torch.save(
            {
                "version": _CHECKPOINT_VERSION,
                "iteration": iteration,
                "algorithm_state": algorithm.state,
                "best_score": algorithm.best_score,
                "best_expression": algorithm.best_expression,
            },
            path,
        )
        logger.debug("Saved final checkpoint to %s", path)
