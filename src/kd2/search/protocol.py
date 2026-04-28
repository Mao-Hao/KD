"""Protocol definitions for search algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kd2.core.evaluator import EvaluationResult, Evaluator
    from kd2.core.executor.context import ExecutionContext
    from kd2.core.expr.executor import PythonExecutor
    from kd2.core.expr.registry import FunctionRegistry
    from kd2.data.schema import PDEDataset
    from kd2.search.recorder import VizRecorder


@dataclass
class PlatformComponents:
    """Container for platform services injected into search algorithms.

    Attributes:
        dataset: PDE dataset with fields and derivatives.
        executor: Expression executor.
        evaluator: Theta builder + solver combiner.
        context: Execution context with variables and derivatives.
        registry: Function/operator registry.
        recorder: Optional visualization recorder.
    """

    dataset: PDEDataset
    executor: PythonExecutor
    evaluator: Evaluator
    context: ExecutionContext
    registry: FunctionRegistry
    recorder: VizRecorder | None = None


@runtime_checkable
class SearchAlgorithm(Protocol):
    """Protocol for search algorithms used by the platform runner.

    Search algorithms own evaluation logic. The runner calls a
    ``prepare -> (propose -> evaluate -> update)*`` loop.
    """

    def prepare(self, components: PlatformComponents) -> None:
        """Initialize algorithm with platform components."""
        ...

    def propose(self, n: int) -> list[str]:
        """Propose candidate expressions.

        ``n`` is a batch-size request for ordinary algorithms. Algorithms with
        staged internal evaluation may return their full evaluated frontier.
        """
        ...

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        """Evaluate candidate expressions."""
        ...

    def update(self, results: list[EvaluationResult]) -> None:
        """Update internal state from evaluation results."""
        ...

    @property
    def best_score(self) -> float:
        """Return the best score found so far."""
        ...

    @property
    def best_expression(self) -> str:
        """Return the best expression found so far."""
        ...

    @property
    def config(self) -> dict[str, Any]:
        """Return a JSON-safe configuration dictionary."""
        ...

    @property
    def state(self) -> dict[str, Any]:
        """Algorithm state for checkpointing (must be pickle-serializable)."""
        ...

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        """Restore algorithm state."""
        ...


@runtime_checkable
class IterativeSearchAlgorithm(SearchAlgorithm, Protocol):
    """Optional extension for algorithms needing inter-iteration work.

    Algorithms implementing this protocol will have between_iterations()
    called by the Runner after each iteration (except the last one and
    when early stopping triggers). Use cases include R-DISCOVER's PINN
    embedding step.
    """

    def between_iterations(self) -> None:
        """Perform inter-iteration work (e.g., PINN refinement)."""
        ...
