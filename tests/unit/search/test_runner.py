"""Tests for ExperimentRunner core loop behavior.

Tests the main execution loop contract:
- prepare -> on_experiment_start -> loop -> on_experiment_end
- Iteration counting, batch_size passing, early stopping semantics
- RunResult correctness
- Edge cases: max_iterations=0, no callbacks, single iteration

Written in TDD red phase -- all tests fail until runner.py is implemented.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import pytest

from kd2.core.evaluator import EvaluationResult
from kd2.search.protocol import PlatformComponents
from kd2.search.runner import ExperimentRunner, RunResult
from tests.unit.search._runner_mocks import (
    ExplodingAlgorithm,
    IterativeRecordingAlgorithm,
    RecordingAlgorithm,
    RecordingCallback,
)

# ============================================================================
# Test Group 1: Smoke -- importability and construction
# ============================================================================


class TestRunnerSmoke:
    """Smoke tests: importability, construction, method existence."""

    @pytest.mark.smoke
    def test_runner_importable(self) -> None:
        """ExperimentRunner and RunResult are importable from kd2.search."""
        from kd2.search import ExperimentRunner as Runner
        from kd2.search import RunResult as Result

        assert Runner is not None
        assert Result is not None

    @pytest.mark.smoke
    def test_run_result_is_dataclass(self) -> None:
        """RunResult is a dataclass."""
        assert dataclasses.is_dataclass(RunResult)

    @pytest.mark.smoke
    def test_run_result_fields(self) -> None:
        """RunResult has exactly the 4 documented fields."""
        field_names = [f.name for f in dataclasses.fields(RunResult)]
        assert field_names == [
            "best_expression",
            "best_score",
            "iterations",
            "early_stopped",
        ]

    @pytest.mark.smoke
    def test_runner_construction(self, recording_algorithm: RecordingAlgorithm) -> None:
        """ExperimentRunner can be constructed with minimal args."""
        runner = ExperimentRunner(algorithm=recording_algorithm)
        assert runner is not None

    @pytest.mark.smoke
    def test_runner_has_run_method(
        self, recording_algorithm: RecordingAlgorithm
    ) -> None:
        """ExperimentRunner has a run() method."""
        runner = ExperimentRunner(algorithm=recording_algorithm)
        assert callable(getattr(runner, "run", None))

    @pytest.mark.smoke
    def test_runner_has_save_checkpoint(
        self, recording_algorithm: RecordingAlgorithm
    ) -> None:
        """ExperimentRunner has a save_checkpoint() method."""
        runner = ExperimentRunner(algorithm=recording_algorithm)
        assert callable(getattr(runner, "save_checkpoint", None))

    @pytest.mark.smoke
    def test_runner_has_load_checkpoint(
        self, recording_algorithm: RecordingAlgorithm
    ) -> None:
        """ExperimentRunner has a load_checkpoint() method."""
        runner = ExperimentRunner(algorithm=recording_algorithm)
        assert callable(getattr(runner, "load_checkpoint", None))

    @pytest.mark.smoke
    def test_runner_no_evaluator_in_init(self) -> None:
        """ExperimentRunner.__init__ does NOT accept 'evaluator' parameter.

        Per /016: Runner is a pure pipeline, no Evaluator ref.
        """
        import inspect

        sig = inspect.signature(ExperimentRunner.__init__)
        params = list(sig.parameters.keys())
        assert "evaluator" not in params, (
            f"Runner must NOT accept evaluator. Params: {params}"
        )


# ============================================================================
# Test Group 2: Core loop behavior
# ============================================================================


class TestRunnerCoreLoop:
    """Tests for the main run() execution loop."""

    @pytest.mark.unit
    def test_prepare_called_first(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """prepare(components) is the first algorithm call in run()."""
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        runner.run(mock_components)
        assert recording_algorithm.call_log[0] == "prepare"

    @pytest.mark.unit
    def test_loop_order_propose_evaluate_update(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """Each iteration follows propose -> evaluate -> update order."""
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=2)
        runner.run(mock_components)

        # Extract only algorithm method calls (not prepare)
        loop_calls = [c for c in recording_algorithm.call_log if c != "prepare"]
        # Should be [propose, evaluate, update, propose, evaluate, update]
        for i in range(2):
            offset = i * 3
            assert loop_calls[offset] == "propose"
            assert loop_calls[offset + 1] == "evaluate"
            assert loop_calls[offset + 2] == "update"

    @pytest.mark.unit
    def test_batch_size_forwarded_to_propose(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """batch_size is passed as n to algorithm.propose()."""
        batch_size = 7
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=3,
            batch_size=batch_size,
        )
        runner.run(mock_components)

        # Every propose call should receive batch_size
        assert all(n == batch_size for n in recording_algorithm.propose_args)
        assert len(recording_algorithm.propose_args) == 3

    @pytest.mark.unit
    def test_evaluate_results_passed_to_update(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """Results from evaluate() are passed directly to update()."""
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=1,
            batch_size=3,
        )
        runner.run(mock_components)

        # update was called once with 3 results (batch_size=3)
        assert len(recording_algorithm.update_args) == 1
        results = recording_algorithm.update_args[0]
        assert len(results) == 3
        assert all(isinstance(r, EvaluationResult) for r in results)

    @pytest.mark.unit
    def test_iterations_count_equals_max(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """When no early stop, iterations == max_iterations."""
        max_iter = 5
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=max_iter,
        )
        result = runner.run(mock_components)
        assert result.iterations == max_iter

    @pytest.mark.unit
    def test_run_returns_result(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """run() returns a result with expected fields."""
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "best_expression")
        assert hasattr(result, "best_score")
        assert hasattr(result, "iterations")
        assert hasattr(result, "early_stopped")

    @pytest.mark.unit
    def test_run_result_best_from_algorithm(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """RunResult's best_score/best_expression come from post-loop state.

        Uses a multi-element sequence to verify the runner reads properties
        AFTER all iterations complete, not at an arbitrary midpoint.
        """
        # M2 fix: multi-element sequence -- score changes across iterations
        algo = RecordingAlgorithm(
            score_sequence=[10.0, 5.0, 1.5],
            expression_sequence=["initial", "improving", "final_best"],
        )
        runner = ExperimentRunner(algorithm=algo, max_iterations=3)
        result = runner.run(mock_components)

        # After 3 iterations, algo._iteration==3, clamped to last element
        assert result.best_score == 1.5
        assert result.best_expression == "final_best"

    @pytest.mark.unit
    def test_not_early_stopped_when_full_run(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """early_stopped is False when loop runs to completion."""
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=3)
        result = runner.run(mock_components)
        assert result.early_stopped is False


# ============================================================================
# Test Group 3: Early stopping
# ============================================================================


class TestRunnerEarlyStopping:
    """Tests for early stopping via callback.should_stop."""

    @pytest.mark.unit
    def test_early_stop_sets_flag(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """early_stopped is True when a callback triggers should_stop."""
        cb = RecordingCallback(stop_at_iteration=1)
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=10,
            callbacks=[cb],
        )
        result = runner.run(mock_components)
        assert result.early_stopped is True

    @pytest.mark.unit
    def test_early_stop_iteration_count(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """iterations reflects actual count, not max_iterations."""
        cb = RecordingCallback(stop_at_iteration=2)
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=100,
            callbacks=[cb],
        )
        result = runner.run(mock_components)
        # stop_at_iteration=2 means should_stop becomes True after iteration 2
        # iterations 0, 1, 2 complete -> break -> 3 iterations completed
        assert result.iterations == 3
        assert result.early_stopped is True

    @pytest.mark.unit
    def test_early_stop_any_callback_triggers(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """any(cb.should_stop) triggers early stop -- not all()."""
        cb_patient = RecordingCallback(stop_at_iteration=None) # never stops
        cb_eager = RecordingCallback(stop_at_iteration=0) # stops immediately
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=10,
            callbacks=[cb_patient, cb_eager],
        )
        result = runner.run(mock_components)
        assert result.early_stopped is True
        # Only iteration 0 completed
        assert result.iterations == 1

    @pytest.mark.unit
    def test_early_stop_at_first_iteration(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """Callback can stop at the very first iteration (iteration 0).

        Verifies that on_iteration_end is called BEFORE the should_stop check,
        so the callback has a chance to record iteration_end:0 even when
        stopping immediately.
        """
        cb = RecordingCallback(stop_at_iteration=0)
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=50,
            callbacks=[cb],
        )
        result = runner.run(mock_components)
        assert result.iterations == 1
        assert result.early_stopped is True

        # M5 fix: verify iteration_end was called for iteration 0
        # (proves iteration_end fires before should_stop check)
        assert "iteration_end:0" in cb.events
        assert 0 in cb.iteration_ends


# ============================================================================
# Test Group 4: Callback lifecycle ordering
# ============================================================================


class TestRunnerCallbackLifecycle:
    """Tests for callback invocation ordering and argument passing."""

    @pytest.mark.unit
    def test_callback_lifecycle_order(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """Callbacks follow: start -> (iter_start, iter_end)* -> end."""
        cb = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            callbacks=[cb],
        )
        runner.run(mock_components)

        assert cb.events[0] == "experiment_start"
        assert cb.events[-1] == "experiment_end"

        # Inner events should alternate iteration_start/iteration_end
        inner = cb.events[1:-1]
        assert len(inner) == 4 # 2 iterations * 2 events each
        assert inner == [
            "iteration_start:0",
            "iteration_end:0",
            "iteration_start:1",
            "iteration_end:1",
        ]

    @pytest.mark.unit
    def test_multiple_callbacks_all_invoked(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """All callbacks in the list are invoked at each lifecycle point."""
        cb1 = RecordingCallback()
        cb2 = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=1,
            callbacks=[cb1, cb2],
        )
        runner.run(mock_components)

        # Both callbacks should have same event sequence
        assert cb1.events == cb2.events
        assert "experiment_start" in cb1.events
        assert "experiment_end" in cb1.events

    @pytest.mark.unit
    def test_iteration_start_before_propose(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """on_iteration_start is called BEFORE propose() in each iteration."""
        # Use a custom algo+callback pair that records interleaved events
        events: list[str] = []

        class _OrderAlgo(RecordingAlgorithm):
            def propose(self, n: int) -> list[str]:
                events.append("algo:propose")
                return super().propose(n)

            def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
                events.append("algo:evaluate")
                return super().evaluate(candidates)

            def update(self, results: list[EvaluationResult]) -> None:
                events.append("algo:update")
                return super().update(results)

        class _OrderCallback(RecordingCallback):
            def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
                events.append("cb:iteration_start")

            def on_iteration_end(
                self,
                iteration: int,
                algorithm: Any,
                candidates: list[str],
                results: list[Any],
            ) -> None:
                events.append("cb:iteration_end")

        algo = _OrderAlgo()
        cb = _OrderCallback()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1, callbacks=[cb])
        runner.run(mock_components)

        # Expected within one iteration:
        # cb:iter_start -> algo:propose -> evaluate -> update -> cb:iter_end
        assert events.index("cb:iteration_start") < events.index("algo:propose")
        assert events.index("algo:update") < events.index("cb:iteration_end")

    @pytest.mark.unit
    def test_iteration_end_receives_candidates_and_results(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """on_iteration_end gets the candidates and results from that iteration."""
        received_candidates: list[list[str]] = []
        received_results: list[list[EvaluationResult]] = []

        class _CapturingCallback(RecordingCallback):
            def on_iteration_end(
                self,
                iteration: int,
                algorithm: Any,
                candidates: list[str],
                results: list[Any],
            ) -> None:
                received_candidates.append(list(candidates))
                received_results.append(list(results))

        cb = _CapturingCallback()
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            batch_size=3,
            callbacks=[cb],
        )
        runner.run(mock_components)

        assert len(received_candidates) == 2
        assert len(received_results) == 2
        # Each iteration proposed 3 candidates
        for cands in received_candidates:
            assert len(cands) == 3
        for ress in received_results:
            assert len(ress) == 3
            assert all(isinstance(r, EvaluationResult) for r in ress)

    @pytest.mark.unit
    def test_no_callbacks_default(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """callbacks=None gives empty-list behavior, no errors."""
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            callbacks=None,
        )
        result = runner.run(mock_components)
        assert result.iterations == 2
        assert result.early_stopped is False


# ============================================================================
# Test Group 5: Edge cases
# ============================================================================


class TestRunnerEdgeCases:
    """Edge case tests for unusual but valid inputs."""

    @pytest.mark.unit
    def test_max_iterations_zero(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """max_iterations=0 should still call prepare + experiment_start/end."""
        cb = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=0,
            callbacks=[cb],
        )
        result = runner.run(mock_components)

        assert result.iterations == 0
        assert result.early_stopped is False
        assert "experiment_start" in cb.events
        assert "experiment_end" in cb.events
        # No iteration events
        assert not any(
            "iteration" in e
            for e in cb.events
            if e not in ("experiment_start", "experiment_end")
        )
        # M4 fix: verify prepare was still called even with 0 iterations
        assert "prepare" in recording_algorithm.call_log

    @pytest.mark.unit
    def test_single_iteration(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """max_iterations=1 executes exactly one loop pass."""
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert result.iterations == 1

        # Algorithm called: prepare, propose, evaluate, update
        assert recording_algorithm.call_log == [
            "prepare",
            "propose",
            "evaluate",
            "update",
        ]

    @pytest.mark.unit
    def test_batch_size_one(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """batch_size=1 proposes single candidate per iteration."""
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=3,
            batch_size=1,
        )
        runner.run(mock_components)
        assert all(n == 1 for n in recording_algorithm.propose_args)

    @pytest.mark.unit
    def test_empty_callbacks_list(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """callbacks=[] should behave same as callbacks=None."""
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            callbacks=[],
        )
        result = runner.run(mock_components)
        assert result.iterations == 2
        assert result.early_stopped is False

    @pytest.mark.unit
    def test_stale_iteration_after_prior_run(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """run() must not leak _current_iteration from a prior run.

        Codex review M1: if _current_iteration is stale from a previous
        run or load_checkpoint, and the new run has max_iterations=0,
        RunResult.iterations would report the stale value.
        """
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=0)
        # Simulate stale state from a prior run or load_checkpoint
        runner._current_iteration = 99

        result = runner.run(mock_components)
        assert result.iterations == 0


# ============================================================================
# Test Group 6: Exception safety (try/finally)
# ============================================================================


class TestRunnerExceptionSafety:
    """Tests that on_experiment_end is called even when exceptions occur."""

    @pytest.mark.unit
    def test_experiment_end_called_on_exception(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """on_experiment_end MUST be called even if algorithm.propose raises."""
        algo = ExplodingAlgorithm(explode_at=1)
        cb = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=5,
            callbacks=[cb],
        )

        with pytest.raises(RuntimeError, match="Algorithm exploded"):
            runner.run(mock_components)

        # Despite the exception, experiment_end must have been called
        assert "experiment_end" in cb.events

    @pytest.mark.unit
    def test_experiment_end_called_on_first_iteration_exception(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """on_experiment_end called even if the very first iteration explodes."""
        algo = ExplodingAlgorithm(explode_at=0)
        cb = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=5,
            callbacks=[cb],
        )

        with pytest.raises(RuntimeError, match="Algorithm exploded"):
            runner.run(mock_components)

        assert "experiment_start" in cb.events
        assert "experiment_end" in cb.events

    @pytest.mark.unit
    def test_all_callbacks_get_experiment_end_on_exception(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """All callbacks receive on_experiment_end even on exception."""
        algo = ExplodingAlgorithm(explode_at=0)
        cb1 = RecordingCallback()
        cb2 = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=5,
            callbacks=[cb1, cb2],
        )

        with pytest.raises(RuntimeError):
            runner.run(mock_components)

        assert "experiment_end" in cb1.events
        assert "experiment_end" in cb2.events

    @pytest.mark.unit
    def test_exception_propagates_after_cleanup(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """The original exception is re-raised after on_experiment_end."""
        algo = ExplodingAlgorithm(explode_at=0)
        runner = ExperimentRunner(
            algorithm=algo,
            max_iterations=5,
            callbacks=[RecordingCallback()],
        )

        with pytest.raises(RuntimeError, match="Algorithm exploded"):
            runner.run(mock_components)


# ============================================================================
# Test Group 7: RunResult dataclass correctness
# ============================================================================


class TestRunResult:
    """Tests for RunResult dataclass itself."""

    @pytest.mark.unit
    def test_run_result_construction(self) -> None:
        """RunResult can be constructed with positional or keyword args."""
        r = RunResult(
            best_expression="mul(u, u_x)",
            best_score=0.001,
            iterations=42,
            early_stopped=True,
        )
        assert r.best_expression == "mul(u, u_x)"
        assert r.best_score == 0.001
        assert r.iterations == 42
        assert r.early_stopped is True

    @pytest.mark.unit
    def test_run_result_equality(self) -> None:
        """Two RunResults with same values are equal (dataclass default)."""
        r1 = RunResult("e", 0.5, 10, False)
        r2 = RunResult("e", 0.5, 10, False)
        assert r1 == r2

    @pytest.mark.unit
    def test_run_result_inequality(self) -> None:
        """Different RunResults are not equal."""
        r1 = RunResult("e", 0.5, 10, False)
        r2 = RunResult("e", 0.5, 10, True)
        assert r1 != r2


# ============================================================================
# Test Group 8: ExperimentResult return type (P4-T1b)
# ============================================================================


class TestRunnerExperimentResult:
    """Tests for Runner.run() returning ExperimentResult instead of RunResult."""

    @pytest.mark.smoke
    def test_experiment_result_importable(self) -> None:
        """ExperimentResult can be imported from kd2.search.result."""
        from kd2.search.result import ExperimentResult

        assert ExperimentResult is not None

    @pytest.mark.unit
    def test_run_returns_experiment_result(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """run() returns an ExperimentResult (not RunResult)."""
        from kd2.search.result import ExperimentResult

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert isinstance(result, ExperimentResult)

    @pytest.mark.unit
    def test_experiment_result_has_final_eval(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """ExperimentResult has a final_eval field (EvaluationResult)."""
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "final_eval")
        assert isinstance(result.final_eval, EvaluationResult)

    @pytest.mark.unit
    def test_experiment_result_has_actual_tensor(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """ExperimentResult has an actual field (Tensor)."""
        import torch

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "actual")
        assert isinstance(result.actual, torch.Tensor)

    @pytest.mark.unit
    def test_experiment_result_has_predicted_tensor(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """ExperimentResult has a predicted field (Tensor)."""
        import torch

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "predicted")
        assert isinstance(result.predicted, torch.Tensor)

    @pytest.mark.unit
    def test_experiment_result_has_recorder(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """ExperimentResult has a recorder field (VizRecorder)."""
        from kd2.search.recorder import VizRecorder

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "recorder")
        assert isinstance(result.recorder, VizRecorder)

    @pytest.mark.unit
    def test_experiment_result_has_config_dict(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """ExperimentResult has a config field (dict)."""
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)
        assert hasattr(result, "config")
        assert isinstance(result.config, dict)

    @pytest.mark.unit
    def test_experiment_result_preserves_best_from_algorithm(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """ExperimentResult best_score/best_expression come from algorithm.

        Same behavioral contract as RunResult: values are read after
        all iterations complete.
        """
        algo = RecordingAlgorithm(
            score_sequence=[10.0, 5.0, 1.5],
            expression_sequence=["initial", "improving", "final_best"],
        )
        runner = ExperimentRunner(algorithm=algo, max_iterations=3)
        result = runner.run(mock_components)

        assert result.best_score == 1.5
        assert result.best_expression == "final_best"


# ============================================================================
# Test Group 9: VizDataCollector auto-injection (P4-T1b)
# ============================================================================


class TestRunnerVizDataCollectorInjection:
    """Tests for automatic VizDataCollector injection in Runner.run()."""

    @pytest.mark.unit
    def test_recorder_has_best_score_after_run(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """After run(), recorder has _best_score data from VizDataCollector."""
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=3)
        result = runner.run(mock_components)

        scores = result.recorder.get("_best_score")
        # One entry per iteration
        assert len(scores) == 3

    @pytest.mark.unit
    def test_recorder_has_best_expr_after_run(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """After run(), recorder has _best_expr data."""
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=2)
        result = runner.run(mock_components)

        exprs = result.recorder.get("_best_expr")
        assert len(exprs) == 2
        # All entries should be non-empty strings
        assert all(isinstance(e, str) for e in exprs)

    @pytest.mark.unit
    def test_recorder_has_n_candidates_after_run(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """After run(), recorder has _n_candidates data."""
        batch_size = 7
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            batch_size=batch_size,
        )
        result = runner.run(mock_components)

        n_cands = result.recorder.get("_n_candidates")
        assert len(n_cands) == 2
        # Each iteration should report batch_size candidates
        assert all(n == batch_size for n in n_cands)

    @pytest.mark.unit
    def test_run_without_recorder_creates_default(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """When components.recorder is None, Runner creates a default recorder.

        The returned ExperimentResult still has a valid recorder with data.
        """
        from kd2.search.recorder import VizRecorder

        # Ensure components does NOT have recorder
        assert mock_components.recorder is None

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=2)
        result = runner.run(mock_components)

        assert isinstance(result.recorder, VizRecorder)
        # Collector still injected, so data should be present
        assert len(result.recorder.get("_best_score")) == 2

    @pytest.mark.unit
    def test_run_with_explicit_recorder(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """When components.recorder is provided, Runner uses it.

        The same VizRecorder instance is returned in ExperimentResult.
        """
        from kd2.search.recorder import VizRecorder

        explicit_recorder = VizRecorder()
        mock_components.recorder = explicit_recorder

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=2)
        result = runner.run(mock_components)

        # Same instance
        assert result.recorder is explicit_recorder
        # Still has data from auto-injected VizDataCollector
        assert len(result.recorder.get("_best_score")) == 2

    @pytest.mark.unit
    def test_user_callbacks_still_invoked(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """User-provided callbacks are still invoked alongside VizDataCollector."""
        cb = RecordingCallback()
        runner = ExperimentRunner(
            algorithm=recording_algorithm,
            max_iterations=2,
            callbacks=[cb],
        )
        result = runner.run(mock_components)

        # User callback ran
        assert "experiment_start" in cb.events
        assert "experiment_end" in cb.events
        # VizDataCollector also ran
        assert len(result.recorder.get("_best_score")) == 2


# ============================================================================
# Test Group 10: ResultBuilder vs Evaluator fallback (P4-T1b)
# ============================================================================


class TestRunnerResultBuilderFallback:
    """Tests for the two end-of-run enrichment paths.

    Path 1: If algorithm implements ResultBuilder, call build_final_result().
    Path 2: Otherwise, fall back to evaluator.evaluate_expression().
    """

    @pytest.mark.unit
    def test_result_builder_path(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """When algorithm implements ResultBuilder, build_final_result() is used.

        Verifies that the final_eval comes from the algorithm, not the evaluator.
        """
        import torch

        from kd2.search.result import ResultBuilder

        final_eval = EvaluationResult(
            mse=0.001,
            nmse=0.001,
            r2=0.999,
            aic=-50.0,
            residuals=torch.zeros(10),
        )

        class _BuilderAlgorithm(RecordingAlgorithm):
            """Algorithm that implements ResultBuilder."""

            @property
            def config(self) -> dict:
                return {"algorithm": "builder_test"}

            def build_final_result(self) -> EvaluationResult:
                return final_eval

        algo = _BuilderAlgorithm()
        assert isinstance(algo, ResultBuilder)

        # Supply a real lhs_target matching residuals.shape so the
        # runner's shape guard (M3) passes for this fallback path.
        mock_components.evaluator.lhs_target = torch.zeros(10)

        runner = ExperimentRunner(algorithm=algo, max_iterations=1)
        result = runner.run(mock_components)

        # final_eval should come from build_final_result, not evaluator
        assert result.final_eval.mse == final_eval.mse
        assert result.final_eval.aic == final_eval.aic

    @pytest.mark.unit
    def test_evaluator_fallback_path(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """When algorithm does NOT implement ResultBuilder, evaluator is used.

        The evaluator.evaluate_expression() should be called with
        algorithm.best_expression.
        """
        import torch

        from kd2.search.result import ResultBuilder

        # Recording algorithm does NOT implement build_final_result
        assert not isinstance(recording_algorithm, ResultBuilder)

        # Set up evaluator mock to return a specific result
        fallback_eval = EvaluationResult(
            mse=0.5,
            nmse=0.5,
            r2=0.5,
            residuals=torch.zeros(10),
        )
        mock_components.evaluator.evaluate_expression.return_value = fallback_eval
        mock_components.evaluator.lhs_target = torch.randn(10)

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)

        # Evaluator was called
        mock_components.evaluator.evaluate_expression.assert_called_once()
        assert result.final_eval.mse == fallback_eval.mse

    @pytest.mark.unit
    def test_predicted_derived_from_actual_and_residuals(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """predicted = actual + residuals (sign convention check).

        Evaluator residuals = predicted - actual (evaluator.py:279).
        So actual + residuals = actual + (predicted - actual) = predicted.
        """
        import torch

        actual_data = torch.tensor([1.0, 2.0, 3.0])
        residuals = torch.tensor([0.1, -0.2, 0.3])
        expected_predicted = actual_data + residuals

        fallback_eval = EvaluationResult(
            mse=0.1,
            nmse=0.1,
            r2=0.9,
            residuals=residuals,
        )
        mock_components.evaluator.evaluate_expression.return_value = fallback_eval
        mock_components.evaluator.lhs_target = actual_data

        algo = RecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1)
        result = runner.run(mock_components)

        torch.testing.assert_close(
            result.predicted, expected_predicted, rtol=1e-7, atol=1e-10
        )

    @pytest.mark.unit
    def test_dataset_name_from_components(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """ExperimentResult.dataset_name comes from components.dataset.name."""
        import torch

        mock_components.dataset.name = "test_burgers"
        fallback_eval = EvaluationResult(
            mse=0.1,
            nmse=0.1,
            r2=0.9,
            residuals=torch.zeros(10),
        )
        mock_components.evaluator.evaluate_expression.return_value = fallback_eval
        mock_components.evaluator.lhs_target = torch.randn(10)

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)

        assert result.dataset_name == "test_burgers"

    @pytest.mark.unit
    def test_algorithm_name_from_class(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """ExperimentResult.algorithm_name is type(algorithm).__name__."""
        import torch

        fallback_eval = EvaluationResult(
            mse=0.1,
            nmse=0.1,
            r2=0.9,
            residuals=torch.zeros(10),
        )
        mock_components.evaluator.evaluate_expression.return_value = fallback_eval
        mock_components.evaluator.lhs_target = torch.randn(10)

        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=1)
        result = runner.run(mock_components)

        assert result.algorithm_name == "RecordingAlgorithm"


# ============================================================================
# Test Group 11: IterativeSearchAlgorithm — between_iterations ()
# ============================================================================


class TestRunnerBetweenIterations:
    """Tests for IterativeSearchAlgorithm.between_iterations() in the run loop.

    The Runner should call between_iterations() after _run_iteration() and
    after the should_stop check, but NOT on the last iteration and NOT
    when early stopping is triggered.

    Expected loop structure:
        for iteration in range(max_iterations):
            _run_iteration(iteration, callbacks)
            _current_iteration = iteration + 1
            if any(cb.should_stop for cb in callbacks):
                break
            if isinstance(algorithm, IterativeSearchAlgorithm):
                algorithm.between_iterations()
    """

    @pytest.mark.unit
    def test_plain_algorithm_no_between_iterations(
        self,
        recording_algorithm: RecordingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """Plain SearchAlgorithm does NOT trigger between_iterations.

        RecordingAlgorithm has no between_iterations method, so
        the Runner must not attempt to call it.
        """
        runner = ExperimentRunner(algorithm=recording_algorithm, max_iterations=3)
        runner.run(mock_components)

        # No between_iterations in call_log
        assert "between_iterations" not in recording_algorithm.call_log

    @pytest.mark.unit
    def test_iterative_algorithm_between_iterations_called(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """IterativeSearchAlgorithm triggers between_iterations in the loop."""
        algo = IterativeRecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=3)
        runner.run(mock_components)

        assert algo.between_iterations_count > 0
        assert "between_iterations" in algo.call_log

    @pytest.mark.unit
    def test_between_iterations_count(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """between_iterations called (iterations - 1) times.

        Not called on the last iteration because there is no next
        iteration to prepare for.
        """
        max_iter = 5
        algo = IterativeRecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=max_iter)
        runner.run(mock_components)

        assert algo.between_iterations_count == max_iter - 1

    @pytest.mark.unit
    def test_between_iterations_not_called_on_early_stop(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """between_iterations NOT called when should_stop triggers.

        When a callback requests early stopping, the loop breaks
        BEFORE between_iterations can run for that iteration.
        """
        algo = IterativeRecordingAlgorithm()
        # Stop at iteration 0 — should_stop becomes True after first iteration
        cb = RecordingCallback(stop_at_iteration=0)
        runner = ExperimentRunner(algorithm=algo, max_iterations=10, callbacks=[cb])
        result = runner.run(mock_components)

        assert result.early_stopped is True
        assert result.iterations == 1
        # between_iterations never called: early stop at iteration 0,
        # no next iteration
        assert algo.between_iterations_count == 0

    @pytest.mark.unit
    def test_between_iterations_not_called_after_early_stop_mid_run(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """between_iterations called for completed iterations, not after stop.

        If early stop triggers at iteration 2 (out of 10):
        - iterations 0, 1: between_iterations called (2 calls)
        - iteration 2: should_stop triggers, break, no between_iterations
        Total: 2 calls (not 3).
        """
        algo = IterativeRecordingAlgorithm()
        cb = RecordingCallback(stop_at_iteration=2)
        runner = ExperimentRunner(algorithm=algo, max_iterations=10, callbacks=[cb])
        result = runner.run(mock_components)

        assert result.early_stopped is True
        assert result.iterations == 3 # iterations 0, 1, 2 completed
        # between_iterations called after iterations 0 and 1, NOT after 2
        assert algo.between_iterations_count == 2

    @pytest.mark.unit
    def test_between_iterations_not_called_single_iteration(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """between_iterations NOT called when max_iterations=1.

        Single iteration = last iteration, so no between_iterations.
        """
        algo = IterativeRecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1)
        runner.run(mock_components)

        assert algo.between_iterations_count == 0

    @pytest.mark.unit
    def test_between_iterations_not_called_zero_iterations(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """between_iterations NOT called when max_iterations=0."""
        algo = IterativeRecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=0)
        runner.run(mock_components)

        assert algo.between_iterations_count == 0

    @pytest.mark.unit
    def test_between_iterations_after_on_iteration_end(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """between_iterations called AFTER on_iteration_end callbacks.

        This ordering is critical: callbacks record pure search state
        first, then algorithm does inter-iteration work (e.g. PINN training).
        """
        global_events: list[str] = []

        class _OrderTrackingAlgo(IterativeRecordingAlgorithm):
            def between_iterations(self) -> None:
                global_events.append("between_iterations")
                super().between_iterations()

        class _OrderTrackingCallback(RecordingCallback):
            def on_iteration_end(
                self,
                iteration: int,
                algorithm: Any,
                candidates: list[str],
                results: list[Any],
            ) -> None:
                global_events.append(f"on_iteration_end:{iteration}")
                super().on_iteration_end(iteration, algorithm, candidates, results)

        algo = _OrderTrackingAlgo()
        cb = _OrderTrackingCallback()
        runner = ExperimentRunner(algorithm=algo, max_iterations=2, callbacks=[cb])
        runner.run(mock_components)

        # For iteration 0 (not the last), both should fire in order:
        # on_iteration_end:0 BEFORE between_iterations
        end_idx = global_events.index("on_iteration_end:0")
        between_idx = global_events.index("between_iterations")
        assert end_idx < between_idx, (
            f"on_iteration_end must precede between_iterations, "
            f"got events: {global_events}"
        )

    @pytest.mark.unit
    def test_between_iterations_loop_ordering(
        self,
        mock_components: PlatformComponents,
    ) -> None:
        """Verify full interleaving: propose/evaluate/update then between.

        For 3 iterations, call_log should show:
        prepare, [propose, evaluate, update, between_iterations] * 2,
        propose, evaluate, update
        (last iteration has no between_iterations)
        """
        algo = IterativeRecordingAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=3)
        runner.run(mock_components)

        log = algo.call_log
        assert log[0] == "prepare"

        # Iterations 0 and 1: propose, evaluate, update, between_iterations
        for i in range(2):
            base = 1 + i * 4
            assert log[base] == "propose", f"iter {i}: expected propose at {base}"
            assert log[base + 1] == "evaluate", f"iter {i}: expected evaluate"
            assert log[base + 2] == "update", f"iter {i}: expected update"
            assert log[base + 3] == "between_iterations", (
                f"iter {i}: expected between_iterations"
            )

        # Last iteration (2): propose, evaluate, update — NO between_iterations
        last_base = 1 + 2 * 4
        assert log[last_base] == "propose"
        assert log[last_base + 1] == "evaluate"
        assert log[last_base + 2] == "update"
        assert len(log) == last_base + 3 # no trailing between_iterations


# ============================================================================
# Test Group: F5 — Runner enforces algorithm.evaluate() batch contract
# ============================================================================


class _MismatchedLengthAlgorithm:
    """Algorithm whose ``evaluate`` returns one fewer result than candidates.

    Mimics a misbehaving plugin that drops a candidate silently — the
    Runner must now raise instead of forwarding the broken pair to update().
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"c_{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        # BAD: drop the last candidate's result.
        return [EvaluationResult(mse=1.0, nmse=1.0, r2=0.0) for _ in candidates[:-1]]

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 1.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "_MismatchedLengthAlgorithm"}

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = value


class _WrongTypeAlgorithm:
    """Algorithm whose ``evaluate`` returns dicts instead of EvaluationResult.

    The Runner must reject the wrong type explicitly — silent forwarding
    pollutes update() and result aggregation with mis-shaped data.
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"c_{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[Any]:
        # BAD: dicts instead of EvaluationResult instances.
        return [{"mse": 1.0} for _ in candidates]

    def update(self, results: list[Any]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 1.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "_WrongTypeAlgorithm"}

    @property
    def state(self) -> dict[str, Any]:
        return self._state

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = value


class TestRunnerEnforcesBatchContract:
    """Runner must validate the propose/evaluate length + type contract.

    F5: ``runner._run_iteration`` previously trusted the plugin to return
    ``len(candidates)`` ``EvaluationResult``-typed entries. A mis-counted
    or mis-typed result silently propagated to ``update()`` and the
    callback chain, corrupting downstream aggregation.
    """

    @pytest.mark.unit
    def test_run_raises_on_length_mismatch(
        self, mock_components: PlatformComponents
    ) -> None:
        """``run()`` raises RuntimeError when evaluate returns fewer results."""
        algo = _MismatchedLengthAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1, batch_size=4)
        with pytest.raises(RuntimeError, match=r"\b3\b.*\b4\b|evaluate"):
            runner.run(mock_components)

    @pytest.mark.unit
    def test_run_raises_on_wrong_result_type(
        self, mock_components: PlatformComponents
    ) -> None:
        """``run()`` raises TypeError when evaluate returns wrong-typed entries."""
        algo = _WrongTypeAlgorithm()
        runner = ExperimentRunner(algorithm=algo, max_iterations=1, batch_size=4)
        with pytest.raises(TypeError, match="EvaluationResult|dict"):
            runner.run(mock_components)
