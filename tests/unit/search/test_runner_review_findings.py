"""Unit tests for ExperimentRunner review findings (M2, M3).

TDD red phase: tests describe the expected fail-fast behavior of
``Runner._actual`` and ``Runner._predicted`` when plugins violate the
``ResultTargetProvider`` contract.

- M2: ``ResultTargetProvider.build_result_target()`` is typed
  ``-> Tensor``. Today, when a plugin returns a non-Tensor (e.g.,
  ``None`` due to a bug), ``_actual`` silently falls through to
  ``evaluator.lhs_target`` — masking the contract violation.
  Expected: raise TypeError with a clear message naming the plugin
  type and the actual returned type.

- M3: ``_predicted`` returns ``actual + final_eval.residuals`` without
  any shape guard. Plugin contract violation (residuals shape !=
  actual shape) crashes deep inside torch broadcasting. Expected:
  raise ValueError mentioning both shapes.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
import torch
from torch import Tensor

from kd2.core.evaluator import EvaluationResult
from kd2.search.protocol import PlatformComponents
from kd2.search.result import ResultTargetProvider
from kd2.search.runner import ExperimentRunner

# ============================================================================
# Mock algorithm: implements ResultTargetProvider but violates the contract
# ============================================================================


class _BadProviderAlgorithm:
    """SearchAlgorithm + ResultTargetProvider that returns a non-Tensor.

    The Protocol declares ``build_result_target() -> Tensor``. Returning
    None mimics a real-world plugin bug. The runner should detect the
    violation and raise rather than silently falling through.
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"e{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return [EvaluationResult(mse=1.0, nmse=1.0, r2=0.0) for _ in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    def build_result_target(self) -> Tensor: # type: ignore[return]
        # Contract violation: returns None despite -> Tensor annotation.
        # Used to assert the runner raises rather than masks the bug.
        return None # type: ignore[return-value]

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return "e0"

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "BadProvider"}

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = dict(value)


class _ShapeMismatchProviderAlgorithm:
    """Plugin returning a target whose shape mismatches final_eval.residuals.

    Used to trigger the ``_predicted`` shape-guard test: ``actual +
    residuals`` would broadcast or crash with a cryptic torch error
    when shapes are incompatible. The runner should fail fast with an
    informative message.
    """

    def __init__(
        self,
        target_shape: tuple[int, ...],
        residual_shape: tuple[int, ...],
    ) -> None:
        self._target_shape = target_shape
        self._residual_shape = residual_shape
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"e{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        # Inject mismatched residuals shape into the EvaluationResult
        residuals = torch.zeros(self._residual_shape)
        return [
            EvaluationResult(
                mse=1.0,
                nmse=1.0,
                r2=0.0,
                residuals=residuals,
            )
            for _ in candidates
        ]

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    def build_result_target(self) -> Tensor:
        return torch.zeros(self._target_shape)

    def build_final_result(self) -> EvaluationResult:
        # Provide residuals of the wrong shape so _predicted hits the
        # broadcasting trap.
        return EvaluationResult(
            mse=1.0,
            nmse=1.0,
            r2=0.0,
            aic=0.0,
            complexity=1,
            coefficients=torch.tensor([1.0]),
            is_valid=True,
            error_message="",
            selected_indices=[0],
            residuals=torch.zeros(self._residual_shape),
            terms=["u"],
            expression="e0",
        )

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return "e0"

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "ShapeMismatch"}

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = dict(value)


# ============================================================================
# M2: build_result_target contract violation must raise, not fall through
# ============================================================================


@pytest.mark.unit
class TestM2BuildResultTargetContract:
    """``Runner._actual`` must raise TypeError when ResultTargetProvider
    violates the ``-> Tensor`` contract by returning a non-Tensor value.

    Today the silent fallback to ``evaluator.lhs_target`` masks bugs in
    plugins. The runner should fail fast so the bug surfaces during
    development rather than producing subtly wrong reports.
    """

    def test_bad_provider_returns_none_raises_typeerror(self) -> None:
        algorithm = _BadProviderAlgorithm()
        # Sanity: BadProvider does implement the runtime-checkable protocol.
        assert isinstance(algorithm, ResultTargetProvider)

        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        # evaluator.lhs_target is a MagicMock (not a Tensor) by default,
        # so we set it to a real tensor to prove the runner does NOT
        # silently fall through to it. The bug we're catching is the
        # ResultTargetProvider returning a non-Tensor.
        components.evaluator.lhs_target = torch.zeros(8)

        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)
        with pytest.raises(TypeError) as exc_info:
            runner.run(components)

        # Error must clearly identify what went wrong.
        msg = str(exc_info.value).lower()
        assert "build_result_target" in msg or "resulttargetprovider" in msg, (
            f"Expected TypeError to mention the protocol method/name, got: "
            f"{exc_info.value}"
        )

    def test_bad_provider_does_not_silently_use_evaluator_target(self) -> None:
        """The fallback path must NOT be exercised when a buggy provider
        returns a non-Tensor. Otherwise the report ships with the wrong
        ``actual`` and the user has no way to detect it.
        """
        algorithm = _BadProviderAlgorithm()
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        # Distinct sentinel so we'd notice it leaking into the result.
        components.evaluator.lhs_target = torch.full((8,), 12345.0)
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises((TypeError, ValueError)):
            runner.run(components)


# ============================================================================
# M3: _predicted must guard against residual/actual shape mismatch
# ============================================================================


@pytest.mark.unit
class TestM3PredictedShapeGuard:
    """``Runner._predicted`` must validate that
    ``actual.shape == final_eval.residuals.shape`` before computing
    ``actual + residuals``. Without the guard, broadcasting either
    (a) silently produces a wrong result, or (b) crashes with a
    cryptic message that doesn't name the offending shapes.
    """

    def test_residual_shape_mismatch_raises_valueerror(self) -> None:
        algorithm = _ShapeMismatchProviderAlgorithm(
            target_shape=(10,),
            residual_shape=(8,),
        )
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        components.evaluator.lhs_target = torch.zeros(10)
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(ValueError) as exc_info:
            runner.run(components)

        msg = str(exc_info.value)
        # Both shapes should appear in the error so the user knows where
        # the contract was broken.
        assert "10" in msg, (
            f"Error message should mention the actual shape (10), got: {msg}"
        )
        assert "8" in msg, (
            f"Error message should mention the residual shape (8), got: {msg}"
        )

    def test_residual_shape_mismatch_2d_raises_valueerror(self) -> None:
        """Multi-dim shape mismatch should also be guarded with a clear
        message. Today actual=(4,5) + residuals=(5,4) silently broadcasts
        to (5,5) — wrong but does not raise.
        """
        algorithm = _ShapeMismatchProviderAlgorithm(
            target_shape=(4, 5),
            residual_shape=(5, 4),
        )
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        components.evaluator.lhs_target = torch.zeros(4, 5)
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(ValueError) as exc_info:
            runner.run(components)
        msg = str(exc_info.value)
        # Shape tuples should appear in the diagnostic.
        assert "(4, 5)" in msg or "[4, 5]" in msg or "4, 5" in msg, (
            f"Error must mention actual shape, got: {msg}"
        )
        assert "(5, 4)" in msg or "[5, 4]" in msg or "5, 4" in msg, (
            f"Error must mention residual shape, got: {msg}"
        )


# ============================================================================
# Round-2: _actual must reject non-Tensor evaluator.lhs_target (Issue D/F)
# ============================================================================


class _PlainAlgorithm:
    """SearchAlgorithm without ResultTargetProvider — exercises the
    evaluator.lhs_target path so we can verify it's strictly type-checked.
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"e{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return [EvaluationResult(mse=1.0, nmse=1.0, r2=0.0) for _ in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return "e0"

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "Plain"}

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = dict(value)


@pytest.mark.unit
class TestActualEvaluatorTargetStrict:
    """``Runner._actual`` must raise TypeError when ``evaluator.lhs_target``
    is not a Tensor and the algorithm does not implement
    ``ResultTargetProvider`` either. Round-1 left a silent ``zeros(0)``
    fallback that masked plugin-side bugs; round-2 removes it.
    """

    def test_non_tensor_lhs_target_raises_typeerror(self) -> None:
        algorithm = _PlainAlgorithm()
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        # Default MagicMock — explicitly not a Tensor.
        components.evaluator.lhs_target = MagicMock()
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(TypeError) as exc_info:
            runner.run(components)
        msg = str(exc_info.value).lower()
        assert "lhs_target" in msg, (
            f"Error must name the offending attribute, got: {exc_info.value}"
        )

    def test_none_lhs_target_raises_typeerror(self) -> None:
        algorithm = _PlainAlgorithm()
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        components.evaluator.lhs_target = None
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(TypeError):
            runner.run(components)


# ============================================================================
# Round-2: _predicted must reject non-Tensor residuals (Issue E)
# ============================================================================


class _BadResidualsAlgorithm:
    """Algorithm whose final eval has residuals of the wrong type.

    Mimics a buggy plugin that stuffs a list / np.ndarray / None into
    ``EvaluationResult.residuals`` despite the field being typed
    ``Tensor``. The runner should fail fast with a clear TypeError.
    """

    def __init__(self, residuals: Any, target_shape: tuple[int, ...]) -> None:
        self._residuals = residuals
        self._target_shape = target_shape
        self._state: dict[str, Any] = {}

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return [f"e{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return [EvaluationResult(mse=1.0, nmse=1.0, r2=0.0) for _ in candidates]

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    def build_result_target(self) -> Tensor:
        return torch.zeros(self._target_shape)

    def build_final_result(self) -> EvaluationResult:
        return EvaluationResult(
            mse=1.0,
            nmse=1.0,
            r2=0.0,
            aic=0.0,
            complexity=1,
            coefficients=torch.tensor([1.0]),
            is_valid=True,
            error_message="",
            selected_indices=[0],
            # Inject the bad residuals — bypasses the dataclass typing.
            residuals=self._residuals,
            terms=["u"],
            expression="e0",
        )

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return "e0"

    @property
    def config(self) -> dict[str, Any]:
        return {"algorithm": "BadResiduals"}

    @property
    def state(self) -> dict[str, Any]:
        return dict(self._state)

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        self._state = dict(value)


@pytest.mark.unit
class TestPredictedResidualsTypeGuard:
    """``Runner._predicted`` must reject non-Tensor residuals before the
    shape compare. A list / np.ndarray could otherwise pass through
    ``actual + residuals`` and crash with a cryptic torch error.
    """

    def test_list_residuals_raises_typeerror(self) -> None:
        import numpy as np

        algorithm = _BadResidualsAlgorithm(
            residuals=[0.0, 0.0, 0.0],
            target_shape=(3,),
        )
        components = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        components.evaluator.lhs_target = torch.zeros(3)
        runner = ExperimentRunner(algorithm=algorithm, max_iterations=1)

        with pytest.raises(TypeError) as exc_info:
            runner.run(components)
        msg = str(exc_info.value).lower()
        assert "residuals" in msg, (
            f"Error must name the offending field, got: {exc_info.value}"
        )

        # numpy.ndarray case — same contract.
        algorithm2 = _BadResidualsAlgorithm(
            residuals=np.zeros(3),
            target_shape=(3,),
        )
        runner2 = ExperimentRunner(algorithm=algorithm2, max_iterations=1)
        with pytest.raises(TypeError):
            runner2.run(components)
