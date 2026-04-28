"""Tests for ExperimentResult + ResultBuilder protocol.

Covers:
- ExperimentResult construction and field access
- to_dict / save / load round-trip
- load() preserves recorder data
- ResultBuilder runtime_checkable protocol
- Negative: invalid path, missing fields
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import Tensor

from kd2.core.evaluator import EvaluationResult
from kd2.search.recorder import VizRecorder
from kd2.search.result import ExperimentResult, ResultBuilder

# Fixtures


@pytest.fixture
def sample_eval_result() -> EvaluationResult:
    """A valid EvaluationResult for testing."""
    return EvaluationResult(
        mse=0.01,
        nmse=0.02,
        r2=0.98,
        aic=-50.0,
        complexity=3,
        coefficients=torch.tensor([1.0, -6.0, 1.0]),
        is_valid=True,
        selected_indices=[0, 1, 2],
        residuals=torch.randn(100),
        terms=["u", "mul(u, u_x)", "u_xx"],
        expression="add(u, add(mul(u, u_x), u_xx))",
    )


@pytest.fixture
def sample_recorder() -> VizRecorder:
    """A VizRecorder with some logged data."""
    rec = VizRecorder()
    rec.log("loss", 0.5)
    rec.log("loss", 0.3)
    rec.log("loss", 0.1)
    rec.log("iteration", 1)
    rec.log("iteration", 2)
    rec.log("iteration", 3)
    return rec


@pytest.fixture
def sample_experiment_result(
    sample_eval_result: EvaluationResult,
    sample_recorder: VizRecorder,
) -> ExperimentResult:
    """A complete ExperimentResult for testing."""
    return ExperimentResult(
        best_expression="add(u, add(mul(u, u_x), u_xx))",
        best_score=0.02,
        iterations=50,
        early_stopped=False,
        final_eval=sample_eval_result,
        actual=torch.randn(100),
        predicted=torch.randn(100),
        dataset_name="burgers_1d",
        algorithm_name="sga",
        config={"max_iter": 100, "threshold": 0.1},
        recorder=sample_recorder,
    )


# Smoke


@pytest.mark.smoke
class TestExperimentResultSmoke:
    """ExperimentResult and ResultBuilder exist and are usable."""

    def test_instantiate(self, sample_experiment_result: ExperimentResult) -> None:
        assert isinstance(sample_experiment_result, ExperimentResult)

    def test_result_builder_is_protocol(self) -> None:
        assert hasattr(ResultBuilder, "build_final_result")


# Unit — construction and field access


class TestExperimentResultFields:
    """Field access and type checks."""

    def test_run_result_fields(
        self, sample_experiment_result: ExperimentResult
    ) -> None:
        r = sample_experiment_result
        assert r.best_expression == "add(u, add(mul(u, u_x), u_xx))"
        assert r.best_score == pytest.approx(0.02)
        assert r.iterations == 50
        assert r.early_stopped is False

    def test_final_eval_accessible(
        self, sample_experiment_result: ExperimentResult
    ) -> None:
        r = sample_experiment_result
        assert r.final_eval.is_valid is True
        assert r.final_eval.r2 == pytest.approx(0.98)
        assert r.final_eval.complexity == 3

    def test_tensor_fields(self, sample_experiment_result: ExperimentResult) -> None:
        r = sample_experiment_result
        assert isinstance(r.actual, Tensor)
        assert isinstance(r.predicted, Tensor)
        assert r.actual.shape == r.predicted.shape

    def test_metadata_fields(self, sample_experiment_result: ExperimentResult) -> None:
        r = sample_experiment_result
        assert r.dataset_name == "burgers_1d"
        assert r.algorithm_name == "sga"
        assert "max_iter" in r.config

    def test_recorder_field(self, sample_experiment_result: ExperimentResult) -> None:
        r = sample_experiment_result
        assert isinstance(r.recorder, VizRecorder)
        assert r.recorder.get("loss") == [0.5, 0.3, 0.1]


# Unit — serialization round-trip


class TestExperimentResultSerialization:
    """to_dict / save / load contract."""

    def test_to_dict_contains_required_keys(
        self, sample_experiment_result: ExperimentResult
    ) -> None:
        d = sample_experiment_result.to_dict()
        assert isinstance(d, dict)
        for key in [
            "best_expression",
            "best_score",
            "iterations",
            "early_stopped",
            "dataset_name",
            "algorithm_name",
            "config",
            "final_eval",
            "actual",
            "predicted",
            "recorder",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_is_json_safe(
        self, sample_experiment_result: ExperimentResult
    ) -> None:
        """to_dict() must produce JSON-serializable output."""
        import json

        d = sample_experiment_result.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_save_load_round_trip(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "result.pt"
        sample_experiment_result.save(fpath)
        assert fpath.exists()

        loaded = ExperimentResult.load(fpath)

        # Scalar fields
        assert loaded.best_expression == sample_experiment_result.best_expression
        assert loaded.best_score == pytest.approx(sample_experiment_result.best_score)
        assert loaded.iterations == sample_experiment_result.iterations
        assert loaded.early_stopped == sample_experiment_result.early_stopped
        assert loaded.dataset_name == sample_experiment_result.dataset_name
        assert loaded.algorithm_name == sample_experiment_result.algorithm_name

    def test_save_load_preserves_tensors(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "result.pt"
        sample_experiment_result.save(fpath)
        loaded = ExperimentResult.load(fpath)

        torch.testing.assert_close(
            loaded.actual,
            sample_experiment_result.actual,
            rtol=1e-5,
            atol=1e-8,
        )
        torch.testing.assert_close(
            loaded.predicted,
            sample_experiment_result.predicted,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_save_load_preserves_final_eval(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "result.pt"
        sample_experiment_result.save(fpath)
        loaded = ExperimentResult.load(fpath)

        orig = sample_experiment_result.final_eval
        fe = loaded.final_eval
        # All scalar fields
        assert fe.mse == pytest.approx(orig.mse)
        assert fe.nmse == pytest.approx(orig.nmse)
        assert fe.r2 == pytest.approx(orig.r2)
        assert fe.aic == pytest.approx(orig.aic)
        assert fe.complexity == orig.complexity
        assert fe.is_valid == orig.is_valid
        assert fe.error_message == orig.error_message
        assert fe.expression == orig.expression
        # List fields
        assert fe.selected_indices == orig.selected_indices
        assert fe.terms == orig.terms
        # Tensor fields
        assert fe.coefficients is not None
        torch.testing.assert_close(fe.coefficients, orig.coefficients)
        assert fe.residuals is not None
        torch.testing.assert_close(fe.residuals, orig.residuals)

    def test_save_load_preserves_recorder(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        """recorder.get() works after load — key behavioral requirement."""
        fpath = tmp_path / "result.pt"
        sample_experiment_result.save(fpath)
        loaded = ExperimentResult.load(fpath)

        assert isinstance(loaded.recorder, VizRecorder)
        assert loaded.recorder.get("loss") == [0.5, 0.3, 0.1]
        assert loaded.recorder.keys() == {"loss", "iteration"}

    def test_save_creates_parent_dirs(
        self, sample_experiment_result: ExperimentResult, tmp_path: Path
    ) -> None:
        fpath = tmp_path / "nested" / "deep" / "result.pt"
        sample_experiment_result.save(fpath)
        assert fpath.exists()


# Unit — ResultBuilder protocol


class TestResultBuilderProtocol:
    """ResultBuilder is runtime_checkable and verifiable."""

    def test_conforming_class_passes_isinstance(self) -> None:
        class MyBuilder:
            def build_final_result(self) -> EvaluationResult:
                return EvaluationResult(mse=0.0, nmse=0.0, r2=1.0)

        assert isinstance(MyBuilder(), ResultBuilder)

    def test_non_conforming_class_fails_isinstance(self) -> None:
        class NotABuilder:
            def some_other_method(self) -> None:
                pass

        assert not isinstance(NotABuilder(), ResultBuilder)

    def test_protocol_not_instantiable_directly(self) -> None:
        """Protocols should not be instantiated — they are structural."""
        with pytest.raises(TypeError):
            ResultBuilder() # type: ignore[call-arg]


# Negative — error handling


@pytest.mark.numerical
class TestExperimentResultNegative:
    """Error cases and edge conditions."""

    def test_load_nonexistent_path_raises(self, tmp_path: Path) -> None:
        with pytest.raises((FileNotFoundError, OSError)):
            ExperimentResult.load(tmp_path / "does_not_exist.pt")

    def test_early_stopped_true(self, sample_eval_result: EvaluationResult) -> None:
        """Verify early_stopped=True is preserved correctly."""
        r = ExperimentResult(
            best_expression="u",
            best_score=1.0,
            iterations=5,
            early_stopped=True,
            final_eval=sample_eval_result,
            actual=torch.tensor([1.0]),
            predicted=torch.tensor([1.0]),
            dataset_name="test",
            algorithm_name="test",
            config={},
            recorder=VizRecorder(),
        )
        assert r.early_stopped is True

    def test_empty_recorder_survives_round_trip(
        self, sample_eval_result: EvaluationResult, tmp_path: Path
    ) -> None:
        """ExperimentResult with empty recorder saves/loads correctly."""
        r = ExperimentResult(
            best_expression="u_xx",
            best_score=0.5,
            iterations=10,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(20),
            predicted=torch.randn(20),
            dataset_name="heat",
            algorithm_name="discover",
            config={"alpha": 0.01},
            recorder=VizRecorder(),
        )
        fpath = tmp_path / "empty_rec.pt"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)
        assert loaded.recorder.keys() == set()
        assert loaded.recorder.get("anything") == []

    def test_save_with_inf_best_score(
        self, sample_eval_result: EvaluationResult, tmp_path: Path
    ) -> None:
        """ExperimentResult with inf best_score (SGA initial value) saves OK.

        F4 fix: ``best_score`` is typed ``float`` on ``RunResult``, so the
        load path must not return ``None`` (would violate the type
        contract). Non-finite floats collapse to NaN on load — RFC 8259
        cannot represent inf/NaN, so the lossy mapping is documented.
        """
        import math

        r = ExperimentResult(
            best_expression="",
            best_score=float("inf"),
            iterations=0,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
        )
        fpath = tmp_path / "inf_score.json"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)
        # inf collapses to NaN on load (lossy but type-stable; RFC 8259 has
        # no representation for inf/NaN).
        assert isinstance(loaded.best_score, float), (
            "loaded.best_score must remain a float to satisfy "
            "RunResult.best_score: float"
        )
        assert math.isnan(loaded.best_score), (
            "non-finite best_score must collapse to NaN on load (was "
            "previously returning None which violates float type)"
        )

    def test_save_with_nan_aic(self, tmp_path: Path) -> None:
        """EvaluationResult with NaN/Inf aic survives save/load.

        F4 fix: ``aic`` is ``Optional[float]`` so None on load remains
        legal here, but the lossy mapping is documented.
        """
        eval_result = EvaluationResult(
            mse=0.01,
            nmse=0.02,
            r2=0.98,
            aic=float("-inf"),
        )
        r = ExperimentResult(
            best_expression="u",
            best_score=0.02,
            iterations=10,
            early_stopped=False,
            final_eval=eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
        )
        fpath = tmp_path / "nan_aic.json"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)
        assert loaded.final_eval.aic is None

    def test_load_preserves_float_type_for_best_score_with_nan(
        self, sample_eval_result: EvaluationResult, tmp_path: Path
    ) -> None:
        """F4 lock: NaN best_score round-trips as NaN float, not None.

        The dataclass declares ``best_score: float``; if save sanitizes
        NaN to None and load forwards None, the loaded ExperimentResult
        violates its own type annotation and downstream code that does
        ``isnan(r.best_score)`` raises TypeError.
        """
        import math

        r = ExperimentResult(
            best_expression="",
            best_score=float("nan"),
            iterations=0,
            early_stopped=False,
            final_eval=sample_eval_result,
            actual=torch.randn(10),
            predicted=torch.randn(10),
            dataset_name="test",
            algorithm_name="sga",
            config={},
            recorder=VizRecorder(),
        )
        fpath = tmp_path / "nan_score.json"
        r.save(fpath)
        loaded = ExperimentResult.load(fpath)
        assert isinstance(loaded.best_score, float)
        assert math.isnan(loaded.best_score)
