"""Unit tests for render_comparison and comparison plot functions."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import torch
from kd2.viz.report import ReportResult

from kd2.core.evaluator import EvaluationResult
from kd2.search.recorder import VizRecorder
from kd2.search.result import ExperimentResult
from kd2.viz import VizEngine


def _make_result(
    name: str,
    r2: float = 0.9,
    nmse: float = 0.05,
    *,
    n_iterations: int = 5,
    has_recorder: bool = True,
) -> ExperimentResult:
    """Create a minimal ExperimentResult for comparison tests."""
    n_samples = 20
    recorder = VizRecorder()
    if has_recorder:
        for i in range(n_iterations):
            recorder.log("_best_score", 1.0 / (i + 1))
            recorder.log("_best_expr", f"expr_{i}")

    return ExperimentResult(
        best_expression=f"expr_{name}",
        best_score=nmse,
        iterations=n_iterations,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=nmse * 2,
            nmse=nmse,
            r2=r2,
            aic=-50.0,
            complexity=2,
            coefficients=torch.tensor([1.0, 0.5]),
            is_valid=True,
            error_message="",
            selected_indices=[0, 1],
            residuals=torch.randn(n_samples) * 0.1,
            terms=["u", "u_x"],
            expression=f"expr_{name}",
        ),
        actual=torch.randn(n_samples),
        predicted=torch.randn(n_samples),
        dataset_name="test",
        algorithm_name=name,
        config={"max_iter": n_iterations},
        recorder=recorder,
    )


class TestRenderComparison:
    """Tests for VizEngine.render_comparison."""

    def test_returns_report_result(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)
        assert isinstance(report, ReportResult)

    def test_creates_comparison_files(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)
        svg_files = list(tmp_path.glob("*.svg"))
        assert len(svg_files) >= 2 # convergence + boxplot + table

    def test_custom_labels(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results, labels=["Run 1", "Run 2"])
        assert isinstance(report, ReportResult)

    def test_all_figures_closed(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        figs_before = plt.get_fignums()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_comparison(results)
        figs_after = plt.get_fignums()
        assert len(figs_after) <= len(figs_before)

    def test_missing_recorder_data_warns(self, tmp_path: Path) -> None:
        """Results with empty recorders should produce warnings, not crash."""
        results = [
            _make_result("A", has_recorder=False),
            _make_result("B", has_recorder=False),
        ]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)
        assert isinstance(report, ReportResult)
        # Should warn about missing convergence data
        assert any(
            "convergence" in w.lower() or "skip" in w.lower() for w in report.warnings
        )

    def test_single_result(self, tmp_path: Path) -> None:
        """Should work with a single result (edge case)."""
        results = [_make_result("A")]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)
        assert isinstance(report, ReportResult)

    def test_figures_exist(self, tmp_path: Path) -> None:
        results = [_make_result("A"), _make_result("B")]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)
        for fig_path in report.figures:
            assert fig_path.exists(), f"Missing: {fig_path}"
