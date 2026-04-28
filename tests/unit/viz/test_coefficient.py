"""Tests for coefficient bar chart plot.

Tests plot_coefficient_bar (Tier 1): visualize fitted coefficients with
optional ground truth overlay.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from kd2.core.evaluator import EvaluationResult
from kd2.search.recorder import VizRecorder
from kd2.search.result import ExperimentResult
from kd2.viz.plots.coefficient import plot_coefficient_bar

# Helpers


def _make_result_with_coefficients(
    coeffs: list[float],
    terms: list[str] | None = None,
    selected_indices: list[int] | None = None,
) -> ExperimentResult:
    """Create ExperimentResult with specific coefficients and terms."""
    n_terms = len(coeffs)
    if terms is None:
        terms = [f"term_{i}" for i in range(n_terms)]
    if selected_indices is None:
        selected_indices = list(range(n_terms))
    n_samples = 50
    actual = torch.randn(n_samples)
    predicted = actual + torch.randn(n_samples) * 0.1
    recorder = VizRecorder()
    recorder.log("_best_score", 1.0)
    recorder.log("_best_expr", "dummy")
    recorder.log("_n_candidates", 10)
    return ExperimentResult(
        best_expression="dummy",
        best_score=1.0,
        iterations=1,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.005,
            r2=0.95,
            aic=-100.0,
            complexity=n_terms,
            coefficients=torch.tensor(coeffs),
            is_valid=True,
            error_message="",
            selected_indices=selected_indices,
            residuals=predicted - actual,
            terms=terms,
            expression="dummy",
        ),
        actual=actual,
        predicted=predicted,
        dataset_name="test",
        algorithm_name="SGA",
        config={},
        recorder=recorder,
    )


# ===========================================================================
# Smoke tests
# ===========================================================================


class TestCoefficientBarSmoke:
    """Basic callable/return-type tests."""

    def test_callable_returns_list(self) -> None:
        """plot_coefficient_bar exists and returns list[str]."""
        result = _make_result_with_coefficients([1.0, -0.5, 0.3])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert isinstance(warnings, list)
            assert all(isinstance(w, str) for w in warnings)
        finally:
            plt.close(fig)

    def test_accepts_ground_truth_kwarg(self) -> None:
        """ground_truth keyword is accepted without error."""
        result = _make_result_with_coefficients([1.0, -0.5])
        gt = torch.tensor([1.1, -0.6])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax, ground_truth=gt)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)


# ===========================================================================
# Happy path
# ===========================================================================


class TestCoefficientBarHappyPath:
    """Core logic: bars drawn, labels present, ground truth overlay."""

    def test_has_bars_for_each_coefficient(self) -> None:
        """Should draw one bar per coefficient (or two if ground truth)."""
        coeffs = [1.0, -0.5, 0.3, 0.0]
        result = _make_result_with_coefficients(coeffs)
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            # At least as many patches (bars) as coefficients
            assert len(ax.patches) >= len(coeffs)
        finally:
            plt.close(fig)

    def test_bars_include_negative_values(self) -> None:
        """Negative coefficients should produce bars below zero."""
        coeffs = [2.0, -3.0]
        result = _make_result_with_coefficients(coeffs)
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            # Check that ylim includes negative territory
            ymin, ymax = ax.get_ylim()
            assert ymin < 0, "y-axis should extend below zero for negative coefficients"
        finally:
            plt.close(fig)

    def test_term_labels_on_xaxis(self) -> None:
        """X-axis tick labels should correspond to term names."""
        terms = ["u", "u_x", "u_xx"]
        result = _make_result_with_coefficients([1.0, -0.5, 0.3], terms=terms)
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            tick_labels = [t.get_text() for t in ax.get_xticklabels()]
            # At least one term name should appear in the tick labels
            found = any(term in label for label in tick_labels for term in terms)
            assert found, f"Expected term labels in {tick_labels}"
        finally:
            plt.close(fig)

    def test_ground_truth_adds_visual_elements(self) -> None:
        """With ground_truth, should have more bars/markers than without."""
        coeffs = [1.0, -0.5]
        result = _make_result_with_coefficients(coeffs)

        fig1, ax1 = plt.subplots()
        fig2, ax2 = plt.subplots()
        try:
            plot_coefficient_bar(result, ax1)
            plot_coefficient_bar(result, ax2, ground_truth=torch.tensor([1.1, -0.6]))
            # Ground truth version should have more visual elements
            # (more patches, or lines, or both)
            elems_without = len(ax1.patches) + len(ax1.get_lines())
            elems_with = len(ax2.patches) + len(ax2.get_lines())
            assert elems_with > elems_without, (
                f"Ground truth should add visual elements: {elems_without} vs {elems_with}"
            )
        finally:
            plt.close(fig1)
            plt.close(fig2)

    def test_title_is_set(self) -> None:
        """Plot should have a non-empty title."""
        result = _make_result_with_coefficients([1.0])
        fig, ax = plt.subplots()
        try:
            plot_coefficient_bar(result, ax)
            assert ax.get_title() != ""
        finally:
            plt.close(fig)


# ===========================================================================
# Edge cases and negative tests (>= 20%)
# ===========================================================================


class TestCoefficientBarEdgeCases:
    """Edge cases: None coefficients, NaN, empty, mismatched shapes."""

    def test_none_coefficients_warns(self) -> None:
        """result.final_eval.coefficients is None -> warning, no crash."""
        result = _make_result_with_coefficients([1.0])
        result.final_eval.coefficients = None
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_none_terms_warns(self) -> None:
        """result.final_eval.terms is None -> warning or fallback labels."""
        result = _make_result_with_coefficients([1.0, -0.5])
        result.final_eval.terms = None
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            # Should not crash
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_nan_coefficient_handled(self) -> None:
        """NaN in coefficients should not crash the plot."""
        result = _make_result_with_coefficients([1.0, float("nan"), 0.5])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_inf_coefficient_handled(self) -> None:
        """Inf in coefficients should not crash the plot."""
        result = _make_result_with_coefficients([1.0, float("inf"), 0.5])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_empty_coefficients(self) -> None:
        """Zero-length coefficients tensor -> warning, no crash."""
        result = _make_result_with_coefficients([])
        result.final_eval.coefficients = torch.tensor([])
        result.final_eval.terms = []
        result.final_eval.selected_indices = []
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_ground_truth_length_mismatch(self) -> None:
        """ground_truth length != coefficients length -> warning."""
        result = _make_result_with_coefficients([1.0, -0.5])
        gt = torch.tensor([1.0]) # Wrong length
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax, ground_truth=gt)
            assert any(
                "mismatch" in w.lower() or "length" in w.lower() for w in warnings
            )
        finally:
            plt.close(fig)

    def test_single_coefficient(self) -> None:
        """Single coefficient should produce one bar."""
        result = _make_result_with_coefficients([42.0])
        fig, ax = plt.subplots()
        try:
            warnings = plot_coefficient_bar(result, ax)
            assert len(ax.patches) >= 1
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_no_figure_leak(self) -> None:
        """Should not leak matplotlib figures."""
        result = _make_result_with_coefficients([1.0, -0.5])
        figs_before = len(plt.get_fignums())
        fig, ax = plt.subplots()
        plot_coefficient_bar(result, ax)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before
