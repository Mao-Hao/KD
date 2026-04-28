"""Unit tests for individual plot functions."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from kd2.search.result import ExperimentResult
from kd2.viz.plots.comparison import (
    plot_score_bar,
    plot_summary_table,
    render_overlaid_convergence,
)
from kd2.viz.plots.convergence import plot_convergence
from kd2.viz.plots.equation import plot_equation
from kd2.viz.plots.parity import plot_parity
from kd2.viz.plots.residual import plot_residual

# ---- plot_convergence ----


class TestPlotConvergence:
    """Tests for convergence plot."""

    def test_renders_line(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_convergence(mock_experiment_result, ax)
        # Should have at least one line
        assert len(ax.get_lines()) >= 1
        plt.close(fig)

    def test_labels_set(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_convergence(mock_experiment_result, ax)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_empty_recorder_warns(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        """No _best_score data should still render without error."""
        from kd2.search.recorder import VizRecorder

        mock_experiment_result.recorder = VizRecorder()
        fig, ax = plt.subplots()
        warnings = plot_convergence(mock_experiment_result, ax)
        assert len(warnings) > 0
        plt.close(fig)


# ---- plot_parity ----


class TestPlotParity:
    """Tests for parity (actual vs predicted) plot."""

    def test_renders_scatter(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_parity(mock_experiment_result, ax)
        # Should have scatter collection
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_has_45_degree_line(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_parity(mock_experiment_result, ax)
        # The 45-degree reference line
        assert len(ax.get_lines()) >= 1
        plt.close(fig)

    def test_r2_annotation(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_parity(mock_experiment_result, ax)
        # Should have R2 text annotation
        texts = [t.get_text() for t in ax.texts]
        assert any("R" in t for t in texts)
        plt.close(fig)

    def test_labels_set(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_parity(mock_experiment_result, ax)
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)

    def test_nan_values_handled(self, mock_experiment_result: ExperimentResult) -> None:
        """NaN in actual/predicted should not crash 45-degree line."""
        mock_experiment_result.actual[0] = float("nan")
        mock_experiment_result.predicted[1] = float("nan")
        fig, ax = plt.subplots()
        warnings = plot_parity(mock_experiment_result, ax)
        # Should still render without error
        assert len(ax.collections) >= 1
        plt.close(fig)

    def test_constant_values(self, mock_experiment_result: ExperimentResult) -> None:
        """All-same values should add padding to 45-degree line."""
        mock_experiment_result.actual = torch.ones(10)
        mock_experiment_result.predicted = torch.ones(10)
        fig, ax = plt.subplots()
        plot_parity(mock_experiment_result, ax)
        # Should have a reference line (not zero-length)
        assert len(ax.get_lines()) >= 1
        plt.close(fig)


# ---- plot_residual ----


class TestPlotResidual:
    """Tests for residual histogram plot."""

    def test_renders_histogram(self, mock_experiment_result: ExperimentResult) -> None:
        fig, warnings = plot_residual(mock_experiment_result)
        # Should have histogram patches on the first (histogram) axes
        ax = fig.axes[0]
        assert len(ax.patches) >= 1
        plt.close(fig)

    def test_has_stats_annotation(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        fig, warnings = plot_residual(mock_experiment_result)
        ax = fig.axes[0]
        texts = [t.get_text() for t in ax.texts]
        # Should annotate mean and std
        assert any("mean" in t.lower() or "std" in t.lower() for t in texts)
        plt.close(fig)

    def test_no_residuals_warns(self, mock_experiment_result: ExperimentResult) -> None:
        """Missing residuals should produce warning."""
        mock_experiment_result.final_eval.residuals = None
        fig, warnings = plot_residual(mock_experiment_result)
        assert len(warnings) > 0
        plt.close(fig)

    def test_nan_residuals_filtered(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        """NaN residuals should be filtered with warning."""
        residuals = torch.randn(20)
        residuals[0] = float("nan")
        residuals[5] = float("inf")
        mock_experiment_result.final_eval.residuals = residuals
        fig, warnings = plot_residual(mock_experiment_result)
        assert any("non-finite" in w.lower() for w in warnings)
        # Should still render histogram with finite data on first axes
        ax = fig.axes[0]
        assert len(ax.patches) >= 1
        plt.close(fig)

    def test_all_nan_residuals(self, mock_experiment_result: ExperimentResult) -> None:
        """All-NaN residuals should produce warning, not crash."""
        mock_experiment_result.final_eval.residuals = torch.full((10,), float("nan"))
        fig, warnings = plot_residual(mock_experiment_result)
        assert any("non-finite" in w.lower() for w in warnings)
        plt.close(fig)

    def test_labels_set(self, mock_experiment_result: ExperimentResult) -> None:
        fig, warnings = plot_residual(mock_experiment_result)
        ax = fig.axes[0]
        assert ax.get_xlabel() != ""
        assert ax.get_ylabel() != ""
        plt.close(fig)


# ---- plot_equation ----


class TestPlotEquation:
    """Tests for equation display plot."""

    def test_renders_text(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_equation(mock_experiment_result, ax)
        # Should have text element with the expression
        assert len(ax.texts) >= 1
        plt.close(fig)

    def test_axis_off(self, mock_experiment_result: ExperimentResult) -> None:
        fig, ax = plt.subplots()
        plot_equation(mock_experiment_result, ax)
        # Axes frame should be invisible
        assert not ax.axison
        plt.close(fig)

    def test_empty_expression_warns(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        mock_experiment_result.best_expression = ""
        fig, ax = plt.subplots()
        warnings = plot_equation(mock_experiment_result, ax)
        assert len(warnings) > 0
        plt.close(fig)

    def test_renders_latex_not_raw_ir(
        self, mock_experiment_result: ExperimentResult
    ) -> None:
        """Equation plot should render LaTeX math, not raw IR funcall strings.
        The rendered text should be in $...$ math mode and should NOT
        contain raw function call syntax like 'add(' or 'mul('."""
        mock_experiment_result.best_expression = "add(u, add(u_x, u_xx))"
        fig, ax = plt.subplots()
        plot_equation(mock_experiment_result, ax)
        texts = [t.get_text() for t in ax.texts]
        assert len(texts) >= 1
        rendered = texts[0]
        # Should be wrapped in $...$ (math mode)
        assert rendered.strip().startswith("$")
        assert rendered.strip().endswith("$")
        # Should NOT contain raw IR function names inside the math text
        inner = rendered.strip("$ ")
        assert "add(" not in inner, f"Raw IR 'add(' found in rendered text: {rendered}"
        assert "mul(" not in inner, f"Raw IR 'mul(' found in rendered text: {rendered}"
        plt.close(fig)


# ---- plot_field_comparison ----
# Old tests removed: signature changed in M2b rewrite.
# New tests are in test_field.py.
