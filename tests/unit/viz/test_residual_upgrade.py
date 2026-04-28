"""Tests for residual plot Tier 2 upgrade.

Tests the NEW Tier 2 signature of plot_residual:
  plot_residual(result, *, style=None, field_shape=None) -> tuple[Figure, list[str]]

The old Tier 1 signature (result, ax) -> list[str] is tested in test_plots.py.
These tests will fail until the upgrade is implemented.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from matplotlib.figure import Figure

from kd2.core.evaluator import EvaluationResult
from kd2.search.recorder import VizRecorder
from kd2.search.result import ExperimentResult
from kd2.viz.plots.residual import plot_residual

# Helpers


def _make_result(n_samples: int = 50) -> ExperimentResult:
    """Minimal ExperimentResult with residuals."""
    actual = torch.randn(n_samples)
    predicted = actual + torch.randn(n_samples) * 0.1
    residuals = predicted - actual
    recorder = VizRecorder()
    recorder.log("_best_score", 1.0)
    recorder.log("_best_expr", "u")
    recorder.log("_n_candidates", 10)
    return ExperimentResult(
        best_expression="u",
        best_score=1.0,
        iterations=1,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.005,
            r2=0.95,
            aic=-100.0,
            complexity=1,
            coefficients=torch.tensor([1.0]),
            is_valid=True,
            error_message="",
            selected_indices=[0],
            residuals=residuals,
            terms=["u"],
            expression="u",
        ),
        actual=actual,
        predicted=predicted,
        dataset_name="test",
        algorithm_name="SGA",
        config={},
        recorder=recorder,
    )


# ===========================================================================
# Smoke: new Tier 2 signature
# ===========================================================================


class TestResidualTier2Smoke:
    """Verify the new Tier 2 signature works."""

    def test_new_signature_returns_figure_tuple(self) -> None:
        """Tier 2 call: plot_residual(result) -> (Figure, list[str]).

        The new signature should NOT require an ax argument.
        It creates its own Figure internally.
        """
        result = _make_result()
        # NEW Tier 2 call: no ax argument, returns (Figure, list[str])
        out = plot_residual(result)
        try:
            assert isinstance(out, tuple), f"Expected tuple, got {type(out)}"
            assert len(out) == 2
            fig, warnings = out
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            if isinstance(out, tuple) and isinstance(out[0], Figure):
                plt.close(out[0])

    def test_accepts_style_kwarg(self) -> None:
        """style keyword should be accepted."""
        result = _make_result()
        fig, _ = plot_residual(result, style={"font.size": 12})
        plt.close(fig)

    def test_accepts_field_shape_kwarg(self) -> None:
        """field_shape keyword should be accepted."""
        result = _make_result(n_samples=100)
        fig, _ = plot_residual(result, field_shape=(10, 10))
        plt.close(fig)


# ===========================================================================
# Happy path: Tier 2 features
# ===========================================================================


class TestResidualTier2HappyPath:
    """Core Tier 2 features: creates figure, has histogram."""

    def test_figure_has_histogram(self) -> None:
        """Tier 2 figure should contain histogram patches."""
        result = _make_result()
        fig, _ = plot_residual(result)
        try:
            # Check for histogram patches in any axes
            has_patches = any(len(ax.patches) > 0 for ax in fig.get_axes())
            assert has_patches, "Expected histogram patches in figure"
        finally:
            plt.close(fig)

    def test_figure_has_stats_annotation(self) -> None:
        """Should annotate mean/std on the figure."""
        result = _make_result()
        fig, _ = plot_residual(result)
        try:
            all_texts = []
            for ax in fig.get_axes():
                all_texts.extend(t.get_text().lower() for t in ax.texts)
            text_joined = " ".join(all_texts)
            assert "mean" in text_joined or "std" in text_joined
        finally:
            plt.close(fig)

    def test_figure_has_labels(self) -> None:
        """Axes should have x/y labels."""
        result = _make_result()
        fig, _ = plot_residual(result)
        try:
            has_labels = any(
                ax.get_xlabel() != "" and ax.get_ylabel() != "" for ax in fig.get_axes()
            )
            assert has_labels
        finally:
            plt.close(fig)


# ===========================================================================
# Edge cases (>= 20%)
# ===========================================================================


class TestResidualTier2EdgeCases:
    """Edge cases for the Tier 2 residual plot."""

    def test_none_residuals_warns(self) -> None:
        """None residuals -> warning, not crash."""
        result = _make_result()
        result.final_eval.residuals = None
        fig, warnings = plot_residual(result)
        try:
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_all_nan_residuals(self) -> None:
        """All-NaN residuals -> warning, not crash."""
        result = _make_result()
        result.final_eval.residuals = torch.full((10,), float("nan"))
        fig, warnings = plot_residual(result)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_inf_residuals(self) -> None:
        """Inf in residuals -> filtered with warning."""
        result = _make_result(n_samples=20)
        result.final_eval.residuals[0] = float("inf")
        fig, warnings = plot_residual(result)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_no_figure_leak(self) -> None:
        """Should not leak matplotlib figures."""
        result = _make_result()
        figs_before = len(plt.get_fignums())
        fig, _ = plot_residual(result)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before
