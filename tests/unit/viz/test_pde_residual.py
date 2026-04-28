"""Tests for PDE residual field with axis-aware rendering.

Tests the extended plot_pde_residual_field signature with dataset parameter:
  plot_pde_residual_field(result, *, dataset=None, ...)

When dataset is provided, rendering should use axis-aware display
(pcolormesh for 1D, heatmap at mid_t for 2D) instead of generic reshape.
"""

from __future__ import annotations

import math

import matplotlib
import matplotlib.pyplot as plt
import pytest
import torch

matplotlib.use("Agg")

from matplotlib.figure import Figure

from kd2.core.evaluator import EvaluationResult
from kd2.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd2.search.recorder import VizRecorder
from kd2.search.result import ExperimentResult
from kd2.viz.plots.pde_residual import plot_pde_residual_field

# Helpers

_TWO_PI = 2.0 * math.pi


def _make_1d_dataset(nx: int = 20, nt: int = 10) -> PDEDataset:
    """1D PDE dataset."""
    x = torch.linspace(0, _TWO_PI, nx)
    t = torch.linspace(0, 1, nt)
    u_field = torch.sin(x).unsqueeze(1) * torch.exp(-t).unsqueeze(0)
    return PDEDataset(
        name="test_1d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_2d_dataset(nx: int = 10, ny: int = 10, nt: int = 5) -> PDEDataset:
    """2D PDE dataset."""
    x = torch.linspace(0, _TWO_PI, nx)
    y = torch.linspace(0, _TWO_PI, ny)
    t = torch.linspace(0, 1, nt)
    u_field = (
        torch.sin(x).reshape(nx, 1, 1)
        * torch.cos(y).reshape(1, ny, 1)
        * torch.exp(-t).reshape(1, 1, nt)
    )
    return PDEDataset(
        name="test_2d",
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "y": AxisInfo(name="y", values=y, is_periodic=True),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "y", "t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_result(n_samples: int = 50) -> ExperimentResult:
    """Minimal ExperimentResult."""
    actual = torch.randn(n_samples)
    predicted = actual + torch.randn(n_samples) * 0.1
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
            residuals=predicted - actual,
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
# Smoke: dataset parameter accepted
# ===========================================================================


class TestPdeResidualDatasetSmoke:
    """Verify dataset parameter is accepted."""

    def test_accepts_dataset_kwarg(self) -> None:
        """dataset keyword should be accepted without error."""
        ds = _make_1d_dataset(nx=10, nt=5)
        # n_samples must match flattened field size for meaningful test
        result = _make_result(n_samples=10 * 5)
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        try:
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_dataset_none_backward_compat(self) -> None:
        """dataset=None (default) should work as before."""
        result = _make_result(n_samples=100)
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 10))
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


# ===========================================================================
# Axis-aware rendering: 1D dataset
# ===========================================================================


class TestPdeResidualAxisAware1D:
    """When dataset is provided with 1D spatial, should use pcolormesh."""

    def test_1d_with_dataset_uses_pcolormesh(self) -> None:
        """1D dataset -> pcolormesh rendering (collections present)."""
        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)
        fig, _ = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        try:
            # pcolormesh creates QuadMesh in ax.collections
            has_collections = any(len(ax.collections) > 0 for ax in fig.get_axes())
            assert has_collections, "1D dataset should trigger pcolormesh (collections)"
        finally:
            plt.close(fig)

    def test_1d_without_dataset_uses_generic(self) -> None:
        """Without dataset, should use generic rendering (imshow or line)."""
        result = _make_result(n_samples=100)
        fig, _ = plot_pde_residual_field(result, field_shape=(10, 10))
        try:
            # Generic path uses imshow (images) or line fallback
            has_images_or_lines = any(
                len(ax.images) > 0 or len(ax.get_lines()) > 0 for ax in fig.get_axes()
            )
            assert has_images_or_lines
        finally:
            plt.close(fig)

    def test_1d_rendering_differs_from_no_dataset(self) -> None:
        """Rendering WITH dataset should produce different visual elements
        than rendering WITHOUT dataset (axis-aware vs generic)."""
        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)

        fig_with, _ = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        fig_without, _ = plot_pde_residual_field(result, field_shape=(10, 5))
        try:
            # Count collections (pcolormesh) in each
            cols_with = sum(len(ax.collections) for ax in fig_with.get_axes())
            cols_without = sum(len(ax.collections) for ax in fig_without.get_axes())
            # With dataset should have more collections (pcolormesh)
            assert cols_with > cols_without, (
                f"Axis-aware should use pcolormesh: "
                f"with={cols_with}, without={cols_without}"
            )
        finally:
            plt.close(fig_with)
            plt.close(fig_without)


# ===========================================================================
# Axis-aware rendering: 2D dataset
# ===========================================================================


class TestPdeResidualAxisAware2D:
    """When dataset is provided with 2D spatial, should use heatmap at mid_t."""

    def test_2d_with_dataset_produces_heatmaps(self) -> None:
        """2D dataset -> heatmap at mid_t (imshow present)."""
        ds = _make_2d_dataset(nx=5, ny=5, nt=4)
        result = _make_result(n_samples=5 * 5 * 4)
        fig, _ = plot_pde_residual_field(result, field_shape=(5, 5, 4), dataset=ds)
        try:
            has_images = any(len(ax.images) > 0 for ax in fig.get_axes())
            assert has_images, "2D dataset should trigger heatmap (imshow)"
        finally:
            plt.close(fig)

    def test_2d_axes_have_titles(self) -> None:
        """Heatmap panels should have descriptive titles."""
        ds = _make_2d_dataset(nx=5, ny=5, nt=4)
        result = _make_result(n_samples=5 * 5 * 4)
        fig, _ = plot_pde_residual_field(result, field_shape=(5, 5, 4), dataset=ds)
        try:
            titles = [ax.get_title() for ax in fig.get_axes() if ax.get_title()]
            assert len(titles) >= 1, "Expected at least one panel title"
        finally:
            plt.close(fig)


# ===========================================================================
# Edge cases (>= 20%)
# ===========================================================================


class TestPdeResidualDatasetEdgeCases:
    """Edge cases for dataset-enhanced PDE residual rendering."""

    def test_dataset_shape_mismatch_with_field_shape(self) -> None:
        """dataset grid size != field_shape -> graceful handling."""
        ds = _make_1d_dataset(nx=20, nt=10) # 200 points
        result = _make_result(n_samples=50) # 50 points
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        try:
            assert isinstance(fig, Figure)
            # Should either produce a warning or fallback gracefully
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_nan_data_with_dataset(self) -> None:
        """NaN in actual data with dataset -> handles gracefully."""
        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)
        result.actual[0] = float("nan")
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_inf_data_with_dataset(self) -> None:
        """Inf in predicted data with dataset -> handles gracefully."""
        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)
        result.predicted[0] = float("inf")
        fig, warnings = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_no_figure_leak(self) -> None:
        """Should not leak matplotlib figures."""
        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)
        figs_before = len(plt.get_fignums())
        fig, _ = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before


# ===========================================================================
# M3/M2: silent fallback warning
# ===========================================================================


class TestPdeResidualSilentFallbackWarning:
    """M3/M2: when dataset is None, fallback to 1D line plot should emit warning.

    Currently pde_residual silently falls back to 1D line plot when dataset
    is not provided (or when reshape fails). This should emit a
    logging.warning so the user knows axis-aware rendering was skipped.
    """

    def test_no_dataset_emits_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """dataset=None -> logging.warning about fallback rendering."""
        import logging

        result = _make_result(n_samples=50)
        with caplog.at_level(logging.WARNING, logger="kd2.viz.plots.pde_residual"):
            fig, _ = plot_pde_residual_field(result, field_shape=None)
            plt.close(fig)

        # Should have at least one warning about fallback
        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert len(warning_messages) > 0, (
            "Expected a warning when dataset is None and falling back to 1D plot"
        )

    def test_no_dataset_square_shape_emits_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """dataset=None with square field_shape -> still emits fallback warning."""
        import logging

        result = _make_result(n_samples=100)
        with caplog.at_level(logging.WARNING, logger="kd2.viz.plots.pde_residual"):
            fig, _ = plot_pde_residual_field(result, field_shape=(10, 10))
            plt.close(fig)

        # Even with valid reshape, no dataset means no axis-aware rendering
        # The warning should indicate the fallback
        warning_messages = [
            r.message for r in caplog.records if r.levelno >= logging.WARNING
        ]
        assert len(warning_messages) > 0, (
            "Expected a warning about fallback even when field_shape is valid"
        )

    def test_with_dataset_no_fallback_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """With proper dataset -> should NOT emit fallback warning."""
        import logging

        ds = _make_1d_dataset(nx=10, nt=5)
        result = _make_result(n_samples=10 * 5)
        with caplog.at_level(logging.WARNING, logger="kd2.viz.plots.pde_residual"):
            fig, _ = plot_pde_residual_field(result, field_shape=(10, 5), dataset=ds)
            plt.close(fig)

        # With dataset, no fallback warning expected
        fallback_warnings = [
            r.message
            for r in caplog.records
            if r.levelno >= logging.WARNING and "fallback" in r.message.lower()
        ]
        assert len(fallback_warnings) == 0, (
            f"Should not warn about fallback with dataset: {fallback_warnings}"
        )
