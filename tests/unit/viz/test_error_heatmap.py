"""Tests for error heatmap plot.

Tests plot_error_heatmap (Tier 2): spatial distribution of prediction error.
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
from kd2.core.integrator import IntegrationResult
from kd2.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd2.search.recorder import VizRecorder
from kd2.search.result import ExperimentResult
from kd2.viz.plots.error_heatmap import plot_error_heatmap

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


def _make_integration_result(
    dataset: PDEDataset,
    *,
    success: bool = True,
    noise_std: float = 0.05,
) -> IntegrationResult:
    """IntegrationResult matching dataset shape."""
    if success:
        true_field = dataset.get_field("u")
        return IntegrationResult(
            success=True,
            predicted_field=true_field + torch.randn_like(true_field) * noise_std,
        )
    return IntegrationResult(success=False, warning="Integration failed")


def _make_experiment_result() -> ExperimentResult:
    """Minimal ExperimentResult."""
    n = 50
    actual = torch.randn(n)
    predicted = actual + torch.randn(n) * 0.1
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
# Smoke tests
# ===========================================================================


class TestErrorHeatmapSmoke:
    """Basic callable/return-type tests."""

    def test_callable_returns_tuple(self) -> None:
        """plot_error_heatmap returns (Figure, list[str])."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, warnings = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert isinstance(warnings, list)
        finally:
            plt.close(fig)

    def test_accepts_style_kwarg(self) -> None:
        """style keyword is accepted."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_error_heatmap(result, ds, ir, style={"font.size": 12})
        plt.close(fig)


# ===========================================================================
# Happy path
# ===========================================================================


class TestErrorHeatmapHappyPath:
    """Core: heatmap rendering with different spatial dims."""

    def test_1d_produces_axes(self) -> None:
        """1D spatial dataset -> at least one axes with heatmap."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_error_heatmap(result, ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_2d_produces_axes(self) -> None:
        """2D spatial dataset -> at least one axes."""
        ds = _make_2d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_error_heatmap(result, ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_has_colorbar_or_colormap(self) -> None:
        """Heatmap should use a colormap (images or pcolormesh present)."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_error_heatmap(result, ds, ir)
        try:
            # Check for images (imshow) or collections (pcolormesh)
            has_visual = any(
                len(ax.images) > 0 or len(ax.collections) > 0 for ax in fig.get_axes()
            )
            assert has_visual, "Expected heatmap/pcolormesh in the figure"
        finally:
            plt.close(fig)

    def test_title_present(self) -> None:
        """Figure or axes should have descriptive title."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_error_heatmap(result, ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            suptitle = fig._suptitle.get_text() if fig._suptitle else ""
            all_text = " ".join(titles) + " " + suptitle
            assert len(all_text.strip()) > 0, "Expected at least one title"
        finally:
            plt.close(fig)


# ===========================================================================
# Edge cases and negative tests (>= 20%)
# ===========================================================================


class TestErrorHeatmapEdgeCases:
    """Edge cases: failed integration, NaN, minimal data."""

    def test_failed_integration_no_crash(self) -> None:
        """Integration failure -> figure with warning."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds, success=False)
        result = _make_experiment_result()
        fig, warnings = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_none_predicted_field(self) -> None:
        """predicted_field=None -> warning, no crash."""
        ds = _make_1d_dataset()
        ir = IntegrationResult(
            success=False,
            predicted_field=None,
            warning="Total failure",
        )
        result = _make_experiment_result()
        fig, warnings = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_diverged_integration(self) -> None:
        """Diverged (field present, success=False) -> handles gracefully."""
        ds = _make_1d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Diverged at t=0.5",
            diverged_at_t=0.5,
        )
        result = _make_experiment_result()
        fig, _ = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_minimal_dataset(self) -> None:
        """Very small dataset (2x2) -> no crash."""
        ds = _make_1d_dataset(nx=2, nt=2)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_no_figure_leak(self) -> None:
        """Should not leak matplotlib figures."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        figs_before = len(plt.get_fignums())
        fig, _ = plot_error_heatmap(result, ds, ir)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before


# ===========================================================================
# P4.5/M3: axis_order=None guard (ODE / incomplete dataset)
# ===========================================================================


def _make_ode_dataset_no_axis_order() -> PDEDataset:
    """ODE dataset with axis_order=None (no spatial info)."""
    nt = 10
    t = torch.linspace(0, 1, nt)
    u_field = torch.sin(t)
    return PDEDataset(
        name="test_ode",
        task_type=TaskType.ODE,
        axes=None,
        axis_order=None,
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


def _make_ode_dataset_n_spatial_0() -> PDEDataset:
    """ODE-like dataset with axis_order=["t"] (n_spatial=0)."""
    nt = 10
    t = torch.linspace(0, 1, nt)
    u_field = torch.sin(t)
    return PDEDataset(
        name="test_ode_t_only",
        task_type=TaskType.ODE,
        axes={"t": AxisInfo(name="t", values=t)},
        axis_order=["t"],
        fields={"u": FieldData(name="u", values=u_field)},
        lhs_field="u",
        lhs_axis="t",
    )


class TestErrorHeatmapAxisOrderNone:
    """P4.5/M3: axis_order=None should not crash."""

    def test_axis_order_none_no_crash(self) -> None:
        """axis_order=None dataset -> returns Figure with warning, not AssertionError."""
        ds = _make_ode_dataset_no_axis_order()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        result = _make_experiment_result()
        fig, warnings = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            # Should produce a warning about missing spatial info
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_axis_order_none_returns_valid_figure(self) -> None:
        """Returned figure should have at least one axes (warning panel)."""
        ds = _make_ode_dataset_no_axis_order()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        result = _make_experiment_result()
        fig, _ = plot_error_heatmap(result, ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_axis_order_none_failed_integration(self) -> None:
        """axis_order=None + failed integration -> still no crash."""
        ds = _make_ode_dataset_no_axis_order()
        ir = IntegrationResult(success=False, warning="Integration failed")
        result = _make_experiment_result()
        fig, warnings = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestErrorHeatmapNSpatialZero:
    """P4.5/M4: n_spatial==0 should not crash."""

    def test_n_spatial_zero_no_crash(self) -> None:
        """Dataset with only time axis -> returns Figure, not IndexError."""
        ds = _make_ode_dataset_n_spatial_0()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        result = _make_experiment_result()
        fig, warnings = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            # Should warn about no spatial dimensions
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_n_spatial_zero_returns_valid_figure(self) -> None:
        """Returned figure should be renderable (has axes)."""
        ds = _make_ode_dataset_n_spatial_0()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        result = _make_experiment_result()
        fig, _ = plot_error_heatmap(result, ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_n_spatial_zero_failed_integration(self) -> None:
        """n_spatial=0 + failed integration -> still no crash."""
        ds = _make_ode_dataset_n_spatial_0()
        ir = IntegrationResult(success=False, warning="Integration failed")
        result = _make_experiment_result()
        fig, warnings = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


# ===========================================================================
# Regression: normal datasets still work
# ===========================================================================


class TestErrorHeatmapNormalRegression:
    """Ensure normal PDE datasets are unaffected by guard additions."""

    def test_1d_normal_dataset_unchanged(self) -> None:
        """1D PDE dataset -> same behavior as before."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, warnings = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 1
            # Success path should not produce guard-related warnings
            guard_warnings = [
                w for w in warnings if "spatial" in w.lower() or "axis" in w.lower()
            ]
            assert len(guard_warnings) == 0, (
                f"Normal dataset should not trigger guards: {guard_warnings}"
            )
        finally:
            plt.close(fig)

    def test_2d_normal_dataset_unchanged(self) -> None:
        """2D PDE dataset -> same behavior as before."""
        ds = _make_2d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, warnings = plot_error_heatmap(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)
