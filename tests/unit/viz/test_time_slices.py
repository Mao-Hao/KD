"""Tests for time-slices comparison plot.

Tests plot_time_slices (Tier 2): true vs predicted field at selected time steps.
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
from kd2.viz.plots.time_slices import plot_time_slices

# Helpers (reuse patterns from test_field.py)

_TWO_PI = 2.0 * math.pi


def _make_1d_dataset(nx: int = 20, nt: int = 10) -> PDEDataset:
    """1D PDE dataset: u(x,t) = sin(x) * exp(-t)."""
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
    """2D PDE dataset: u(x,y,t) = sin(x)*cos(y)*exp(-t)."""
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
    noise_std: float = 0.01,
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


class TestTimeSlicesSmoke:
    """Basic callable/return-type tests."""

    def test_callable_returns_tuple(self) -> None:
        """plot_time_slices returns (Figure, list[str])."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
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
        fig, _ = plot_time_slices(result, ds, ir, style={"font.size": 12})
        plt.close(fig)

    def test_accepts_n_slices_kwarg(self) -> None:
        """n_slices keyword is accepted."""
        ds = _make_1d_dataset(nt=10)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=5)
        plt.close(fig)


# ===========================================================================
# Happy path
# ===========================================================================


class TestTimeSlicesHappyPath:
    """Core functionality: multi-panel time slice comparison."""

    def test_1d_produces_panels(self) -> None:
        """1D dataset: should produce panels for each time slice."""
        ds = _make_1d_dataset(nt=10)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            # Should have at least 3 panels (one per slice, possibly true+pred pairs)
            assert len(fig.get_axes()) >= 3
        finally:
            plt.close(fig)

    def test_2d_produces_panels(self) -> None:
        """2D dataset: should produce panels for each time slice."""
        ds = _make_2d_dataset(nt=5)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            assert len(fig.get_axes()) >= 3
        finally:
            plt.close(fig)

    def test_n_slices_affects_panel_count(self) -> None:
        """More slices should produce more panels."""
        ds = _make_1d_dataset(nt=20)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()

        fig2, _ = plot_time_slices(result, ds, ir, n_slices=2)
        fig5, _ = plot_time_slices(result, ds, ir, n_slices=5)
        try:
            assert len(fig5.get_axes()) > len(fig2.get_axes())
        finally:
            plt.close(fig2)
            plt.close(fig5)

    def test_panels_have_titles(self) -> None:
        """Each panel should have a title (typically with time value)."""
        ds = _make_1d_dataset(nt=10)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            non_empty = [t for t in titles if t]
            assert len(non_empty) >= 3
        finally:
            plt.close(fig)


# ===========================================================================
# Edge cases and negative tests (>= 20%)
# ===========================================================================


class TestTimeSlicesEdgeCases:
    """Edge cases: failed integration, NaN, minimal data."""

    def test_failed_integration_no_crash(self) -> None:
        """Integration failure -> figure with warning, not crash."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds, success=False)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
            assert len(warnings) > 0
        finally:
            plt.close(fig)

    def test_single_time_step(self) -> None:
        """Dataset with nt=1 -> should not crash."""
        ds = _make_1d_dataset(nx=10, nt=1)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_n_slices_larger_than_nt(self) -> None:
        """n_slices > nt -> clamp to nt, no crash."""
        ds = _make_1d_dataset(nt=2)
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir, n_slices=10)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)

    def test_diverged_integration(self) -> None:
        """Diverged (field present but success=False) -> handles gracefully."""
        ds = _make_1d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Diverged at t=0.5",
            diverged_at_t=0.5,
        )
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
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
        fig, _ = plot_time_slices(result, ds, ir)
        plt.close(fig)
        figs_after = len(plt.get_fignums())
        assert figs_after <= figs_before


# ===========================================================================
# P4.5/M3: axis_order=None guard (ODE / incomplete dataset)
# ===========================================================================


def _make_ode_dataset_no_axis_order() -> PDEDataset:
    """ODE dataset with axis_order=None (no spatial info).

    This simulates an ODE or incomplete dataset where axis_order
    is not provided, triggering the assert crash in the current code.
    """
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
    """ODE-like dataset with axis_order=["t"] (n_spatial=0).

    This simulates a dataset where the only axis is time, so there
    are zero spatial dimensions. The current code hits IndexError
    when accessing spatial_axes[0].
    """
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


class TestTimeSlicesAxisOrderNone:
    """P4.5/M3: axis_order=None should not crash (ODE dataset)."""

    def test_axis_order_none_no_crash(self) -> None:
        """axis_order=None dataset -> returns Figure with warning, not AssertionError."""
        ds = _make_ode_dataset_no_axis_order()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
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
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_axis_order_none_failed_integration(self) -> None:
        """axis_order=None + failed integration -> still no crash."""
        ds = _make_ode_dataset_no_axis_order()
        ir = IntegrationResult(success=False, warning="Integration failed")
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


class TestTimeSlicesNSpatialZero:
    """P4.5/M4: n_spatial==0 should not crash (time-only dataset)."""

    def test_n_spatial_zero_no_crash(self) -> None:
        """Dataset with only time axis -> returns Figure, not IndexError."""
        ds = _make_ode_dataset_n_spatial_0()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(success=True, predicted_field=pred)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
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
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            assert len(fig.get_axes()) >= 1
        finally:
            plt.close(fig)

    def test_n_spatial_zero_failed_integration(self) -> None:
        """n_spatial=0 + failed integration -> still no crash."""
        ds = _make_ode_dataset_n_spatial_0()
        ir = IntegrationResult(success=False, warning="Integration failed")
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir)
        try:
            assert isinstance(fig, Figure)
        finally:
            plt.close(fig)


# ===========================================================================
# P4.5/M2: diverged title text verification
# ===========================================================================


class TestTimeSlicesDivergedTitle:
    """P4.5/M2: diverged tests must verify DIVERGED appears in titles."""

    def test_diverged_1d_title_contains_diverged(self) -> None:
        """1D diverged: at least one panel title should contain 'DIVERGED'."""
        ds = _make_1d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Diverged at t=0.5",
            diverged_at_t=0.5,
        )
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" in all_titles, (
                f"Expected 'DIVERGED' in panel titles, got: {titles}"
            )
        finally:
            plt.close(fig)

    def test_diverged_2d_title_contains_diverged(self) -> None:
        """2D diverged: at least one panel title should contain 'DIVERGED'."""
        ds = _make_2d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Diverged at t=0.3",
            diverged_at_t=0.3,
        )
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" in all_titles, (
                f"Expected 'DIVERGED' in panel titles, got: {titles}"
            )
        finally:
            plt.close(fig)

    def test_diverged_without_at_t_still_has_tag(self) -> None:
        """diverged_at_t=None -> title should still contain 'DIVERGED'."""
        ds = _make_1d_dataset()
        pred = ds.get_field("u").clone()
        ir = IntegrationResult(
            success=False,
            predicted_field=pred,
            warning="Integration did not succeed",
            diverged_at_t=None,
        )
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" in all_titles, (
                f"Expected 'DIVERGED' in panel titles, got: {titles}"
            )
        finally:
            plt.close(fig)

    def test_non_diverged_no_diverged_tag(self) -> None:
        """Non-diverged integration should NOT have DIVERGED in titles."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds, success=True)
        result = _make_experiment_result()
        fig, _ = plot_time_slices(result, ds, ir)
        try:
            titles = [ax.get_title() for ax in fig.get_axes()]
            all_titles = " ".join(titles)
            assert "DIVERGED" not in all_titles, (
                f"Non-diverged should not have DIVERGED tag, got: {titles}"
            )
        finally:
            plt.close(fig)


# ===========================================================================
# Regression: normal datasets still work
# ===========================================================================


class TestTimeSlicesNormalRegression:
    """Ensure normal PDE datasets are unaffected by guard additions."""

    def test_1d_normal_dataset_unchanged(self) -> None:
        """1D PDE dataset -> same behavior as before (panels, no extra warnings)."""
        ds = _make_1d_dataset()
        ir = _make_integration_result(ds)
        result = _make_experiment_result()
        fig, warnings = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 3
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
        fig, warnings = plot_time_slices(result, ds, ir, n_slices=3)
        try:
            assert isinstance(fig, Figure)
            assert len(fig.get_axes()) >= 3
        finally:
            plt.close(fig)
