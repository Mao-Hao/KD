"""Unit tests for VizEngine."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

from pathlib import Path

import matplotlib.pyplot as plt
import pytest

from kd2.data.schema import PDEDataset
from kd2.search.result import ExperimentResult
from kd2.viz import VizEngine
from kd2.viz.report import ReportResult


class TestVizEngineInit:
    """Tests for VizEngine construction."""

    def test_creates_output_dir(self, tmp_path: Path) -> None:
        out = tmp_path / "viz_output"
        engine = VizEngine(output_dir=out)
        assert out.exists()

    def test_accepts_custom_style(self, tmp_path: Path) -> None:
        engine = VizEngine(output_dir=tmp_path, style={"font.size": 20})
        assert engine._style["font.size"] == 20


class TestRenderUniversal:
    """Tests for render_universal (5 universal plots)."""

    def test_returns_report_result(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_universal(mock_experiment_result)
        assert isinstance(report, ReportResult)

    def test_creates_svg_files(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_universal(mock_experiment_result)
        # Should produce convergence, parity, residual, equation SVGs
        svg_files = list(tmp_path.glob("*.svg"))
        assert len(svg_files) >= 4

    def test_all_figures_closed(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        figs_before = plt.get_fignums()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_universal(mock_experiment_result)
        figs_after = plt.get_fignums()
        # No new unclosed figures
        assert len(figs_after) <= len(figs_before)

    def test_figures_in_report(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_universal(mock_experiment_result)
        # All paths in report.figures should exist
        for fig_path in report.figures:
            assert fig_path.exists(), f"Missing: {fig_path}"

    def test_report_has_no_unexpected_warnings(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_universal(mock_experiment_result)
        # With valid mock data, should have no warnings
        assert len(report.warnings) == 0, f"Unexpected warnings: {report.warnings}"


class TestRenderAll:
    """Tests for render_all (universal + field comparison if dataset given)."""

    def test_without_dataset(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result)
        assert isinstance(report, ReportResult)
        # Without dataset, no field comparison
        field_files = [f for f in report.figures if "field" in str(f)]
        assert len(field_files) == 0

    def test_with_dataset_renders_field(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """render_all with real PDEDataset should produce field comparison."""
        import torch

        from kd2.data.schema import (
            AxisInfo,
            DataTopology,
            FieldData,
            PDEDataset,
            TaskType,
        )

        nx, nt = 10, 5
        x_vals = torch.linspace(0, 1, nx)
        t_vals = torch.linspace(0, 1, nt)
        u_field = torch.randn(nx, nt, dtype=torch.float64)

        ds = PDEDataset(
            name="test_1d",
            task_type=TaskType.PDE,
            topology=DataTopology.GRID,
            axes={
                "x": AxisInfo(name="x", values=x_vals, is_periodic=True),
                "t": AxisInfo(name="t", values=t_vals),
            },
            axis_order=["x", "t"],
            fields={"u": FieldData(name="u", values=u_field)},
            lhs_field="u",
            lhs_axis="t",
        )

        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=ds)
        field_files = [f for f in report.figures if f.name == "field_comparison.svg"]
        assert len(field_files) == 1

    def test_dataset_without_proper_api_warns(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """Non-PDEDataset object should produce warnings via error isolation."""

        class _EmptyDataset:
            pass

        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=_EmptyDataset())
        # Should have warnings from failed field comparison / pde_residual
        assert any("failed" in w.lower() for w in report.warnings)

    def test_all_figures_closed(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        figs_before = plt.get_fignums()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_all(mock_experiment_result)
        figs_after = plt.get_fignums()
        assert len(figs_after) <= len(figs_before)

    def test_universal_plot_error_isolation(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """A failing universal plot should not prevent others."""
        from unittest.mock import patch

        def _raise_on_equation(result: ExperimentResult, ax: object) -> list[str]:
            raise RuntimeError("Simulated plot failure")

        engine = VizEngine(output_dir=tmp_path)
        with patch("kd2.viz.engine.plot_equation", _raise_on_equation):
            report = engine.render_all(mock_experiment_result)
        # Should still have some figures (convergence, parity, residual)
        assert len(report.figures) >= 2
        # Should have a warning about the failed plot
        assert any("failed" in w.lower() for w in report.warnings)


# ===========================================================================
# M3: render_all with new Tier 2 plots
# ===========================================================================


def _make_pde_dataset_for_engine() -> PDEDataset:
    """Create a PDEDataset for engine integration tests."""
    import torch

    from kd2.data.schema import AxisInfo, FieldData, PDEDataset, TaskType

    nx, nt = 10, 5
    x = torch.linspace(0, 1, nx)
    t = torch.linspace(0, 1, nt)
    u_field = torch.randn(nx, nt, dtype=torch.float64)
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


class TestRenderAllTier2Plots:
    """render_all should produce new Tier 2 plots when dataset given."""

    def test_coefficient_bar_in_render_all(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """render_all should produce coefficient_bar SVG."""
        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=ds)
        coeff_files = [f for f in report.figures if "coefficient" in f.name.lower()]
        assert len(coeff_files) >= 1, (
            f"Expected coefficient plot in output, got: "
            f"{[f.name for f in report.figures]}"
        )

    def test_time_slices_in_render_all(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """render_all should produce time_slices SVG."""
        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=ds)
        slice_files = [f for f in report.figures if "time_slice" in f.name.lower()]
        assert len(slice_files) >= 1, (
            f"Expected time_slices plot in output, got: "
            f"{[f.name for f in report.figures]}"
        )

    def test_error_heatmap_in_render_all(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """render_all should produce error_heatmap SVG."""
        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result, dataset=ds)
        heatmap_files = [f for f in report.figures if "error_heatmap" in f.name.lower()]
        assert len(heatmap_files) >= 1, (
            f"Expected error_heatmap plot in output, got: "
            f"{[f.name for f in report.figures]}"
        )

    def test_pde_residual_passes_dataset(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """render_all should pass dataset to _render_pde_residual (not just field_shape)."""
        from unittest.mock import patch

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)

        calls: list[dict] = []
        original = engine._render_pde_residual

        def _spy(result, dataset, report):
            calls.append({"dataset": dataset})
            return original(result, dataset, report)

        with patch.object(engine, "_render_pde_residual", _spy):
            engine.render_all(mock_experiment_result, dataset=ds)

        assert len(calls) >= 1, "Expected _render_pde_residual to be called"
        assert calls[0]["dataset"] is ds

    def test_new_plots_do_not_break_without_dataset(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """render_all without dataset should still work (no new Tier 2 plots)."""
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(mock_experiment_result)
        assert isinstance(report, ReportResult)
        # Only universal plots should be present
        names = [f.name for f in report.figures]
        assert not any("coefficient" in n.lower() for n in names)
        assert not any("time_slice" in n.lower() for n in names)
        assert not any("error_heatmap" in n.lower() for n in names)

    def test_tier2_error_isolation(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """A failing Tier 2 plot should not prevent other plots."""
        from unittest.mock import patch

        import matplotlib.pyplot as mpl_plt
        from matplotlib.figure import Figure as MplFigure

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)

        def _failing_plot(**kwargs):
            raise RuntimeError("Simulated Tier 2 failure")

        with patch("kd2.viz.engine.plot_field_comparison", _failing_plot):
            report = engine.render_all(mock_experiment_result, dataset=ds)
        # Should still have universal plots
        assert len(report.figures) >= 3
        # Should have a warning about the failed field comparison
        assert any("failed" in w.lower() for w in report.warnings)

    def test_all_figures_closed_with_m3(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """render_all with dataset should close all figures."""
        ds = _make_pde_dataset_for_engine()
        figs_before = plt.get_fignums()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_all(mock_experiment_result, dataset=ds)
        figs_after = plt.get_fignums()
        assert len(figs_after) <= len(figs_before)


# ===========================================================================
# _get_integration_result try/except narrowing
# ===========================================================================


class TestGetIntegrationResultTryExcept:
    """try/except should only catch integrate_pde errors, not programmer bugs.

    Currently _get_integration_result wraps everything (format_pde, attribute
    access, integrate_pde) in one big try/except. This silently swallows bugs
    in format_pde or result attribute access, making them look like integration
    failures. The fix should narrow the try/except to only wrap integrate_pde().
    """

    def test_integrate_pde_error_caught(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """integrate_pde raising an exception -> captured, returns failed IntegrationResult."""
        from unittest.mock import patch

        from kd2.core.integrator import IntegrationResult

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)

        # integrate_pde is locally imported inside _get_integration_result,
        # so we mock it at the source module level
        with patch(
            "kd2.core.integrator.integrate_pde",
            side_effect=RuntimeError("Solver diverged"),
        ):
            result = engine._get_integration_result(mock_experiment_result, ds)

        assert isinstance(result, IntegrationResult)
        assert not result.success
        # Warning should mention the actual error
        assert result.warning is not None

    def test_format_pde_bug_not_swallowed(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """format_pde raising a bug (e.g. TypeError) should NOT be silently caught.

        Currently this is swallowed by the broad try/except. After the fix,
        a bug in format_pde (outside integrate_pde) should propagate.
        """
        from unittest.mock import patch

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)

        # format_pde is locally imported inside _get_integration_result
        with (
            patch(
                "kd2.core.expr.sympy_bridge.format_pde",
                side_effect=TypeError("BUG: wrong argument type"),
            ),
            pytest.raises(TypeError, match="BUG"),
        ):
            engine._get_integration_result(mock_experiment_result, ds)

    def test_attribute_access_bug_not_swallowed(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """Accessing result.final_eval.terms when terms is wrong type should propagate.

        This tests that programmer errors in data access before integrate_pde
        are not silently swallowed.
        """
        from unittest.mock import PropertyMock, patch

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)

        # Simulate a bug where final_eval.terms property raises
        bad_result = mock_experiment_result
        original_final_eval = bad_result.final_eval

        class _BrokenEval:
            """Simulates a final_eval with a broken property."""

            @property
            def terms(self):
                raise AttributeError("BUG: terms property broken")

            @property
            def coefficients(self):
                return original_final_eval.coefficients

            @property
            def selected_indices(self):
                return original_final_eval.selected_indices

        bad_result.final_eval = _BrokenEval()
        try:
            with pytest.raises(AttributeError, match="BUG"):
                engine._get_integration_result(bad_result, ds)
        finally:
            bad_result.final_eval = original_final_eval

    def test_missing_terms_still_handled(
        self, tmp_path: Path, mock_experiment_result: ExperimentResult
    ) -> None:
        """terms=None -> should return failed IntegrationResult (not crash).

        This is a valid data-level error (missing terms), not a programmer bug.
        The current code handles it correctly via the explicit None check.
        """
        from kd2.core.integrator import IntegrationResult

        ds = _make_pde_dataset_for_engine()
        engine = VizEngine(output_dir=tmp_path)

        # Set terms to None (valid case: no terms selected)
        original_terms = mock_experiment_result.final_eval.terms
        mock_experiment_result.final_eval.terms = None
        try:
            result = engine._get_integration_result(mock_experiment_result, ds)
            assert isinstance(result, IntegrationResult)
            assert not result.success
        finally:
            mock_experiment_result.final_eval.terms = original_terms
