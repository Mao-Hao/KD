"""Integration tests for the visualization pipeline.

Tests the full flow: ExperimentResult -> VizEngine -> SVG + HTML report.
Uses mock data (no real SGA execution).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import torch

from kd2.core.evaluator import EvaluationResult
from kd2.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd2.search.recorder import VizRecorder
from kd2.search.result import ExperimentResult
from kd2.viz import VizEngine
from kd2.viz.extension import PlotInfo
from kd2.viz.report import ReportResult

# ---- Helpers ----


def _make_experiment_result(
    *,
    seed: int = 42,
    algorithm_name: str = "SGA",
    dataset_name: str = "burgers",
    n_samples: int = 50,
    n_iters: int = 10,
    best_score: float = 0.3,
    r2: float = 0.95,
) -> ExperimentResult:
    """Build a synthetic ExperimentResult."""
    torch.manual_seed(seed)
    actual = torch.sin(torch.linspace(0, 6.28, n_samples))
    predicted = actual + torch.randn(n_samples) * 0.1
    residuals = predicted - actual

    recorder = VizRecorder()
    scores = [10.0 / (i + 1) for i in range(n_iters)]
    for s in scores:
        recorder.log("_best_score", s)
        recorder.log("_best_expr", "expr")
        recorder.log("_n_candidates", 20)

    return ExperimentResult(
        best_expression="add(u, u_x)",
        best_score=best_score,
        iterations=n_iters,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.005,
            r2=r2,
            aic=-100.0,
            complexity=2,
            coefficients=torch.tensor([1.0, -0.5]),
            is_valid=True,
            error_message="",
            selected_indices=[0, 1],
            residuals=residuals,
            terms=["u", "u_x"],
            expression="add(u, u_x)",
        ),
        actual=actual,
        predicted=predicted,
        dataset_name=dataset_name,
        algorithm_name=algorithm_name,
        config={
            "algorithm": "sga",
            "use_autograd": False,
            "max_iter": n_iters,
            "population_size": 20,
            "seed": seed,
        },
        recorder=recorder,
    )


class _MockVizExtension:
    """Mock algorithm implementing VizExtension."""

    def list_plots(self) -> list[PlotInfo]:
        return [
            PlotInfo(name="aic_landscape", title="AIC vs Complexity"),
            PlotInfo(name="diversity", title="Population Diversity"),
        ]

    def render_plot(self, name: str, ax: Any) -> None:
        if name == "aic_landscape":
            ax.plot([1, 2, 3], [100, 50, 30])
            ax.set_title("AIC Landscape")
        elif name == "diversity":
            ax.bar([1, 2, 3], [0.8, 0.5, 0.3])
            ax.set_title("Diversity")

    def get_plot_data(self, name: str) -> Any:
        if name == "aic_landscape":
            return {"complexities": [1, 2, 3], "aic_scores": [100, 50, 30]}
        return {"values": [0.8, 0.5, 0.3]}


class _FailingVizExtension:
    """Mock extension where one plot always fails."""

    def list_plots(self) -> list[PlotInfo]:
        return [
            PlotInfo(name="good_plot", title="Good Plot"),
            PlotInfo(name="bad_plot", title="Bad Plot"),
        ]

    def render_plot(self, name: str, ax: Any) -> None:
        if name == "bad_plot":
            raise RuntimeError("Intentional failure in bad_plot")
        ax.plot([1, 2], [3, 4])

    def get_plot_data(self, name: str) -> Any:
        return {}


# ---- Single-run pipeline ----


class TestSingleRunPipeline:
    """Test: ExperimentResult -> VizEngine.render_all -> SVGs + HTML."""

    def test_render_all_produces_svgs(self, tmp_path: Path) -> None:
        result = _make_experiment_result()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result)

        assert isinstance(report, ReportResult)
        svg_files = list(tmp_path.glob("*.svg"))
        assert len(svg_files) >= 4 # convergence, parity, residual, equation

    def test_render_all_produces_html_report(self, tmp_path: Path) -> None:
        result = _make_experiment_result()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result)

        assert report.report is not None
        assert report.report.exists()
        content = report.report.read_text()
        assert "Experiment Report" in content
        assert "SGA" in content
        assert "burgers" in content

    def test_html_contains_inline_svgs(self, tmp_path: Path) -> None:
        result = _make_experiment_result()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result)

        content = report.report.read_text()
        # Should embed SVGs as base64 data URIs
        assert "data:image/svg+xml;base64," in content

    def test_html_contains_json_summary(self, tmp_path: Path) -> None:
        result = _make_experiment_result()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result)

        content = report.report.read_text()
        # Collapsible JSON dump section (commit 9973884).
        assert "Full result (JSON, for debugging)" in content
        assert 'class="data-summary"' in content

    def test_all_figures_closed(self, tmp_path: Path) -> None:
        import matplotlib.pyplot as plt

        figs_before = plt.get_fignums()
        result = _make_experiment_result()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_all(result)
        figs_after = plt.get_fignums()
        assert len(figs_after) <= len(figs_before)


# ---- Plugin plots via VizExtension ----


class TestPluginPlotsPipeline:
    """Test: render_all with VizExtension algorithm."""

    def test_plugin_plots_rendered(self, tmp_path: Path) -> None:
        result = _make_experiment_result()
        ext = _MockVizExtension()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_all(result, algorithm=ext)

        # Should have universal plots + plugin plots
        svg_files = list(tmp_path.glob("*.svg"))
        plugin_svgs = [f for f in svg_files if "plugin_" in f.name]
        assert len(plugin_svgs) == 2

    def test_plugin_plots_in_report_figures(self, tmp_path: Path) -> None:
        result = _make_experiment_result()
        ext = _MockVizExtension()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result, algorithm=ext)

        plugin_figs = [f for f in report.figures if f.name.startswith("plugin_")]
        assert len(plugin_figs) == 2

    def test_per_plot_error_isolation(self, tmp_path: Path) -> None:
        """One failing plugin plot should not prevent others."""
        result = _make_experiment_result()
        ext = _FailingVizExtension()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result, algorithm=ext)

        # Should still have universal plots + the good plugin plot
        assert any("good_plot" in str(f) for f in report.figures)
        # bad_plot should appear in warnings
        assert any("bad_plot" in w for w in report.warnings)

    def test_non_extension_algorithm_ignored(self, tmp_path: Path) -> None:
        """Algorithm without VizExtension = universal plots only."""
        result = _make_experiment_result()

        class PlainAlgorithm:
            pass

        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result, algorithm=PlainAlgorithm())

        plugin_svgs = [f for f in report.figures if f.name.startswith("plugin_")]
        assert len(plugin_svgs) == 0


# ---- Multi-seed comparison ----


class TestMultiSeedComparison:
    """Test: multiple runs -> render_comparison -> comparison plots."""

    def test_render_comparison_produces_plots(self, tmp_path: Path) -> None:
        results = [
            _make_experiment_result(seed=42, r2=0.95),
            _make_experiment_result(seed=43, r2=0.92),
            _make_experiment_result(seed=44, r2=0.97),
        ]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)

        assert isinstance(report, ReportResult)
        svg_files = list(tmp_path.glob("*.svg"))
        assert len(svg_files) >= 3 # convergence overlay, score boxplot, summary table

    def test_render_comparison_with_labels(self, tmp_path: Path) -> None:
        results = [
            _make_experiment_result(seed=42),
            _make_experiment_result(seed=43),
        ]
        labels = ["seed_42", "seed_43"]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results, labels=labels)

        assert isinstance(report, ReportResult)
        assert len(report.figures) >= 3

    def test_render_comparison_all_figures_exist(self, tmp_path: Path) -> None:
        results = [
            _make_experiment_result(seed=42),
            _make_experiment_result(seed=43),
        ]
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_comparison(results)

        for fig_path in report.figures:
            assert fig_path.exists(), f"Missing: {fig_path}"


# ---- Field comparison with real PDEDataset ----

_TWO_PI = 2.0 * 3.141592653589793


def _make_1d_pde_dataset(nx: int = 20, nt: int = 10) -> PDEDataset:
    """Create 1D PDE dataset (Burgers-like) for integration tests."""
    x = torch.linspace(0, _TWO_PI, nx)
    t = torch.linspace(0, 1, nt)
    u_field = torch.sin(x).unsqueeze(1) * torch.exp(-t).unsqueeze(0)
    return PDEDataset(
        name="test_burgers_1d",
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


class TestFieldComparisonPipeline:
    """Test: render_all with real PDEDataset exercises field comparison."""

    def test_render_all_with_dataset_produces_field_comparison(
        self, tmp_path: Path
    ) -> None:
        """Full pipeline: engine → format_pde → integrate_pde → field plot."""
        result = _make_experiment_result()
        dataset = _make_1d_pde_dataset()
        engine = VizEngine(output_dir=tmp_path)
        engine.render_all(result, dataset=dataset)

        svg_names = {f.name for f in tmp_path.glob("*.svg")}
        # Both Tier 2 field plots should be rendered (even if integration
        # falls back to warning panels — that's correct error isolation)
        assert "field_comparison.svg" in svg_names
        assert "pde_residual_field.svg" in svg_names

    def test_render_all_with_dataset_no_crash(self, tmp_path: Path) -> None:
        """Dataset integration should not crash even if expression is nonsensical."""
        result = _make_experiment_result()
        dataset = _make_1d_pde_dataset()
        engine = VizEngine(output_dir=tmp_path)
        # Should not raise — integration failure is caught gracefully
        report = engine.render_all(result, dataset=dataset)
        assert report.report is not None
        assert report.report.exists()

    def test_render_all_without_dataset_skips_field(self, tmp_path: Path) -> None:
        """Without dataset, no field comparison plots generated."""
        result = _make_experiment_result()
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result)
        svg_names = {f.name for f in report.figures}
        assert "field_comparison.svg" not in svg_names
        assert "pde_residual_field.svg" not in svg_names
