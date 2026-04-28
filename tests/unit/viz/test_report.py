"""Unit tests for HTML report generation."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest
import torch

from kd2.core.evaluator import EvaluationResult
from kd2.search.recorder import VizRecorder
from kd2.search.result import ExperimentResult
from kd2.viz.report import ReportResult, generate_report

# ---- Fixtures ----


@pytest.fixture()
def svg_files(tmp_path: Path) -> list[Path]:
    """Create minimal SVG files for testing."""
    svg_content = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">'
        '<rect width="100" height="100" fill="blue"/>'
        "</svg>"
    )
    paths: list[Path] = []
    for name in ["convergence", "parity", "residual", "equation"]:
        p = tmp_path / f"{name}.svg"
        p.write_text(svg_content)
        paths.append(p)
    return paths


# ---- generate_report basic ----


class TestGenerateReportBasic:
    """Basic tests for generate_report."""

    def test_returns_path(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        output = tmp_path / "report.html"
        result = generate_report(mock_experiment_result, svg_files, output)
        assert result == output

    def test_creates_html_file(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        assert output.exists()
        assert output.stat().st_size > 0

    def test_html_is_self_contained(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        """HTML should not reference external files (except CDN MathJax)."""
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        content = output.read_text()
        # Should contain inline CSS (style tag)
        assert "<style>" in content
        # Should not reference external CSS
        assert 'rel="stylesheet"' not in content

    def test_creates_parent_dirs(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        output = tmp_path / "sub" / "dir" / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        assert output.exists()


# ---- HTML content ----


class TestReportContent:
    """Tests for report HTML content structure."""

    def test_contains_metadata_table(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        content = output.read_text()
        # Metadata: algorithm_name, dataset_name
        assert "SGA" in content
        assert "test_dataset" in content

    def test_contains_metrics(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        content = output.read_text()
        # Should show R2, NMSE, iterations, early_stopped
        assert "0.95" in content # R2
        assert "10" in content # iterations

    def test_contains_inline_svg(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        content = output.read_text()
        # SVGs should be base64-encoded or raw inline
        assert "data:image/svg+xml;base64," in content or "<svg" in content

    def test_contains_json_summary(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        content = output.read_text()
        # Should have a collapsible JSON summary section.
        assert "Full result (JSON, for debugging)" in content
        assert 'class="data-summary"' in content
        assert "json-summary" in content

    def test_contains_experiment_report_title(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        content = output.read_text()
        assert "Experiment Report" in content

    def test_contains_all_section_headings(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        content = output.read_text()
        for heading in ["Convergence", "Parity", "Residual", "Equation"]:
            assert heading in content

    def test_empty_figures_list(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
    ) -> None:
        """Report should still generate with no figures."""
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, [], output)
        assert output.exists()
        content = output.read_text()
        assert "Experiment Report" in content

    def test_config_displayed(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        content = output.read_text()
        # Config keys should appear
        assert "max_iter" in content or "population_size" in content

    def test_mathjax_script_present(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
        svg_files: list[Path],
    ) -> None:
        """HTML report should include a MathJax script tag for LaTeX rendering.
        MathJax is loaded from CDN to render equation LaTeX in the browser."""
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, svg_files, output)
        content = output.read_text()
        # Must contain a <script> tag referencing MathJax
        assert "mathjax" in content.lower(), (
            "Report HTML should include MathJax for LaTeX rendering"
        )


# ---- Edge cases ----


class TestReportEdgeCases:
    """Edge case tests for generate_report."""

    def test_missing_svg_file_skipped_gracefully(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
    ) -> None:
        """Non-existent figure paths should be skipped, not crash."""
        fake = [tmp_path / "nonexistent.svg"]
        output = tmp_path / "report.html"
        # Should not raise
        generate_report(mock_experiment_result, fake, output)
        assert output.exists()

    def test_non_svg_figures_handled(
        self,
        tmp_path: Path,
        mock_experiment_result: ExperimentResult,
    ) -> None:
        """Non-SVG figure files should be handled gracefully."""
        png = tmp_path / "test.png"
        png.write_bytes(b"\x89PNG\r\n\x1a\n")
        output = tmp_path / "report.html"
        generate_report(mock_experiment_result, [png], output)
        assert output.exists()
