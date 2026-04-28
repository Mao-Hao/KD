"""HTML report generation and report result types.

Generates HTML reports with inline SVG figures, metadata tables and
JSON data summaries using Jinja2 templates. The report bundles every
figure as inline SVG so it travels as a single file, but LaTeX in the
equation block is rendered by MathJax loaded from a public CDN, so
opening the report fully offline drops the rendered equation back to
its raw TeX source.
"""

from __future__ import annotations

import base64
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader
from markupsafe import Markup

from kd2.core.expr.sympy_bridge import format_pde, to_latex

if TYPE_CHECKING:
    from kd2.search.result import ExperimentResult

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"
_JSON_INDENT = 2


@dataclass
class ReportResult:
    """Metadata from a visualization rendering pass.

    Attributes:
        figures: Paths to generated figure files (SVG/PNG).
        data_files: Paths to exported data files (JSON).
        report: Path to HTML report, if generated.
        warnings: Non-fatal issues encountered during rendering.
    """

    figures: list[Path] = field(default_factory=list)
    data_files: list[Path] = field(default_factory=list)
    report: Path | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass
class _FigureEntry:
    """Internal: figure metadata for template rendering."""

    title: str
    data_uri: str


def _figure_title_from_path(path: Path) -> str:
    """Derive a human-readable title from a figure filename."""
    stem = path.stem
    return stem.replace("_", " ").replace("-", " ").title()


def _encode_figure(path: Path) -> str | None:
    """Read a figure file and return a base64 data URI.

    Returns None if the file does not exist or cannot be read.
    """
    if not path.exists():
        logger.warning("Figure file not found: %s", path)
        return None

    suffix = path.suffix.lower()
    raw = path.read_bytes()

    if suffix == ".svg":
        mime = "image/svg+xml"
    elif suffix == ".png":
        mime = "image/png"
    else:
        mime = "application/octet-stream"

    encoded = base64.b64encode(raw).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _build_json_summary(result: ExperimentResult) -> str:
    """Build a pretty-printed JSON summary of scalar/small data."""
    data = result.to_dict()
    # Remove large arrays for readability
    for key in ("actual", "predicted"):
        if key in data:
            arr = data[key]
            if isinstance(arr, list) and len(arr) > 20:
                data[key] = f"[{len(arr)} elements]"
    # Truncate nested large arrays (e.g. final_eval.residuals)
    final_eval = data.get("final_eval")
    if isinstance(final_eval, dict):
        for key in ("residuals", "coefficients"):
            arr = final_eval.get(key)
            if isinstance(arr, list) and len(arr) > 20:
                final_eval[key] = f"[{len(arr)} elements]"
    return json.dumps(data, indent=_JSON_INDENT, default=str)


def _best_expression_latex(result: ExperimentResult) -> str:
    """Return a LaTeX string for the best discovered equation."""
    final_eval = result.final_eval
    if final_eval.terms is not None and final_eval.coefficients is not None:
        try:
            return format_pde(
                final_eval.terms,
                final_eval.coefficients,
                lhs=result.lhs_label,
                selected_indices=final_eval.selected_indices,
            ).latex
        except ValueError:
            logger.exception("Failed to format report equation as full PDE")
    return to_latex(result.best_expression, strict=False)


def generate_report(
    result: ExperimentResult,
    figures: list[Path],
    output_path: Path,
    *,
    plugin_figures: list[Path] | None = None,
    warnings: list[str] | None = None,
) -> Path:
    """Generate an HTML report with inline SVG figures.

    The HTML embeds every figure as inline SVG so it travels as a single
    file, but the equation block uses MathJax loaded from a public CDN;
    opening the report fully offline keeps the figures and falls back to
    raw TeX in the equation panel.

    Args:
        result: Completed experiment result (data source).
        figures: Paths to universal plot SVG/PNG files.
        output_path: Where to write the HTML report.
        plugin_figures: Optional paths to plugin-generated plot files.
        warnings: Optional list of warnings to display in report.

    Returns:
        The output_path where the report was written.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build figure entries
    fig_entries: list[_FigureEntry] = []
    for fig_path in figures:
        data_uri = _encode_figure(fig_path)
        if data_uri is None:
            continue
        fig_entries.append(
            _FigureEntry(
                title=_figure_title_from_path(fig_path),
                data_uri=data_uri,
            )
        )

    # Build plugin figure entries
    plugin_entries: list[_FigureEntry] = []
    if plugin_figures:
        for fig_path in plugin_figures:
            data_uri = _encode_figure(fig_path)
            if data_uri is None:
                continue
            plugin_entries.append(
                _FigureEntry(
                    title=_figure_title_from_path(fig_path),
                    data_uri=data_uri,
                )
            )

    # Build JSON summary (Markup to prevent autoescape double-encoding)
    json_summary = Markup(_build_json_summary(result))

    # Render template
    env = Environment(
        loader=FileSystemLoader(str(_TEMPLATES_DIR)),
        autoescape=True,
    )
    template = env.get_template("report.html")

    html = template.render(
        dataset_name=result.dataset_name,
        algorithm_name=result.algorithm_name,
        best_expression=result.best_expression,
        best_expression_latex=_best_expression_latex(result),
        best_score=f"{result.best_score:.2f}",
        r2=f"{result.final_eval.r2:.6f}",
        nmse=f"{result.final_eval.nmse:.4g}",
        iterations=result.iterations,
        early_stopped=result.early_stopped,
        config=result.config,
        figures=fig_entries,
        plugin_figures=plugin_entries,
        warnings=warnings or [],
        json_summary=json_summary,
    )

    output_path.write_text(html, encoding="utf-8")
    logger.debug("Generated HTML report at %s", output_path)
    return output_path
