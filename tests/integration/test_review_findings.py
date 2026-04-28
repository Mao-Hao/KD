"""Integration tests for review findings (M1, L2).

TDD red phase: tests describe expected behavior for issues found in
post-merge review of HTML autograd warning, BYOD/ResultTargetProvider,
and grid-uniformity predicate consistency.

- M1: Autograd domain warning must appear EXACTLY ONCE in rendered HTML,
  even when integration fails and Tier 2 plots forward the annotated
  IntegrationResult.warning. Currently reproduces 4x because forward
  paths string-concatenate the note while the engine-level dedup
  ``if autograd_note not in report.warnings`` checks raw equality.

- L2: viz pipeline test fixtures must explicitly set ``algorithm`` and
  ``use_autograd`` keys. Today they default to None and the warning
  branch silently no-ops, hiding regressions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import pytest
import torch

from kd2.core.evaluator import EvaluationResult
from kd2.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd2.search.recorder import VizRecorder
from kd2.search.result import ExperimentResult
from kd2.viz import VizEngine

# Fixtures (local — keep this file self-contained for integration scope)

_TWO_PI = 2.0 * 3.141592653589793


def _make_pde_dataset(nx: int = 20, nt: int = 10) -> PDEDataset:
    """1D PDE dataset (Burgers-like) used for integrator-driven warnings."""
    x = torch.linspace(0.0, _TWO_PI, nx)
    t = torch.linspace(0.0, 1.0, nt)
    u_field = torch.sin(x).unsqueeze(1) * torch.exp(-t).unsqueeze(0)
    return PDEDataset(
        name="autograd_warning_dataset",
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


def _make_diverging_sga_autograd_result(
    dataset: PDEDataset,
) -> ExperimentResult:
    """Build a synthetic ExperimentResult that:

    1. Carries ``config["algorithm"]="sga"`` and ``config["use_autograd"]=True``
       so the engine emits the autograd domain note.
    2. Has ``terms=['u', 'u**3']`` with a 100x coefficient that causes
       ``integrate_pde`` to fail (Required step size < spacing) — exercising
       the failure path that forwards ``IntegrationResult.warning`` from
       Tier 2 plots back to ``ReportResult.warnings``.
    """
    # Use a local generator so this fixture doesn't mutate the global
    # torch RNG (would otherwise leak into adjacent tests, e.g.
    # tests/unit/search/sga/test_plugin.py::test_propose_returns_kd2_format).
    rng = torch.Generator()
    rng.manual_seed(0)
    n_samples = dataset.get_shape()[0] * dataset.get_shape()[1]
    actual = torch.randn(n_samples, generator=rng)
    predicted = actual + torch.randn(n_samples, generator=rng) * 0.01
    residuals = predicted - actual

    recorder = VizRecorder()
    recorder.log("best_aic", 1.0)
    recorder.log("_best_score", 1.0)
    recorder.log("_best_expr", "u**3")
    recorder.log("_n_candidates", 1)

    return ExperimentResult(
        best_expression="u**3",
        best_score=1.0,
        iterations=1,
        early_stopped=False,
        final_eval=EvaluationResult(
            mse=0.01,
            nmse=0.01,
            r2=0.5,
            aic=-50.0,
            complexity=1,
            coefficients=torch.tensor([0.0, 100.0]),
            is_valid=True,
            error_message="",
            selected_indices=[0, 1],
            residuals=residuals,
            terms=["u", "u**3"],
            expression="u**3",
        ),
        actual=actual,
        predicted=predicted,
        dataset_name=dataset.name,
        algorithm_name="SGA",
        config={"algorithm": "sga", "use_autograd": True},
        recorder=recorder,
    )


# M1: Domain note must appear exactly once even when Tier 2 plots forward it


@pytest.mark.integration
class TestM1AutogradWarningDeduplication:
    """The SGA-autograd domain note must be emitted exactly once.

    Engine-level dedup at ``engine.py:323`` only catches identical strings.
    Tier 2 plots (field_comparison, time_slices, error_heatmap) forward
    the annotated ``integration_result.warning`` directly from the
    failure paths, so the same note enters ``report.warnings`` 3 more
    times — totaling 4 in the HTML body.
    """

    def test_autograd_domain_note_appears_once_in_html(self, tmp_path: Path) -> None:
        """HTML must contain the 'Domain note: ...use_autograd=True' substring
        exactly once even when integration fails.

        Reproduction:
        - SGA + use_autograd=True
        - Coefficients designed to make ``integrate_pde`` diverge
        - Failure path in ``_extract_pred`` (time_slices/error_heatmap) and
          ``_extract_predicted`` (field_comparison) appends
          ``integration_result.warning`` (which is the annotated note +
          failure reason) to ``warnings``
        - Engine then appends the bare note via the
          ``if autograd_note not in report.warnings`` guard, but the guard
          uses raw string equality, missing the annotated forwards.

        Expected: dedup must collapse all forwards into a single warning.
        """
        dataset = _make_pde_dataset()
        result = _make_diverging_sga_autograd_result(dataset)
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result, dataset=dataset)

        html = report.report.read_text() if report.report is not None else ""
        # The autograd domain note's signature substring.
        sentinel = "Domain note: SGA was run with use_autograd=True"
        occurrences = html.count(sentinel)
        assert occurrences == 1, (
            f"Expected the autograd domain note to appear exactly once in "
            f"the rendered HTML, got {occurrences} occurrences. The Tier 2 "
            f"plot warning-forwarding paths bypass the engine-level dedup. "
            f"Warnings collected: {report.warnings}"
        )

    def test_autograd_domain_note_appears_once_in_report_warnings(
        self, tmp_path: Path
    ) -> None:
        """report.warnings must contain the annotated note at most once
        (regardless of suffix), so the HTML 'Warnings' section is clean.
        """
        dataset = _make_pde_dataset()
        result = _make_diverging_sga_autograd_result(dataset)
        engine = VizEngine(output_dir=tmp_path)
        report = engine.render_all(result, dataset=dataset)

        sentinel = "Domain note: SGA was run with use_autograd=True"
        matching = [w for w in report.warnings if sentinel in w]
        assert len(matching) == 1, (
            f"Expected exactly one warning containing the autograd domain "
            f"note in report.warnings; got {len(matching)}: {matching}"
        )

        # Lock that the integration-failure path is actually exercised. Without
        # this assertion, a future change that stops integration from diverging
        # would silently turn the dedup test into a no-op (it would pass but
        # measure nothing).
        assert any(
            "integrate" in w.lower() or "step size" in w.lower()
            for w in report.warnings
        ), (
            f"integration-failure path was not exercised; expected an "
            f"'integrate' or 'step size' warning, got: {report.warnings}"
        )


# L2: Viz fixtures must set algorithm + use_autograd explicitly


@pytest.mark.integration
class TestL2VizFixtureExplicitConfig:
    """Regression: viz integration fixtures must set ``algorithm`` and
    ``use_autograd`` config keys explicitly.

    The ``_make_experiment_result`` helper in
    ``tests/integration/test_viz_pipeline.py`` currently sets
    ``config={"max_iter": ..., "population_size": ..., "seed": ...}`` —
    no ``algorithm`` key, no ``use_autograd``. The HTML autograd warning
    code reads ``config.get("algorithm")`` and ``config.get("use_autograd")``
    so the warning branch silently no-ops. Brittle: if a future refactor
    breaks the warning emit logic, these tests cannot detect it.
    """

    def test_viz_pipeline_helper_sets_algorithm_key(self) -> None:
        """``_make_experiment_result`` must populate ``config['algorithm']``."""
        from tests.integration.test_viz_pipeline import (
            _make_experiment_result,
        )

        result = _make_experiment_result()
        assert "algorithm" in result.config, (
            "Viz pipeline test fixture is missing config['algorithm']. "
            "Without it, the autograd-warning branch in VizEngine cannot "
            "be exercised by these tests."
        )
        # Must be one of the known algorithm strings.
        assert result.config["algorithm"] in {"sga", "discover", "dlga"}, (
            f"config['algorithm'] should be a known algorithm name, got "
            f"{result.config['algorithm']!r}"
        )

    def test_viz_pipeline_helper_sets_use_autograd_key(self) -> None:
        """``_make_experiment_result`` must populate ``config['use_autograd']``."""
        from tests.integration.test_viz_pipeline import (
            _make_experiment_result,
        )

        result = _make_experiment_result()
        assert "use_autograd" in result.config, (
            "Viz pipeline test fixture is missing config['use_autograd']. "
            "Without it, the autograd-warning branch silently no-ops."
        )
        assert isinstance(result.config["use_autograd"], bool), (
            f"config['use_autograd'] must be a bool, got "
            f"{type(result.config['use_autograd']).__name__}"
        )
