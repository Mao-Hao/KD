"""Integration tests for SGA autograd mode

Locks the three-layer derivative semantics defined in - Layer 1 (raw u leaf): ``data_dict["u"]`` always equals raw
                          ``dataset.fields["u"]`` regardless of
                          ``use_autograd``.
- Layer 2 (terminal u_x): ``data_dict["u_x"]`` differs between
                          ``use_autograd=True`` and ``False`` when the
                          surrogate is non-trivial — proves Layer 2
                          actually routes through the AutogradProvider.
- Layer 3 (tree d): SGA's internal ``_finite_diff_torch`` is the
                          sole derivative engine for tree ``d`` / ``d^2``
                          operators, so ``execute_tree(d(u, x))`` is
                          identical between modes. ``execute_tree(d(u_x, x))``
                          differs because its leaf ``u_x`` source differs
                          (AD-then-FD mix, intentional per paper / the predecessor).

Plus:
- ``components.context.derivative_provider`` is unchanged after
  ``prepare(use_autograd=True)`` — the surrogate provider lives only on
  the plugin instance (no platform pollution).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from kd2.core.evaluator import Evaluator
from kd2.core.executor.context import ExecutionContext
from kd2.core.expr import FunctionRegistry, PythonExecutor
from kd2.core.linear_solve.least_squares import LeastSquaresSolver
from kd2.data.derivatives.autograd import AutogradProvider
from kd2.data.derivatives.finite_diff import FiniteDiffProvider
from kd2.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd2.models import FieldModel
from kd2.search.protocol import PlatformComponents
from kd2.search.runner import ExperimentRunner
from kd2.search.sga import SGAConfig, SGAPlugin
from kd2.search.sga.evaluate import execute_tree
from kd2.search.sga.tree import Node, Tree
from kd2.viz.report import ReportResult

_NX = 16
_NT = 8
_X_MAX = 2.0 * math.pi
_T_MAX = 1.0
_SEED = 42


def _build_dataset() -> PDEDataset:
    """Small smooth dataset: u = sin(x) * exp(-t)."""
    dtype = torch.float64
    x = torch.linspace(0.0, _X_MAX, _NX + 1, dtype=dtype)[:-1]
    t = torch.linspace(0.0, _T_MAX, _NT, dtype=dtype)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    u = torch.sin(gx) * torch.exp(-gt)
    return PDEDataset(
        name="autograd-test",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=True),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _build_components(dataset: PDEDataset) -> PlatformComponents:
    provider = FiniteDiffProvider(dataset, max_order=2)
    context = ExecutionContext(dataset=dataset, derivative_provider=provider)
    registry = FunctionRegistry.create_default()
    executor = PythonExecutor(registry)
    solver = LeastSquaresSolver()
    u_t = provider.get_derivative("u", "t", order=1).flatten()
    evaluator = Evaluator(
        executor=executor,
        solver=solver,
        context=context,
        lhs=u_t,
    )
    return PlatformComponents(
        dataset=dataset,
        executor=executor,
        evaluator=evaluator,
        context=context,
        registry=registry,
    )


def _untrained_field_model(dataset: PDEDataset) -> FieldModel:
    """A FieldModel with random init (no training).

    Avoids slow training in CI while still proving the AD wiring: random
    weights still produce non-trivial gradients via ``torch.autograd.grad``,
    and the resulting Layer 2 tensor is detectably different from FD.
    """
    torch.manual_seed(_SEED)
    model = FieldModel(
        coord_names=list(dataset.axes.keys()),
        field_names=list(dataset.fields.keys()),
        hidden_sizes=[8, 8],
    ).to(dtype=torch.float64)
    model.eval()
    return model


def _config(use_autograd: bool, dataset: PDEDataset) -> SGAConfig:
    return SGAConfig(
        num=4,
        depth=3,
        width=3,
        seed=_SEED,
        use_autograd=use_autograd,
        field_model=_untrained_field_model(dataset) if use_autograd else None,
    )


@pytest.fixture(scope="module")
def dataset() -> PDEDataset:
    return _build_dataset()


@pytest.fixture(scope="module")
def fd_components(dataset: PDEDataset) -> PlatformComponents:
    return _build_components(dataset)


@pytest.fixture(scope="module")
def ad_components(dataset: PDEDataset) -> PlatformComponents:
    """Separate components instance to verify provider non-pollution."""
    return _build_components(dataset)


@pytest.fixture(scope="module")
def fd_plugin(dataset: PDEDataset, fd_components: PlatformComponents) -> SGAPlugin:
    plugin = SGAPlugin(_config(use_autograd=False, dataset=dataset))
    plugin.prepare(fd_components)
    return plugin


@pytest.fixture(scope="module")
def ad_plugin(dataset: PDEDataset, ad_components: PlatformComponents) -> SGAPlugin:
    plugin = SGAPlugin(_config(use_autograd=True, dataset=dataset))
    plugin.prepare(ad_components)
    return plugin


@pytest.mark.integration
class TestThreeLayerSemantics:
    """Lock the three-layer derivative invariants from."""

    def test_layer1_raw_u_invariant_under_use_autograd(
        self,
        dataset: PDEDataset,
        fd_plugin: SGAPlugin,
        ad_plugin: SGAPlugin,
    ) -> None:
        """Layer 1: data_dict['u'] is raw dataset field, regardless of mode."""
        raw_u_flat = dataset.fields["u"].values.flatten()
        # Both plugins must store identical raw u
        torch.testing.assert_close(
            fd_plugin._data_dict["u"], raw_u_flat, rtol=0, atol=0
        )
        torch.testing.assert_close(
            ad_plugin._data_dict["u"], raw_u_flat, rtol=0, atol=0
        )

    def test_layer2_terminal_routes_through_autograd(
        self, fd_plugin: SGAPlugin, ad_plugin: SGAPlugin
    ) -> None:
        """Layer 2: u_x terminal differs FD vs AD (proves AD path is live)."""
        fd_ux = fd_plugin._data_dict["u_x"]
        ad_ux = ad_plugin._data_dict["u_x"]
        assert fd_ux.shape == ad_ux.shape
        # Random-init NN's AD derivatives are nowhere near FD u_x
        max_abs_diff = (fd_ux - ad_ux).abs().max().item()
        assert max_abs_diff > 1e-3, (
            f"Layer 2 appears not to route through AutogradProvider: "
            f"FD u_x and AD u_x differ by only {max_abs_diff:.3e}"
        )

    def test_layer2_lhs_target_routes_through_autograd(
        self, fd_plugin: SGAPlugin, ad_plugin: SGAPlugin
    ) -> None:
        """LHS target u_t also routes through plugin's AD provider."""
        assert fd_plugin._y is not None and ad_plugin._y is not None
        max_abs_diff = (fd_plugin._y - ad_plugin._y).abs().max().item()
        assert max_abs_diff > 1e-3, (
            f"LHS target appears not to use AD provider when use_autograd=True: "
            f"max diff {max_abs_diff:.3e}"
        )

    def test_autograd_layer2_values_are_detached(self, ad_plugin: SGAPlugin) -> None:
        """SGA stores numeric AD terminal values, not PyTorch graphs."""
        assert ad_plugin._y is not None
        assert ad_plugin._y.requires_grad is False
        assert ad_plugin._data_dict["u_x"].requires_grad is False

    def test_layer3_tree_d_on_raw_u_invariant(
        self, fd_plugin: SGAPlugin, ad_plugin: SGAPlugin
    ) -> None:
        """Layer 3: tree d(u, x) is identical between modes.

        Reasoning: leaf 'u' is raw (Layer 1 invariant), and tree's 'd'
        operator goes through SGA's internal _finite_diff_torch regardless
        of provider — so d(u, x) results must be bit-identical.
        """
        tree = Tree(
            root=Node(
                name="d",
                arity=2,
                children=[Node(name="u", arity=0), Node(name="x", arity=0)],
            )
        )
        fd_result = execute_tree(tree, fd_plugin._data_dict, fd_plugin._diff_ctx)
        ad_result = execute_tree(tree, ad_plugin._data_dict, ad_plugin._diff_ctx)
        torch.testing.assert_close(fd_result, ad_result, rtol=0, atol=0)

    def test_layer3_tree_d_on_u_x_differs_due_to_leaf_source(
        self, fd_plugin: SGAPlugin, ad_plugin: SGAPlugin
    ) -> None:
        """d(u_x, x) DIFFERS because Layer 2 leaf u_x source differs.

        This is the documented 'AD-then-FD' mixed semantics — the outer
        d operator is the same FD code, but the inner u_x leaf is fed
        from the AD provider in one case and from FD in the other.
        """
        tree = Tree(
            root=Node(
                name="d",
                arity=2,
                children=[Node(name="u_x", arity=0), Node(name="x", arity=0)],
            )
        )
        fd_result = execute_tree(tree, fd_plugin._data_dict, fd_plugin._diff_ctx)
        ad_result = execute_tree(tree, ad_plugin._data_dict, ad_plugin._diff_ctx)
        max_abs_diff = (fd_result - ad_result).abs().max().item()
        assert max_abs_diff > 1e-3, (
            f"Expected d(u_x, x) to differ between FD and AD modes "
            f"(leaf u_x source differs); got max diff {max_abs_diff:.3e}"
        )

    def test_use_autograd_does_not_replace_context_provider(
        self, ad_components: PlatformComponents, ad_plugin: SGAPlugin
    ) -> None:
        """Platform-shared provider must remain untouched (-D2)."""
        assert isinstance(
            ad_components.context.derivative_provider, FiniteDiffProvider
        ), (
            "use_autograd=True should not replace the platform-shared "
            "context.derivative_provider"
        )
        assert isinstance(ad_plugin._autograd_provider, AutogradProvider)
        assert (
            ad_plugin._autograd_provider
            is not ad_components.context.derivative_provider
        )

    def test_runner_result_actual_uses_autograd_target(
        self, dataset: PDEDataset
    ) -> None:
        """ExperimentResult target must match the SGA final-eval domain."""
        components = _build_components(dataset)
        plugin = SGAPlugin(_config(use_autograd=True, dataset=dataset))
        runner = ExperimentRunner(plugin, max_iterations=1, batch_size=4)

        result = runner.run(components)

        assert plugin._y is not None
        torch.testing.assert_close(result.actual, plugin._y, rtol=0, atol=0)
        assert result.final_eval.residuals is not None
        torch.testing.assert_close(
            result.predicted,
            result.actual + result.final_eval.residuals,
            rtol=0,
            atol=0,
        )
        fd_target = components.evaluator.lhs_target
        assert isinstance(fd_target, torch.Tensor)
        assert not torch.allclose(result.actual, fd_target)


@pytest.mark.integration
class TestAutogradConfigValidation:
    """SGAConfig + prepare() validation around use_autograd."""

    def test_field_model_coord_mismatch_rejected(self, dataset: PDEDataset) -> None:
        """Pre-trained model with wrong coord_names must raise."""
        bad_model = FieldModel(
            coord_names=["x", "y"], # wrong: dataset has [x, t]
            field_names=["u"],
            hidden_sizes=[4],
        )
        cfg = SGAConfig(use_autograd=True, field_model=bad_model)
        plugin = SGAPlugin(cfg)
        components = _build_components(dataset)
        with pytest.raises(ValueError, match="coord_names"):
            plugin.prepare(components)

    def test_field_model_field_mismatch_rejected(self, dataset: PDEDataset) -> None:
        """Pre-trained model with wrong field_names must raise."""
        bad_model = FieldModel(
            coord_names=["x", "t"],
            field_names=["v"], # wrong: dataset has u
            hidden_sizes=[4],
        )
        cfg = SGAConfig(use_autograd=True, field_model=bad_model)
        plugin = SGAPlugin(cfg)
        components = _build_components(dataset)
        with pytest.raises(ValueError, match="field_names"):
            plugin.prepare(components)

    def test_use_autograd_false_leaves_provider_untouched(
        self, dataset: PDEDataset
    ) -> None:
        """Default use_autograd=False does not allocate AutogradProvider."""
        components = _build_components(dataset)
        plugin = SGAPlugin(SGAConfig(use_autograd=False, seed=_SEED))
        plugin.prepare(components)
        assert plugin._autograd_provider is None


@pytest.mark.integration
class TestHtmlAutogradDomainWarning:
    """: HTML report warns when SGA autograd fit domain != FD integrator.

    When SGAConfig.use_autograd=True, SGA fits coefficients in the AD
    domain (NN-smoothed first-order derivatives) but the field-comparison
    integrator uses finite-difference spatial derivatives at every
    timestep. Final-eval metrics and field-comparison metrics therefore
    measure different things and may disagree on noisy data; the HTML
    report must say so explicitly.
    """

    def _run_sga_and_render(
        self,
        dataset: PDEDataset,
        use_autograd: bool,
        tmp_path: Path,
    ) -> ReportResult:
        from kd2.viz import VizEngine

        components = _build_components(dataset)
        plugin = SGAPlugin(_config(use_autograd=use_autograd, dataset=dataset))
        runner = ExperimentRunner(plugin, max_iterations=1, batch_size=4)
        result = runner.run(components)
        engine = VizEngine(output_dir=tmp_path)
        return engine.render_all(result, dataset=dataset)

    def test_autograd_true_emits_domain_warning(
        self, dataset: PDEDataset, tmp_path: Path
    ) -> None:
        report = self._run_sga_and_render(dataset, use_autograd=True, tmp_path=tmp_path)
        joined = "\n".join(report.warnings).lower()
        assert "autograd" in joined, (
            f"Expected autograd domain note in HTML warnings; got: {report.warnings}"
        )
        assert "domain" in joined
        # And it must reach the rendered HTML body too.
        html = report.report.read_text() if report.report is not None else ""
        assert "use_autograd=True" in html

    def test_autograd_false_omits_domain_warning(
        self, dataset: PDEDataset, tmp_path: Path
    ) -> None:
        report = self._run_sga_and_render(
            dataset, use_autograd=False, tmp_path=tmp_path
        )
        joined = "\n".join(report.warnings).lower()
        assert "use_autograd=true" not in joined, (
            f"Did not expect autograd domain note for FD-only run; "
            f"got: {report.warnings}"
        )
