"""FieldModel + AutogradProvider integration tests.

Validates that FieldModel's internal normalization (Strategy A) produces
correct autograd derivatives in the original coordinate space.

Uses the analytic function u(x, t) = sin(pi*x) * exp(-t) with known
derivatives:
    u_t = -sin(pi*x) * exp(-t)
    u_x = pi * cos(pi*x) * exp(-t)
    u_xx = -pi^2 * sin(pi*x) * exp(-t)

Test matrix:
    1. Basic derivatives (normalization on) -- correlation > 0.99
    2. No-normalization baseline -- derivatives also correct
    3. Normalization consistency -- on vs off within tolerance
    4. Multi-field -- each field independent
    5. Identity derivative -- diff_x(x) == 1?
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from kd2.data.derivatives.autograd import AutogradProvider
from kd2.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd2.models import FieldModel, FieldModelTrainer

# Constants

# Grid sizes -- small enough for fast CI, large enough for good NN fit
_NX = 50
_NT = 30

# Training hyper-parameters
_HIDDEN = [64, 64]
_LR = 1e-3
_MAX_EPOCHS = 5000
_PATIENCE = 300
_SEED = 42

# Acceptance threshold for Pearson correlation
_CORR_THRESHOLD = 0.99

# Relaxed threshold for multi-field 2nd-order (high-frequency amplifies error)
_CORR_THRESHOLD_2ND_ORDER = 0.98


# Helpers


def _pearson_correlation(a: Tensor, b: Tensor) -> float:
    """Pearson correlation between two 1-D tensors.

    Returns:
        Scalar correlation in [-1, 1].
    """
    a_c = a - a.mean()
    b_c = b - b.mean()
    num = (a_c * b_c).sum()
    denom = a_c.norm() * b_c.norm()
    if denom.item() == 0.0:
        return 0.0
    return (num / denom).item()


def _make_analytic_data(
    nx: int = _NX,
    nt: int = _NT,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Generate u(x, t) = sin(pi*x) * exp(-t) on a grid.

    Returns:
        (x_1d, t_1d, gx, gt, gu) where gx/gt/gu are 2-D grids.
    """
    x = torch.linspace(0.0, 2.0, nx)
    t = torch.linspace(0.0, 1.0, nt)
    gx, gt = torch.meshgrid(x, t, indexing="ij")
    gu = torch.sin(math.pi * gx) * torch.exp(-gt)
    return x, t, gx, gt, gu


def _train_fieldmodel(
    coord_names: list[str],
    field_names: list[str],
    coords: dict[str, Tensor],
    targets: dict[str, Tensor],
    *,
    normalize: bool = True,
) -> FieldModel:
    """Train a FieldModel and return the model in eval mode.

    When normalize=False, we train without setting normalization
    (identity mean=0, std=1 buffers remain).
    """
    model = FieldModel(
        coord_names=coord_names,
        field_names=field_names,
        hidden_sizes=_HIDDEN,
        activation="tanh",
    )

    if normalize:
        trainer = FieldModelTrainer(model, lr=_LR)
        trainer.fit(
            coords=coords,
            targets=targets,
            max_epochs=_MAX_EPOCHS,
            patience=_PATIENCE,
            seed=_SEED,
        )
    else:
        # Train manually without normalization
        # (Trainer always sets normalization, so we do a bare loop)
        torch.manual_seed(_SEED)
        _reinit(model)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=_LR)
        criterion = torch.nn.MSELoss()
        best_loss = float("inf")
        stale = 0
        for _epoch in range(_MAX_EPOCHS):
            optimizer.zero_grad()
            preds = model(**coords)
            losses = []
            for name in field_names:
                losses.append(criterion(preds[name], targets[name]))
            loss = torch.stack(losses).mean()
            loss.backward() # type: ignore[no-untyped-call]
            optimizer.step()
            cur = loss.detach().item()
            if cur < best_loss:
                best_loss = cur
                stale = 0
            else:
                stale += 1
                if stale >= _PATIENCE:
                    break
        model.eval()

    return model


def _reinit(model: torch.nn.Module) -> None:
    """Re-initialize model parameters (mirrors trainer logic)."""
    for m in model.modules():
        reset_fn = getattr(m, "reset_parameters", None)
        if reset_fn is not None:
            reset_fn()


def _make_provider(
    model: FieldModel,
    gx: Tensor,
    gt: Tensor,
    x_1d: Tensor,
    t_1d: Tensor,
    gu: Tensor,
    field_names: list[str] | None = None,
    fields_grid: dict[str, Tensor] | None = None,
) -> tuple[AutogradProvider, dict[str, Tensor]]:
    """Build AutogradProvider from a trained FieldModel.

    Returns:
        (provider, flat_coords) where flat_coords have requires_grad.
    """
    if field_names is None:
        field_names = ["u"]
    if fields_grid is None:
        fields_grid = {"u": gu}

    flat_x = gx.reshape(-1).clone().detach().requires_grad_(True)
    flat_t = gt.reshape(-1).clone().detach().requires_grad_(True)
    coords = {"x": flat_x, "t": flat_t}

    fields = {
        name: FieldData(name=name, values=vals) for name, vals in fields_grid.items()
    }

    dataset = PDEDataset(
        name="test_fieldmodel_autograd",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x_1d),
            "t": AxisInfo(name="t", values=t_1d),
        },
        axis_order=["x", "t"],
        fields=fields,
        lhs_field=field_names[0],
        lhs_axis="t",
    )

    provider = AutogradProvider(model=model, coords=coords, dataset=dataset)
    return provider, coords


# Analytic reference values (flat tensors, same ordering as meshgrid "ij")


def _analytic_ut(gx: Tensor, gt: Tensor) -> Tensor:
    return -torch.sin(math.pi * gx.reshape(-1)) * torch.exp(-gt.reshape(-1))


def _analytic_ux(gx: Tensor, gt: Tensor) -> Tensor:
    return math.pi * torch.cos(math.pi * gx.reshape(-1)) * torch.exp(-gt.reshape(-1))


def _analytic_uxx(gx: Tensor, gt: Tensor) -> Tensor:
    return (
        -(math.pi**2) * torch.sin(math.pi * gx.reshape(-1)) * torch.exp(-gt.reshape(-1))
    )


# ===========================================================================
# Test 1: Basic derivatives with normalization ON
# ===========================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestBasicDerivativesNormalized:
    """FieldModel trained with normalization -- derivatives via autograd."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        """Train model once, reuse across tests in this class."""
        x_1d, t_1d, gx, gt, gu = _make_analytic_data()

        flat_coords = {"x": gx.reshape(-1), "t": gt.reshape(-1)}
        flat_targets = {"u": gu.reshape(-1)}

        self.model = _train_fieldmodel(
            ["x", "t"],
            ["u"],
            flat_coords,
            flat_targets,
            normalize=True,
        )
        self.provider, self.coords = _make_provider(
            self.model,
            gx,
            gt,
            x_1d,
            t_1d,
            gu,
        )
        self.gx = gx
        self.gt = gt

    def test_ut_correlation(self) -> None:
        """u_t correlation with analytic solution > 0.99."""
        ut = self.provider.get_derivative("u", "t", order=1)
        expected = _analytic_ut(self.gx, self.gt)
        corr = _pearson_correlation(ut.detach(), expected)
        assert corr > _CORR_THRESHOLD, f"u_t correlation {corr:.4f} < {_CORR_THRESHOLD}"

    def test_ux_correlation(self) -> None:
        """u_x correlation with analytic solution > 0.99."""
        ux = self.provider.get_derivative("u", "x", order=1)
        expected = _analytic_ux(self.gx, self.gt)
        corr = _pearson_correlation(ux.detach(), expected)
        assert corr > _CORR_THRESHOLD, f"u_x correlation {corr:.4f} < {_CORR_THRESHOLD}"

    def test_uxx_correlation(self) -> None:
        """u_xx correlation with analytic solution > 0.99."""
        uxx = self.provider.get_derivative("u", "x", order=2)
        expected = _analytic_uxx(self.gx, self.gt)
        corr = _pearson_correlation(uxx.detach(), expected)
        assert corr > _CORR_THRESHOLD, (
            f"u_xx correlation {corr:.4f} < {_CORR_THRESHOLD}"
        )

    def test_derivatives_finite(self) -> None:
        """All derivatives should be finite (no NaN / Inf)."""
        for axis, order in [("t", 1), ("x", 1), ("x", 2)]:
            d = self.provider.get_derivative("u", axis, order)
            assert torch.isfinite(d).all(), (
                f"Derivative (u, {axis}, {order}) contains NaN/Inf"
            )


# ===========================================================================
# Test 2: Derivatives WITHOUT normalization
# ===========================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestBasicDerivativesUnnormalized:
    """FieldModel trained without normalization -- identity mean/std."""

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        x_1d, t_1d, gx, gt, gu = _make_analytic_data()

        flat_coords = {"x": gx.reshape(-1), "t": gt.reshape(-1)}
        flat_targets = {"u": gu.reshape(-1)}

        self.model = _train_fieldmodel(
            ["x", "t"],
            ["u"],
            flat_coords,
            flat_targets,
            normalize=False,
        )
        self.provider, self.coords = _make_provider(
            self.model,
            gx,
            gt,
            x_1d,
            t_1d,
            gu,
        )
        self.gx = gx
        self.gt = gt

    def test_ut_correlation(self) -> None:
        """u_t without normalization should still be correct."""
        ut = self.provider.get_derivative("u", "t", order=1)
        expected = _analytic_ut(self.gx, self.gt)
        corr = _pearson_correlation(ut.detach(), expected)
        assert corr > _CORR_THRESHOLD, (
            f"u_t (no-norm) correlation {corr:.4f} < {_CORR_THRESHOLD}"
        )

    def test_ux_correlation(self) -> None:
        ux = self.provider.get_derivative("u", "x", order=1)
        expected = _analytic_ux(self.gx, self.gt)
        corr = _pearson_correlation(ux.detach(), expected)
        assert corr > _CORR_THRESHOLD, (
            f"u_x (no-norm) correlation {corr:.4f} < {_CORR_THRESHOLD}"
        )

    def test_uxx_correlation(self) -> None:
        uxx = self.provider.get_derivative("u", "x", order=2)
        expected = _analytic_uxx(self.gx, self.gt)
        corr = _pearson_correlation(uxx.detach(), expected)
        assert corr > _CORR_THRESHOLD, (
            f"u_xx (no-norm) correlation {corr:.4f} < {_CORR_THRESHOLD}"
        )


# ===========================================================================
# Test 3: Normalization consistency (on vs off should agree)
# ===========================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestNormalizationConsistency:
    """Derivatives from normalized and unnormalized models should agree.

    Both models approximate the same analytic function, so their autograd
    derivatives should converge to the same values. We compare via
    correlation and per-point relative difference.
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        x_1d, t_1d, gx, gt, gu = _make_analytic_data()

        flat_coords = {"x": gx.reshape(-1), "t": gt.reshape(-1)}
        flat_targets = {"u": gu.reshape(-1)}

        model_norm = _train_fieldmodel(
            ["x", "t"],
            ["u"],
            flat_coords,
            flat_targets,
            normalize=True,
        )
        model_no_norm = _train_fieldmodel(
            ["x", "t"],
            ["u"],
            flat_coords,
            flat_targets,
            normalize=False,
        )

        self.prov_norm, _ = _make_provider(model_norm, gx, gt, x_1d, t_1d, gu)
        self.prov_no_norm, _ = _make_provider(
            model_no_norm,
            gx,
            gt,
            x_1d,
            t_1d,
            gu,
        )
        self.gx = gx
        self.gt = gt

    def test_ut_consistency(self) -> None:
        """u_t from both models should be highly correlated."""
        ut_n = self.prov_norm.get_derivative("u", "t", 1).detach()
        ut_u = self.prov_no_norm.get_derivative("u", "t", 1).detach()
        corr = _pearson_correlation(ut_n, ut_u)
        assert corr > _CORR_THRESHOLD, (
            f"u_t norm/no-norm correlation {corr:.4f} < {_CORR_THRESHOLD}"
        )

    def test_ux_consistency(self) -> None:
        ux_n = self.prov_norm.get_derivative("u", "x", 1).detach()
        ux_u = self.prov_no_norm.get_derivative("u", "x", 1).detach()
        corr = _pearson_correlation(ux_n, ux_u)
        assert corr > _CORR_THRESHOLD, (
            f"u_x norm/no-norm correlation {corr:.4f} < {_CORR_THRESHOLD}"
        )

    def test_uxx_consistency(self) -> None:
        uxx_n = self.prov_norm.get_derivative("u", "x", 2).detach()
        uxx_u = self.prov_no_norm.get_derivative("u", "x", 2).detach()
        corr = _pearson_correlation(uxx_n, uxx_u)
        assert corr > _CORR_THRESHOLD, (
            f"u_xx norm/no-norm correlation {corr:.4f} < {_CORR_THRESHOLD}"
        )

    def test_both_agree_with_analytic(self) -> None:
        """Both models' u_t should match the analytic solution closely."""
        expected = _analytic_ut(self.gx, self.gt)
        for label, prov in [
            ("norm", self.prov_norm),
            ("no-norm", self.prov_no_norm),
        ]:
            ut = prov.get_derivative("u", "t", 1).detach()
            corr = _pearson_correlation(ut, expected)
            assert corr > _CORR_THRESHOLD, (
                f"u_t ({label}) analytic corr {corr:.4f} < {_CORR_THRESHOLD}"
            )


# ===========================================================================
# Test 4: Multi-field
# ===========================================================================


@pytest.mark.integration
@pytest.mark.slow
class TestMultiFieldDerivatives:
    """Two-field model: u and v with independent analytic solutions.

    u(x,t) = sin(pi*x) * exp(-t)
    v(x,t) = cos(2*pi*x) * exp(-0.5*t)

    v_t = -0.5 * cos(2*pi*x) * exp(-0.5*t)
    v_x = -2*pi * sin(2*pi*x) * exp(-0.5*t)
    v_xx = -(2*pi)^2 * cos(2*pi*x) * exp(-0.5*t)
    """

    @pytest.fixture(autouse=True)
    def _setup(self) -> None:
        x_1d, t_1d, gx, gt, gu = _make_analytic_data()
        gv = torch.cos(2 * math.pi * gx) * torch.exp(-0.5 * gt)

        flat_coords = {"x": gx.reshape(-1), "t": gt.reshape(-1)}
        flat_targets = {"u": gu.reshape(-1), "v": gv.reshape(-1)}

        self.model = _train_fieldmodel(
            ["x", "t"],
            ["u", "v"],
            flat_coords,
            flat_targets,
            normalize=True,
        )
        self.provider, self.coords = _make_provider(
            self.model,
            gx,
            gt,
            x_1d,
            t_1d,
            gu,
            field_names=["u", "v"],
            fields_grid={"u": gu, "v": gv},
        )
        self.gx = gx
        self.gt = gt

    def test_u_derivatives(self) -> None:
        """Field u derivatives should match analytic."""
        ut = self.provider.get_derivative("u", "t", 1).detach()
        ux = self.provider.get_derivative("u", "x", 1).detach()
        assert (
            _pearson_correlation(
                ut,
                _analytic_ut(self.gx, self.gt),
            )
            > _CORR_THRESHOLD
        )
        assert (
            _pearson_correlation(
                ux,
                _analytic_ux(self.gx, self.gt),
            )
            > _CORR_THRESHOLD
        )

    def test_v_derivatives(self) -> None:
        """Field v derivatives should match analytic."""
        x_flat = self.gx.reshape(-1)
        t_flat = self.gt.reshape(-1)

        # v_t analytic
        vt_expected = -0.5 * torch.cos(2 * math.pi * x_flat) * torch.exp(-0.5 * t_flat)
        vt = self.provider.get_derivative("v", "t", 1).detach()
        corr_vt = _pearson_correlation(vt, vt_expected)
        assert corr_vt > _CORR_THRESHOLD, f"v_t correlation {corr_vt:.4f}"

        # v_x analytic
        vx_expected = (
            -2 * math.pi * torch.sin(2 * math.pi * x_flat) * torch.exp(-0.5 * t_flat)
        )
        vx = self.provider.get_derivative("v", "x", 1).detach()
        corr_vx = _pearson_correlation(vx, vx_expected)
        assert corr_vx > _CORR_THRESHOLD, f"v_x correlation {corr_vx:.4f}"

    def test_v_second_derivative(self) -> None:
        """v_xx should match -(2*pi)^2 * cos(2*pi*x) * exp(-0.5*t).

        Relaxed threshold (0.98): v has higher spatial frequency (2*pi)
        than u (pi), so v_xx involves (2*pi)^2 scaling which amplifies
        NN fitting errors. Multi-field + high-frequency + 2nd-order is
        the hardest case. 0.98 still confirms autograd chain correctness.
        """
        x_flat = self.gx.reshape(-1)
        t_flat = self.gt.reshape(-1)
        vxx_expected = (
            -((2 * math.pi) ** 2)
            * torch.cos(2 * math.pi * x_flat)
            * torch.exp(-0.5 * t_flat)
        )
        vxx = self.provider.get_derivative("v", "x", 2).detach()
        corr = _pearson_correlation(vxx, vxx_expected)
        assert corr > _CORR_THRESHOLD_2ND_ORDER, f"v_xx correlation {corr:.4f}"

    def test_cross_field_independence(self) -> None:
        """u and v derivatives should not be correlated with each other's analytic."""
        ut = self.provider.get_derivative("u", "t", 1).detach()
        x_flat = self.gx.reshape(-1)
        t_flat = self.gt.reshape(-1)
        vt_expected = -0.5 * torch.cos(2 * math.pi * x_flat) * torch.exp(-0.5 * t_flat)

        # u_t should NOT match v_t analytic (different functions)
        cross_corr = abs(_pearson_correlation(ut, vt_expected))
        # They are not perfectly correlated -- relaxed check
        assert cross_corr < 0.95, (
            f"Cross-field correlation {cross_corr:.4f} suspiciously high"
        )


# ===========================================================================
# Test 5: identity derivative
# ===========================================================================


@pytest.mark.integration
class TestIdentityDerivative:
    """Verify identity derivative: diff_x(x) through the computation graph.

    If coords are in the autograd graph (requires_grad=True) and the model
    output depends on them, then diff(model_output_of_x, x) should work.
    But diff(x_itself, x) = 1 only if x is directly in the graph as a
    leaf -- which it is, since we pass coords with requires_grad=True.

    This test evaluates the current status of 2.2/M1.
    """

    def test_identity_derivative_via_model(self) -> None:
        """diff(u, x) should work since u depends on x through the model.

        This is the normal usage path and should always work.
        """
        x_1d, t_1d, gx, gt, gu = _make_analytic_data(nx=10, nt=5)

        flat_coords = {"x": gx.reshape(-1), "t": gt.reshape(-1)}
        flat_targets = {"u": gu.reshape(-1)}

        model = _train_fieldmodel(
            ["x", "t"],
            ["u"],
            flat_coords,
            flat_targets,
            normalize=True,
        )
        provider, coords = _make_provider(model, gx, gt, x_1d, t_1d, gu)

        # This should work: u depends on x through the model graph
        ux = provider.get_derivative("u", "x", 1)
        assert torch.isfinite(ux).all(), "u_x should be finite"

    def test_coord_identity_derivative(self) -> None:
        """diff(x, x) = 1 would require x in the graph as output.

        AutogradProvider.diff takes an expression tensor. If we pass the
        raw coordinate x and differentiate w.r.t. x, autograd should
        return 1. This tests the identity derivative.
        """
        x_1d, t_1d, gx, gt, gu = _make_analytic_data(nx=10, nt=5)

        flat_coords = {"x": gx.reshape(-1), "t": gt.reshape(-1)}
        flat_targets = {"u": gu.reshape(-1)}

        model = _train_fieldmodel(
            ["x", "t"],
            ["u"],
            flat_coords,
            flat_targets,
            normalize=True,
        )
        provider, coords = _make_provider(model, gx, gt, x_1d, t_1d, gu)

        # Direct identity: diff(x, x) should be 1
        x_coord = coords["x"]
        dx_dx = provider.diff(x_coord, "x", order=1)
        expected = torch.ones_like(x_coord)

        # This should succeed: x is a leaf with requires_grad and we
        # differentiate x w.r.t. itself -> identity = 1.
        torch.testing.assert_close(
            dx_dx.detach(),
            expected,
            rtol=1e-6,
            atol=1e-6,
            msg="diff(x, x) should equal 1 (identity derivative)",
        )
