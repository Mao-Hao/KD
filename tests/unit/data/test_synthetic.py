"""Tests for synthetic PDE data generation."""

import pytest
import torch

from kd2.data.schema import DataTopology, PDEDataset, TaskType
from kd2.data.synthetic import generate_burgers_data

# =============================================================================
# Smoke Tests
# =============================================================================


class TestSyntheticSmoke:
    """Smoke tests: basic function existence and returns."""

    @pytest.mark.smoke
    def test_generate_burgers_exists(self) -> None:
        """generate_burgers_data function is callable."""
        assert callable(generate_burgers_data)

    @pytest.mark.smoke
    def test_generate_burgers_returns_pde_dataset(self) -> None:
        """generate_burgers_data returns PDEDataset."""
        dataset = generate_burgers_data(nx=16, nt=11)
        assert isinstance(dataset, PDEDataset)

    @pytest.mark.smoke
    def test_generate_burgers_default_params(self) -> None:
        """generate_burgers_data works with default parameters."""
        dataset = generate_burgers_data()
        assert dataset is not None
        assert dataset.name is not None


# =============================================================================
# Unit Tests: Shape and Structure
# =============================================================================


class TestBurgersShape:
    """Tests for Burgers data shape and structure."""

    @pytest.mark.unit
    def test_shape_matches_nx_nt(self) -> None:
        """Output shape matches (nx, nt) parameters."""
        nx, nt = 128, 51
        dataset = generate_burgers_data(nx=nx, nt=nt)
        assert dataset.get_shape() == (nx, nt)

    @pytest.mark.unit
    def test_field_shape_matches(self) -> None:
        """Field 'u' has correct shape."""
        nx, nt = 64, 31
        dataset = generate_burgers_data(nx=nx, nt=nt)
        u = dataset.get_field("u")
        assert u.shape == (nx, nt)

    @pytest.mark.unit
    def test_x_coords_length(self) -> None:
        """X coordinates have correct length."""
        nx = 100
        dataset = generate_burgers_data(nx=nx, nt=21)
        x = dataset.get_coords("x")
        assert x.shape == (nx,)

    @pytest.mark.unit
    def test_t_coords_length(self) -> None:
        """T coordinates have correct length."""
        nt = 50
        dataset = generate_burgers_data(nx=32, nt=nt)
        t = dataset.get_coords("t")
        assert t.shape == (nt,)

    @pytest.mark.unit
    def test_x_coords_range(self) -> None:
        """X coordinates span [-1, 1) for proper periodic grid."""
        dataset = generate_burgers_data(nx=64, nt=21)
        x = dataset.get_coords("x")
        assert x.min().item() == pytest.approx(-1.0, abs=1e-6)
        # Periodic grid: [x_min, x_max) — right endpoint excluded (006/H2 fix)
        assert x.max().item() < 1.0
        assert len(x) == 64

    @pytest.mark.unit
    def test_t_coords_range(self) -> None:
        """T coordinates span [0, 1]."""
        dataset = generate_burgers_data(nx=32, nt=51)
        t = dataset.get_coords("t")
        assert t.min().item() == pytest.approx(0.0, abs=1e-6)
        assert t.max().item() == pytest.approx(1.0, abs=1e-6)


# =============================================================================
# Unit Tests: Dataset Metadata
# =============================================================================


class TestBurgersMetadata:
    """Tests for Burgers dataset metadata."""

    @pytest.mark.unit
    def test_task_type_is_pde(self) -> None:
        """Task type is PDE."""
        dataset = generate_burgers_data(nx=16, nt=11)
        assert dataset.task_type == TaskType.PDE

    @pytest.mark.unit
    def test_topology_is_grid(self) -> None:
        """Topology is GRID."""
        dataset = generate_burgers_data(nx=16, nt=11)
        assert dataset.topology == DataTopology.GRID

    @pytest.mark.unit
    def test_lhs_field_is_u(self) -> None:
        """LHS field is 'u'."""
        dataset = generate_burgers_data(nx=16, nt=11)
        assert dataset.lhs_field == "u"

    @pytest.mark.unit
    def test_lhs_axis_is_t(self) -> None:
        """LHS axis is 't' (for u_t = RHS)."""
        dataset = generate_burgers_data(nx=16, nt=11)
        assert dataset.lhs_axis == "t"

    @pytest.mark.unit
    def test_axis_order_is_x_t(self) -> None:
        """Axis order is ['x', 't']."""
        dataset = generate_burgers_data(nx=16, nt=11)
        assert dataset.axis_order == ["x", "t"]

    @pytest.mark.unit
    def test_ground_truth_string(self) -> None:
        """Ground truth equation string is provided."""
        dataset = generate_burgers_data(nx=16, nt=11, nu=0.1)
        assert dataset.ground_truth is not None
        # Should contain the equation terms
        assert "u" in dataset.ground_truth

    @pytest.mark.unit
    def test_noise_level_stored(self) -> None:
        """Noise level is stored in metadata."""
        noise = 0.05
        dataset = generate_burgers_data(nx=16, nt=11, noise_level=noise)
        assert dataset.noise_level == noise

    @pytest.mark.unit
    def test_x_axis_is_periodic(self) -> None:
        """X axis should be marked as periodic."""
        dataset = generate_burgers_data(nx=32, nt=21)
        assert dataset.axes is not None
        assert dataset.axes["x"].is_periodic is True

    @pytest.mark.unit
    def test_t_axis_is_not_periodic(self) -> None:
        """T axis should not be periodic."""
        dataset = generate_burgers_data(nx=32, nt=21)
        assert dataset.axes is not None
        assert dataset.axes["t"].is_periodic is False


# =============================================================================
# Unit Tests: Physics Validation
# =============================================================================


class TestBurgersPhysics:
    """Tests for physical correctness of Burgers data."""

    @pytest.mark.unit
    def test_initial_condition_shape(self) -> None:
        """Initial condition u(x, 0) = -sin(pi * x)."""
        dataset = generate_burgers_data(nx=64, nt=51, nu=0.1, noise_level=0.0)
        x = dataset.get_coords("x")
        u = dataset.get_field("u")
        u_initial = u[:, 0] # t=0 slice
        expected = -torch.sin(torch.pi * x)
        torch.testing.assert_close(u_initial, expected, rtol=1e-4, atol=1e-4)

    @pytest.mark.unit
    def test_data_is_finite(self) -> None:
        """All data values should be finite (no NaN/Inf)."""
        dataset = generate_burgers_data(nx=128, nt=101, nu=0.1)
        u = dataset.get_field("u")
        assert torch.isfinite(u).all(), "Data contains NaN or Inf"

    @pytest.mark.unit
    def test_data_bounded(self) -> None:
        """Data should remain bounded (no blow-up)."""
        dataset = generate_burgers_data(nx=128, nt=101, nu=0.1)
        u = dataset.get_field("u")
        # Initial condition has max magnitude 1, viscosity prevents blow-up
        assert u.abs().max() < 10.0, "Data appears to have blown up"

    @pytest.mark.unit
    def test_viscosity_effect(self) -> None:
        """Higher viscosity should produce smoother solution."""
        # Generate two datasets with different viscosity
        ds_low_nu = generate_burgers_data(nx=128, nt=51, nu=0.01, noise_level=0.0)
        ds_high_nu = generate_burgers_data(nx=128, nt=51, nu=0.5, noise_level=0.0)

        # At final time, high viscosity solution should be smoother
        u_low = ds_low_nu.get_field("u")[:, -1]
        u_high = ds_high_nu.get_field("u")[:, -1]

        # Measure smoothness by total variation
        tv_low = (u_low[1:] - u_low[:-1]).abs().sum()
        tv_high = (u_high[1:] - u_high[:-1]).abs().sum()

        assert tv_high < tv_low, "Higher viscosity should produce smoother solution"

    @pytest.mark.integration
    def test_burgers_equation_residual(self) -> None:
        """Generated data should approximately satisfy Burgers equation.

        u_t + u * u_x = nu * u_xx

        We compute finite difference approximations and check the residual.
        """
        nu = 0.1
        nx, nt = 128, 101
        dataset = generate_burgers_data(nx=nx, nt=nt, nu=nu, noise_level=0.0)

        x = dataset.get_coords("x")
        t = dataset.get_coords("t")
        u = dataset.get_field("u")

        dx = (x[1] - x[0]).item()
        dt = (t[1] - t[0]).item()

        # Interior points only (avoid boundary issues)
        i_slice = slice(2, nx - 2)
        t_slice = slice(1, nt - 1)

        u_interior = u[i_slice, t_slice]

        # Finite differences (central for space, forward for time)
        # u_t: forward difference
        u_t = (u[i_slice, 2:nt] - u[i_slice, 0: nt - 2]) / (2 * dt)

        # u_x: central difference
        u_x = (u[3: nx - 1, t_slice] - u[1: nx - 3, t_slice]) / (2 * dx)

        # u_xx: central second difference
        u_xx = (
            u[3: nx - 1, t_slice] - 2 * u[i_slice, t_slice] + u[1: nx - 3, t_slice]
        ) / (dx**2)

        # Residual: u_t + u * u_x - nu * u_xx
        residual = u_t + u_interior * u_x - nu * u_xx

        # Residual should be small (allowing for FD truncation error)
        mean_residual = residual.abs().mean().item()
        assert mean_residual < 0.1, f"Mean residual {mean_residual} too large"


# =============================================================================
# Unit Tests: Noise
# =============================================================================


class TestBurgersNoise:
    """Tests for noise addition."""

    @pytest.mark.unit
    def test_zero_noise_deterministic(self) -> None:
        """Zero noise with same seed produces identical data."""
        ds1 = generate_burgers_data(nx=32, nt=21, noise_level=0.0, seed=42)
        ds2 = generate_burgers_data(nx=32, nt=21, noise_level=0.0, seed=42)
        torch.testing.assert_close(ds1.get_field("u"), ds2.get_field("u"))

    @pytest.mark.unit
    def test_noise_adds_variation(self) -> None:
        """Non-zero noise produces different data with different seeds."""
        ds1 = generate_burgers_data(nx=32, nt=21, noise_level=0.1, seed=42)
        ds2 = generate_burgers_data(nx=32, nt=21, noise_level=0.1, seed=123)
        # Data should be different
        assert not torch.allclose(ds1.get_field("u"), ds2.get_field("u"))

    @pytest.mark.unit
    def test_noise_magnitude(self) -> None:
        """Noise magnitude approximately matches noise_level."""
        noise_level = 0.1
        # Generate clean and noisy data
        ds_clean = generate_burgers_data(nx=128, nt=101, noise_level=0.0, seed=42)
        ds_noisy = generate_burgers_data(
            nx=128, nt=101, noise_level=noise_level, seed=42
        )

        u_clean = ds_clean.get_field("u")
        u_noisy = ds_noisy.get_field("u")
        noise = u_noisy - u_clean

        # Noise std should be approximately noise_level
        actual_std = noise.std().item()
        assert actual_std == pytest.approx(noise_level, rel=0.3), (
            f"Noise std {actual_std} differs from expected {noise_level}"
        )

    @pytest.mark.unit
    def test_seed_reproducibility(self) -> None:
        """Same seed produces identical results."""
        ds1 = generate_burgers_data(nx=32, nt=21, noise_level=0.05, seed=999)
        ds2 = generate_burgers_data(nx=32, nt=21, noise_level=0.05, seed=999)
        torch.testing.assert_close(ds1.get_field("u"), ds2.get_field("u"))


# =============================================================================
# Device Tests
# =============================================================================


class TestBurgersDevice:
    """Tests for device-aware tensor creation."""

    @pytest.mark.numerical
    def test_default_device_is_cpu(self) -> None:
        """Default device should be CPU."""
        dataset = generate_burgers_data(nx=16, nt=11)
        u = dataset.get_field("u")
        assert u.device == torch.device("cpu")

    @pytest.mark.numerical
    def test_explicit_cpu_device(self) -> None:
        """Explicit CPU device works."""
        device = torch.device("cpu")
        dataset = generate_burgers_data(nx=16, nt=11, device=device)
        u = dataset.get_field("u")
        assert u.device == device

    @pytest.mark.numerical
    def test_field_on_specified_device(self, device: torch.device) -> None:
        """Field tensor is on specified device."""
        dataset = generate_burgers_data(nx=32, nt=21, device=device)
        u = dataset.get_field("u")
        assert u.device.type == device.type

    @pytest.mark.numerical
    def test_coords_on_specified_device(self, device: torch.device) -> None:
        """Coordinate tensors are on specified device."""
        dataset = generate_burgers_data(nx=32, nt=21, device=device)
        x = dataset.get_coords("x")
        t = dataset.get_coords("t")
        assert x.device.type == device.type
        assert t.device.type == device.type

    @pytest.mark.numerical
    def test_all_tensors_same_device(self, device: torch.device) -> None:
        """All tensors in dataset are on the same device."""
        dataset = generate_burgers_data(nx=32, nt=21, device=device)

        # Check all axes
        assert dataset.axes is not None
        for axis_info in dataset.axes.values():
            assert axis_info.values.device.type == device.type

        # Check all fields
        assert dataset.fields is not None
        for field_data in dataset.fields.values():
            assert field_data.values.device.type == device.type


# =============================================================================
# Edge Cases
# =============================================================================


class TestBurgersEdgeCases:
    """Edge case tests for Burgers data generation."""

    @pytest.mark.unit
    def test_minimum_grid_size(self) -> None:
        """Minimum reasonable grid size works."""
        dataset = generate_burgers_data(nx=8, nt=5)
        assert dataset.get_shape() == (8, 5)

    @pytest.mark.unit
    def test_large_grid_size(self) -> None:
        """Large grid size works (performance check)."""
        dataset = generate_burgers_data(nx=512, nt=201)
        assert dataset.get_shape() == (512, 201)
        assert torch.isfinite(dataset.get_field("u")).all()

    @pytest.mark.unit
    def test_high_viscosity(self) -> None:
        """High viscosity (nu=1.0) produces stable solution."""
        dataset = generate_burgers_data(nx=64, nt=51, nu=1.0)
        u = dataset.get_field("u")
        assert torch.isfinite(u).all()

    @pytest.mark.unit
    def test_low_viscosity(self) -> None:
        """Low viscosity (nu=0.01) produces valid solution.

        Note: Very low viscosity may cause numerical instability.
        The implementation should handle this gracefully.
        """
        dataset = generate_burgers_data(nx=256, nt=101, nu=0.01)
        u = dataset.get_field("u")
        # Should at least be finite
        assert torch.isfinite(u).all()

    @pytest.mark.unit
    def test_different_nu_values(self) -> None:
        """Various viscosity values work correctly."""
        for nu in [0.001, 0.01, 0.1, 0.5, 1.0]:
            dataset = generate_burgers_data(nx=64, nt=51, nu=nu)
            assert torch.isfinite(dataset.get_field("u")).all()


# =============================================================================
# Parameter Validation Tests
# =============================================================================


class TestBurgersParameterValidation:
    """Tests for input parameter validation."""

    @pytest.mark.unit
    def test_nx_zero_raises(self) -> None:
        """nx=0 should raise ValueError."""
        with pytest.raises(ValueError, match="nx.*positive"):
            generate_burgers_data(nx=0, nt=10)

    @pytest.mark.unit
    def test_nx_negative_raises(self) -> None:
        """Negative nx should raise ValueError."""
        with pytest.raises(ValueError, match="nx.*positive"):
            generate_burgers_data(nx=-10, nt=10)

    @pytest.mark.unit
    def test_nt_zero_raises(self) -> None:
        """nt=0 should raise ValueError."""
        with pytest.raises(ValueError, match="nt.*positive"):
            generate_burgers_data(nx=10, nt=0)

    @pytest.mark.unit
    def test_nt_negative_raises(self) -> None:
        """Negative nt should raise ValueError."""
        with pytest.raises(ValueError, match="nt.*positive"):
            generate_burgers_data(nx=10, nt=-5)

    @pytest.mark.unit
    def test_nu_zero_raises(self) -> None:
        """nu=0 (inviscid) should raise ValueError.

        Inviscid Burgers equation forms shocks and is numerically
        unstable with standard finite difference methods.
        """
        with pytest.raises(ValueError, match="nu.*positive"):
            generate_burgers_data(nx=64, nt=51, nu=0.0)

    @pytest.mark.unit
    def test_nu_negative_raises(self) -> None:
        """Negative nu should raise ValueError."""
        with pytest.raises(ValueError, match="nu.*positive"):
            generate_burgers_data(nx=64, nt=51, nu=-0.1)

    @pytest.mark.unit
    def test_noise_level_negative_raises(self) -> None:
        """Negative noise_level should raise ValueError."""
        with pytest.raises(ValueError, match="noise_level.*non-negative"):
            generate_burgers_data(nx=32, nt=21, noise_level=-0.1)

    @pytest.mark.unit
    def test_nx_exceeds_limit_raises(self) -> None:
        """nx exceeding 10000 should raise ValueError.

        H2: Resource exhaustion attack - nx=100000, nt=100000 would
        allocate ~40GB of memory, causing OOM. We cap at 10000 (~400MB).

        Note: Without the fix, this test will be very slow as it tries
        to generate the full dataset. With the fix, it should fail fast.
        """
        with pytest.raises(ValueError, match="nx"):
            generate_burgers_data(
                nx=10001, nt=5
            ) # Use small nt to limit slowness if fix missing

    @pytest.mark.unit
    def test_nt_exceeds_limit_raises(self) -> None:
        """nt exceeding 10000 should raise ValueError.

        H2: Resource exhaustion attack - large nt values can also
        cause OOM. We cap at 10000.
        """
        with pytest.raises(ValueError, match="nt"):
            generate_burgers_data(
                nx=5, nt=10001
            ) # Use small nx to limit slowness if fix missing

    @pytest.mark.slow
    @pytest.mark.unit
    def test_nx_at_limit_is_valid(self) -> None:
        """nx=10000 should be accepted (boundary case).

        Marked slow because generating 10000 spatial points takes time.
        """
        # This should not raise, just verify it doesn't error
        # We use small nt to keep the test fast
        dataset = generate_burgers_data(nx=10000, nt=5)
        assert dataset.get_shape()[0] == 10000

    @pytest.mark.slow
    @pytest.mark.unit
    def test_nt_at_limit_is_valid(self) -> None:
        """nt=10000 should be accepted (boundary case).

        Marked slow because generating 10000 time steps takes time.
        """
        # This should not raise, just verify it doesn't error
        dataset = generate_burgers_data(nx=5, nt=10000)
        assert dataset.get_shape()[1] == 10000

    @pytest.mark.unit
    def test_noise_level_exceeds_limit_raises(self) -> None:
        """noise_level exceeding 10.0 should raise ValueError.

        H3: Extreme noise levels like 1e308 cause Inf values.
        We cap at 10.0 which is already extreme (1000% noise).
        """
        with pytest.raises(ValueError, match="noise_level"):
            generate_burgers_data(nx=32, nt=21, noise_level=10.1)

    @pytest.mark.unit
    def test_noise_level_at_limit_is_valid(self) -> None:
        """noise_level=10.0 should be accepted (boundary case)."""
        dataset = generate_burgers_data(nx=32, nt=21, noise_level=10.0)
        # Data should still be finite even with extreme noise
        u = dataset.get_field("u")
        assert torch.isfinite(u).all()


# =============================================================================
# RK4-IF accuracy (006/H1 regression)
# =============================================================================


class TestBurgersRK4IFAccuracy:
    """Direct accuracy tests for the RK4 Integrating-Factor stepper.

    Regression for 006/H1: Stage 3 must NOT multiply k2_hat by exp_half.
    Verified by single-mode IC where the linear part dominates and the
    analytic solution u(x,t) = amp * exp(-nu*k^2*t) * cos(k*x) is known.
    """

    @pytest.mark.numerical
    def test_rk4if_step_matches_linear_decay_single_mode(self) -> None:
        """Single low-amplitude Fourier mode decays per analytic exp(-nu*k^2*t).

        The Burgers nonlinearity N = -u*u_x scales as O(amp^2). Choosing
        amp=1e-3 makes the nonlinear contribution O(1e-6), well below the
        1e-4-relative tolerance, so the stepper output is governed by the
        integrating-factor logic (Stages 1-4).

        With the 006/H1 bug (extra exp_half on k2_hat in Stage 3), this
        decay test still passes for low-k smooth IC because most energy
        sits at low k where exp_half ≈ 1. The test is calibrated tight
        enough to catch the math but is NOT a smoking gun by itself —
        the value is preventing future regression of the explicit derivation.
        """
        from kd2.data.synthetic._burgers import _rk4if_step

        nx = 64
        L = 2.0
        dx = L / nx
        x = torch.linspace(-1.0, 1.0 - dx, nx, dtype=torch.float64)

        amp = 1e-3
        k_idx = 4 # mode = cos(k_idx * pi * x)
        u0 = amp * torch.cos(k_idx * torch.pi * x)

        nu = 0.1
        dt = 0.005
        nsteps = 10
        t_final = nsteps * dt

        k = torch.fft.fftfreq(nx, d=dx) * 2 * torch.pi
        nu_k2 = nu * k * k
        dealias_mask = torch.ones(nx, dtype=torch.float64)

        u = u0.clone()
        for _ in range(nsteps):
            u = _rk4if_step(u, k, nu_k2, dealias_mask, dt)

        # Linear-only analytic solution (omits O(amp^2) nonlinear correction)
        u_exact = u0 * torch.exp(
            torch.tensor(-nu * (k_idx * torch.pi) ** 2 * t_final, dtype=torch.float64)
        )
        max_err = (u - u_exact).abs().max().item()
        rel_err = max_err / amp
        # Linear evolution should be exact to machine precision aside from
        # the small O(amp^2) nonlinear term and FFT roundoff.
        assert rel_err < 1e-3, (
            f"RK4-IF linear decay error {rel_err:.4e} exceeds tolerance"
        )

    @pytest.mark.numerical
    def test_rk4if_step_stage3_no_extra_exp_half(self) -> None:
        """Stage 3 must use ``+ dt/2 * k2_hat`` (not ``* exp_half * k2_hat``).

        Direct contract test: build u_hat such that k1=k2 (constant nonlinearity
        throughout the half step), then the analytic Stage 3 estimate is
        v3_hat = exp_half * u_hat + dt/2 * k2_hat. Comparing the kd2 Stage 3
        output against this expression isolates the bug from any other
        compounding error.

        This is the regression assertion for 006/H1.
        """
        from kd2.data.synthetic._burgers import _nonlinear_term

        nx = 32
        L = 2.0
        dx = L / nx
        x = torch.linspace(-1.0, 1.0 - dx, nx, dtype=torch.float64)
        nu = 0.5 # large nu to make exp_half meaningfully different from 1
        dt = 0.01
        amp = 1.0 # large enough for nonlinear k2_hat to register
        k_idx = 6
        u0 = amp * torch.cos(k_idx * torch.pi * x)
        k = torch.fft.fftfreq(nx, d=dx) * 2 * torch.pi
        nu_k2 = nu * k * k
        dealias_mask = torch.ones(nx, dtype=torch.float64)

        # Reproduce one stage of the RK4-IF step manually
        exp_half = torch.exp(-nu_k2 * dt / 2)
        u_hat = torch.fft.fft(u0)
        n1 = _nonlinear_term(u0, k, dealias_mask)
        k1_hat = torch.fft.fft(n1)
        u2_hat = exp_half * u_hat + (dt / 2) * exp_half * k1_hat
        u2 = torch.fft.ifft(u2_hat).real
        n2 = _nonlinear_term(u2, k, dealias_mask)
        k2_hat = torch.fft.fft(n2)

        # Correct Stage 3 (per integrating-factor derivation):
        # v3_hat = exp_half * u_hat + dt/2 * k2_hat (NO exp_half on k2_hat)
        u3_hat_correct = exp_half * u_hat + (dt / 2) * k2_hat

        # Buggy Stage 3 (pre-fix):
        u3_hat_buggy = exp_half * u_hat + (dt / 2) * exp_half * k2_hat

        # The two must differ at high k (where exp_half is meaningfully < 1).
        # Extract energy difference at the active mode.
        diff = (u3_hat_correct - u3_hat_buggy).abs()
        # At the active mode k_idx, exp_half should be << 1 (we chose nu=0.5)
        # so the contributions differ.
        active_idx = k_idx
        assert diff[active_idx].item() > 0.0, (
            "Correct vs buggy Stage 3 must differ at active mode"
        )

        # Verify code uses the correct formula by reproducing _rk4if_step's
        # Stage 3 inline (mirror lines in src) and matching the corrected form.
        # If a future edit reintroduces the bug, this comparison fails.
        from kd2.data.synthetic._burgers import _rk4if_step

        # Run one full step and ensure the result matches an independent
        # reproduction using the corrected Stage 3 formula throughout.
        n3 = _nonlinear_term(torch.fft.ifft(u3_hat_correct).real, k, dealias_mask)
        k3_hat = torch.fft.fft(n3)
        exp_full = torch.exp(-nu_k2 * dt)
        u4_hat = exp_full * u_hat + dt * exp_half * k3_hat
        u4 = torch.fft.ifft(u4_hat).real
        n4 = _nonlinear_term(u4, k, dealias_mask)
        k4_hat = torch.fft.fft(n4)
        u_new_hat_expected = exp_full * u_hat + (dt / 6) * (
            exp_full * k1_hat + 2 * exp_half * k2_hat + 2 * exp_half * k3_hat + k4_hat
        )
        u_new_expected = torch.fft.ifft(u_new_hat_expected).real

        u_new_actual = _rk4if_step(u0, k, nu_k2, dealias_mask, dt)

        torch.testing.assert_close(u_new_actual, u_new_expected, rtol=1e-10, atol=1e-12)
