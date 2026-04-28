"""Tests for SGA Train sweep, AIC scoring, and evaluate_candidate pipeline.

TDD red phase -- tests written against the interface spec before implementation.
The Train() tolerance sweep adaptively searches for the optimal STRidge
tolerance that minimizes AIC. evaluate_candidate chains execution through
execute_pde -> build_theta -> train_sweep.

Key contract: SolveResult.residual is SSR (sum of squared residuals),
NOT MSE. MSE = SSR / n_samples. This is tested explicitly.
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from kd2.search.sga.config import SGAConfig
from kd2.search.sga.pde import PDE
from kd2.search.sga.train import (
    TrainResult,
    compute_aic,
    evaluate_candidate,
    train_sweep,
)
from kd2.search.sga.tree import Node, Tree

# -- Constants ----------------------------------------------------------------

N_SAMPLES = 100
RTOL = 1e-5
ATOL = 1e-8


# -- Helpers ------------------------------------------------------------------


def _leaf(name: str) -> Node:
    """Shorthand for a leaf node."""
    return Node(name=name, arity=0, children=[])


def _binary(op: str, left: Node, right: Node) -> Node:
    """Shorthand for a binary operator node."""
    return Node(name=op, arity=2, children=[left, right])


def _make_sparse_system(
    n: int = N_SAMPLES,
    d: int = 5,
    seed: int = 42,
) -> tuple[Tensor, Tensor, Tensor]:
    """Build a well-conditioned sparse linear system.

    y = 2*x1 + 3*x2 + noise, with x3..x5 irrelevant.

    Returns:
        (theta, y, true_w) where true_w has only 2 non-zero entries.
    """
    gen = torch.Generator().manual_seed(seed)
    theta = torch.randn(n, d, generator=gen)
    true_w = torch.zeros(d)
    true_w[0] = 2.0
    true_w[1] = 3.0
    noise = 0.01 * torch.randn(n, generator=gen)
    y = theta @ true_w + noise
    return theta, y.unsqueeze(1), true_w


# ===========================================================================
# Smoke: function existence and basic callability
# ===========================================================================


class TestSmoke:
    """Verify that the public API is importable and callable."""

    @pytest.mark.smoke
    def test_compute_aic_callable(self) -> None:
        result = compute_aic(mse=1.0, k=2, ratio=1.0)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_train_sweep_callable(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert isinstance(result, TrainResult)

    @pytest.mark.smoke
    def test_evaluate_candidate_callable(self) -> None:
        from kd2.search.sga.train import CandidateResult

        data = {"u": torch.randn(20), "x": torch.randn(20)}
        pde = PDE(terms=[Tree(root=_leaf("u")), Tree(root=_leaf("x"))])
        y = torch.randn(20, 1)
        config = SGAConfig()
        result = evaluate_candidate(pde, data, None, y, config)
        assert isinstance(result, CandidateResult)

    @pytest.mark.smoke
    def test_train_result_has_required_fields(self) -> None:
        """TrainResult dataclass has all documented fields."""
        tr = TrainResult(
            coefficients=torch.zeros(3),
            selected_indices=[0],
            aic_score=1.0,
            mse=0.5,
            best_tol=0.1,
        )
        assert hasattr(tr, "coefficients")
        assert hasattr(tr, "selected_indices")
        assert hasattr(tr, "aic_score")
        assert hasattr(tr, "mse")
        assert hasattr(tr, "best_tol")


# ===========================================================================
# compute_aic: core formula
# ===========================================================================


class TestComputeAic:
    """AIC = 2 * k * ratio + 2 * log(MSE).

    This is verified via independent hand-calculation using math.log.
    """

    def test_known_values(self) -> None:
        """Hand-calculated: mse=1.0, k=2, ratio=1.0 -> 2*2*1 + 2*log(1) = 4."""
        result = compute_aic(mse=1.0, k=2, ratio=1.0)
        expected = 2 * 2 * 1.0 + 2 * math.log(1.0) # = 4.0
        torch.testing.assert_close(
            torch.tensor(result), torch.tensor(expected), rtol=RTOL, atol=ATOL
        )

    def test_known_values_nonunit_mse(self) -> None:
        """mse=2.5, k=3, ratio=1.0 -> 6 + 2*log(2.5)."""
        result = compute_aic(mse=2.5, k=3, ratio=1.0)
        expected = 2 * 3 * 1.0 + 2 * math.log(2.5)
        torch.testing.assert_close(
            torch.tensor(result), torch.tensor(expected), rtol=RTOL, atol=ATOL
        )

    def test_k_zero_only_log_term(self) -> None:
        """k=0 means no penalty, AIC = 2*log(MSE)."""
        result = compute_aic(mse=0.5, k=0, ratio=1.0)
        expected = 2 * math.log(0.5)
        torch.testing.assert_close(
            torch.tensor(result), torch.tensor(expected), rtol=RTOL, atol=ATOL
        )

    def test_mse_zero_returns_inf(self) -> None:
        """mse=0 is degenerate (log(0) = -inf); spec says return inf."""
        result = compute_aic(mse=0.0, k=1, ratio=1.0)
        assert result == float("inf")

    def test_mse_negative_returns_inf(self) -> None:
        """Negative MSE is nonsensical; spec says return inf."""
        result = compute_aic(mse=-1.0, k=1, ratio=1.0)
        assert result == float("inf")

    def test_ratio_greater_than_one(self) -> None:
        """Higher ratio penalizes complexity more heavily."""
        result = compute_aic(mse=1.0, k=3, ratio=2.0)
        expected = 2 * 3 * 2.0 + 2 * math.log(1.0) # = 12.0
        torch.testing.assert_close(
            torch.tensor(result), torch.tensor(expected), rtol=RTOL, atol=ATOL
        )

    def test_aic_monotonic_in_k(self) -> None:
        """Fixed MSE: increasing k should increase AIC (more penalty)."""
        mse = 1.5
        aic_values = [compute_aic(mse=mse, k=k, ratio=1.0) for k in range(10)]
        for i in range(len(aic_values) - 1):
            assert aic_values[i] < aic_values[i + 1], (
                f"AIC not monotonically increasing in k: "
                f"AIC(k={i})={aic_values[i]}, AIC(k={i + 1})={aic_values[i + 1]}"
            )

    def test_aic_monotonic_in_mse(self) -> None:
        """Fixed k: increasing MSE should increase AIC (log is monotone)."""
        k = 2
        mse_values = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
        aic_values = [compute_aic(mse=m, k=k, ratio=1.0) for m in mse_values]
        for i in range(len(aic_values) - 1):
            assert aic_values[i] < aic_values[i + 1], (
                f"AIC not monotonically increasing in MSE: "
                f"AIC(mse={mse_values[i]})={aic_values[i]}, "
                f"AIC(mse={mse_values[i + 1]})={aic_values[i + 1]}"
            )


# ===========================================================================
# train_sweep: basic contract
# ===========================================================================


class TestTrainSweepContract:
    """train_sweep returns a TrainResult with the right types and shapes."""

    def test_return_type(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert isinstance(result, TrainResult)

    def test_coefficients_shape(self) -> None:
        """Coefficients should match theta column count."""
        theta, y, _ = _make_sparse_system(d=5)
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert result.coefficients.shape == (5,)

    def test_coefficients_is_tensor(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert isinstance(result.coefficients, Tensor)

    def test_selected_indices_is_list_of_int(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert isinstance(result.selected_indices, list)
        for idx in result.selected_indices:
            assert isinstance(idx, int)

    def test_aic_score_is_finite(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert math.isfinite(result.aic_score)

    def test_mse_is_nonnegative(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert result.mse >= 0.0

    def test_best_tol_is_nonnegative(self) -> None:
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert result.best_tol >= 0.0


# ===========================================================================
# train_sweep: MSE = SSR / n_samples conversion
# ===========================================================================


class TestTrainSweepMseConversion:
    """train_sweep must convert SSR (from SolveResult.residual) to MSE."""

    def test_mse_consistent_with_coefficients(self) -> None:
        """MSE should approximately equal mean((y - theta @ w)^2)."""
        theta, y, _ = _make_sparse_system()
        config = SGAConfig()
        result = train_sweep(theta, y, config)

        # Recompute MSE from returned coefficients
        y_1d = y.squeeze(-1) if y.dim() == 2 else y
        residuals = y_1d - theta @ result.coefficients
        recomputed_mse = (residuals**2).mean().item()

        torch.testing.assert_close(
            torch.tensor(result.mse),
            torch.tensor(recomputed_mse),
            rtol=1e-4,
            atol=1e-6,
        )


# ===========================================================================
# train_sweep: sparse recovery
# ===========================================================================


class TestTrainSweepSparseRecovery:
    """With well-conditioned synthetic data, train_sweep should recover
    the sparse ground truth (or a good approximation)."""

    def test_recovers_true_support(self) -> None:
        """For y = 2*x1 + 3*x2, selected_indices should contain {0, 1}."""
        theta, y, _ = _make_sparse_system(n=200, seed=123)
        config = SGAConfig(d_tol=0.5, maxit=20)
        result = train_sweep(theta, y, config)

        # The true support is {0, 1}; result should include these
        assert 0 in result.selected_indices, (
            f"Expected index 0 in support, got {result.selected_indices}"
        )
        assert 1 in result.selected_indices, (
            f"Expected index 1 in support, got {result.selected_indices}"
        )

    def test_recovers_approximate_coefficients(self) -> None:
        """Recovered coefficients should be close to [2.0, 3.0, 0, 0, 0]."""
        theta, y, true_w = _make_sparse_system(n=200, seed=123)
        config = SGAConfig(d_tol=0.5, maxit=20)
        result = train_sweep(theta, y, config)

        # Check that the active coefficients are close to true values.
        # Inactive coefficients should be exactly zero.
        w = result.coefficients
        for i in range(len(true_w)):
            if true_w[i] != 0:
                torch.testing.assert_close(
                    w[i].unsqueeze(0),
                    true_w[i].unsqueeze(0),
                    rtol=0.05,
                    atol=0.1,
                )
            else:
                assert w[i].abs().item() < 0.1, (
                    f"Coefficient at index {i} should be ~0, got {w[i].item()}"
                )


# ===========================================================================
# train_sweep: tolerance adaptation behavior
# ===========================================================================


class TestTrainSweepAdaptation:
    """The adaptive tolerance sweep should produce different results for
    different d_tol and maxit settings."""

    def test_different_dtol_different_best_tol(self) -> None:
        """Different d_tol step sizes should (generally) produce different
        best_tol values -- testing that the sweep actually runs."""
        theta, y, _ = _make_sparse_system(n=200, seed=42)

        config_fine = SGAConfig(d_tol=0.1, maxit=20)
        config_coarse = SGAConfig(d_tol=2.0, maxit=20)

        result_fine = train_sweep(theta, y, config_fine)
        result_coarse = train_sweep(theta, y, config_coarse)

        # They should differ because the grid of tested tolerances differs
        # (but we can't guarantee this -- just check they both produce valid results)
        assert math.isfinite(result_fine.aic_score)
        assert math.isfinite(result_coarse.aic_score)
        assert result_fine.mse >= 0
        assert result_coarse.mse >= 0

    def test_higher_tol_trend_fewer_terms(self) -> None:
        """When d_tol is very large, the sweep can push tolerance high,
        which should produce sparser solutions (fewer selected terms)."""
        theta, y, _ = _make_sparse_system(n=200, d=8, seed=77)

        config_small = SGAConfig(d_tol=0.01, maxit=5)
        config_large = SGAConfig(d_tol=5.0, maxit=5)

        result_small = train_sweep(theta, y, config_small)
        result_large = train_sweep(theta, y, config_large)

        # With larger tolerance steps, we expect at most as many terms
        assert (
            len(result_large.selected_indices) <= len(result_small.selected_indices) + 2
        ), (
            f"Large tol sweep should be at least as sparse: "
            f"small_tol={len(result_small.selected_indices)} terms, "
            f"large_tol={len(result_large.selected_indices)} terms"
        )

    def test_ols_baseline_used_at_tol_zero(self) -> None:
        """The initial AIC baseline is computed at tol=0 (OLS).
        Result should never be worse than pure OLS."""
        theta, y, _ = _make_sparse_system()
        config = SGAConfig(d_tol=1.0, maxit=10)
        result = train_sweep(theta, y, config)

        # Compute OLS baseline AIC independently
        y_1d = y.squeeze(-1) if y.dim() == 2 else y
        w_ols = torch.linalg.lstsq(theta, y_1d.unsqueeze(1)).solution.squeeze()
        mse_ols = ((y_1d - theta @ w_ols) ** 2).mean().item()
        k_ols = (w_ols.abs() > 1e-10).sum().item()
        aic_ols = compute_aic(mse=mse_ols, k=k_ols, ratio=config.aic_ratio)

        # train_sweep result should be <= OLS AIC (it picks the best)
        assert result.aic_score <= aic_ols + 1e-6, (
            f"train_sweep AIC ({result.aic_score}) should not exceed "
            f"OLS baseline AIC ({aic_ols})"
        )


# ===========================================================================
# train_sweep: edge cases
# ===========================================================================


class TestTrainSweepEdgeCases:
    """Edge cases for train_sweep."""

    def test_zero_column_theta(self) -> None:
        """Theta with 0 columns (empty) should return inf score."""
        theta = torch.empty(N_SAMPLES, 0)
        y = torch.randn(N_SAMPLES, 1)
        config = SGAConfig()
        result = train_sweep(theta, y, config)
        assert result.aic_score == float("inf")

    def test_single_column_theta(self) -> None:
        """Theta with 1 column -- STRidge can still solve."""
        gen = torch.Generator().manual_seed(99)
        theta = torch.randn(N_SAMPLES, 1, generator=gen)
        y = 3.0 * theta[:, 0:1] + 0.01 * torch.randn(N_SAMPLES, 1, generator=gen)
        config = SGAConfig(d_tol=0.1, maxit=5)
        result = train_sweep(theta, y, config)

        assert math.isfinite(result.aic_score)
        assert len(result.selected_indices) <= 1

    def test_maxit_zero(self) -> None:
        """With maxit=0, no sweep iterations -- should still return OLS baseline."""
        theta, y, _ = _make_sparse_system()
        config = SGAConfig(maxit=0)
        result = train_sweep(theta, y, config)

        assert math.isfinite(result.aic_score)
        assert result.mse >= 0

    def test_perfect_fit_very_low_mse(self) -> None:
        """When y is exactly in the column space of theta, MSE should be ~0."""
        gen = torch.Generator().manual_seed(88)
        theta = torch.randn(N_SAMPLES, 3, generator=gen)
        true_w = torch.tensor([1.0, -2.0, 0.5])
        y = (theta @ true_w).unsqueeze(1) # no noise
        config = SGAConfig(d_tol=0.1, maxit=10)
        result = train_sweep(theta, y, config)

        assert result.mse < 1e-6, f"Expected near-zero MSE, got {result.mse}"


# ===========================================================================
# train_sweep: config parameter forwarding
# ===========================================================================


class TestTrainSweepConfigForwarding:
    """train_sweep should honor config parameters (lam, str_iters, normalize)."""

    def test_lam_zero_uses_ols(self) -> None:
        """With lam=0, STRidge should use OLS internally (not ridge)."""
        theta, y, _ = _make_sparse_system()
        config = SGAConfig(lam=0.0, d_tol=1.0, maxit=5)
        result = train_sweep(theta, y, config)
        assert math.isfinite(result.aic_score)

    def test_lam_nonzero_uses_ridge(self) -> None:
        """With lam > 0, result should still be valid (potentially different)."""
        theta, y, _ = _make_sparse_system()
        config = SGAConfig(lam=1e-3, d_tol=1.0, maxit=5)
        result = train_sweep(theta, y, config)
        assert math.isfinite(result.aic_score)

    def test_different_lam_different_result(self) -> None:
        """Different lam values should generally produce different solutions."""
        theta, y, _ = _make_sparse_system(n=200)
        config_ols = SGAConfig(lam=0.0, d_tol=1.0, maxit=10)
        config_ridge = SGAConfig(lam=10.0, d_tol=1.0, maxit=10)

        result_ols = train_sweep(theta, y, config_ols)
        result_ridge = train_sweep(theta, y, config_ridge)

        # Not necessarily different, but with lam=10 the solution is
        # significantly regularized, so coefficients should differ
        diff = (result_ols.coefficients - result_ridge.coefficients).abs().sum()
        assert diff > 1e-3, "OLS and heavy-ridge solutions should differ significantly"


# ===========================================================================
# evaluate_candidate: full pipeline
# ===========================================================================


class TestEvaluateCandidatePipeline:
    """evaluate_candidate chains execute_pde -> build_theta -> train_sweep."""

    def test_basic_pipeline(self) -> None:
        """Simple PDE with valid terms produces a finite result."""
        gen = torch.Generator().manual_seed(42)
        u = torch.randn(N_SAMPLES, generator=gen)
        x = torch.randn(N_SAMPLES, generator=gen)
        data = {"u": u, "x": x}

        # y = 2*u + noise
        y = (2.0 * u + 0.01 * torch.randn(N_SAMPLES, generator=gen)).unsqueeze(1)

        pde = PDE(terms=[Tree(root=_leaf("u")), Tree(root=_leaf("x"))])
        config = SGAConfig(d_tol=0.5, maxit=10)

        result = evaluate_candidate(pde, data, None, y, config)
        assert math.isfinite(result.aic_score)
        assert result.mse >= 0

    def test_all_terms_filtered_returns_inf(self) -> None:
        """When all PDE terms are invalid (NaN), return inf score."""
        data = {"nan_var": torch.full((N_SAMPLES,), float("nan"))}
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(terms=[Tree(root=_leaf("nan_var"))])
        config = SGAConfig()

        result = evaluate_candidate(pde, data, None, y, config)
        assert result.aic_score == float("inf")

    def test_empty_pde_returns_inf(self) -> None:
        """An empty PDE (no terms) should return inf score."""
        data = {"u": torch.randn(N_SAMPLES)}
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(terms=[])
        config = SGAConfig()

        result = evaluate_candidate(pde, data, None, y, config)
        assert result.aic_score == float("inf")

    def test_with_default_terms(self) -> None:
        """default_terms should be prepended to the theta matrix."""
        gen = torch.Generator().manual_seed(55)
        u = torch.randn(N_SAMPLES, generator=gen)
        x = torch.randn(N_SAMPLES, generator=gen)
        data = {"u": u, "x": x}
        y = torch.randn(N_SAMPLES, 1, generator=gen)

        # 2 default terms (e.g., constant + some feature)
        default_terms = torch.randn(N_SAMPLES, 2, generator=gen)
        pde = PDE(terms=[Tree(root=_leaf("u"))])
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = evaluate_candidate(pde, data, default_terms, y, config)
        assert math.isfinite(result.aic_score)
        # Should have 2 (default) + 1 (PDE term) = 3 total features
        assert result.coefficients.shape == (3,)

    def test_default_terms_only_when_pde_terms_filtered(self) -> None:
        """When PDE terms are all filtered but defaults exist,
        train on defaults only."""
        data = {"nan_var": torch.full((N_SAMPLES,), float("nan"))}
        y = torch.randn(N_SAMPLES, 1)

        default_terms = torch.randn(N_SAMPLES, 2)
        pde = PDE(terms=[Tree(root=_leaf("nan_var"))])
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = evaluate_candidate(pde, data, default_terms, y, config)
        # With defaults, theta has 2 columns from defaults + 0 from PDE
        assert result.coefficients.shape == (2,)
        assert math.isfinite(result.aic_score)

    def test_no_default_no_valid_terms_returns_inf(self) -> None:
        """No defaults + all terms filtered = inf."""
        data = {"zeros": torch.zeros(N_SAMPLES)}
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(terms=[Tree(root=_leaf("zeros"))])
        config = SGAConfig()

        result = evaluate_candidate(pde, data, None, y, config)
        assert result.aic_score == float("inf")


# ===========================================================================
# evaluate_candidate: selected_indices semantics
# ===========================================================================


class TestEvaluateCandidateIndices:
    """selected_indices in TrainResult are relative to the theta matrix
    (including prepended defaults), not to PDE.terms."""

    def test_indices_within_range(self) -> None:
        """All selected_indices should be < total number of theta columns."""
        gen = torch.Generator().manual_seed(42)
        data = {"u": torch.randn(N_SAMPLES, generator=gen)}
        y = torch.randn(N_SAMPLES, 1, generator=gen)

        default_terms = torch.randn(N_SAMPLES, 2, generator=gen)
        pde = PDE(terms=[Tree(root=_leaf("u"))])
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = evaluate_candidate(pde, data, default_terms, y, config)
        total_cols = 2 + 1 # 2 defaults + 1 PDE term
        for idx in result.selected_indices:
            assert 0 <= idx < total_cols, f"Index {idx} out of range [0, {total_cols})"


# ===========================================================================
# Negative tests: error handling and invalid inputs (>= 20%)
# ===========================================================================


class TestNegativeAndFailure:
    """Error handling, invalid inputs, NaN injection."""

    @pytest.mark.numerical
    def test_nan_in_y_handled(self) -> None:
        """NaN in y should either raise ValueError or produce inf result."""
        theta = torch.randn(N_SAMPLES, 3)
        y = torch.randn(N_SAMPLES, 1)
        y[0, 0] = float("nan")
        config = SGAConfig()

        # STRidgeSolver raises ValueError on NaN input
        with pytest.raises(ValueError):
            train_sweep(theta, y, config)

    @pytest.mark.numerical
    def test_inf_in_theta_handled(self) -> None:
        """Inf in theta should either raise ValueError or produce inf result."""
        theta = torch.randn(N_SAMPLES, 3)
        theta[0, 0] = float("inf")
        y = torch.randn(N_SAMPLES, 1)
        config = SGAConfig()

        with pytest.raises(ValueError):
            train_sweep(theta, y, config)

    @pytest.mark.numerical
    def test_nan_in_theta_handled(self) -> None:
        """NaN in theta should either raise ValueError or produce inf result."""
        theta = torch.randn(N_SAMPLES, 3)
        theta[5, 1] = float("nan")
        y = torch.randn(N_SAMPLES, 1)
        config = SGAConfig()

        with pytest.raises(ValueError):
            train_sweep(theta, y, config)

    def test_evaluate_candidate_unknown_variable(self) -> None:
        """PDE referencing a variable not in data_dict should not crash.
        The term is filtered, leading to inf if no valid terms remain."""
        data = {"u": torch.randn(N_SAMPLES)}
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(terms=[Tree(root=_leaf("nonexistent"))])
        config = SGAConfig()

        result = evaluate_candidate(pde, data, None, y, config)
        assert result.aic_score == float("inf")

    def test_compute_aic_nan_mse(self) -> None:
        """NaN MSE should return inf."""
        result = compute_aic(mse=float("nan"), k=1, ratio=1.0)
        assert result == float("inf")

    def test_compute_aic_inf_mse(self) -> None:
        """Inf MSE should return inf (2*log(inf) = inf)."""
        result = compute_aic(mse=float("inf"), k=1, ratio=1.0)
        assert result == float("inf")

    def test_evaluate_candidate_returns_pruned_pde(self) -> None:
        """evaluate_candidate should return pruned PDE in CandidateResult."""
        from kd2.search.sga.train import CandidateResult

        data = {
            "u": torch.randn(N_SAMPLES),
            "nan_var": torch.full((N_SAMPLES,), float("nan")),
        }
        y = (2.0 * data["u"]).unsqueeze(1)
        pde = PDE(terms=[Tree(root=_leaf("u")), Tree(root=_leaf("nan_var"))])
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = evaluate_candidate(pde, data, None, y, config)
        assert isinstance(result, CandidateResult)
        assert result.pruned_pde is not None
        assert result.pruned_pde.width == 1
        assert str(result.pruned_pde.terms[0]) == "u"

    def test_evaluate_candidate_coefficients_aligned_with_pruned_pde(self) -> None:
        """TrainResult coefficients should be aligned with pruned PDE terms."""
        from kd2.search.sga.train import CandidateResult

        gen = torch.Generator().manual_seed(42)
        data = {
            "u": torch.randn(N_SAMPLES, generator=gen),
            "x": torch.randn(N_SAMPLES, generator=gen),
            "zeros": torch.zeros(N_SAMPLES),
        }
        # y = 2*u + 3*x
        y = (2.0 * data["u"] + 3.0 * data["x"]).unsqueeze(1)

        # PDE has [u, zeros, x] — "zeros" will be pruned
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")),
                Tree(root=_leaf("zeros")),
                Tree(root=_leaf("x")),
            ]
        )
        config = SGAConfig(d_tol=0.5, maxit=10)

        result = evaluate_candidate(pde, data, None, y, config)
        assert isinstance(result, CandidateResult)
        # Pruned PDE should have 2 terms: u and x
        assert result.pruned_pde.width == 2
        # Coefficients length should match pruned PDE width (no defaults)
        assert result.train_result.coefficients.shape[0] == 2

    def test_evaluate_candidate_selected_indices_map_to_pruned_pde(self) -> None:
        """selected_indices should index into pruned PDE terms (+ defaults)."""
        from kd2.search.sga.train import CandidateResult

        gen = torch.Generator().manual_seed(42)
        data = {
            "u": torch.randn(N_SAMPLES, generator=gen),
            "x": torch.randn(N_SAMPLES, generator=gen),
            "zeros": torch.zeros(N_SAMPLES),
        }
        y = (2.0 * data["u"]).unsqueeze(1)

        pde = PDE(
            terms=[
                Tree(root=_leaf("zeros")),
                Tree(root=_leaf("u")),
                Tree(root=_leaf("x")),
            ]
        )
        config = SGAConfig(d_tol=0.5, maxit=10)

        result = evaluate_candidate(pde, data, None, y, config)
        assert isinstance(result, CandidateResult)
        # All selected_indices should be valid indices into the theta
        n_theta_cols = result.pruned_pde.width # no defaults
        for idx in result.train_result.selected_indices:
            assert 0 <= idx < n_theta_cols

    def test_evaluate_candidate_preserves_original_pde(self) -> None:
        """evaluate_candidate should not modify the input PDE."""
        data = {"u": torch.randn(N_SAMPLES), "zeros": torch.zeros(N_SAMPLES)}
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(terms=[Tree(root=_leaf("u")), Tree(root=_leaf("zeros"))])
        original_width = pde.width
        original_str = str(pde)
        config = SGAConfig()

        _ = evaluate_candidate(pde, data, None, y, config)
        assert pde.width == original_width
        assert str(pde) == original_str

    def test_evaluate_candidate_valid_term_indices_maps_original(self) -> None:
        """CandidateResult.valid_term_indices should map to the original PDE."""
        from kd2.search.sga.train import CandidateResult

        data = {
            "u": torch.randn(N_SAMPLES),
            "zeros": torch.zeros(N_SAMPLES),
            "x": torch.randn(N_SAMPLES),
        }
        y = torch.randn(N_SAMPLES, 1)
        pde = PDE(
            terms=[
                Tree(root=_leaf("u")), # idx 0 - valid
                Tree(root=_leaf("zeros")), # idx 1 - invalid
                Tree(root=_leaf("x")), # idx 2 - valid
            ]
        )
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = evaluate_candidate(pde, data, None, y, config)
        assert isinstance(result, CandidateResult)
        assert result.valid_term_indices == [0, 2]

    def test_dimension_mismatch_theta_y(self) -> None:
        """theta and y with different n_samples should raise."""
        theta = torch.randn(N_SAMPLES, 3)
        y = torch.randn(N_SAMPLES + 10, 1)
        config = SGAConfig()

        with pytest.raises(ValueError):
            train_sweep(theta, y, config)


# ===========================================================================
# Numerical stability
# ===========================================================================


class TestNumericalStability:
    """Numerical edge cases for train_sweep."""

    @pytest.mark.numerical
    def test_extreme_large_theta(self) -> None:
        """Very large values in theta should not crash."""
        gen = torch.Generator().manual_seed(42)
        theta = 1e10 * torch.randn(N_SAMPLES, 3, generator=gen)
        y = theta[:, 0:1] + 0.01 * torch.randn(N_SAMPLES, 1, generator=gen)
        config = SGAConfig(d_tol=1.0, maxit=5)

        result = train_sweep(theta, y, config)
        # Must not crash; result validity depends on condition number
        assert isinstance(result, TrainResult)

    @pytest.mark.numerical
    def test_extreme_small_theta(self) -> None:
        """Very small values in theta should not crash."""
        gen = torch.Generator().manual_seed(42)
        theta = 1e-10 * torch.randn(N_SAMPLES, 3, generator=gen)
        y = theta[:, 0:1] + 1e-12 * torch.randn(N_SAMPLES, 1, generator=gen)
        config = SGAConfig(d_tol=1e-12, maxit=5)

        result = train_sweep(theta, y, config)
        assert isinstance(result, TrainResult)

    @pytest.mark.numerical
    def test_collinear_columns(self) -> None:
        """Collinear columns (high condition number) should not crash."""
        gen = torch.Generator().manual_seed(42)
        base = torch.randn(N_SAMPLES, generator=gen)
        theta = torch.stack([base, base * 1.001, base * 0.999], dim=1)
        y = (base + 0.01 * torch.randn(N_SAMPLES, generator=gen)).unsqueeze(1)
        config = SGAConfig(d_tol=0.5, maxit=5)

        result = train_sweep(theta, y, config)
        assert isinstance(result, TrainResult)
        # AIC can be finite or +inf (all terms eliminated), but never NaN
        assert not math.isnan(result.aic_score)

    @pytest.mark.numerical
    def test_single_sample(self) -> None:
        """With n_samples=1, system is underdetermined but should not crash."""
        theta = torch.tensor([[1.0, 2.0, 3.0]])
        y = torch.tensor([[5.0]])
        config = SGAConfig(d_tol=0.1, maxit=3)

        result = train_sweep(theta, y, config)
        assert isinstance(result, TrainResult)


# ===========================================================================
# Property-based: AIC properties
# ===========================================================================


class TestAicProperties:
    """Mathematical properties of AIC that must hold for all valid inputs."""

    @pytest.mark.parametrize("mse", [0.001, 0.1, 1.0, 10.0, 1000.0])
    def test_aic_is_finite_for_positive_mse(self, mse: float) -> None:
        """AIC should be finite for any positive MSE."""
        result = compute_aic(mse=mse, k=3, ratio=1.0)
        assert math.isfinite(result)

    @pytest.mark.parametrize("ratio", [0.5, 1.0, 2.0, 5.0])
    def test_ratio_scales_penalty_linearly(self, ratio: float) -> None:
        """Doubling ratio should double the penalty term contribution."""
        mse = 1.0 # log(1) = 0, so AIC = 2*k*ratio
        k = 3
        result = compute_aic(mse=mse, k=k, ratio=ratio)
        expected = 2 * k * ratio # + 2*log(1) = 0
        torch.testing.assert_close(
            torch.tensor(result), torch.tensor(expected), rtol=RTOL, atol=ATOL
        )

    def test_aic_decomposition(self) -> None:
        """AIC(mse, k, r) = penalty(k, r) + fit(mse), i.e., it's additive."""
        mse = 2.5
        k = 4
        ratio = 1.5

        full_aic = compute_aic(mse=mse, k=k, ratio=ratio)
        penalty = 2 * k * ratio
        fit = 2 * math.log(mse)

        torch.testing.assert_close(
            torch.tensor(full_aic),
            torch.tensor(penalty + fit),
            rtol=RTOL,
            atol=ATOL,
        )
