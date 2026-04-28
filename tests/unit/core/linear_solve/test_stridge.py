"""Unit tests for STRidgeSolver.

1. Known sparse system exact recovery
2. lam=0 OLS path
3. Column normalization
4. Zero columns handling
5. Empty biginds (j==0 vs j>0)
6. Early stopping on support convergence
7. Debiased OLS final coefficients
8. solve_with_tol different tol -> different sparsity
9. Fat matrix (n < d)
10. NaN theta rejection
11. High condition number warning
12. (Equivalence test in tests/equivalence/test_stridge_equiv.py)
"""

import logging

import pytest
import torch

from kd2.core.linear_solve.base import SolveResult, SparseSolver

# ============================================================
# Helpers
# ============================================================


def _make_sparse_system(
    n: int = 100,
    d: int = 5,
    true_coeffs: list[float] | None = None,
    noise_std: float = 0.0,
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a test linear system y = theta @ xi + noise.

    Args:
        n: Number of samples.
        d: Number of features (columns).
        true_coeffs: Ground truth coefficients. Length must equal d.
        noise_std: Standard deviation of additive noise.
        seed: Random seed for reproducible data generation.

    Returns:
        (theta, y, true_xi) where theta is (n, d), y is (n,),
        true_xi is (d,).
    """
    if true_coeffs is None:
        true_coeffs = [2.0, 0.0, 3.0, 0.0, 0.0]
    true_xi = torch.tensor(true_coeffs, dtype=torch.float64)
    assert true_xi.shape[0] == d, f"true_coeffs length {len(true_coeffs)} != d={d}"

    rng = torch.Generator().manual_seed(seed)
    theta = torch.randn(n, d, dtype=torch.float64, generator=rng)
    y = theta @ true_xi
    if noise_std > 0:
        y = y + noise_std * torch.randn(n, dtype=torch.float64, generator=rng)
    return theta, y, true_xi


def _make_multiscale_system(
    n: int = 200,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create system with columns of very different magnitudes.

    Returns system where col0 ~ O(1e3), col1 ~ O(1), col2 ~ O(1e-3),
    with true coefficients [1.0, 2.0, 3.0].
    Without normalization, thresholding would incorrectly drop columns.
    """
    gen = torch.Generator().manual_seed(123)
    x0 = 1e3 * torch.randn(n, 1, dtype=torch.float64, generator=gen)
    x1 = torch.randn(n, 1, dtype=torch.float64, generator=gen)
    x2 = 1e-3 * torch.randn(n, 1, dtype=torch.float64, generator=gen)
    theta = torch.cat([x0, x1, x2], dim=1)
    true_xi = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
    y = theta @ true_xi
    return theta, y, true_xi


# ============================================================
# Test class
# ============================================================


@pytest.mark.unit
class TestSTRidgeSolver:
    """Unit tests for STRidgeSolver."""

    # ---- AC1: Known sparse system exact recovery ----
    def test_sparse_recovery_exact(self) -> None:
        """y = 2*x1 + 0*x2 + 3*x3 + 0*x4 + 0*x5 should be recovered."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=42)
        result = solver.solve(theta, y)

        assert isinstance(result, SolveResult)
        torch.testing.assert_close(result.coefficients, true_xi, rtol=1e-5, atol=1e-8)
        # selected_indices should be [0, 2] (non-zero positions)
        assert result.selected_indices is not None
        assert set(result.selected_indices) == {0, 2}

    # ---- AC2: lam=0 OLS path ----
    def test_lam_zero_uses_ols(self) -> None:
        """lam=0 should use lstsq (OLS), not ridge."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1, lam=0.0)
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=43)
        result = solver.solve(theta, y)

        # OLS with thresholding should still recover the sparse solution
        torch.testing.assert_close(result.coefficients, true_xi, rtol=1e-5, atol=1e-8)

    def test_lam_nonzero_uses_ridge(self) -> None:
        """lam>0 should use ridge regression in the iterative steps."""
        from kd2.core.linear_solve import STRidgeSolver

        solver_ols = STRidgeSolver(tol=0.1, lam=0.0)
        solver_ridge = STRidgeSolver(tol=0.1, lam=1.0)
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=44)

        result_ols = solver_ols.solve(theta, y)
        result_ridge = solver_ridge.solve(theta, y)

        # With large lam, ridge coefficients should differ from OLS
        # (they won't be exactly equal due to regularization)
        # Both should still identify the sparse pattern
        assert result_ols.selected_indices is not None
        assert result_ridge.selected_indices is not None

    # ---- AC3: Column normalization ----
    def test_column_normalization_multiscale(self) -> None:
        """Columns with different magnitudes should be handled by normalization."""
        from kd2.core.linear_solve import STRidgeSolver

        # Use low tol so that all columns survive thresholding even after
        # normalization (O(1e-3) column has small normalized coefficients)
        solver = STRidgeSolver(tol=0.01, normalize=2)
        theta, y, true_xi = _make_multiscale_system(n=200)
        result = solver.solve(theta, y)

        # All 3 columns are relevant; normalization ensures none are
        # incorrectly thresholded due to scale
        torch.testing.assert_close(result.coefficients, true_xi, rtol=1e-4, atol=1e-6)
        assert result.selected_indices is not None
        assert set(result.selected_indices) == {0, 1, 2}

    def test_no_normalization(self) -> None:
        """normalize=0 should skip normalization."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1, normalize=0)
        # Use well-scaled system so no normalization is fine
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=45)
        result = solver.solve(theta, y)

        # Should still produce a valid result
        assert isinstance(result, SolveResult)
        assert result.coefficients.shape == (5,)

    # ---- AC4: Zero columns ----
    def test_zero_column_no_crash(self) -> None:
        """Zero columns should not cause division by zero or crashes."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(
            n=100, d=5, true_coeffs=[2.0, 0.0, 3.0, 0.0, 0.0], seed=46
        )
        # Make column 1 and column 3 all zeros
        theta[:, 1] = 0.0
        theta[:, 3] = 0.0

        result = solver.solve(theta, y)

        assert isinstance(result, SolveResult)
        assert result.coefficients.shape == (5,)
        # Zero columns should have zero coefficients
        assert abs(result.coefficients[1].item()) < 1e-10
        assert abs(result.coefficients[3].item()) < 1e-10

    def test_zero_column_selected_indices_map_to_original(self) -> None:
        """selected_indices must map back to original column indices."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(
            n=100, d=5, true_coeffs=[2.0, 0.0, 3.0, 0.0, 0.0], seed=47
        )
        # Zero out columns 1 and 3
        theta[:, 1] = 0.0
        theta[:, 3] = 0.0

        result = solver.solve(theta, y)

        # selected_indices should reference original columns, not
        # the reduced matrix indices
        assert result.selected_indices is not None
        assert set(result.selected_indices) == {0, 2}

    # ---- AC5: biginds empty ----
    def test_biginds_empty_j0_returns_initial(self) -> None:
        """When tol is so high that all coeffs are small at j=0,
        should return initial w (not crash)."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=1e6, lam=0.0, max_iter=10)
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=48)
        result = solver.solve(theta, y)

        # Should not crash; returns the initial (pre-threshold) solution
        assert isinstance(result, SolveResult)
        assert result.coefficients.shape == (5,)
        # The initial w should be the OLS solution (before thresholding)
        # So coefficients should be non-trivially non-zero
        assert result.coefficients.abs().sum().item() > 0

    def test_biginds_empty_after_j0_uses_previous(self) -> None:
        """When biginds becomes empty after j>0, should break and use
        the last valid w."""
        from kd2.core.linear_solve import STRidgeSolver

        # Create system where iterative thresholding eventually
        # eliminates everything after a few rounds
        theta, y, _ = _make_sparse_system(
            n=100, d=5, true_coeffs=[0.5, 0.0, 0.5, 0.0, 0.0], seed=49
        )
        # Use a tolerance that's close to the coefficient magnitudes
        # so that after a few iterations, thresholding kills everything
        solver = STRidgeSolver(tol=0.4, max_iter=20)
        result = solver.solve(theta, y)

        # Should produce a valid result (not crash)
        assert isinstance(result, SolveResult)

    # ---- AC6: Early stopping on support convergence ----
    def test_early_stopping_support_unchanged(self) -> None:
        """Should stop early when support size doesn't change."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1, max_iter=100)
        theta, y, true_xi = _make_sparse_system(n=200, d=5, seed=50)
        result = solver.solve(theta, y)

        # With clean data, support should converge quickly
        # (well before 100 iterations). The result should be correct.
        torch.testing.assert_close(result.coefficients, true_xi, rtol=1e-5, atol=1e-8)

    # ---- AC7: Debiased OLS ----
    def test_debiased_ols_final_coefficients(self) -> None:
        """Final coefficients should be debiased (OLS on selected support)."""
        from kd2.core.linear_solve import STRidgeSolver

        # Use nonzero lam to introduce ridge bias during iteration
        solver = STRidgeSolver(tol=0.1, lam=0.5, max_iter=10)
        theta, y, true_xi = _make_sparse_system(n=200, d=5, seed=51)
        result = solver.solve(theta, y)

        # Despite using ridge during iteration, final coefficients
        # should be debiased (OLS on selected support)
        # So they should closely match the true values
        assert result.selected_indices is not None
        for idx in result.selected_indices:
            torch.testing.assert_close(
                result.coefficients[idx],
                true_xi[idx],
                rtol=1e-4,
                atol=1e-6,
            )

    # ---- AC8: solve_with_tol ----
    def test_solve_with_tol_different_sparsity(self) -> None:
        """Different tol values should produce different sparsity levels."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1) # default tol doesn't matter
        theta, y, _ = _make_sparse_system(
            n=200,
            d=10,
            true_coeffs=[3.0, 2.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            seed=52,
        )

        result_low_tol = solver.solve_with_tol(theta, y, tol=0.01)
        result_high_tol = solver.solve_with_tol(theta, y, tol=2.5)

        # Lower tol -> more terms selected; higher tol -> fewer terms
        assert result_low_tol.selected_indices is not None
        assert result_high_tol.selected_indices is not None
        assert len(result_low_tol.selected_indices) >= len(
            result_high_tol.selected_indices
        )

    def test_solve_with_tol_overrides_default(self) -> None:
        """solve_with_tol should use the provided tol, not self.tol."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=1e6) # absurdly high default
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=53)

        # Using a reasonable tol should recover the solution
        result = solver.solve_with_tol(theta, y, tol=0.1)
        torch.testing.assert_close(result.coefficients, true_xi, rtol=1e-5, atol=1e-8)

    def test_solve_uses_default_tol(self) -> None:
        """solve() should use self.tol."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=54)

        result_solve = solver.solve(theta, y)
        result_with_tol = solver.solve_with_tol(theta, y, tol=0.1)

        # Results should be identical
        torch.testing.assert_close(
            result_solve.coefficients,
            result_with_tol.coefficients,
            rtol=1e-10,
            atol=1e-10,
        )

    # ---- tol=0 keeps all ----
    def test_tol_zero_keeps_all_coefficients(self) -> None:
        """tol=0 should keep all coefficients (no thresholding)."""
        rng = torch.Generator().manual_seed(200)
        theta = torch.randn(100, 5, generator=rng, dtype=torch.float64)
        true_w = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
        y = theta @ true_w

        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.0)
        result = solver.solve(theta, y)

        assert result.selected_indices is not None
        assert len(result.selected_indices) == 5, "tol=0 should select all terms"
        torch.testing.assert_close(result.coefficients, true_w, rtol=1e-5, atol=1e-8)

    # ---- AC9: Fat matrix (n < d) ----
    def test_fat_matrix_no_crash(self) -> None:
        """Underdetermined system (more columns than rows) should not crash."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        n, d = 10, 50
        rng = torch.Generator().manual_seed(55)
        theta = torch.randn(n, d, dtype=torch.float64, generator=rng)
        true_xi = torch.zeros(d, dtype=torch.float64)
        true_xi[0] = 2.0
        true_xi[5] = 3.0
        y = theta @ true_xi

        result = solver.solve(theta, y)

        assert isinstance(result, SolveResult)
        assert result.coefficients.shape == (d,)

    # ---- AC10: NaN rejection ----
    def test_nan_theta_raises(self) -> None:
        """theta containing NaN should raise ValueError."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta = torch.tensor([[1.0, float("nan")], [3.0, 4.0]], dtype=torch.float64)
        y = torch.tensor([1.0, 2.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="NaN"):
            solver.solve(theta, y)

    def test_nan_y_raises(self) -> None:
        """y containing NaN should raise ValueError."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        y = torch.tensor([1.0, float("nan")], dtype=torch.float64)

        with pytest.raises(ValueError, match="NaN"):
            solver.solve(theta, y)

    def test_inf_theta_raises(self) -> None:
        """theta containing Inf should raise ValueError."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta = torch.tensor([[1.0, float("inf")], [3.0, 4.0]], dtype=torch.float64)
        y = torch.tensor([1.0, 2.0], dtype=torch.float64)

        with pytest.raises(ValueError, match="Inf"):
            solver.solve(theta, y)

    # ---- AC11: High condition number warning ----
    def test_high_condition_number_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Should log a warning when theta has high condition number."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.01)
        n = 100
        # Create an ill-conditioned matrix
        rng = torch.Generator().manual_seed(56)
        theta = torch.randn(n, 5, dtype=torch.float64, generator=rng)
        # Make columns nearly linearly dependent
        theta[:, 4] = theta[:, 0] + 1e-12 * theta[:, 1]
        y = torch.randn(n, dtype=torch.float64, generator=rng)

        with caplog.at_level(logging.WARNING):
            solver.solve(theta, y)

        # Should have logged a condition number warning
        assert any(
            "condition" in record.message.lower() for record in caplog.records
        ), (
            f"Expected condition number warning, got: {[r.message for r in caplog.records]}"
        )

    # ---- Interface tests ----
    def test_inherits_sparse_solver(self) -> None:
        """STRidgeSolver should be a subclass of SparseSolver."""
        from kd2.core.linear_solve import STRidgeSolver

        assert issubclass(STRidgeSolver, SparseSolver)

    def test_solve_returns_solve_result(self) -> None:
        """solve() should return a SolveResult."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=57)
        result = solver.solve(theta, y)

        assert isinstance(result, SolveResult)
        assert isinstance(result.coefficients, torch.Tensor)
        assert isinstance(result.residual, float)
        assert isinstance(result.r2, float)
        assert isinstance(result.condition_number, float)

    def test_result_has_selected_indices(self) -> None:
        """SolveResult should include selected_indices."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(n=100, d=5, seed=58)
        result = solver.solve(theta, y)

        assert result.selected_indices is not None
        assert isinstance(result.selected_indices, list)
        assert all(isinstance(i, int) for i in result.selected_indices)

    def test_r2_close_to_one_for_clean_data(self) -> None:
        """R2 should be close to 1.0 for noiseless data."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(n=200, d=5, noise_std=0.0, seed=59)
        result = solver.solve(theta, y)

        assert result.r2 > 0.999

    def test_residual_near_zero_for_clean_data(self) -> None:
        """Residual should be near zero for noiseless data."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, _ = _make_sparse_system(n=200, d=5, noise_std=0.0, seed=60)
        result = solver.solve(theta, y)

        assert result.residual < 1e-10

    def test_y_2d_accepted(self) -> None:
        """y as (n, 1) should work the same as (n,)."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        theta, y, true_xi = _make_sparse_system(n=100, d=5, seed=61)

        result_1d = solver.solve(theta, y)
        result_2d = solver.solve(theta, y.unsqueeze(1))

        torch.testing.assert_close(
            result_1d.coefficients, result_2d.coefficients, rtol=1e-10, atol=1e-10
        )

    def test_default_parameters(self) -> None:
        """Default constructor parameters should match design spec."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        assert solver.tol == 0.1
        assert solver.lam == 0.0
        assert solver.max_iter == 10
        assert solver.normalize == 2


@pytest.mark.unit
class TestSTRidgeSolverEdgeCases:
    """Edge case tests for STRidgeSolver."""

    def test_single_column(self) -> None:
        """d=1 system should work."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.01)
        rng = torch.Generator().manual_seed(62)
        theta = torch.randn(50, 1, dtype=torch.float64, generator=rng)
        y = 3.0 * theta.squeeze()
        result = solver.solve(theta, y)

        assert abs(result.coefficients[0].item() - 3.0) < 1e-4

    def test_all_zero_y(self) -> None:
        """y = 0 should give zero coefficients."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        rng = torch.Generator().manual_seed(63)
        theta = torch.randn(50, 5, dtype=torch.float64, generator=rng)
        y = torch.zeros(50, dtype=torch.float64)
        result = solver.solve(theta, y)

        assert result.coefficients.abs().max().item() < 1e-10

    def test_selected_indices_empty_list_not_none(self) -> None:
        """When all coefficients are zero, selected_indices should be []
        (sparse solver selected zero terms), NOT None (dense/unreported)."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver(tol=0.1)
        rng = torch.Generator().manual_seed(64)
        theta = torch.randn(50, 5, dtype=torch.float64, generator=rng)
        y = torch.zeros(50, dtype=torch.float64)
        result = solver.solve(theta, y)

        # selected_indices must be a list (possibly empty), never None
        assert result.selected_indices is not None
        assert isinstance(result.selected_indices, list)
        assert len(result.selected_indices) == 0

    def test_dimension_mismatch_raises(self) -> None:
        """Mismatched theta and y dimensions should raise ValueError."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta = torch.randn(10, 5, dtype=torch.float64)
        y = torch.randn(20, dtype=torch.float64)

        with pytest.raises(ValueError, match="dimension"):
            solver.solve(theta, y)

    def test_empty_theta_raises(self) -> None:
        """Empty theta should raise ValueError."""
        from kd2.core.linear_solve import STRidgeSolver

        solver = STRidgeSolver()
        theta = torch.zeros(0, 5, dtype=torch.float64)
        y = torch.zeros(0, dtype=torch.float64)

        with pytest.raises(ValueError):
            solver.solve(theta, y)
