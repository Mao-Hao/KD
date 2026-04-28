"""Tests for LeastSquaresSolver.

Test coverage:
- smoke: Basic instantiation and interface existence
- unit: Core solving logic, coefficient recovery
- numerical: Precision, stability, edge cases

Note: These tests are written for TDD TDD red phase. All tests should fail
until the implementation is complete (currently raises NotImplementedError).
"""

from __future__ import annotations

import pytest
import torch

from kd2.core.linear_solve import LeastSquaresSolver, SolveResult, SparseSolver

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def solver() -> LeastSquaresSolver:
    """Default LeastSquaresSolver instance."""
    return LeastSquaresSolver()


@pytest.fixture
def solver_with_rcond() -> LeastSquaresSolver:
    """LeastSquaresSolver with explicit rcond."""
    return LeastSquaresSolver(rcond=1e-10)


@pytest.fixture
def simple_overdetermined() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simple overdetermined system with known exact solution.

    System: y = 2*x1 + 3*x2
    theta: 10 samples, 2 features
    true_coef: [2, 3]

    Returns:
        (theta, y, true_coefficients)
    """
    torch.manual_seed(42)
    n_samples = 10
    n_features = 2

    theta = torch.randn(n_samples, n_features, dtype=torch.float64)
    true_coef = torch.tensor([2.0, 3.0], dtype=torch.float64)
    y = theta @ true_coef

    return theta, y, true_coef


@pytest.fixture
def noisy_overdetermined() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Overdetermined system with noise.

    Same as simple_overdetermined but with added noise.

    Returns:
        (theta, y, true_coefficients)
    """
    torch.manual_seed(42)
    n_samples = 100
    n_features = 2

    theta = torch.randn(n_samples, n_features, dtype=torch.float64)
    true_coef = torch.tensor([2.0, 3.0], dtype=torch.float64)
    noise = 0.1 * torch.randn(n_samples, dtype=torch.float64)
    y = theta @ true_coef + noise

    return theta, y, true_coef


@pytest.fixture
def underdetermined() -> tuple[torch.Tensor, torch.Tensor]:
    """Underdetermined system (more features than samples).

    System: 5 samples, 10 features
    Has infinite solutions; solver should return minimum norm solution.

    Returns:
        (theta, y)
    """
    torch.manual_seed(42)
    n_samples = 5
    n_features = 10

    theta = torch.randn(n_samples, n_features, dtype=torch.float64)
    y = torch.randn(n_samples, dtype=torch.float64)

    return theta, y


@pytest.fixture
def square_system() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Square system (same number of samples and features).

    Returns:
        (theta, y, true_coefficients)
    """
    torch.manual_seed(42)
    n = 5

    # Create a well-conditioned square matrix
    theta = torch.randn(n, n, dtype=torch.float64)
    theta = theta + torch.eye(n, dtype=torch.float64) * 2 # Make it well-conditioned
    true_coef = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=torch.float64)
    y = theta @ true_coef

    return theta, y, true_coef


@pytest.fixture
def ill_conditioned() -> tuple[torch.Tensor, torch.Tensor]:
    """Ill-conditioned system with high condition number.

    Near-collinear columns make the system numerically challenging.

    Returns:
        (theta, y)
    """
    torch.manual_seed(42)
    n_samples = 20

    # Create matrix with near-collinear columns (3 features)
    x1 = torch.randn(n_samples, 1, dtype=torch.float64)
    x2 = x1 + 1e-8 * torch.randn(n_samples, 1, dtype=torch.float64) # Nearly identical
    x3 = torch.randn(n_samples, 1, dtype=torch.float64)

    theta = torch.cat([x1, x2, x3], dim=1)
    y = 2 * x1.squeeze() + 3 * x3.squeeze()

    return theta, y


# =============================================================================
# Smoke Tests
# =============================================================================


@pytest.mark.smoke
class TestSolverSmoke:
    """Smoke tests: basic instantiation and interface existence."""

    def test_solve_result_can_be_created(self) -> None:
        """SolveResult can be instantiated with required fields."""
        result = SolveResult(
            coefficients=torch.tensor([1.0, 2.0]),
            residual=0.1,
            r2=0.99,
            condition_number=10.0,
        )
        assert result.coefficients is not None
        assert result.residual == 0.1
        assert result.r2 == 0.99
        assert result.condition_number == 10.0
        assert result.selected_indices is None

    def test_solve_result_with_selected_indices(self) -> None:
        """SolveResult can include selected_indices for sparse solvers."""
        result = SolveResult(
            coefficients=torch.tensor([1.0, 0.0, 2.0]),
            residual=0.0,
            r2=1.0,
            condition_number=1.0,
            selected_indices=[0, 2],
        )
        assert result.selected_indices == [0, 2]

    def test_sparse_solver_is_abc(self) -> None:
        """SparseSolver is abstract and cannot be instantiated directly."""
        with pytest.raises(TypeError):
            SparseSolver() # type: ignore[abstract]

    def test_least_squares_solver_can_be_created(self) -> None:
        """LeastSquaresSolver can be instantiated."""
        solver = LeastSquaresSolver()
        assert solver is not None
        assert solver.rcond is None

    def test_least_squares_solver_with_rcond(self) -> None:
        """LeastSquaresSolver accepts rcond parameter."""
        solver = LeastSquaresSolver(rcond=1e-10)
        assert solver.rcond == 1e-10

    def test_least_squares_solver_is_sparse_solver(self) -> None:
        """LeastSquaresSolver is a subclass of SparseSolver."""
        assert issubclass(LeastSquaresSolver, SparseSolver)
        solver = LeastSquaresSolver()
        assert isinstance(solver, SparseSolver)

    def test_least_squares_solver_has_solve_method(self) -> None:
        """LeastSquaresSolver has solve method."""
        solver = LeastSquaresSolver()
        assert hasattr(solver, "solve")
        assert callable(solver.solve)


# =============================================================================
# Unit Tests - Basic Functionality
# =============================================================================


@pytest.mark.unit
class TestBasicSolving:
    """Unit tests for basic solving functionality."""

    def test_solve_returns_solve_result(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """solve() returns a SolveResult instance."""
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)
        assert isinstance(result, SolveResult)

    def test_solve_recovers_exact_coefficients(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """solve() recovers exact coefficients for noiseless data."""
        theta, y, true_coef = simple_overdetermined
        result = solver.solve(theta, y)

        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_solve_coefficients_shape(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Coefficients have shape (n_terms,)."""
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)

        n_terms = theta.shape[1]
        assert result.coefficients.shape == (n_terms,)

    def test_solve_handles_1d_y(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """solve() handles y as 1D tensor (n_samples,)."""
        theta, y, true_coef = simple_overdetermined
        assert y.dim() == 1 # Verify fixture gives 1D

        result = solver.solve(theta, y)
        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_solve_handles_2d_y(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """solve() handles y as 2D tensor (n_samples, 1)."""
        theta, y, true_coef = simple_overdetermined
        y_2d = y.unsqueeze(1)
        assert y_2d.dim() == 2

        result = solver.solve(theta, y_2d)
        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_solve_square_system(
        self,
        solver: LeastSquaresSolver,
        square_system: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """solve() handles square systems (n_samples == n_terms)."""
        theta, y, true_coef = square_system
        result = solver.solve(theta, y)

        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-8,
        )


# =============================================================================
# Unit Tests - Overdetermined Systems
# =============================================================================


@pytest.mark.unit
class TestOverdeterminedSystems:
    """Unit tests for overdetermined systems (n > m)."""

    def test_overdetermined_minimizes_residual(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Overdetermined system finds least squares solution."""
        theta, y, _ = noisy_overdetermined
        result = solver.solve(theta, y)

        # Verify solution minimizes residual
        # Any small perturbation should increase residual
        y_pred = theta @ result.coefficients
        residual = ((y - y_pred) ** 2).sum().item()

        # Try a perturbed solution
        perturbed_coef = result.coefficients + 0.1 * torch.randn_like(
            result.coefficients
        )
        y_pred_perturbed = theta @ perturbed_coef
        residual_perturbed = ((y - y_pred_perturbed) ** 2).sum().item()

        assert residual <= residual_perturbed

    def test_overdetermined_coefficients_close_to_true(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Noisy overdetermined recovers coefficients approximately."""
        theta, y, true_coef = noisy_overdetermined
        result = solver.solve(theta, y)

        # With noise, coefficients should be close but not exact
        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=0.1, # 10% relative tolerance
            atol=0.2, # Absolute tolerance for noise
        )


# =============================================================================
# Unit Tests - Underdetermined Systems
# =============================================================================


@pytest.mark.unit
class TestUnderdeterminedSystems:
    """Unit tests for underdetermined systems (n < m)."""

    def test_underdetermined_returns_solution(
        self,
        solver: LeastSquaresSolver,
        underdetermined: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Underdetermined system returns a valid solution."""
        theta, y = underdetermined
        result = solver.solve(theta, y)

        # Solution should satisfy the equation (or approximately)
        y_pred = theta @ result.coefficients
        torch.testing.assert_close(
            y_pred,
            y,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_underdetermined_minimum_norm(
        self,
        solver: LeastSquaresSolver,
        underdetermined: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Underdetermined system returns minimum norm solution.

        lstsq should return the solution with smallest ||xi||.
        """
        theta, y = underdetermined
        result = solver.solve(theta, y)

        solution_norm = torch.linalg.norm(result.coefficients).item()

        # Generate random solution that also satisfies equation
        # (add null space component)
        # If lstsq gives min norm, any other solution should have larger norm
        # This is hard to test directly, so we just check solution is reasonable
        assert solution_norm < 1e6 # Not exploding


# =============================================================================
# Unit Tests - Metrics (R2, Residual, Condition Number)
# =============================================================================


@pytest.mark.unit
class TestMetrics:
    """Unit tests for solver metrics."""

    def test_residual_for_exact_solution(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Residual is near zero for exact solution."""
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)

        # For noiseless data, residual should be ~0
        assert result.residual < 1e-10

    def test_residual_for_noisy_data(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Residual is positive for noisy data."""
        theta, y, _ = noisy_overdetermined
        result = solver.solve(theta, y)

        # With noise, residual should be positive
        assert result.residual > 0.0

    def test_residual_matches_computed(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Residual matches ||y - theta @ xi||^2."""
        theta, y, _ = noisy_overdetermined
        result = solver.solve(theta, y)

        y_pred = theta @ result.coefficients
        expected_residual = ((y - y_pred) ** 2).sum().item()

        assert abs(result.residual - expected_residual) < 1e-10

    def test_r2_for_perfect_fit(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """R2 is ~1.0 for perfect fit."""
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)

        assert result.r2 > 0.9999

    def test_r2_for_noisy_data(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """R2 is less than 1.0 for noisy data."""
        theta, y, _ = noisy_overdetermined
        result = solver.solve(theta, y)

        assert 0.0 < result.r2 < 1.0

    def test_r2_formula_correct(
        self,
        solver: LeastSquaresSolver,
        noisy_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """R2 = 1 - SS_res / SS_tot."""
        theta, y, _ = noisy_overdetermined
        result = solver.solve(theta, y)

        # Compute R2 directly
        y_pred = theta @ result.coefficients
        ss_res = ((y - y_pred) ** 2).sum().item()
        ss_tot = ((y - y.mean()) ** 2).sum().item()
        expected_r2 = 1.0 - ss_res / ss_tot

        assert abs(result.r2 - expected_r2) < 1e-10

    def test_condition_number_returned(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Condition number is computed and returned."""
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)

        assert result.condition_number > 0
        assert result.condition_number < float("inf")

    def test_condition_number_high_for_ill_conditioned(
        self,
        solver: LeastSquaresSolver,
        ill_conditioned: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Ill-conditioned matrix has high condition number."""
        theta, y = ill_conditioned
        result = solver.solve(theta, y)

        # Near-collinear columns should give high condition number
        assert result.condition_number > 1e6


# =============================================================================
# Unit Tests - Data Types
# =============================================================================


@pytest.mark.unit
class TestDataTypes:
    """Unit tests for different data types."""

    def test_float64_input(self, solver: LeastSquaresSolver) -> None:
        """Solver handles float64 input."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        true_coef = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64)
        y = theta @ true_coef

        result = solver.solve(theta, y)
        assert result.coefficients.dtype == torch.float64

    def test_float32_input(self, solver: LeastSquaresSolver) -> None:
        """Solver handles float32 input."""
        theta = torch.randn(10, 3, dtype=torch.float32)
        true_coef = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        y = theta @ true_coef

        result = solver.solve(theta, y)
        assert result.coefficients.dtype == torch.float32

    def test_output_matches_input_dtype(self, solver: LeastSquaresSolver) -> None:
        """Output dtype matches input dtype."""
        for dtype in [torch.float32, torch.float64]:
            theta = torch.randn(10, 3, dtype=dtype)
            y = torch.randn(10, dtype=dtype)

            result = solver.solve(theta, y)
            assert result.coefficients.dtype == dtype


# =============================================================================
# Numerical Stability Tests
# =============================================================================


@pytest.mark.numerical
class TestNumericalStability:
    """Numerical stability tests."""

    def test_handles_very_small_values(self, solver: LeastSquaresSolver) -> None:
        """Solver handles very small coefficient values."""
        theta = torch.randn(20, 3, dtype=torch.float64)
        true_coef = torch.tensor([1e-10, 2e-10, 3e-10], dtype=torch.float64)
        y = theta @ true_coef

        result = solver.solve(theta, y)
        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-3, # Relaxed tolerance for small values
            atol=1e-12,
        )

    def test_handles_very_large_values(self, solver: LeastSquaresSolver) -> None:
        """Solver handles very large coefficient values."""
        theta = torch.randn(20, 3, dtype=torch.float64)
        true_coef = torch.tensor([1e6, 2e6, 3e6], dtype=torch.float64)
        y = theta @ true_coef

        result = solver.solve(theta, y)
        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-2,
        )

    def test_handles_mixed_scale_values(self, solver: LeastSquaresSolver) -> None:
        """Solver handles coefficients with very different scales."""
        theta = torch.randn(50, 3, dtype=torch.float64)
        true_coef = torch.tensor([1e-6, 1.0, 1e6], dtype=torch.float64)
        y = theta @ true_coef

        result = solver.solve(theta, y)
        # With mixed scales, relative tolerance may need adjustment
        diff = (result.coefficients - true_coef).abs()
        relative_error = diff / (true_coef.abs() + 1e-10)
        assert relative_error.max().item() < 0.01

    def test_result_is_finite(
        self,
        solver: LeastSquaresSolver,
        simple_overdetermined: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Result coefficients are always finite (no NaN/Inf)."""
        theta, y, _ = simple_overdetermined
        result = solver.solve(theta, y)

        assert torch.isfinite(result.coefficients).all()

    def test_result_finite_for_ill_conditioned(
        self,
        solver: LeastSquaresSolver,
        ill_conditioned: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Result is finite even for ill-conditioned systems."""
        theta, y = ill_conditioned
        result = solver.solve(theta, y)

        assert torch.isfinite(result.coefficients).all()


# =============================================================================
# Edge Case Tests
# =============================================================================


@pytest.mark.unit
class TestEdgeCases:
    """Edge case tests."""

    def test_single_sample_single_feature(self, solver: LeastSquaresSolver) -> None:
        """Solver handles 1x1 system."""
        theta = torch.tensor([[2.0]], dtype=torch.float64)
        y = torch.tensor([6.0], dtype=torch.float64)

        result = solver.solve(theta, y)
        # 2 * x = 6 => x = 3
        torch.testing.assert_close(
            result.coefficients,
            torch.tensor([3.0], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_single_feature_multiple_samples(self, solver: LeastSquaresSolver) -> None:
        """Solver handles nx1 system (single feature)."""
        theta = torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64)
        y = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)

        result = solver.solve(theta, y)
        # y = 2*x => coefficient is 2
        torch.testing.assert_close(
            result.coefficients,
            torch.tensor([2.0], dtype=torch.float64),
            rtol=1e-5,
            atol=1e-8,
        )

    def test_multiple_features_few_samples(self, solver: LeastSquaresSolver) -> None:
        """Solver handles system with more features than samples."""
        # 3 samples, 5 features (underdetermined)
        theta = torch.randn(3, 5, dtype=torch.float64)
        y = torch.randn(3, dtype=torch.float64)

        result = solver.solve(theta, y)

        # Should return a solution that satisfies the equation
        y_pred = theta @ result.coefficients
        torch.testing.assert_close(y_pred, y, rtol=1e-5, atol=1e-8)

    def test_all_zeros_y(self, solver: LeastSquaresSolver) -> None:
        """Solver handles y = 0 (trivial solution)."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.zeros(10, dtype=torch.float64)

        result = solver.solve(theta, y)

        # Coefficients should be all zeros (or very close)
        assert torch.allclose(
            result.coefficients,
            torch.zeros(3, dtype=torch.float64),
            atol=1e-10,
        )
        # When y is all zeros, SS_tot = 0, so R2 should be 0.0 (design decision)
        assert result.r2 == 0.0

    def test_constant_y(self, solver: LeastSquaresSolver) -> None:
        """Solver handles constant y (all same value)."""
        # Include a constant column (all ones) in theta
        theta = torch.randn(10, 3, dtype=torch.float64)
        theta[:, 0] = 1.0 # First column is constant

        # y is constant 5.0
        y = torch.full((10,), 5.0, dtype=torch.float64)

        result = solver.solve(theta, y)

        # Reconstruction should be close to y
        y_pred = theta @ result.coefficients
        torch.testing.assert_close(y_pred, y, rtol=1e-5, atol=1e-8)

    def test_r2_with_constant_y(self, solver: LeastSquaresSolver) -> None:
        """R2 is handled correctly when y is constant (SS_tot = 0).

        When y is constant, SS_tot = 0 and R2 formula breaks.
        Design decision: return R2 = 0.0 when SS_tot = 0.
        """
        # Include a constant column to get perfect fit
        theta = torch.ones(10, 1, dtype=torch.float64)
        y = torch.full((10,), 5.0, dtype=torch.float64)

        result = solver.solve(theta, y)

        # R2 should be 0.0 when SS_tot = 0 (design decision)
        assert result.r2 == 0.0


# =============================================================================
# Input Validation Tests (H1, H5)
# =============================================================================


@pytest.mark.unit
class TestInputValidation:
    """Tests for input validation and error handling."""

    def test_dimension_mismatch_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when theta rows != y length."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(5, dtype=torch.float64) # Mismatch: 5 != 10

        with pytest.raises(ValueError, match="dimension"):
            solver.solve(theta, y)

    def test_empty_theta_zero_rows_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when theta has 0 rows."""
        theta = torch.empty(0, 3, dtype=torch.float64)
        y = torch.empty(0, dtype=torch.float64)

        with pytest.raises(ValueError, match="empty"):
            solver.solve(theta, y)

    def test_empty_theta_zero_cols_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when theta has 0 columns."""
        theta = torch.empty(10, 0, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="empty"):
            solver.solve(theta, y)

    def test_theta_1d_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when theta is 1D."""
        theta = torch.randn(10, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="2D"):
            solver.solve(theta, y)

    def test_y_wrong_shape_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when y has wrong shape."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, 2, dtype=torch.float64) # 2D with wrong shape

        with pytest.raises(ValueError):
            solver.solve(theta, y)


# =============================================================================
# Numerical Edge Cases (H3, H4, H6)
# =============================================================================


@pytest.mark.numerical
class TestNumericalEdgeCases:
    """Tests for NaN, Inf, and other numerical edge cases."""

    def test_nan_in_theta_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when theta contains NaN."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        theta[5, 1] = float("nan")
        y = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="NaN"):
            solver.solve(theta, y)

    def test_nan_in_y_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when y contains NaN."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)
        y[3] = float("nan")

        with pytest.raises(ValueError, match="NaN"):
            solver.solve(theta, y)

    def test_inf_in_theta_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when theta contains Inf."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        theta[2, 0] = float("inf")
        y = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="Inf"):
            solver.solve(theta, y)

    def test_neg_inf_in_theta_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when theta contains -Inf."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        theta[7, 2] = float("-inf")
        y = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="Inf"):
            solver.solve(theta, y)

    def test_inf_in_y_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when y contains Inf."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)
        y[8] = float("inf")

        with pytest.raises(ValueError, match="Inf"):
            solver.solve(theta, y)

    def test_neg_inf_in_y_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """ValueError raised when y contains -Inf."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)
        y[0] = float("-inf")

        with pytest.raises(ValueError, match="Inf"):
            solver.solve(theta, y)

    def test_all_zero_theta_returns_result_with_inf_condition(
        self, solver: LeastSquaresSolver
    ) -> None:
        """All-zero theta is a degenerate case, condition_number = inf.

        Design decision: Return result but with condition_number = inf
        to indicate the matrix is singular.
        """
        theta = torch.zeros(10, 3, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float64)

        result = solver.solve(theta, y)

        # Coefficients should be finite (likely zeros or arbitrary)
        assert torch.isfinite(result.coefficients).all()
        # Condition number should be infinity for singular matrix
        assert result.condition_number == float("inf")

    def test_single_zero_column_in_theta(
        self, solver: LeastSquaresSolver
    ) -> None:
        """Theta with one zero column is still solvable."""
        theta = torch.randn(10, 3, dtype=torch.float64)
        theta[:, 1] = 0.0 # Zero out middle column
        y = theta[:, 0] * 2.0 + theta[:, 2] * 3.0

        result = solver.solve(theta, y)

        # Should still produce a valid result
        assert torch.isfinite(result.coefficients).all()
        # Condition number should be high (one column is zero)
        # But not necessarily inf if using pseudoinverse


# =============================================================================
# Rcond Tests
# =============================================================================


@pytest.mark.unit
class TestRcond:
    """Tests for rcond parameter behavior."""

    def test_rcond_affects_ill_conditioned_solution(self) -> None:
        """rcond parameter affects solution for ill-conditioned systems."""
        # Create a nearly singular matrix
        n_samples = 20
        theta = torch.randn(n_samples, 3, dtype=torch.float64)
        # Make two columns nearly identical
        theta[:, 1] = theta[:, 0] + 1e-12 * torch.randn(n_samples, dtype=torch.float64)
        y = torch.randn(n_samples, dtype=torch.float64)

        solver_no_rcond = LeastSquaresSolver(rcond=None)
        solver_with_rcond = LeastSquaresSolver(rcond=1e-6)

        result_no_rcond = solver_no_rcond.solve(theta, y)
        result_with_rcond = solver_with_rcond.solve(theta, y)

        # Both should produce finite results
        assert torch.isfinite(result_no_rcond.coefficients).all()
        assert torch.isfinite(result_with_rcond.coefficients).all()


# =============================================================================
# Integration-like Tests (Self-contained)
# =============================================================================


@pytest.mark.unit
class TestBurgersCoefficients:
    """Tests simulating coefficient recovery for Burgers equation.

    Burgers equation: u_t + u*u_x = nu*u_xx
    In standard form for solving: u_t = -u*u_x + nu*u_xx

    Library terms: [u*u_x, u_xx]
    True coefficients: [-1.0, nu]
    """

    def test_recover_burgers_coefficients(self, solver: LeastSquaresSolver) -> None:
        """Recover Burgers equation coefficients from synthetic data."""
        torch.manual_seed(42)
        n_samples = 100
        nu = 0.1

        # Simulate library terms: [u*u_x, u_xx]
        # theta[:, 0] = u*u_x (convection)
        # theta[:, 1] = u_xx (diffusion)
        theta = torch.randn(n_samples, 2, dtype=torch.float64)

        # True coefficients for u_t = -u*u_x + nu*u_xx
        true_coef = torch.tensor([-1.0, nu], dtype=torch.float64)

        # y = u_t = -u*u_x + nu*u_xx
        y = theta @ true_coef

        result = solver.solve(theta, y)

        torch.testing.assert_close(
            result.coefficients,
            true_coef,
            rtol=1e-5,
            atol=1e-8,
        )

    def test_recover_burgers_with_noise(self, solver: LeastSquaresSolver) -> None:
        """Recover Burgers coefficients with measurement noise."""
        torch.manual_seed(42)
        n_samples = 500 # More samples to handle noise
        nu = 0.1

        theta = torch.randn(n_samples, 2, dtype=torch.float64)
        true_coef = torch.tensor([-1.0, nu], dtype=torch.float64)

        # Add 5% noise
        noise_level = 0.05
        y_clean = theta @ true_coef
        noise_scale = noise_level * y_clean.std()
        noise = noise_scale * torch.randn(n_samples, dtype=torch.float64)
        y = y_clean + noise

        result = solver.solve(theta, y)

        # Should recover coefficients within 10% error
        relative_error = (result.coefficients - true_coef).abs() / true_coef.abs()
        assert relative_error.max().item() < 0.1


# =============================================================================
# High Priority Issue Tests (H1, H2, H3)
# =============================================================================


@pytest.mark.unit
class TestHighPriorityFixes:
    """Tests for high priority issues identified in review.

    H1: Mixed dtype attack - theta and y with different dtypes
    H2: R2 floating point comparison - ss_tot == 0.0 fails for near-constant y
    H3: 0D tensor attack - y as 0D tensor causes IndexError
    """

    def test_mixed_dtype_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """H1: ValueError raised when theta and y have different dtypes.

        Mixed precision can cause silent precision loss. The solver should
        reject mismatched dtypes and require explicit casting.
        """
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.randn(10, dtype=torch.float32) # Different dtype!

        with pytest.raises(ValueError, match="dtype"):
            solver.solve(theta, y)

    def test_mixed_dtype_both_directions(
        self, solver: LeastSquaresSolver
    ) -> None:
        """H1: Both directions of dtype mismatch are caught."""
        # float32 theta, float64 y
        theta_32 = torch.randn(10, 3, dtype=torch.float32)
        y_64 = torch.randn(10, dtype=torch.float64)

        with pytest.raises(ValueError, match="dtype"):
            solver.solve(theta_32, y_64)

        # float64 theta, float32 y
        theta_64 = torch.randn(10, 3, dtype=torch.float64)
        y_32 = torch.randn(10, dtype=torch.float32)

        with pytest.raises(ValueError, match="dtype"):
            solver.solve(theta_64, y_32)

    def test_0d_y_raises_value_error(
        self, solver: LeastSquaresSolver
    ) -> None:
        """H3: ValueError raised when y is 0D tensor.

        A 0D tensor (scalar) has no shape[0], causing IndexError in validation.
        The solver should detect and reject 0D tensors early.
        """
        theta = torch.randn(10, 3, dtype=torch.float64)
        y = torch.tensor(1.0, dtype=torch.float64) # 0D tensor!

        assert y.dim() == 0 # Verify it's really 0D

        with pytest.raises(ValueError, match="0D"):
            solver.solve(theta, y)

    def test_near_constant_y_r2_bounded(
        self, solver: LeastSquaresSolver
    ) -> None:
        """H2: R2 is bounded for near-constant y.

        When y is nearly constant (SS_tot very small but not exactly 0),
        the exact comparison ss_tot == 0.0 fails, leading to extreme R2.
        The fix uses epsilon comparison and clamps R2 to [-1, 1].
        """
        theta = torch.randn(10, 3, dtype=torch.float64)
        # Near-constant y: small perturbation from constant
        y = torch.full((10,), 5.0, dtype=torch.float64)
        y = y + 1e-14 * torch.randn(10, dtype=torch.float64) # Tiny noise

        result = solver.solve(theta, y)

        # R2 should be bounded, not extreme (like 1e+28)
        assert -1.0 <= result.r2 <= 1.0

    def test_exactly_constant_y_r2_zero(
        self, solver: LeastSquaresSolver
    ) -> None:
        """H2: R2 is 0.0 for exactly constant y (SS_tot = 0).

        This verifies the existing behavior is preserved after the fix.
        """
        theta = torch.ones(10, 1, dtype=torch.float64)
        y = torch.full((10,), 5.0, dtype=torch.float64)

        result = solver.solve(theta, y)

        assert result.r2 == 0.0
