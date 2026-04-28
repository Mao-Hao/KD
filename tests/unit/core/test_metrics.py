"""Tests for core/metrics.py -- standalone model selection metrics.

TDD red phase: tests written against the interface spec (, ).
Implementation does not exist yet; all tests are expected to fail on import.

Test coverage:
- smoke: Imports, callability, ScorerFn type
- unit: Known-value correctness, mathematical identities
- edge cases: MSE=0, MSE<0, MSE=NaN, MSE=Inf, k=0, AICc degenerate n
- ranking: aic and aic_no_n rank candidates identically (same dataset)
- factory: make_aic_scorer, make_sga_scorer, make_bic_scorer
- property-based (hypothesis): finiteness, monotonicity, penalty ordering
- negative tests: invalid inputs, degenerate conditions (~25%)
"""

from __future__ import annotations

import math

import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from kd2.core.metrics import (
    ScorerFn,
    aic,
    aic_no_n,
    aicc,
    bic,
    make_aic_scorer,
    make_bic_scorer,
    make_sga_scorer,
    nmse,
)

# -- Hypothesis strategies ---------------------------------------------------

# Positive MSE values (the valid-input domain for information criteria)
positive_mse = st.floats(
    min_value=1e-12, max_value=1e8, allow_nan=False, allow_infinity=False
)

# Complexity: non-negative integers
complexity_k = st.integers(min_value=0, max_value=50)

# Sample size: at least 2 (need n > 1 for AIC to be meaningful)
sample_n = st.integers(min_value=2, max_value=10000)

# Ratio: positive float for SGA penalty scaling
aic_ratio = st.floats(
    min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False
)


# ===========================================================================
# Smoke: imports and callability
# ===========================================================================


class TestSmoke:
    """Verify that the public API is importable and callable."""

    @pytest.mark.smoke
    def test_aic_callable(self) -> None:
        result = aic(mse=1.0, k=2, n=100)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_aic_no_n_callable(self) -> None:
        result = aic_no_n(mse=1.0, k=2)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_aicc_callable(self) -> None:
        result = aicc(mse=1.0, k=2, n=100)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_bic_callable(self) -> None:
        result = bic(mse=1.0, k=2, n=100)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_nmse_callable(self) -> None:
        result = nmse(mse=0.5, target_var=1.0)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_scorer_fn_type_is_callable(self) -> None:
        """ScorerFn should be a type alias for Callable[[float, int], float]."""
        # Verify the type alias exists and is a callable specification
        assert ScorerFn is not None

        # A conforming function should be assignable
        def my_scorer(mse: float, k: int) -> float:
            return mse + k

        scorer: ScorerFn = my_scorer
        assert scorer(1.0, 2) == 3.0

    @pytest.mark.smoke
    def test_make_aic_scorer_callable(self) -> None:
        scorer = make_aic_scorer(n=100)
        result = scorer(1.0, 2)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_make_sga_scorer_callable(self) -> None:
        scorer = make_sga_scorer(ratio=1.0)
        result = scorer(1.0, 2)
        assert isinstance(result, float)

    @pytest.mark.smoke
    def test_make_bic_scorer_callable(self) -> None:
        scorer = make_bic_scorer(n=100)
        result = scorer(1.0, 2)
        assert isinstance(result, float)


# ===========================================================================
# Unit: known-value correctness via independent computation
# ===========================================================================


class TestAICKnownValues:
    """Test aic() against independently computed values.

    Formula: aic = n * log(MSE) + 2 * k
    """

    def test_mse_one_k_zero(self) -> None:
        """log(1) = 0, so aic = n*0 + 2*0 = 0."""
        result = aic(mse=1.0, k=0, n=100)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_mse_one_k_positive(self) -> None:
        """log(1) = 0, so aic = 0 + 2*k = 2k."""
        result = aic(mse=1.0, k=5, n=50)
        assert result == pytest.approx(10.0, abs=1e-12)

    def test_mse_e_squared(self) -> None:
        """mse = e^2, log(e^2) = 2, so aic = n*2 + 2*k."""
        e2 = math.e**2
        result = aic(mse=e2, k=3, n=50)
        expected = 50.0 * 2.0 + 2.0 * 3.0
        assert result == pytest.approx(expected, abs=1e-10)

    def test_k_zero_reduces_to_n_log_mse(self) -> None:
        """When k=0, aic = n * log(MSE) exactly."""
        result = aic(mse=0.01, k=0, n=200)
        expected = 200.0 * math.log(0.01)
        assert result == pytest.approx(expected, abs=1e-10)


class TestAICNoNKnownValues:
    """Test aic_no_n() against independently computed values.

    Formula: aic_no_n = 2 * k * ratio + 2 * log(MSE)
    """

    def test_mse_one_ratio_one(self) -> None:
        """log(1) = 0, so aic_no_n = 2*k + 0 = 2k."""
        result = aic_no_n(mse=1.0, k=3, ratio=1.0)
        assert result == pytest.approx(6.0, abs=1e-12)

    def test_k_zero(self) -> None:
        """k=0 means no penalty: aic_no_n = 2*log(MSE)."""
        result = aic_no_n(mse=0.5, k=0, ratio=1.0)
        expected = 2.0 * math.log(0.5)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_custom_ratio(self) -> None:
        """Ratio scales the complexity penalty."""
        result = aic_no_n(mse=1.0, k=4, ratio=2.5)
        expected = 2.0 * 4 * 2.5 + 2.0 * math.log(1.0)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_default_ratio_is_one(self) -> None:
        """Default ratio should be 1.0."""
        with_default = aic_no_n(mse=2.0, k=3)
        with_explicit = aic_no_n(mse=2.0, k=3, ratio=1.0)
        assert with_default == pytest.approx(with_explicit, abs=1e-15)


class TestAICcKnownValues:
    """Test aicc() against independently computed values.

    Formula: aicc = n*log(MSE) + 2k + 2k(k+1)/(n-k-1)
    """

    def test_large_n_converges_to_aic(self) -> None:
        """For n >> k, the correction term vanishes: AICc -> AIC."""
        n = 10000
        k = 3
        mse = 0.5
        result_aicc = aicc(mse=mse, k=k, n=n)
        result_aic = aic(mse=mse, k=k, n=n)
        # Correction = 2*3*4 / (10000-4) = 24/9996 ~ 0.0024
        assert abs(result_aicc - result_aic) < 0.01

    def test_k_zero_equals_aic(self) -> None:
        """When k=0, correction = 2*0*1/(n-0-1) = 0, so AICc = AIC."""
        result_aicc = aicc(mse=2.0, k=0, n=100)
        result_aic = aic(mse=2.0, k=0, n=100)
        assert result_aicc == pytest.approx(result_aic, abs=1e-12)

    def test_correction_positive(self) -> None:
        """AICc correction is always non-negative: AICc >= AIC."""
        mse, k, n = 0.5, 5, 50
        result_aicc = aicc(mse=mse, k=k, n=n)
        result_aic = aic(mse=mse, k=k, n=n)
        assert result_aicc >= result_aic

    def test_independent_computation(self) -> None:
        """Verify against step-by-step manual calculation."""
        mse, k, n = 0.3, 4, 20
        # Step 1: base AIC
        base = n * math.log(mse) + 2 * k
        # Step 2: correction
        correction = 2.0 * k * (k + 1) / (n - k - 1)
        expected = base + correction
        result = aicc(mse=mse, k=k, n=n)
        assert result == pytest.approx(expected, abs=1e-10)


class TestBICKnownValues:
    """Test bic() against independently computed values.

    Formula: bic = n * log(MSE) + k * log(n)
    """

    def test_mse_one_k_zero(self) -> None:
        """log(1) = 0, k=0 => bic = 0."""
        result = bic(mse=1.0, k=0, n=100)
        assert result == pytest.approx(0.0, abs=1e-12)

    def test_k_zero_equals_aic(self) -> None:
        """When k=0, both BIC and AIC reduce to n*log(MSE)."""
        mse = 0.25
        n = 100
        result_bic = bic(mse=mse, k=0, n=n)
        result_aic = aic(mse=mse, k=0, n=n)
        assert result_bic == pytest.approx(result_aic, abs=1e-12)

    def test_bic_penalty_heavier_than_aic_for_large_n(self) -> None:
        """For n > e^2 ~ 7.39, BIC penalizes more per term than AIC.

        AIC penalty per term: 2
        BIC penalty per term: log(n)
        log(n) > 2 when n > e^2.
        """
        mse = 0.5
        k = 5
        n = 100 # log(100) ~ 4.6 > 2
        result_bic = bic(mse=mse, k=k, n=n)
        result_aic = aic(mse=mse, k=k, n=n)
        # BIC should be larger (more penalized) than AIC for same k
        assert result_bic > result_aic

    def test_independent_computation(self) -> None:
        """Verify against step-by-step manual calculation."""
        mse, k, n = 0.1, 3, 200
        expected = n * math.log(mse) + k * math.log(n)
        result = bic(mse=mse, k=k, n=n)
        assert result == pytest.approx(expected, abs=1e-10)


class TestNMSEKnownValues:
    """Test nmse() against independently computed values.

    Formula: nmse = MSE / target_var (if target_var > eps, else MSE)
    """

    def test_unit_variance(self) -> None:
        """target_var = 1 => nmse = mse."""
        result = nmse(mse=0.3, target_var=1.0)
        assert result == pytest.approx(0.3, abs=1e-12)

    def test_normalization(self) -> None:
        """nmse = mse / target_var."""
        result = nmse(mse=0.5, target_var=2.0)
        assert result == pytest.approx(0.25, abs=1e-12)

    def test_perfect_fit(self) -> None:
        """mse = 0 => nmse = 0."""
        result = nmse(mse=0.0, target_var=1.0)
        assert result == pytest.approx(0.0, abs=1e-15)

    def test_nmse_one_means_as_bad_as_mean_model(self) -> None:
        """nmse = 1.0 when MSE equals variance (mean-predictor baseline)."""
        var = 3.14
        result = nmse(mse=var, target_var=var)
        assert result == pytest.approx(1.0, abs=1e-12)


# ===========================================================================
# Edge cases and error handling (~25% negative tests)
# ===========================================================================


class TestAICEdgeCases:
    """Edge cases for aic()."""

    def test_mse_zero_returns_inf_or_neg_inf(self) -> None:
        """MSE = 0 is a perfect fit. log(0) is undefined.

        Current evaluator returns -inf for mse <= 1e-15.
        Implementation should return -inf (perfect fit is best possible).
        """
        result = aic(mse=0.0, k=2, n=100)
        assert result == float("-inf")

    def test_mse_very_small_returns_neg_inf(self) -> None:
        """MSE below machine epsilon should be treated as perfect fit."""
        result = aic(mse=1e-16, k=2, n=100)
        assert result == float("-inf")

    def test_mse_negative_returns_inf(self) -> None:
        """Negative MSE is invalid; should return +inf (worst score)."""
        result = aic(mse=-1.0, k=2, n=100)
        assert result == float("inf")

    def test_mse_nan_returns_inf(self) -> None:
        """NaN MSE is invalid; should return +inf."""
        result = aic(mse=float("nan"), k=2, n=100)
        assert result == float("inf")

    def test_mse_inf_returns_inf(self) -> None:
        """Infinite MSE is invalid; should return +inf."""
        result = aic(mse=float("inf"), k=2, n=100)
        assert result == float("inf")

    def test_mse_neg_inf_returns_inf(self) -> None:
        """-Inf MSE is invalid; should return +inf."""
        result = aic(mse=float("-inf"), k=2, n=100)
        assert result == float("inf")


class TestAICNoNEdgeCases:
    """Edge cases for aic_no_n()."""

    def test_mse_zero_returns_inf(self) -> None:
        """SGA convention: mse <= 0 returns +inf (invalid)."""
        result = aic_no_n(mse=0.0, k=2)
        assert result == float("inf")

    def test_mse_negative_returns_inf(self) -> None:
        result = aic_no_n(mse=-0.5, k=1)
        assert result == float("inf")

    def test_mse_nan_returns_inf(self) -> None:
        result = aic_no_n(mse=float("nan"), k=1)
        assert result == float("inf")

    def test_mse_inf_returns_inf(self) -> None:
        result = aic_no_n(mse=float("inf"), k=1)
        assert result == float("inf")


class TestAICcEdgeCases:
    """Edge cases for aicc()."""

    def test_n_equals_k_plus_one_returns_inf(self) -> None:
        """When n = k + 1, denominator (n - k - 1) = 0 => division by zero.

        Should return +inf (correction blows up, model is overparameterized).
        """
        result = aicc(mse=1.0, k=5, n=6)
        assert result == float("inf")

    def test_n_less_than_k_plus_one_returns_inf(self) -> None:
        """When n < k + 1, denominator is negative => degenerate.

        Should return +inf (model has more params than data points).
        """
        result = aicc(mse=1.0, k=10, n=5)
        assert result == float("inf")

    def test_n_equals_k_plus_two(self) -> None:
        """n = k + 2 => denominator = 1. Large but finite correction."""
        k = 3
        n = k + 2 # n = 5, denominator = 1
        result = aicc(mse=1.0, k=k, n=n)
        # Should be finite (correction = 2*3*4/1 = 24)
        assert math.isfinite(result)

    def test_mse_zero_returns_neg_inf_or_inf(self) -> None:
        """MSE = 0 with valid n: should follow aic convention (-inf for perfect fit)
        unless n = k+1 makes it degenerate (inf)."""
        # Valid n >> k
        result = aicc(mse=0.0, k=2, n=100)
        assert result == float("-inf")

    def test_mse_nan_returns_inf(self) -> None:
        result = aicc(mse=float("nan"), k=2, n=100)
        assert result == float("inf")


class TestBICEdgeCases:
    """Edge cases for bic()."""

    def test_mse_zero_returns_neg_inf(self) -> None:
        result = bic(mse=0.0, k=2, n=100)
        assert result == float("-inf")

    def test_mse_nan_returns_inf(self) -> None:
        result = bic(mse=float("nan"), k=2, n=100)
        assert result == float("inf")

    def test_mse_inf_returns_inf(self) -> None:
        result = bic(mse=float("inf"), k=2, n=100)
        assert result == float("inf")

    def test_mse_negative_returns_inf(self) -> None:
        result = bic(mse=-0.5, k=2, n=100)
        assert result == float("inf")


class TestNMSEEdgeCases:
    """Edge cases for nmse()."""

    def test_target_var_zero_returns_mse(self) -> None:
        """When target variance is zero (constant target), fallback to raw MSE."""
        result = nmse(mse=0.5, target_var=0.0)
        assert result == pytest.approx(0.5, abs=1e-12)

    def test_target_var_below_eps_returns_mse(self) -> None:
        """Tiny variance below eps should also fallback to raw MSE."""
        result = nmse(mse=0.3, target_var=1e-20, eps=1e-15)
        assert result == pytest.approx(0.3, abs=1e-12)

    def test_target_var_negative_returns_mse(self) -> None:
        """Negative variance is degenerate; fallback to raw MSE."""
        result = nmse(mse=0.5, target_var=-1.0)
        assert result == pytest.approx(0.5, abs=1e-12)

    def test_mse_zero_target_var_positive(self) -> None:
        """Perfect fit: nmse = 0 regardless of variance."""
        result = nmse(mse=0.0, target_var=5.0)
        assert result == pytest.approx(0.0, abs=1e-15)


# ===========================================================================
# Ranking equivalence: aic and aic_no_n produce the same ordering
# ===========================================================================


class TestRankingEquivalence:
    """Relationship between aic and aic_no_n.

    aic(mse, k, n) = n*log(MSE) + 2k
    aic_no_n(mse, k, ratio=1.0) = 2k + 2*log(MSE)

    Key insight: aic_no_n is exactly aic with n=2. So they produce
    identical rankings when n=2, but CAN diverge for other n values
    because the MSE-vs-complexity tradeoff weight changes.

    The claim "排序等价" is about same-formula comparisons:
    within aic, n is a constant that doesn't change relative ordering
    (all candidates share same dataset → same n). It does NOT mean
    aic and aic_no_n always agree on ranking.
    """

    def test_aic_no_n_equals_aic_at_n_2(self) -> None:
        """aic_no_n(mse, k, ratio=1.0) == aic(mse, k, n=2) exactly."""
        cases = [
            (0.01, 1),
            (0.001, 5),
            (1.0, 0),
            (0.5, 3),
            (100.0, 10),
        ]
        for mse_val, k_val in cases:
            assert aic(mse=mse_val, k=k_val, n=2) == pytest.approx(
                aic_no_n(mse=mse_val, k=k_val, ratio=1.0)
            ), f"mse={mse_val}, k={k_val}"

    def test_ranking_can_diverge_for_large_n(self) -> None:
        """When n >> 2, aic weights MSE more heavily → ranking can reverse."""
        # Candidate A: lower MSE, higher k
        # Candidate B: higher MSE, lower k
        mse_a, k_a = 0.01, 5
        mse_b, k_b = 0.1, 1

        # With n=1000: aic heavily weights MSE → A wins (lower MSE)
        n_large = 1000
        assert aic(mse=mse_a, k=k_a, n=n_large) < aic(mse=mse_b, k=k_b, n=n_large)

        # With aic_no_n (n=2 effective): complexity matters more → B wins
        assert aic_no_n(mse=mse_a, k=k_a) > aic_no_n(mse=mse_b, k=k_b)

    @given(
        mse_a=positive_mse,
        mse_b=positive_mse,
        k_a=st.integers(min_value=0, max_value=20),
        k_b=st.integers(min_value=0, max_value=20),
    )
    @settings(max_examples=300)
    def test_aic_no_n_is_aic_at_n_2_property(
        self, mse_a: float, mse_b: float, k_a: int, k_b: int
    ) -> None:
        """Property: aic(n=2) and aic_no_n(ratio=1) always agree on ordering."""
        score_a_full = aic(mse=mse_a, k=k_a, n=2)
        score_b_full = aic(mse=mse_b, k=k_b, n=2)
        score_a_no_n = aic_no_n(mse=mse_a, k=k_a, ratio=1.0)
        score_b_no_n = aic_no_n(mse=mse_b, k=k_b, ratio=1.0)

        # Skip ties
        assume(score_a_full != score_b_full)
        assume(score_a_no_n != score_b_no_n)

        full_a_wins = score_a_full < score_b_full
        no_n_a_wins = score_a_no_n < score_b_no_n
        assert full_a_wins == no_n_a_wins


# ===========================================================================
# Scorer factory tests
# ===========================================================================


class TestMakeAICScorer:
    """Test make_aic_scorer factory."""

    def test_returns_callable(self) -> None:
        scorer = make_aic_scorer(n=100)
        assert callable(scorer)

    def test_signature_matches_scorer_fn(self) -> None:
        """Scorer accepts (mse: float, k: int) and returns float."""
        scorer = make_aic_scorer(n=100)
        result = scorer(1.0, 3)
        assert isinstance(result, float)

    def test_binds_n_correctly(self) -> None:
        """Scorer(mse, k) should equal aic(mse, k, n) for the bound n."""
        n = 250
        scorer = make_aic_scorer(n=n)
        mse, k = 0.3, 4
        assert scorer(mse, k) == pytest.approx(aic(mse, k, n), abs=1e-12)

    def test_different_n_different_result(self) -> None:
        """Different n values produce different absolute scores."""
        scorer_100 = make_aic_scorer(n=100)
        scorer_200 = make_aic_scorer(n=200)
        mse, k = 0.5, 3
        # n * log(0.5) is negative; doubling n doubles the log term
        assert scorer_100(mse, k) != scorer_200(mse, k)

    def test_edge_case_forwarded(self) -> None:
        """Factory scorer should handle edge cases like the raw function."""
        scorer = make_aic_scorer(n=100)
        assert scorer(0.0, 2) == float("-inf")
        assert scorer(float("nan"), 2) == float("inf")


class TestMakeSGAScorer:
    """Test make_sga_scorer factory."""

    def test_returns_callable(self) -> None:
        scorer = make_sga_scorer(ratio=1.0)
        assert callable(scorer)

    def test_binds_ratio_correctly(self) -> None:
        """Scorer(mse, k) should equal aic_no_n(mse, k, ratio) for bound ratio."""
        ratio = 1.5
        scorer = make_sga_scorer(ratio=ratio)
        mse, k = 0.4, 6
        assert scorer(mse, k) == pytest.approx(aic_no_n(mse, k, ratio), abs=1e-12)

    def test_default_ratio_one(self) -> None:
        """Default ratio should be 1.0."""
        scorer = make_sga_scorer()
        result = scorer(1.0, 3)
        expected = aic_no_n(1.0, 3, 1.0)
        assert result == pytest.approx(expected, abs=1e-12)

    def test_different_ratio_different_result(self) -> None:
        """Different ratios scale the penalty differently."""
        scorer_1 = make_sga_scorer(ratio=1.0)
        scorer_2 = make_sga_scorer(ratio=2.0)
        # k > 0 so the penalty term differs
        assert scorer_1(0.5, 5) != scorer_2(0.5, 5)

    def test_edge_case_forwarded(self) -> None:
        """Factory scorer should handle edge cases like the raw function."""
        scorer = make_sga_scorer(ratio=1.0)
        assert scorer(0.0, 2) == float("inf")
        assert scorer(float("nan"), 2) == float("inf")


class TestMakeBICScorer:
    """Test make_bic_scorer factory."""

    def test_returns_callable(self) -> None:
        scorer = make_bic_scorer(n=100)
        assert callable(scorer)

    def test_binds_n_correctly(self) -> None:
        """Scorer(mse, k) should equal bic(mse, k, n) for the bound n."""
        n = 300
        scorer = make_bic_scorer(n=n)
        mse, k = 0.2, 3
        assert scorer(mse, k) == pytest.approx(bic(mse, k, n), abs=1e-12)

    def test_edge_case_forwarded(self) -> None:
        """Factory scorer should handle edge cases like the raw function."""
        scorer = make_bic_scorer(n=100)
        assert scorer(0.0, 2) == float("-inf")
        assert scorer(float("nan"), 2) == float("inf")


# ===========================================================================
# Property-based tests (hypothesis)
# ===========================================================================


@pytest.mark.numerical
class TestAICProperties:
    """Mathematical properties of aic()."""

    @given(mse=positive_mse, k=complexity_k, n=sample_n)
    @settings(max_examples=500)
    def test_finite_output(self, mse: float, k: int, n: int) -> None:
        """Positive MSE + finite params => finite output."""
        result = aic(mse=mse, k=k, n=n)
        assert math.isfinite(result)

    @given(mse=positive_mse, n=sample_n)
    @settings(max_examples=200)
    def test_monotonic_in_k(self, mse: float, n: int) -> None:
        """More complex models have higher (worse) AIC, all else equal."""
        scores = [aic(mse=mse, k=k, n=n) for k in range(10)]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    @given(k=complexity_k, n=sample_n)
    @settings(max_examples=200)
    def test_monotonic_in_mse(self, k: int, n: int) -> None:
        """Higher MSE => higher (worse) AIC, all else equal."""
        mse_values = [0.01, 0.1, 0.5, 1.0, 5.0, 100.0]
        scores = [aic(mse=m, k=k, n=n) for m in mse_values]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    @given(mse=positive_mse, k=complexity_k, n=sample_n)
    @settings(max_examples=200)
    def test_aic_decomposition(self, mse: float, k: int, n: int) -> None:
        """aic = goodness_of_fit_term + penalty_term."""
        result = aic(mse=mse, k=k, n=n)
        gof = n * math.log(mse)
        penalty = 2.0 * k
        assert result == pytest.approx(gof + penalty, abs=1e-8)


@pytest.mark.numerical
class TestAICNoNProperties:
    """Mathematical properties of aic_no_n()."""

    @given(mse=positive_mse, k=complexity_k, ratio=aic_ratio)
    @settings(max_examples=500)
    def test_finite_output(self, mse: float, k: int, ratio: float) -> None:
        """Positive MSE => finite output."""
        result = aic_no_n(mse=mse, k=k, ratio=ratio)
        assert math.isfinite(result)

    @given(mse=positive_mse, ratio=aic_ratio)
    @settings(max_examples=200)
    def test_monotonic_in_k(self, mse: float, ratio: float) -> None:
        """More complex models have higher (worse) score."""
        scores = [aic_no_n(mse=mse, k=k, ratio=ratio) for k in range(10)]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    @given(k=complexity_k, ratio=aic_ratio)
    @settings(max_examples=200)
    def test_monotonic_in_mse(self, k: int, ratio: float) -> None:
        """Higher MSE => higher (worse) score."""
        mse_values = [0.01, 0.1, 0.5, 1.0, 5.0, 100.0]
        scores = [aic_no_n(mse=m, k=k, ratio=ratio) for m in mse_values]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]


@pytest.mark.numerical
class TestAICcProperties:
    """Mathematical properties of aicc()."""

    @given(
        mse=positive_mse,
        k=st.integers(min_value=0, max_value=10),
        n=st.integers(min_value=20, max_value=5000),
    )
    @settings(max_examples=300)
    def test_aicc_geq_aic(self, mse: float, k: int, n: int) -> None:
        """AICc >= AIC always (correction is non-negative) when n > k + 1."""
        assume(n > k + 1)
        result_aicc = aicc(mse=mse, k=k, n=n)
        result_aic = aic(mse=mse, k=k, n=n)
        assert result_aicc >= result_aic - 1e-10 # small tolerance for float

    @given(mse=positive_mse, k=st.integers(min_value=1, max_value=10))
    @settings(max_examples=200)
    def test_correction_decreases_with_n(self, mse: float, k: int) -> None:
        """As n grows, AICc correction shrinks toward zero."""
        n_values = [k + 5, k + 20, k + 100, k + 1000]
        corrections = []
        for n in n_values:
            diff = aicc(mse=mse, k=k, n=n) - aic(mse=mse, k=k, n=n)
            corrections.append(diff)
        # Corrections should be decreasing
        for i in range(len(corrections) - 1):
            assert corrections[i] > corrections[i + 1]


@pytest.mark.numerical
class TestBICProperties:
    """Mathematical properties of bic()."""

    @given(mse=positive_mse, k=complexity_k, n=sample_n)
    @settings(max_examples=500)
    def test_finite_output(self, mse: float, k: int, n: int) -> None:
        """Positive MSE + finite params => finite output."""
        result = bic(mse=mse, k=k, n=n)
        assert math.isfinite(result)

    @given(mse=positive_mse, n=sample_n)
    @settings(max_examples=200)
    def test_monotonic_in_k(self, mse: float, n: int) -> None:
        """More terms => higher BIC."""
        scores = [bic(mse=mse, k=k, n=n) for k in range(10)]
        for i in range(len(scores) - 1):
            assert scores[i] < scores[i + 1]

    @given(mse=positive_mse, k=st.integers(min_value=1, max_value=20))
    @settings(max_examples=200)
    def test_bic_penalty_scales_with_n(self, mse: float, k: int) -> None:
        """BIC penalty grows with n (via log(n) factor).

        The penalty term k*log(n) grows with n. We verify this directly
        from the formula: bic(n2) - bic(n1) includes k*(log(n2) - log(n1)).
        """
        n_small, n_large = 10, 1000
        score_small = bic(mse=mse, k=k, n=n_small)
        score_large = bic(mse=mse, k=k, n=n_large)
        # Full difference decomposes into GoF scaling + penalty scaling.
        # Verify penalty contribution is positive (k >= 1 by strategy).
        penalty_diff = k * (math.log(n_large) - math.log(n_small))
        assert penalty_diff > 0
        # Verify total diff matches decomposition.
        gof_diff = (n_large - n_small) * math.log(mse)
        expected_diff = gof_diff + penalty_diff
        assert score_large - score_small == pytest.approx(expected_diff, abs=1e-8)


@pytest.mark.numerical
class TestNMSEProperties:
    """Mathematical properties of nmse()."""

    @given(
        mse=st.floats(
            min_value=0.0, max_value=1e8, allow_nan=False, allow_infinity=False
        ),
        var=st.floats(
            min_value=1e-6, max_value=1e8, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=300)
    def test_non_negative(self, mse: float, var: float) -> None:
        """NMSE is non-negative for non-negative MSE."""
        result = nmse(mse=mse, target_var=var)
        assert result >= 0.0

    @given(
        var=st.floats(
            min_value=1e-6, max_value=1e8, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=200)
    def test_monotonic_in_mse(self, var: float) -> None:
        """Higher MSE => higher NMSE."""
        mse_values = [0.0, 0.01, 0.1, 0.5, 1.0, 10.0]
        scores = [nmse(mse=m, target_var=var) for m in mse_values]
        for i in range(len(scores) - 1):
            assert scores[i] <= scores[i + 1]

    @given(
        mse=st.floats(
            min_value=1e-10, max_value=1e4, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(max_examples=200)
    def test_scales_inversely_with_variance(self, mse: float) -> None:
        """Doubling variance halves NMSE."""
        var = 2.0
        result_1 = nmse(mse=mse, target_var=var)
        result_2 = nmse(mse=mse, target_var=2.0 * var)
        assert result_2 == pytest.approx(result_1 / 2.0, rel=1e-10)


# ===========================================================================
# Cross-metric consistency
# ===========================================================================


class TestCrossMetricConsistency:
    """Verify relationships between different metrics."""

    def test_aic_and_bic_agree_on_simpler_better_at_same_mse(self) -> None:
        """Both AIC and BIC prefer simpler models when fit is identical."""
        mse = 0.5
        n = 100
        simple = (aic(mse, k=1, n=n), bic(mse, k=1, n=n))
        complex_ = (aic(mse, k=5, n=n), bic(mse, k=5, n=n))
        # Both should prefer k=1
        assert simple[0] < complex_[0] # AIC prefers simpler
        assert simple[1] < complex_[1] # BIC prefers simpler

    def test_bic_more_conservative_than_aic(self) -> None:
        """BIC penalizes complexity more than AIC for n > e^2.

        Given two models where the complex one fits slightly better,
        BIC is more likely to prefer the simple one.
        """
        n = 100 # log(100) ~ 4.6 > 2
        # Simple model: k=1, slightly worse fit
        mse_simple = 0.12
        k_simple = 1
        # Complex model: k=5, slightly better fit
        mse_complex = 0.10
        k_complex = 5

        # AIC difference (complex - simple)
        aic_diff = aic(mse_complex, k_complex, n) - aic(mse_simple, k_simple, n)
        # BIC difference (complex - simple)
        bic_diff = bic(mse_complex, k_complex, n) - bic(mse_simple, k_simple, n)

        # BIC should penalize the complex model more (larger diff)
        assert bic_diff > aic_diff

    def test_aicc_and_aic_agree_on_best_for_large_n(self) -> None:
        """For large n, AICc and AIC should select the same best model."""
        n = 10000
        candidates = [
            (0.01, 1),
            (0.005, 3),
            (0.001, 8),
            (0.1, 0),
        ]
        aic_scores = [aic(m, k, n) for m, k in candidates]
        aicc_scores = [aicc(m, k, n) for m, k in candidates]

        best_aic = min(range(len(candidates)), key=lambda i: aic_scores[i])
        best_aicc = min(range(len(candidates)), key=lambda i: aicc_scores[i])

        assert best_aic == best_aicc
