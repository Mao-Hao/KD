"""Tests for uniform-grid predicate consistency between FD and integrator.

TDD red phase (Medium #4): The two grid-uniformity checks use mathematically
inequivalent predicates:

- ``finite_diff._check_uniform_grid``:
  ``abs(max(diffs) - min(diffs)) > abs(dx0) * UNIFORM_GRID_RTOL``
  (range / max-min span)

- ``integrator._check_spatial_uniformity``:
  ``not np.allclose(diffs, dx0, rtol=UNIFORM_GRID_RTOL, atol=0.0)``
  i.e. ``max(abs(diffs - dx0)) > UNIFORM_GRID_RTOL * abs(dx0)``
  (max abs deviation from dx0)

These predicates are mathematically NOT equivalent. With a symmetric
drift around dx0, ``max - min`` can reach ``2 * max(abs(d - dx0))`` so
the FD predicate may reject grids that the integrator predicate
accepts. This causes confusing user-facing inconsistency: the same
dataset that builds a ``FiniteDiffProvider`` cannot integrate, or vice
versa.

Resolution path: extract a shared ``is_uniform_grid(coords, rtol)``
helper and route both call sites through it. The tests here lock the
contract that the two predicates agree on the same input.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from kd2.data.derivatives.finite_diff import (
    UNIFORM_GRID_RTOL,
    _check_uniform_grid,
)
from kd2.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)


def _fd_accepts(coords: torch.Tensor) -> bool:
    """Run the FD-side predicate; return True if accepted, False if rejected."""
    try:
        _check_uniform_grid(coords, "x")
        return True
    except ValueError:
        return False


def _integrator_accepts(coords: torch.Tensor) -> bool:
    """Run the integrator-side predicate via _check_spatial_uniformity."""
    from kd2.core.integrator import _check_spatial_uniformity

    # Build a 1D-spatial dataset with the test coords as the spatial axis.
    # _check_spatial_uniformity returns None on accept, a warning string
    # on reject.
    nx = coords.shape[0]
    nt = 4
    t = torch.linspace(0.0, 1.0, nt, dtype=coords.dtype)
    u = torch.zeros(nx, nt, dtype=coords.dtype)
    dataset = PDEDataset(
        name="uniformity_consistency_check",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=coords),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )
    warning = _check_spatial_uniformity(dataset, ["x"])
    return warning is None


# ============================================================================
# Test cases — pairs of inputs that should produce identical decisions
# ============================================================================


@pytest.mark.unit
class TestPredicateAgreement:
    """Both predicates must agree on every reasonable input grid.

    Each test constructs a borderline grid that exercises the disagreement
    region: drifts comparable to ``rtol * |dx0|``. The assertion compares
    accept/reject decisions between the two implementations.
    """

    def test_uniform_linspace_both_accept(self) -> None:
        """Plain ``linspace`` is uniform and must be accepted by both."""
        coords = torch.linspace(0.0, 1.0, 100, dtype=torch.float64)
        assert _fd_accepts(coords) == _integrator_accepts(coords)
        assert _fd_accepts(coords), "uniform linspace should be accepted"

    def test_geometric_spacing_both_reject(self) -> None:
        """Geometric spacing diverges by O(1) and must be rejected by both."""
        coords = torch.tensor([2.0**i for i in range(10)], dtype=torch.float64)
        assert _fd_accepts(coords) == _integrator_accepts(coords)
        assert not _fd_accepts(coords), "geometric grid should be rejected"

    def test_log_spacing_both_reject(self) -> None:
        """logspace is rejected by both."""
        coords = torch.logspace(0.0, 2.0, 50, dtype=torch.float64)
        assert _fd_accepts(coords) == _integrator_accepts(coords)
        assert not _fd_accepts(coords), "log-spaced grid should be rejected"

    def test_symmetric_drift_predicates_agree(self) -> None:
        """Symmetric drift around dx0 — the canonical disagreement case.

        Construct a grid whose successive diffs are
        ``[dx0, dx0+δ, dx0-δ, dx0]`` with
        ``rtol * |dx0| < 2δ <= 2 * rtol * |dx0|``.

        - FD predicate sees ``max-min = 2δ > rtol*|dx0|`` → REJECT.
        - Integrator predicate sees ``max(|d-dx0|) = δ ≤ rtol*|dx0|`` → ACCEPT.

        Today this disagrees. Both should give the same decision.
        """
        rtol = UNIFORM_GRID_RTOL
        dx0 = 0.1
        # Pick δ at 0.8 * rtol*|dx0| → δ = 8e-6 (since rtol*|dx0| = 1e-5)
        delta = 0.8 * rtol * abs(dx0)
        # Build coords so diffs come out [dx0, dx0+δ, dx0-δ, dx0]
        coords = torch.tensor(
            [
                0.0,
                dx0,
                dx0 + (dx0 + delta),
                dx0 + (dx0 + delta) + (dx0 - delta),
                dx0 + (dx0 + delta) + (dx0 - delta) + dx0,
            ],
            dtype=torch.float64,
        )

        # Sanity: confirm the constructed diffs fall in the disagreement zone.
        diffs = (coords[1:] - coords[:-1]).numpy()
        max_minus_min = float(diffs.max() - diffs.min())
        max_abs_dev = float(np.max(np.abs(diffs - diffs[0])))
        assert max_minus_min > rtol * abs(dx0), (
            f"setup error: max-min={max_minus_min} should exceed {rtol * abs(dx0)}"
        )
        assert max_abs_dev <= rtol * abs(dx0), (
            f"setup error: max-abs-dev={max_abs_dev} should be <= {rtol * abs(dx0)}"
        )

        # CONTRACT: predicates must agree on the same input.
        assert _fd_accepts(coords) == _integrator_accepts(coords), (
            f"FD and integrator predicates disagree on the same grid. "
            f"FD accepts: {_fd_accepts(coords)}, "
            f"integrator accepts: {_integrator_accepts(coords)}. "
            f"Both should use the same uniform-grid predicate."
        )

    def test_one_sided_drift_predicates_agree(self) -> None:
        """One-sided drift: all diffs >= dx0. Both predicates degenerate
        to ``max-dx0 > rtol*|dx0|``, so they should already agree here —
        but we lock the property to prevent regressions.
        """
        rtol = UNIFORM_GRID_RTOL
        dx0 = 0.1
        # All diffs slightly above dx0 by 0.5 * rtol*|dx0|.
        delta = 0.5 * rtol * abs(dx0)
        diffs = [dx0, dx0 + delta, dx0 + delta, dx0]
        coord_list = [0.0]
        for d in diffs:
            coord_list.append(coord_list[-1] + d)
        coords = torch.tensor(coord_list, dtype=torch.float64)
        assert _fd_accepts(coords) == _integrator_accepts(coords)

    @pytest.mark.parametrize("scale", [1.0, 1e-6, 1e6])
    def test_scale_invariance(self, scale: float) -> None:
        """Changing the grid scale must not flip either predicate's decision.

        Both predicates use ``rtol * |dx0|`` so scaling all coords keeps
        the relative drift fixed — and decisions must not depend on
        absolute magnitude.
        """
        coords = torch.linspace(0.0, 1.0, 100, dtype=torch.float64) * scale
        assert _fd_accepts(coords) == _integrator_accepts(coords)


@pytest.mark.unit
class TestSharedHelperContract:
    """When the disagreement is fixed, ``is_uniform_grid(coords, rtol)``
    or an equivalent shared helper should exist and both call sites
    should use it.

    Today these tests fail because the helper does not exist; the
    failing import documents the desired refactor.
    """

    def test_shared_helper_exists(self) -> None:
        """A public-ish helper must be importable from one canonical
        module so both finite_diff and integrator can call it.

        Import is the contract: the test fails until the helper is
        extracted. After extraction the two existing predicate
        implementations should delegate to the helper.
        """
        try:
            from kd2.data.derivatives.finite_diff import ( # noqa: F401
                is_uniform_grid,
            )

            helper_exists = True
        except ImportError:
            helper_exists = False

        assert helper_exists, (
            "Expected ``is_uniform_grid`` helper to be exported from "
            "``kd2.data.derivatives.finite_diff`` so both "
            "``_check_uniform_grid`` and ``integrator._check_spatial_uniformity``"
            " can share a single predicate. Until the helper exists the two "
            "predicates remain mathematically inequivalent."
        )

    def test_shared_helper_returns_bool(self) -> None:
        """Helper signature: ``is_uniform_grid(coords, rtol) -> bool``.

        - True when the coordinate vector is uniformly spaced.
        - False when it is not (the helper must NOT raise; raising
          should be left to the call site).
        """
        from kd2.data.derivatives.finite_diff import ( # type: ignore[attr-defined]
            is_uniform_grid,
        )

        uniform = torch.linspace(0.0, 1.0, 50, dtype=torch.float64)
        nonuniform = torch.tensor([0.0, 0.1, 0.21, 0.33, 0.46], dtype=torch.float64)
        assert is_uniform_grid(uniform, rtol=UNIFORM_GRID_RTOL) is True
        assert is_uniform_grid(nonuniform, rtol=UNIFORM_GRID_RTOL) is False


# ============================================================================
# Negative / failure-injection tests (>= 20% of file)
# ============================================================================


@pytest.mark.unit
class TestPredicateNegativeCases:
    """Edge inputs that should be rejected by both predicates."""

    def test_constant_coordinate_axis_both_reject(self) -> None:
        """All-zero coord vector → degenerate spacing. Both must reject."""
        coords = torch.zeros(10, dtype=torch.float64)
        # FD path raises with 'degenerate' message.
        with pytest.raises(ValueError, match="degenerate"):
            _check_uniform_grid(coords, "x")
        # Integrator path also rejects.
        assert not _integrator_accepts(coords)

    def test_subnormal_dx_both_reject(self) -> None:
        """dx below DX_ZERO_FLOOR=1e-30 → both must reject."""
        coords = torch.linspace(0.0, 1e-31, 10, dtype=torch.float64)
        with pytest.raises(ValueError, match="degenerate"):
            _check_uniform_grid(coords, "x")
        assert not _integrator_accepts(coords)

    def test_two_point_grid_predicates_agree(self) -> None:
        """Two-point grids are degenerate-uniform (single dx). Both must
        produce the same decision (likely accept — there's only one diff).
        """
        coords = torch.tensor([0.0, 0.1], dtype=torch.float64)
        # FD requires len >= 2 and accepts; integrator skips len <= 2.
        # Either way, they must agree on this input.
        try:
            _check_uniform_grid(coords, "x")
            fd = True
        except ValueError:
            fd = False
        intg = _integrator_accepts(coords)
        assert fd == intg, f"Two-point grid disagreement: FD={fd}, integrator={intg}"

    @pytest.mark.parametrize(
        "coords_list",
        [
            [0.0, 0.1, 0.21, 0.33, 0.46], # quadratic spacing
            [0.0, math.pi, 6.28, 9.42], # accumulated rounding (uniform)
        ],
    )
    def test_predicates_agree_on_parametric_inputs(
        self, coords_list: list[float]
    ) -> None:
        """Property: predicate decisions match across a small input set.

        Parametrising over different drift patterns ensures the property
        holds beyond the single hand-crafted symmetric-drift case.
        """
        coords = torch.tensor(coords_list, dtype=torch.float64)
        assert _fd_accepts(coords) == _integrator_accepts(coords), (
            f"Predicates disagree on {coords_list}: "
            f"FD={_fd_accepts(coords)}, integrator={_integrator_accepts(coords)}"
        )


# ============================================================================
# T1: ``inf`` spacing must be rejected (np.diff over near-MAX_DOUBLE produces
# inf, which np.allclose blindly accepts as "uniform" pre-fix).
# ============================================================================


@pytest.mark.unit
class TestUniformGridRejectsInfiniteSpacing:
    """``np.diff([-MAX_DOUBLE, MAX_DOUBLE])`` is ``[inf]`` (overflow); the
    pre-fix predicate then checked ``np.allclose([inf], inf)`` which returns
    True, falsely accepting a degenerate grid. Both helper paths must reject.
    """

    def test_is_uniform_grid_rejects_inf_dx0(self) -> None:
        """``is_uniform_grid`` returns False on inf-overflow spacing."""
        from kd2.data.derivatives.finite_diff import is_uniform_grid

        coords = np.array([-1.7e308, 1.7e308], dtype=np.float64)
        assert is_uniform_grid(coords) is False, (
            "is_uniform_grid must reject coords whose diff overflows to inf; "
            "np.allclose([inf], inf) returns True, so the helper needs an "
            "explicit np.isfinite guard."
        )

    def test_check_uniform_grid_raises_on_inf_dx0(self) -> None:
        """``_check_uniform_grid`` must raise (not silently return inf) on
        inf-overflow spacing. Without the guard the function returns ``inf``
        as the FD step, which then divides downstream tensors by inf and
        silently produces zeros instead of derivatives.
        """
        coords = torch.tensor([-1.7e308, 1.7e308], dtype=torch.float64)
        with pytest.raises(ValueError):
            _check_uniform_grid(coords, "x")


# ============================================================================
# T4: descending coordinates must be rejected (FD stencils silently flip
# the sign of derivatives when dx is negative).
# ============================================================================


@pytest.mark.unit
class TestUniformGridRejectsDescendingAxes:
    """A descending grid (``[0.4, 0.3, 0.2, 0.1, 0]``) has constant dx=-0.1.
    The pre-fix predicate uses ``abs(dx0)`` and returns True, so descending
    grids slip past the FD provider. The stencils then divide by negative dx
    and produce sign-reversed derivatives without warning.

    PDEDataset BYOD safeguards already enforce monotonic increasing axes,
    so the helper rejecting descending coords keeps stencil contracts valid
    (the caller must flip).
    """

    def test_is_uniform_grid_rejects_descending(self) -> None:
        """``is_uniform_grid`` returns False on descending coords."""
        from kd2.data.derivatives.finite_diff import is_uniform_grid

        coords = np.array([0.4, 0.3, 0.2, 0.1, 0.0], dtype=np.float64)
        assert is_uniform_grid(coords) is False, (
            "is_uniform_grid must reject descending coords (dx0 < 0). "
            "FD stencils require positive dx; the existing abs(dx0) check "
            "incorrectly accepts negative spacing."
        )

    def test_check_uniform_grid_raises_on_descending(self) -> None:
        """``_check_uniform_grid`` raises ValueError on descending coords."""
        coords = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0], dtype=torch.float64)
        with pytest.raises(ValueError):
            _check_uniform_grid(coords, "x")
