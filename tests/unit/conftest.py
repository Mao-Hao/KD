"""Shared test fixtures for unit tests."""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest
import torch

from kd2.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd2.search.protocol import PlatformComponents
from tests.unit.search._runner_mocks import RecordingAlgorithm, StatefulAlgorithm

# ============================================================================
# Runner-related shared fixtures
# ============================================================================


@pytest.fixture
def mock_components() -> PlatformComponents:
    """PlatformComponents with MagicMock fields -- Runner only forwards it.

    ``evaluator.lhs_target`` is set to an empty Tensor by default so the
    Runner's strict ``_actual`` contract (no silent zero-fallback) is
    satisfied for tests that don't explicitly stub it. Tests that need
    a specific shape override this attribute themselves.
    """
    components = PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
    )
    components.evaluator.lhs_target = torch.zeros(0)
    return components


@pytest.fixture
def recording_algorithm() -> RecordingAlgorithm:
    """Fresh RecordingAlgorithm instance."""
    return RecordingAlgorithm()


@pytest.fixture
def stateful_algorithm() -> StatefulAlgorithm:
    """Fresh StatefulAlgorithm instance."""
    return StatefulAlgorithm()


@pytest.fixture
def simple_1d_dataset() -> PDEDataset:
    """1D grid dataset for derivative testing.

    Creates a dataset with:
    - 100 points, x in [0, 2*pi]
    - Field u = sin(x)

    Expected derivatives:
    - u_x = cos(x)
    - u_xx = -sin(x)
    - u_xxx = -cos(x)
    """
    n_points = 100
    x = torch.linspace(0, 2 * math.pi, n_points, dtype=torch.float64)
    u = torch.sin(x)

    return PDEDataset(
        name="test_1d_sin",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="x",
    )


@pytest.fixture
def simple_2d_dataset() -> PDEDataset:
    """2D grid dataset (x, t) for derivative testing.

    Creates a dataset with:
    - x: 64 points in [0, 2*pi]
    - t: 32 points in [0, 1]
    - Field u = sin(x) * exp(-t)

    Expected derivatives:
    - u_x = cos(x) * exp(-t)
    - u_t = -sin(x) * exp(-t)
    - u_xx = -sin(x) * exp(-t)
    """
    n_x = 64
    n_t = 32

    x = torch.linspace(0, 2 * math.pi, n_x, dtype=torch.float64)
    t = torch.linspace(0, 1, n_t, dtype=torch.float64)

    # Create 2D meshgrid: shape (n_x, n_t)
    X, T = torch.meshgrid(x, t, indexing="ij") # noqa: N806 — math convention
    u = torch.sin(X) * torch.exp(-T)

    return PDEDataset(
        name="test_2d_sin_exp",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


@pytest.fixture
def polynomial_1d_dataset() -> PDEDataset:
    """1D grid dataset with polynomial field for exact derivative testing.

    Creates a dataset with:
    - 100 points, x in [0, 1]
    - Field u = x^3

    Expected derivatives (exact):
    - u_x = 3*x^2
    - u_xx = 6*x
    - u_xxx = 6
    """
    n_points = 100
    x = torch.linspace(0, 1, n_points, dtype=torch.float64)
    u = x**3

    return PDEDataset(
        name="test_1d_polynomial",
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="x",
    )


@pytest.fixture
def scattered_dataset() -> PDEDataset:
    """Scattered topology dataset (not Grid).

    This is used to test that FiniteDiffProvider rejects non-Grid data.
    """
    # For scattered, we don't need valid grid axes
    # Just create minimal valid dataset with SCATTERED topology
    x = torch.linspace(0, 1, 10, dtype=torch.float64)
    u = torch.sin(x)

    return PDEDataset(
        name="test_scattered",
        task_type=TaskType.PDE,
        topology=DataTopology.SCATTERED,
        axes={"x": AxisInfo(name="x", values=x)},
        axis_order=["x"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="x",
    )
