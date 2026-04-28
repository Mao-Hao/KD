"""Tests for ``PDEDataset.from_arrays`` factory."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import kd2
from kd2.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)

# Helpers


def _heat_field(x: torch.Tensor, t: torch.Tensor, alpha: float = 0.1) -> torch.Tensor:
    """Analytic heat-equation solution u(x, t) = sin(x) * exp(-alpha * t)."""
    xx, tt = torch.meshgrid(x, t, indexing="ij")
    return torch.exp(-alpha * tt) * torch.sin(xx)


# 1. Basic 1D heat equation


def test_from_arrays_basic() -> None:
    """1D heat equation: validates dataset is constructed correctly."""
    nx, nt = 32, 16
    x = torch.linspace(0.0, 2 * math.pi, nx + 1, dtype=torch.float64)[:-1]
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = _heat_field(x, t)

    ds = PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"u": u},
        lhs="u_t",
        periodic={"x"},
        name="heat",
    )

    assert isinstance(ds, PDEDataset)
    assert ds.name == "heat"
    assert ds.task_type is TaskType.PDE
    assert ds.topology is DataTopology.GRID
    assert ds.axis_order == ["x", "t"]
    assert ds.lhs_field == "u"
    assert ds.lhs_axis == "t"
    assert ds.axes is not None and ds.fields is not None
    assert ds.axes["x"].is_periodic is True
    assert ds.axes["t"].is_periodic is False
    assert ds.fields["u"].values.shape == (nx, nt)


# 2. 2D field with 2 spatial + 1 time axis


def test_from_arrays_2d() -> None:
    """3D dataset: 2 spatial + 1 temporal axis, axis_order from insertion order."""
    nx, ny, nt = 8, 6, 4
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    y = torch.linspace(0.0, 1.0, ny, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.randn(nx, ny, nt, dtype=torch.float64)

    ds = PDEDataset.from_arrays(
        coords={"x": x, "y": y, "t": t},
        fields={"u": u},
        lhs="u_t",
    )

    assert ds.axis_order == ["x", "y", "t"]
    assert ds.fields is not None
    assert ds.fields["u"].values.shape == (nx, ny, nt)
    assert ds.lhs_axis == "t"


# 3. NumPy input gets converted to torch


def test_from_arrays_numpy_input() -> None:
    """Numpy ndarray inputs are converted to torch tensors with the requested dtype."""
    nx, nt = 12, 8
    x_np = np.linspace(0.0, 1.0, nx) # float64 by default
    t_np = np.linspace(0.0, 1.0, nt)
    u_np = np.random.RandomState(0).randn(nx, nt)

    ds = PDEDataset.from_arrays(
        coords={"x": x_np, "t": t_np},
        fields={"u": u_np},
        lhs="u_t",
    )

    assert ds.axes is not None
    assert isinstance(ds.axes["x"].values, torch.Tensor)
    assert ds.axes["x"].values.dtype == torch.float64
    assert ds.fields is not None
    assert isinstance(ds.fields["u"].values, torch.Tensor)
    assert ds.fields["u"].values.dtype == torch.float64


# 4. Malformed lhs raises with axes available


def test_from_arrays_lhs_unparseable() -> None:
    """``lhs`` without an underscore raises ValueError mentioning available axes."""
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    u = torch.randn(8, 6, dtype=torch.float64)

    with pytest.raises(ValueError) as exc_info:
        PDEDataset.from_arrays(
            coords={"x": x, "t": t},
            fields={"u": u},
            lhs="notanaxis",
        )
    msg = str(exc_info.value)
    assert "notanaxis" in msg
    # Must surface available axes/fields so the user can fix the spec
    assert "x" in msg and "t" in msg


# 5. Field referenced in lhs is missing from fields dict


def test_from_arrays_lhs_field_missing() -> None:
    """lhs='v_t' but only 'u' present in fields -> ValueError listing fields."""
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    u = torch.randn(8, 6, dtype=torch.float64)

    with pytest.raises(ValueError) as exc_info:
        PDEDataset.from_arrays(
            coords={"x": x, "t": t},
            fields={"u": u},
            lhs="v_t",
        )
    msg = str(exc_info.value)
    assert "v" in msg
    # The available field 'u' must be mentioned for guidance
    assert "u" in msg


# 6. periodic={"x"} -> is_periodic=True on the axis


def test_from_arrays_periodic() -> None:
    """``periodic`` set marks the matching axis as periodic."""
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    u = torch.randn(8, 6, dtype=torch.float64)

    ds = PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"u": u},
        lhs="u_t",
        periodic={"x"},
    )
    assert ds.axes is not None
    assert ds.axes["x"].is_periodic is True
    assert ds.axes["t"].is_periodic is False


# 7. dtype cast: float32 -> float64 by default


def test_from_arrays_dtype_cast() -> None:
    """A float32 input array is cast to the default (float64)."""
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float32)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float32)
    u = torch.randn(8, 6, dtype=torch.float32)

    ds = PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"u": u},
        lhs="u_t",
    )
    assert ds.axes is not None and ds.fields is not None
    assert ds.axes["x"].values.dtype == torch.float64
    assert ds.fields["u"].values.dtype == torch.float64

    # Also verify explicit float32 keeps float32.
    ds32 = PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"u": u},
        lhs="u_t",
        dtype=torch.float32,
    )
    assert ds32.axes is not None and ds32.fields is not None
    assert ds32.axes["x"].values.dtype == torch.float32
    assert ds32.fields["u"].values.dtype == torch.float32


# 8. Field shape mismatch raises with both shapes named


def test_from_arrays_shape_mismatch() -> None:
    """Field shape that doesn't match coords -> ValueError naming both."""
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    u_wrong = torch.randn(8, 7, dtype=torch.float64) # nt should be 6

    with pytest.raises(ValueError) as exc_info:
        PDEDataset.from_arrays(
            coords={"x": x, "t": t},
            fields={"u": u_wrong},
            lhs="u_t",
        )
    msg = str(exc_info.value)
    # The enriched message must mention coord lengths AND the field shape
    # so the user can spot the off-by-one immediately.
    assert "u" in msg
    assert "(8, 7)" in msg or "8, 7" in msg
    assert "x" in msg and "t" in msg


# 9. Coordinate monotonicity (BYOD safeguard)


def test_from_arrays_rejects_non_monotonic_coord() -> None:
    """Non-monotonic coordinate values must raise, not silently produce
    garbage finite-difference derivatives downstream."""
    x_bad = torch.tensor([0.0, 0.5, 0.1, 0.6, 1.0], dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    u = torch.randn(5, 5, dtype=torch.float64)

    with pytest.raises(ValueError, match="strictly increasing"):
        PDEDataset.from_arrays(
            coords={"x": x_bad, "t": t},
            fields={"u": u},
            lhs="u_t",
        )


def test_from_arrays_rejects_descending_coord() -> None:
    """Descending coordinates must raise; downstream FD requires positive dx."""
    x_desc = torch.linspace(1.0, 0.0, 8, dtype=torch.float64) # 1 -> 0
    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    u = torch.randn(8, 5, dtype=torch.float64)

    with pytest.raises(ValueError, match="strictly increasing"):
        PDEDataset.from_arrays(
            coords={"x": x_desc, "t": t},
            fields={"u": u},
            lhs="u_t",
        )


def test_from_arrays_rejects_duplicate_coord() -> None:
    """Repeated coordinate values violate strict monotonicity (dx=0)."""
    x_dup = torch.tensor([0.0, 0.25, 0.25, 0.5, 0.75], dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    u = torch.randn(5, 5, dtype=torch.float64)

    with pytest.raises(ValueError, match="strictly increasing"):
        PDEDataset.from_arrays(
            coords={"x": x_dup, "t": t},
            fields={"u": u},
            lhs="u_t",
        )


def test_from_arrays_accepts_single_point_axis() -> None:
    """Single-element coord axis can't be checked for monotonicity; allow it."""
    x = torch.tensor([0.5], dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    u = torch.randn(1, 5, dtype=torch.float64)

    # Should not raise — single-element axis has nothing to compare.
    ds = PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"u": u},
        lhs="u_t",
    )
    assert ds.axes is not None
    assert ds.axes["x"].values.numel() == 1


# 10. Compatibility with kd2.Model.fit


def test_from_arrays_compatible_with_model() -> None:
    """The factory output drops cleanly into ``kd2.Model.fit``."""
    nx, nt = 24, 12
    x = torch.linspace(0.0, 2 * math.pi, nx + 1, dtype=torch.float64)[:-1]
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = _heat_field(x, t)

    ds = PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"u": u},
        lhs="u_t",
        periodic={"x"},
        name="heat_for_fit",
    )

    model = kd2.Model(
        algorithm="sga",
        generations=2,
        population=3,
        depth=3,
        width=3,
        seed=0,
        verbose=False,
    )
    model.fit(ds)
    assert isinstance(model.best_expr_, str)
    assert model.best_expr_ # non-empty


# 10. Multi-field dataset


def test_from_arrays_multifield() -> None:
    """Two fields registered, lhs uses one of them, both reach the dataset."""
    nx, nt = 12, 8
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.randn(nx, nt, dtype=torch.float64)
    v = torch.randn(nx, nt, dtype=torch.float64)

    ds = PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"u": u, "v": v},
        lhs="v_t",
    )
    assert ds.fields is not None
    assert set(ds.fields.keys()) == {"u", "v"}
    assert ds.lhs_field == "v"
    assert ds.lhs_axis == "t"


# 11. lhs with multi-underscore field name parses on LAST underscore


def test_from_arrays_lhs_multi_underscore_field() -> None:
    """``lhs='my_field_t'`` parses to field='my_field', axis='t'."""
    nx, nt = 6, 4
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    f = torch.randn(nx, nt, dtype=torch.float64)

    ds = PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"my_field": f},
        lhs="my_field_t",
    )
    assert ds.lhs_field == "my_field"
    assert ds.lhs_axis == "t"


# 12. List input for coords gets converted


def test_from_arrays_list_coords() -> None:
    """Plain Python lists for coords are accepted (covers Sequence[float])."""
    x_list = [0.0, 0.5, 1.0, 1.5, 2.0]
    t_list = [0.0, 0.5, 1.0]
    u = torch.randn(len(x_list), len(t_list), dtype=torch.float64)

    ds = PDEDataset.from_arrays(
        coords={"x": x_list, "t": t_list},
        fields={"u": u},
        lhs="u_t",
    )
    assert ds.axes is not None
    assert ds.axes["x"].values.numel() == 5


# 13. Equivalent to manual construction


def test_from_arrays_equivalent_to_manual() -> None:
    """from_arrays yields the same fields/axes as the manual recipe."""
    nx, nt = 10, 6
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.randn(nx, nt, dtype=torch.float64)

    ds_factory = PDEDataset.from_arrays(
        coords={"x": x, "t": t},
        fields={"u": u},
        lhs="u_t",
        periodic={"x"},
        name="manual_compare",
    )

    ds_manual = PDEDataset(
        name="manual_compare",
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

    assert ds_factory.name == ds_manual.name
    assert ds_factory.axis_order == ds_manual.axis_order
    assert ds_factory.lhs_field == ds_manual.lhs_field
    assert ds_factory.lhs_axis == ds_manual.lhs_axis
    assert ds_factory.axes is not None and ds_manual.axes is not None
    assert ds_factory.axes["x"].is_periodic == ds_manual.axes["x"].is_periodic
    assert ds_factory.fields is not None and ds_manual.fields is not None
    assert torch.equal(ds_factory.fields["u"].values, ds_manual.fields["u"].values)
