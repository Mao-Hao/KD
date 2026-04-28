"""Tests for ``kd2.preview`` (sanity-check tool)."""

from __future__ import annotations

import io

import pytest
import torch

from kd2.data.schema import (
    AxisInfo,
    DataTopology,
    FieldData,
    PDEDataset,
    TaskType,
)
from kd2.data.synthetic import generate_burgers_data
from kd2.inspect import preview

# Helpers


def _make_dataset(
    *,
    nx: int = 32,
    nt: int = 16,
    periodic_x: bool = False,
    nonuniform: bool = False,
    name: str = "synthetic",
) -> PDEDataset:
    """Build a small synthetic PDEDataset for tests.

    Args:
        nx, nt: Grid sizes.
        periodic_x: Whether the x-axis is marked periodic.
        nonuniform: If True, use a quadratically-spaced x-axis.
        name: Dataset name.
    """
    if nonuniform:
        # Quadratic spacing → diff is non-constant.
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64) ** 2
    else:
        x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.randn(nx, nt, dtype=torch.float64)
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        topology=DataTopology.GRID,
        axes={
            "x": AxisInfo(name="x", values=x, is_periodic=periodic_x),
            "t": AxisInfo(name="t", values=t, is_periodic=False),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def _capture_preview(ds: PDEDataset) -> str:
    """Run ``preview`` against an in-memory text stream and return output."""
    buf = io.StringIO()
    preview(ds, file=buf)
    return buf.getvalue()


# 1. Basic output structure


@pytest.mark.smoke
def test_preview_basic() -> None:
    """Output contains the four mandatory section headers + dataset name."""
    ds = generate_burgers_data(nx=32, nt=16, nu=0.1, seed=0)
    text = _capture_preview(ds)
    # Section headers
    assert "Dataset:" in text
    assert "Axes:" in text
    assert "Fields:" in text
    assert "LHS:" in text
    # Dataset name
    assert ds.name in text
    # Axes are listed
    assert "x " in text or "x|" in text or "x |" in text
    assert "t " in text or "t|" in text or "t |" in text


# 2. Uniform spacing -> "uniform" appears, no warning


def test_preview_uniform_spacing() -> None:
    """Uniformly-spaced axes are reported as ``uniform`` and emit no warning."""
    ds = _make_dataset(nx=32, nt=20)
    text = _capture_preview(ds)
    assert "uniform" in text
    assert "NON-UNIFORM" not in text
    assert "WARNING:" not in text # no warnings on a clean uniform grid


# 3. Non-uniform spacing -> NON-UNIFORM tag + warning


def test_preview_non_uniform_warning() -> None:
    """A non-uniform grid surfaces a NON-UNIFORM tag plus a warning line."""
    ds = _make_dataset(nx=32, nt=20, nonuniform=True)
    text = _capture_preview(ds)
    assert "NON-UNIFORM" in text
    assert "WARNING:" in text
    assert "not uniformly spaced" in text or "uniform" in text


# 4. NaN / Inf in field -> warning + counts


def test_preview_nan_warning() -> None:
    """A field with NaN entries surfaces a warning even though PDEDataset's
    construction validators forbid NaN.

    We bypass the validator by mutating ``field.values`` after construction
    so the printer logic itself is exercised.
    """
    ds = _make_dataset(nx=20, nt=12)
    assert ds.fields is not None
    # Inject NaN/Inf post-construction (validators only run on __post_init__).
    f = ds.fields["u"].values
    f[0, 0] = float("nan")
    f[1, 1] = float("inf")

    text = _capture_preview(ds)
    assert "NaN=1" in text or "NaN=" in text
    # Warning line for NaN
    assert "NaN" in text
    # The status line must reflect the warning(s)
    assert "warning" in text.lower()


# 5. Small grid (n<16) -> warning


def test_preview_small_grid_warning() -> None:
    """An axis with fewer than 16 points triggers the small-grid warning."""
    ds = _make_dataset(nx=8, nt=20)
    text = _capture_preview(ds)
    assert "WARNING:" in text
    # Match the warning's spirit — implementation phrases it as small grid.
    assert "small grid" in text or "8 points" in text


# 6. Clean dataset -> "ready to fit" status


def test_preview_no_warnings_clean_data() -> None:
    """A fully clean Burgers dataset prints ``Status: ready to fit``."""
    ds = generate_burgers_data(nx=64, nt=32, nu=0.1, seed=0)
    text = _capture_preview(ds)
    assert "Status: ready to fit" in text
    assert "WARNING:" not in text


# 7. Defaults to stdout via capsys


def test_preview_default_stdout(capsys: pytest.CaptureFixture[str]) -> None:
    """Calling preview without ``file=`` writes to stdout (capsys works)."""
    ds = generate_burgers_data(nx=32, nt=16, nu=0.1, seed=0)
    preview(ds) # no explicit file
    out = capsys.readouterr().out
    assert "Dataset:" in out
    assert ds.name in out


# 8. preview is exported at the top level


def test_preview_top_level_export() -> None:
    """``kd2.preview`` is the same callable as ``kd2.inspect.preview``."""
    import kd2
    import kd2.inspect as kd2_inspect

    assert kd2.preview is kd2_inspect.preview


# 9. preview's uniformity verdict matches FiniteDiffProvider's
# (F3: inspect.py used a stricter local rtol=1e-6 + custom spread, causing
# false NON-UNIFORM warnings on float32 grids the FD provider accepts.
# Also: descending grids are reported as "uniform" because the local logic
# uses sign-blind spread.)


def _make_dataset_with_x(x: torch.Tensor, name: str = "f3") -> PDEDataset:
    """Build a 1D-spatial dataset with an arbitrary x axis for verdict tests."""
    nt = 8
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.zeros(x.shape[0], nt, dtype=torch.float64)
    return PDEDataset(
        name=name,
        task_type=TaskType.PDE,
        axes={
            "x": AxisInfo(name="x", values=x),
            "t": AxisInfo(name="t", values=t),
        },
        axis_order=["x", "t"],
        fields={"u": FieldData(name="u", values=u)},
        lhs_field="u",
        lhs_axis="t",
    )


def test_preview_float32_linspace_does_not_warn() -> None:
    """Pre-fix: ``inspect`` used rtol=1e-6 (a stricter local threshold) and
    flagged float32 ``linspace`` as NON-UNIFORM, even though the FD provider
    accepts the same grid at rtol=1e-4. The verdicts must agree on grids
    the FD path treats as uniform.

    n=2000 keeps drift at ~1e-4 which the shared predicate accepts; the
    pre-fix local logic at rtol=1e-6 rejected it.
    """
    x = torch.linspace(0.0, 1.0, 2000, dtype=torch.float32)
    ds = _make_dataset_with_x(x, name="float32_linspace")
    text = _capture_preview(ds)
    # Mirror the FD provider's verdict — accept this drift.
    assert "NON-UNIFORM" not in text, (
        "preview must use the same uniform-grid predicate as "
        "FiniteDiffProvider (UNIFORM_GRID_RTOL=1e-4); the local rtol=1e-6 "
        "was rejecting float32 linspace drift the FD path accepts."
    )
    assert "uniform" in text


def test_preview_descending_axis_warns_explicitly() -> None:
    """Pre-fix: a descending axis ``[0.4, 0.3, 0.2, 0.1, 0]`` had constant
    spacing of -0.1, and the local spread logic (max-min / abs(mean))
    blessed it as ``uniform``. FD provider rejects it (signed dx<0), so
    preview must surface a warning instead of silently passing.
    """
    x = torch.tensor([0.4, 0.3, 0.2, 0.1, 0.0], dtype=torch.float64)
    ds = _make_dataset_with_x(x, name="descending")
    text = _capture_preview(ds)
    assert "WARNING:" in text, (
        "preview must warn on descending coords — FD stencils reject them "
        "(dx<0 silently flips derivative signs), so the verdict cannot be "
        "'uniform'."
    )
    # The implementation should explicitly call out the direction (or
    # otherwise mark the grid as not acceptable for FD).
    assert "decreasing" in text.lower() or "NON-UNIFORM" in text


def test_preview_pathological_inf_dx_warns() -> None:
    """A coord vector that overflows on np.diff (e.g. ``[0, 0.1, 0.2,
    1e308, 0.4]``) used to slip through the local logic as "uniform"
    in some configurations. The shared FD predicate rejects it (inf
    diff), so preview must too.
    """
    x = torch.tensor([0.0, 0.1, 0.2, 1.0e308, 0.4], dtype=torch.float64)
    ds = _make_dataset_with_x(x, name="inf_drift")
    text = _capture_preview(ds)
    assert "WARNING:" in text, (
        "preview must reject grids with inf or extreme drift the FD "
        "predicate rejects, not display them as uniform."
    )
