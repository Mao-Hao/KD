"""Smoke tests for ``examples/0*.py``.

Each test runs one example as a subprocess and asserts that:
- the script exits cleanly (returncode == 0),
- stdout contains a sentinel string ("Discovered" or "[kd2] Done.").

We intentionally do NOT assert on the discovered expression - search is
seeded but stochastic across PyTorch versions, so locking the exact
expression would be flaky. Exit code + sentinel is enough to catch
import errors, API breakage, and broken examples.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

# Repository root: tests/examples/ -> tests/ -> repo root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# Default timeout for fast examples (01, 02, 05).
_FAST_TIMEOUT_SEC = 120

# Looser timeout for the slow examples (03, 04) so contention does not
# cause spurious timeouts on shared CI hardware.
_SLOW_TIMEOUT_SEC = 300

# Sentinel substrings - either implies the script ran end-to-end.
_SENTINELS = ("Discovered", "[kd2] Done.")


def _run_example(
    script_name: str, timeout_sec: int
) -> subprocess.CompletedProcess[str]:
    """Run a single example script as a subprocess and return the result."""
    script_path = _EXAMPLES_DIR / script_name
    assert script_path.exists(), f"Missing example script: {script_path}"
    return subprocess.run( # noqa: S603 - args are repo-controlled paths
        [sys.executable, str(script_path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=timeout_sec,
        cwd=_REPO_ROOT,
    )


def _assert_sentinel(stdout: str) -> None:
    """Assert that ``stdout`` contains at least one expected sentinel string."""
    assert any(s in stdout for s in _SENTINELS), (
        f"No sentinel found in example stdout. Expected one of {_SENTINELS}.\n"
        f"--- stdout ---\n{stdout}"
    )


@pytest.mark.smoke
def test_example_01_quickstart() -> None:
    """Example 01 runs end-to-end and prints a discovered expression."""
    result = _run_example("01_quickstart.py", _FAST_TIMEOUT_SEC)
    _assert_sentinel(result.stdout)


@pytest.mark.smoke
def test_example_02_your_data() -> None:
    """Example 02 (BYOD) wraps numpy arrays via from_arrays and fits."""
    result = _run_example("02_your_data.py", _FAST_TIMEOUT_SEC)
    _assert_sentinel(result.stdout)


@pytest.mark.smoke
@pytest.mark.slow
def test_example_03_visualize() -> None:
    """Example 03 fits on bundled data and renders the HTML viz report."""
    result = _run_example("03_visualize.py", _SLOW_TIMEOUT_SEC)
    _assert_sentinel(result.stdout)
    assert "Report" in result.stdout, (
        "Example 03 did not report an HTML report path.\n"
        f"--- stdout ---\n{result.stdout}"
    )


@pytest.mark.smoke
@pytest.mark.slow
def test_example_04_noisy_data() -> None:
    """Example 04 compares finite_diff vs autograd on noisy data."""
    result = _run_example("04_noisy_data.py", _SLOW_TIMEOUT_SEC)
    _assert_sentinel(result.stdout)


@pytest.mark.smoke
def test_example_05_save_load() -> None:
    """Example 05 round-trips a result through JSON."""
    result = _run_example("05_save_load.py", _FAST_TIMEOUT_SEC)
    _assert_sentinel(result.stdout)
    assert "Round-trip OK" in result.stdout, (
        f"Example 05 did not report round-trip status.\n--- stdout ---\n{result.stdout}"
    )
