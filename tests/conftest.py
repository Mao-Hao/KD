"""Shared test fixtures for kd2."""

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Default test device - auto-detects best available: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Auto-assign markers based on test directory location.

    Mapping:
    - tests/unit/ → @pytest.mark.unit
    - tests/integration/ → @pytest.mark.integration
    - tests/equivalence/ → @pytest.mark.equivalence
    - tests/validation/ → @pytest.mark.validation

    Other markers (smoke, numerical, slow, etc.) are applied manually
    via @pytest.mark decorators in individual test files.

    Only adds marker if not already present (respects manual annotations).
    """
    marker_map = {
        "unit": "unit",
        "integration": "integration",
        "equivalence": "equivalence",
        "validation": "validation",
    }

    for item in items:
        # Get test file path relative to project root
        test_path = str(item.fspath)

        # Check which directory the test is in
        for directory, marker_name in marker_map.items():
            if f"/tests/{directory}/" in test_path:
                # Check if marker already exists
                marker_names = {mark.name for mark in item.iter_markers()}
                if marker_name not in marker_names:
                    item.add_marker(getattr(pytest.mark, marker_name))
                break
