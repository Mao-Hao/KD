"""Unit tests for SGAPlugin review findings (L1).

TDD red phase: tests the contract that ``SGAPlugin.build_result_target()``
returns a copy that callers can mutate without aliasing the plugin's
private ``self._y`` storage.

Today the implementation returns ``self._y.detach()`` which still
shares the underlying storage with ``self._y`` — an in-place mutation
on the returned tensor (e.g., ``.zero_()``) silently overwrites the
plugin's internal LHS target.
"""

from __future__ import annotations

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
from kd2.search.sga.config import SGAConfig

# ============================================================================
# Fixtures (compact, shared with test_plugin.py via duplicated helpers —
# keep this file self-contained so it can be moved to a refactor later).
# ============================================================================


_GRID_SIZE = 10
_TIME_SIZE = 5

# Local generator so fixture randomness doesn't pollute the global torch
# RNG state — adjacent tests in this directory rely on a deterministic
# global state (notably ``test_plugin.py::test_propose_returns_kd2_format``).
_FIXTURE_RNG = torch.Generator()
_FIXTURE_RNG.manual_seed(0)


def _make_dataset() -> PDEDataset:
    x = torch.linspace(0.0, 1.0, _GRID_SIZE)
    t = torch.linspace(0.0, 1.0, _TIME_SIZE)
    u = torch.randn(_GRID_SIZE, _TIME_SIZE, generator=_FIXTURE_RNG)
    return PDEDataset(
        name="byod_test",
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


def _make_context(dataset: PDEDataset) -> MagicMock:
    context = MagicMock()
    context.dataset = dataset

    def get_variable(name: str) -> torch.Tensor:
        if dataset.fields is not None and name in dataset.fields:
            return dataset.fields[name].values
        if dataset.axes is not None and name in dataset.axes:
            return dataset.axes[name].values
        raise KeyError(name)

    def get_derivative(field: str, axis: str, order: int) -> torch.Tensor:
        # Use a per-call seeded local generator instead of mutating the
        # global RNG (would otherwise leak into adjacent SGA tests in this
        # directory and produce empty PDE candidates).
        local_rng = torch.Generator()
        local_rng.manual_seed(hash((field, axis, order)) % (2**31))
        return torch.randn(_GRID_SIZE, _TIME_SIZE, generator=local_rng)

    context.get_variable = get_variable
    context.get_derivative = get_derivative
    return context


@pytest.fixture
def prepared_plugin() -> SGAPlugin: # noqa: F821
    """SGAPlugin that has been ``prepare()``-d on a small dataset."""
    from kd2.search.sga.plugin import SGAPlugin

    dataset = _make_dataset()
    components = PlatformComponents(
        dataset=dataset,
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=_make_context(dataset),
        registry=MagicMock(),
    )
    plugin = SGAPlugin(SGAConfig(num=4, depth=3, width=3, seed=7, maxit=2))
    plugin.prepare(components)
    return plugin


# ============================================================================
# L1: build_result_target must return a non-aliased copy
# ============================================================================


@pytest.mark.unit
class TestL1BuildResultTargetNonAliased:
    """``SGAPlugin.build_result_target()`` must NOT alias ``self._y``.

    The Runner's ``_actual`` wraps the returned tensor in
    ``actual.detach()`` and exposes it through ``ExperimentResult.actual``.
    Callers (downstream viz code, user notebooks) routinely run
    in-place ops on result tensors; aliased storage means SGA's
    private state silently mutates and breaks subsequent re-evaluation
    (e.g., ``build_final_result`` on a stale ``_y``).
    """

    def test_returned_tensor_does_not_share_storage_with_internal_y(
        self,
        prepared_plugin: SGAPlugin, # noqa: F821
    ) -> None:
        """Returned tensor and ``_y`` must not share underlying storage."""
        target = prepared_plugin.build_result_target()
        assert prepared_plugin._y is not None
        # ``data_ptr`` reveals storage aliasing even when the tensor is a
        # detached view.
        assert target.data_ptr() != prepared_plugin._y.data_ptr(), (
            "build_result_target() returned a tensor sharing storage with "
            "the plugin's private _y. In-place mutation by the caller "
            "would silently corrupt SGA state."
        )

    def test_in_place_mutation_does_not_affect_internal_y(
        self,
        prepared_plugin: SGAPlugin, # noqa: F821
    ) -> None:
        """Zeroing the returned tensor in-place must leave ``_y`` intact."""
        assert prepared_plugin._y is not None
        original = prepared_plugin._y.clone()

        target = prepared_plugin.build_result_target()
        target.zero_() # explicit in-place mutation

        # _y must be untouched: clone semantics, not view semantics.
        torch.testing.assert_close(prepared_plugin._y, original, rtol=0.0, atol=0.0)

    def test_in_place_mutation_does_not_affect_subsequent_call(
        self,
        prepared_plugin: SGAPlugin, # noqa: F821
    ) -> None:
        """Two successive ``build_result_target`` calls return independent
        tensors, so mutating one cannot poison the other.
        """
        first = prepared_plugin.build_result_target()
        # Mutate aggressively.
        first.fill_(float("nan"))

        second = prepared_plugin.build_result_target()
        assert torch.isfinite(second).all(), (
            "Second build_result_target() returned a tensor poisoned by "
            "the in-place mutation of the first — they must be independent."
        )

    def test_returned_tensor_is_detached(
        self,
        prepared_plugin: SGAPlugin, # noqa: F821
    ) -> None:
        """Returned tensor must not carry an autograd graph (regression
        check for the broader 'detach + clone' contract)."""
        target = prepared_plugin.build_result_target()
        assert target.requires_grad is False
