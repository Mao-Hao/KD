"""Runtime recorder for visualization-friendly experiment data."""

from __future__ import annotations

import dataclasses
import json
import logging
import math
from collections import defaultdict
from typing import Any

from torch import Tensor

logger = logging.getLogger(__name__)

_LARGE_TENSOR_WARNING_THRESHOLD = 10_000


def _detach_tensor(value: Tensor) -> Tensor:
    """Detach a tensor, clone storage, and move to CPU."""
    tensor = value.detach().clone()
    if tensor.device.type != "cpu":
        tensor = tensor.cpu()
    return tensor


def _is_dataclass_instance(value: Any) -> bool:
    """True if ``value`` is a dataclass instance (not the class itself)."""
    return dataclasses.is_dataclass(value) and not isinstance(value, type)


def _detach_field_value(value: Any) -> Any:
    """Detach a dataclass field value while preserving its type contract.

    Differs from ``_detach_recursive`` only in the top-level Tensor
    branch: a 1-element Tensor field is *not* demoted to a Python
    scalar, since dataclass annotations (e.g. ``EvaluationResult.
    coefficients: torch.Tensor``) require the value to remain a Tensor
    for downstream ``.detach() / matmul`` calls (cross-model review).
    """
    if isinstance(value, Tensor):
        return _detach_tensor(value)
    return _detach_recursive(value)


def _detach_recursive(value: Any) -> Any:
    """Walk container types and detach+clone any nested ``Tensor``.

    A direct ``Tensor`` value is already isolated by ``log()`` (which
    calls ``_detach_tensor`` or ``.item()``), but nested containers
    would otherwise alias the caller's tensor and let an in-place edit
    after ``log()`` rewrite recorded history.

    Supported containers: ``dict``, ``list``, ``tuple`` (incl.
    ``namedtuple``), ``set`` / ``frozenset``, and dataclass instances
    (e.g. ``EvaluationResult``, ``IntegrationResult``). Namedtuples
    preserve their type via ``type(value)._make``; dataclasses via
    ``dataclasses.replace`` over ``init=True`` fields with
    ``_detach_field_value`` to preserve Tensor type contracts.
    """
    if isinstance(value, Tensor):
        if value.numel() == 1:
            return value.detach().item()
        return _detach_tensor(value)
    if isinstance(value, dict):
        return {k: _detach_recursive(v) for k, v in value.items()}
    # Namedtuple detection must precede plain-tuple branch — `_make`
    # preserves the named-tuple subclass that plain tuple() would lose.
    if isinstance(value, tuple) and hasattr(value, "_fields"):
        return type(value)._make(_detach_recursive(v) for v in value) # type: ignore[attr-defined]
    if isinstance(value, list):
        return [_detach_recursive(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_detach_recursive(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return type(value)(_detach_recursive(v) for v in value)
    if _is_dataclass_instance(value):
        detached = {
            f.name: _detach_field_value(getattr(value, f.name))
            for f in dataclasses.fields(value)
            if f.init
        }
        return dataclasses.replace(value, **detached)
    return value


def _is_json_serializable(value: Any) -> bool:
    """Return whether a value can be serialized by ``json``."""
    try:
        json.dumps(value)
    except TypeError:
        return False
    return True


def _sanitize_float(value: float) -> float | None:
    """Replace NaN/Inf with None for RFC 8259 compliance."""
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _make_json_safe(value: Any, *, key: str) -> Any:
    """Convert a value into a JSON-safe (RFC 8259) representation.

    NaN/Inf floats are replaced with None (JSON null).
    """
    if isinstance(value, float):
        return _sanitize_float(value)

    if isinstance(value, Tensor):
        safe_list = _detach_tensor(value).tolist()
        # Recursively sanitize NaN/Inf in tensor-derived lists
        return _make_json_safe(safe_list, key=key)

    if isinstance(value, dict):
        return {
            str(child_key): _make_json_safe(child_value, key=key)
            for child_key, child_value in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_make_json_safe(item, key=key) for item in value]

    if _is_json_serializable(value):
        return value

    logger.warning(
        "VizRecorder: key '%s' uses string fallback for %s",
        key,
        type(value).__name__,
    )
    return str(value)


class VizRecorder:
    """Platform-provided accumulator for visualization data."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self._store: defaultdict[str, list[Any]] = defaultdict(list)

    def log(self, key: str, value: Any) -> None:
        """Append a value to a named series with tensor-safe handling."""
        if not self.enabled:
            return

        if isinstance(value, Tensor):
            if value.numel() > _LARGE_TENSOR_WARNING_THRESHOLD:
                logger.warning(
                    "VizRecorder: large tensor for key '%s' (%d elements)",
                    key,
                    value.numel(),
                )
            if value.numel() == 1:
                value = value.detach().item()
            else:
                value = _detach_tensor(value)
        elif isinstance(
            value, (list, tuple, dict, set, frozenset)
        ) or _is_dataclass_instance(value):
            value = _detach_recursive(value)

        self._store[key].append(value)

    def get(self, key: str) -> list[Any]:
        """Return a copy of the recorded series for ``key``."""
        return list(self._store.get(key, []))

    def keys(self) -> set[str]:
        """Return all recorded series names."""
        return set(self._store.keys())

    def to_dict(self) -> dict[str, list[Any]]:
        """Return a JSON-safe serialization of the recorder contents."""
        return {
            key: [_make_json_safe(value, key=key) for value in values]
            for key, values in self._store.items()
        }

    @classmethod
    def from_dict(cls, data: dict[str, list[Any]]) -> VizRecorder:
        """Reconstruct a recorder from serialized data."""
        recorder = cls()
        recorder._store = defaultdict(
            list,
            {key: list(values) for key, values in data.items()},
        )
        return recorder
