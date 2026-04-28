"""Tests for VizRecorder — platform-provided accumulator for viz data.

Covers:
- log / get / keys basic contract
- enabled=False disables accumulation
- Tensor handling: scalar .item(), non-scalar .detach().cpu()
- Large tensor warning (>10k elements)
- to_dict / from_dict round-trip
- Non-serializable values degrade to str() in to_dict
"""

from __future__ import annotations

import logging

import pytest
import torch

from kd2.search.recorder import VizRecorder

# Smoke


@pytest.mark.smoke
class TestVizRecorderSmoke:
    """VizRecorder exists and basic API is callable."""

    def test_instantiate(self) -> None:
        rec = VizRecorder()
        assert rec.enabled is True

    def test_log_get_keys_exist(self) -> None:
        rec = VizRecorder()
        rec.log("x", 1)
        assert "x" in rec.keys()
        assert rec.get("x") == [1]


# Unit — core contract


class TestVizRecorderCore:
    """Happy-path and basic contract tests."""

    def test_log_appends_multiple_values(self) -> None:
        rec = VizRecorder()
        rec.log("loss", 0.5)
        rec.log("loss", 0.3)
        rec.log("loss", 0.1)
        assert rec.get("loss") == [0.5, 0.3, 0.1]

    def test_multiple_keys_independent(self) -> None:
        rec = VizRecorder()
        rec.log("a", 1)
        rec.log("b", 2)
        rec.log("a", 3)
        assert rec.get("a") == [1, 3]
        assert rec.get("b") == [2]
        assert rec.keys() == {"a", "b"}

    def test_get_missing_key_returns_empty_list(self) -> None:
        rec = VizRecorder()
        result = rec.get("nonexistent")
        assert result == []

    def test_keys_returns_set(self) -> None:
        rec = VizRecorder()
        rec.log("x", 1)
        rec.log("y", 2)
        keys = rec.keys()
        assert isinstance(keys, set)
        assert keys == {"x", "y"}

    def test_log_accepts_diverse_types(self) -> None:
        """log() works with plain Python values: int, float, str, dict."""
        rec = VizRecorder()
        rec.log("int", 42)
        rec.log("float", 3.14)
        rec.log("str", "hello")
        rec.log("dict", {"a": 1})
        assert len(rec.get("int")) == 1
        assert len(rec.get("str")) == 1


# Unit — enabled=False


class TestVizRecorderDisabled:
    """When enabled=False, no data should accumulate."""

    def test_disabled_skips_logging(self) -> None:
        rec = VizRecorder(enabled=False)
        rec.log("loss", 1.0)
        rec.log("loss", 2.0)
        assert rec.get("loss") == []
        assert rec.keys() == set()

    def test_disabled_skips_tensor_logging(self) -> None:
        rec = VizRecorder(enabled=False)
        rec.log("t", torch.tensor(1.0))
        assert rec.get("t") == []


# Unit — tensor handling


class TestVizRecorderTensorHandling:
    """Tensor values are converted: scalar -> .item(), non-scalar -> detach+cpu."""

    def test_scalar_tensor_stored_as_python_number(self) -> None:
        rec = VizRecorder()
        rec.log("val", torch.tensor(3.14))
        stored = rec.get("val")[0]
        # Should be a plain Python float, not a Tensor
        assert not isinstance(stored, torch.Tensor)
        assert isinstance(stored, float)
        torch.testing.assert_close(
            torch.tensor(stored), torch.tensor(3.14), rtol=1e-5, atol=1e-8
        )

    def test_nonscalar_tensor_is_detached(self) -> None:
        x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
        y = x * 2 # has grad_fn
        rec = VizRecorder()
        rec.log("vec", y)
        stored = rec.get("vec")[0]
        assert isinstance(stored, torch.Tensor)
        assert not stored.requires_grad
        torch.testing.assert_close(stored, torch.tensor([2.0, 4.0, 6.0]))

    def test_nonscalar_tensor_moved_to_cpu(self) -> None:
        """Non-scalar tensor on any device ends up on CPU after log."""
        # We test on CPU — the code path still calls .cpu()
        t = torch.tensor([1.0, 2.0])
        rec = VizRecorder()
        rec.log("t", t)
        stored = rec.get("t")[0]
        assert stored.device == torch.device("cpu")

    def test_large_tensor_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Tensors with >10_000 elements produce a warning."""
        big = torch.zeros(10_001)
        rec = VizRecorder()
        with caplog.at_level(logging.WARNING):
            rec.log("big", big)
        assert any("large tensor" in m.lower() for m in caplog.messages)
        assert any("10001" in m for m in caplog.messages)

    def test_exactly_10k_no_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """Tensors with exactly 10_000 elements should NOT trigger a warning."""
        t = torch.zeros(10_000)
        rec = VizRecorder()
        with caplog.at_level(logging.WARNING):
            rec.log("ok", t)
        assert not any("large tensor" in m.lower() for m in caplog.messages)

    def test_tensor_in_list_isolated_from_inplace_mutation(self) -> None:
        """Tensors nested in a list/tuple/dict must not alias the caller's
        tensor — otherwise an in-place edit after ``log()`` silently
        rewrites recorded history."""
        t = torch.tensor([1.0, 2.0, 3.0])
        rec = VizRecorder()
        rec.log("listed", [t, "meta"])
        t[0] = 999.0
        stored = rec.get("listed")[0]
        torch.testing.assert_close(stored[0], torch.tensor([1.0, 2.0, 3.0]))

    def test_tensor_in_dict_isolated_from_inplace_mutation(self) -> None:
        """Same invariant for dict containers."""
        t = torch.tensor([5.0, 6.0])
        rec = VizRecorder()
        rec.log("dicted", {"data": t, "tag": "x"})
        t[0] = 777.0
        stored = rec.get("dicted")[0]
        torch.testing.assert_close(stored["data"], torch.tensor([5.0, 6.0]))

    def test_tensor_in_dataclass_isolated_from_inplace_mutation(self) -> None:
        """Tensors nested in a dataclass must not alias the caller's
        tensor — relevant when callbacks log full
        ``EvaluationResult`` / ``IntegrationResult`` instances."""
        from dataclasses import dataclass

        @dataclass
        class _Result:
            tag: str
            data: torch.Tensor

        t = torch.tensor([1.0, 2.0, 3.0])
        rec = VizRecorder()
        rec.log("dc", _Result(tag="x", data=t))
        t[0] = 999.0
        stored = rec.get("dc")[0]
        assert isinstance(stored, _Result)
        assert stored.tag == "x"
        torch.testing.assert_close(stored.data, torch.tensor([1.0, 2.0, 3.0]))

    def test_one_element_tensor_in_dataclass_keeps_tensor_type(self) -> None:
        """A 1-element Tensor field in a dataclass must remain a Tensor
        after detach (cross-model review): demoting to ``.item()`` would
        break the dataclass type contract — e.g. ``EvaluationResult.
        coefficients=tensor([0.5])`` (single-term selection) becoming a
        Python float crashes downstream ``.detach()`` / matmul calls.
        """
        from dataclasses import dataclass

        @dataclass
        class _Eval:
            mse: float
            coefficients: torch.Tensor

        coef = torch.tensor([0.5]) # single-term selection
        rec = VizRecorder()
        rec.log("eval", _Eval(mse=0.1, coefficients=coef))
        coef[0] = 999.0 # mutate caller's tensor
        stored = rec.get("eval")[0]
        assert isinstance(stored, _Eval)
        # Type contract preserved
        assert isinstance(stored.coefficients, torch.Tensor)
        # Value isolated from caller's mutation
        torch.testing.assert_close(stored.coefficients, torch.tensor([0.5]))

    def test_tensor_in_namedtuple_preserves_type_and_isolated(self) -> None:
        """Namedtuples must not be downgraded to plain tuples and the
        nested tensor must be isolated."""
        from collections import namedtuple

        Pair = namedtuple("Pair", ["data", "tag"])
        t = torch.tensor([5.0, 6.0])
        rec = VizRecorder()
        rec.log("nt", Pair(data=t, tag="y"))
        t[0] = 777.0
        stored = rec.get("nt")[0]
        assert isinstance(stored, Pair)
        assert stored.tag == "y"
        torch.testing.assert_close(stored.data, torch.tensor([5.0, 6.0]))

    def test_scalar_tensors_in_set_isolated_from_inplace_mutation(self) -> None:
        """Scalar tensors in a set become hashable Python scalars and
        survive in-place mutation of the originals."""
        a = torch.tensor(1.0)
        b = torch.tensor(2.0)
        rec = VizRecorder()
        rec.log("s", {a, b})
        a[...] = 999.0
        b[...] = 888.0
        stored = rec.get("s")[0]
        assert isinstance(stored, set)
        assert stored == {1.0, 2.0}


# Unit — NaN/Inf handling


class TestVizRecorderNanInf:
    """NaN and Inf values are sanitized in to_dict for RFC 8259 compliance."""

    def test_nan_float_becomes_none_in_to_dict(self) -> None:
        rec = VizRecorder()
        rec.log("score", float("nan"))
        d = rec.to_dict()
        assert d["score"][0] is None

    def test_inf_float_becomes_none_in_to_dict(self) -> None:
        rec = VizRecorder()
        rec.log("score", float("inf"))
        rec.log("score", float("-inf"))
        d = rec.to_dict()
        assert d["score"][0] is None
        assert d["score"][1] is None

    def test_nan_tensor_scalar_becomes_none(self) -> None:
        rec = VizRecorder()
        rec.log("val", torch.tensor(float("nan")))
        d = rec.to_dict()
        # scalar tensor → .item() → float('nan') → None in to_dict
        assert d["val"][0] is None

    def test_tensor_with_nan_elements(self) -> None:
        """Non-scalar tensor with NaN elements: NaN → null in JSON."""
        import json

        rec = VizRecorder()
        t = torch.tensor([1.0, float("nan"), 3.0])
        rec.log("vec", t)
        d = rec.to_dict()
        # Must be JSON-safe
        serialized = json.dumps(d)
        assert isinstance(serialized, str)
        # NaN element should be None
        assert d["vec"][0][1] is None


# Unit — serialization round-trip


class TestVizRecorderSerialization:
    """to_dict / from_dict contract."""

    def test_round_trip_plain_values(self) -> None:
        rec = VizRecorder()
        rec.log("loss", 0.5)
        rec.log("loss", 0.3)
        rec.log("iter", 1)
        rec.log("iter", 2)

        d = rec.to_dict()
        restored = VizRecorder.from_dict(d)

        assert restored.get("loss") == [0.5, 0.3]
        assert restored.get("iter") == [1, 2]
        assert restored.keys() == {"loss", "iter"}

    def test_round_trip_tensor_values(self) -> None:
        """Tensors survive to_dict -> from_dict (as lists or tensors)."""
        rec = VizRecorder()
        original = torch.tensor([1.0, 2.0, 3.0])
        rec.log("vec", original)

        d = rec.to_dict()
        # to_dict should convert tensors to JSON-safe form (list)
        series = d["vec"]
        assert len(series) == 1
        # The serialized form should be a list, not a tensor
        item = series[0]
        assert isinstance(item, list)
        torch.testing.assert_close(torch.tensor(item), original, rtol=1e-5, atol=1e-8)

    def test_to_dict_non_serializable_degrades_to_str(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Objects that cannot be JSON-serialized fall back to str() + warning."""

        class Opaque:
            def __str__(self) -> str:
                return "opaque-object"

        rec = VizRecorder()
        rec.log("obj", Opaque())

        with caplog.at_level(logging.WARNING):
            d = rec.to_dict()

        # Value should be the str() fallback
        assert d["obj"][0] == "opaque-object"
        # Warning must be emitted for lossy serialization
        assert any("obj" in m for m in caplog.messages)

    def test_to_dict_is_json_safe(self) -> None:
        """to_dict() output must be JSON-serializable."""
        import json

        rec = VizRecorder()
        rec.log("scalar", torch.tensor(1.0))
        rec.log("vec", torch.tensor([1.0, 2.0]))
        rec.log("plain", 42)
        rec.log("text", "hello")

        d = rec.to_dict()
        # Must not raise
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_from_dict_round_trip_with_tensor_derived_lists(self) -> None:
        """from_dict correctly restores list-valued data (from tensor to_dict)."""
        rec = VizRecorder()
        rec.log("vec", torch.tensor([10.0, 20.0, 30.0]))
        rec.log("vec", torch.tensor([40.0, 50.0]))

        d = rec.to_dict()
        restored = VizRecorder.from_dict(d)

        assert len(restored.get("vec")) == 2
        # First entry should be a list of 3
        assert len(restored.get("vec")[0]) == 3

    def test_from_dict_creates_enabled_recorder(self) -> None:
        rec = VizRecorder.from_dict({"a": [1, 2]})
        assert rec.enabled is True
        assert rec.get("a") == [1, 2]

    def test_round_trip_empty_recorder(self) -> None:
        rec = VizRecorder()
        d = rec.to_dict()
        restored = VizRecorder.from_dict(d)
        assert restored.keys() == set()

    def test_round_trip_preserves_scalar_tensors_as_numbers(self) -> None:
        """Scalar tensors logged as .item() survive round-trip as numbers."""
        rec = VizRecorder()
        rec.log("scalar", torch.tensor(42.0))
        d = rec.to_dict()
        restored = VizRecorder.from_dict(d)
        val = restored.get("scalar")[0]
        assert isinstance(val, (int, float))
        assert val == pytest.approx(42.0)
