"""Tests for SearchAlgorithm Protocol and PlatformComponents.

Tests the Protocol interface contract, PlatformComponents dataclass structure,
behavioral lifecycle, and negative/edge cases. Written in TDD red phase -- these
tests define the expected API before implementation.
"""

from __future__ import annotations

import dataclasses
import pickle
from typing import Any
from unittest.mock import MagicMock

import pytest

from kd2.core.evaluator import EvaluationResult

# ── Imports under test (will fail until implemented) ─────────────────────
from kd2.search.protocol import PlatformComponents, SearchAlgorithm

# ═══════════════════════════════════════════════════════════════════════════
# Helpers: Mock implementation of the Protocol
# ═══════════════════════════════════════════════════════════════════════════


class _ConformingAlgorithm:
    """A minimal mock that satisfies the SearchAlgorithm protocol."""

    def __init__(self) -> None:
        self._best_score: float = float("inf")
        self._best_expression: str = ""
        self._state: dict[str, Any] = {}
        self._prepared = False

    # -- protocol methods --------------------------------------------------

    def prepare(self, components: PlatformComponents) -> None:
        self._prepared = True

    def propose(self, n: int) -> list[str]:
        return [f"expr_{i}" for i in range(n)]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return [
            EvaluationResult(mse=0.1 * i, nmse=0.1 * i, r2=1.0 - 0.1 * i)
            for i, _ in enumerate(candidates)
        ]

    def update(self, results: list[EvaluationResult]) -> None:
        for r in results:
            if r.is_valid and r.mse < self._best_score:
                self._best_score = r.mse
                self._best_expression = r.expression

    # -- protocol properties -----------------------------------------------

    @property
    def best_score(self) -> float:
        return self._best_score

    @property
    def best_expression(self) -> str:
        return self._best_expression

    @property
    def config(self) -> dict:
        return {"algorithm": "ConformingAlgorithm"}

    @property
    def state(self) -> dict:
        return self._state

    @state.setter
    def state(self, value: dict) -> None:
        self._state = value


class _MissingPrepare:
    """Missing prepare() method."""

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingPropose:
    """Missing propose() method."""

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingEvaluate:
    """Missing evaluate() method."""

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingUpdate:
    """Missing update() method."""

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingBestScore:
    """Missing best_score property."""

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingBestExpression:
    """Missing best_expression property."""

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}

    @state.setter
    def state(self, value: dict) -> None:
        pass


class _MissingState:
    """Missing state property entirely."""

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}


class _NoStateSetter:
    """Has state getter but no setter."""

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        return []

    def update(self, results: list[EvaluationResult]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def config(self) -> dict:
        return {}

    @property
    def state(self) -> dict:
        return {}


# ═══════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════


@pytest.fixture
def mock_components() -> PlatformComponents:
    """Create PlatformComponents with mock objects for all 5 fields."""
    return PlatformComponents(
        dataset=MagicMock(),
        executor=MagicMock(),
        evaluator=MagicMock(),
        context=MagicMock(),
        registry=MagicMock(),
    )


@pytest.fixture
def conforming_algorithm() -> _ConformingAlgorithm:
    """Return a mock implementation that fully conforms to the protocol."""
    return _ConformingAlgorithm()


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 1: Protocol Conformance (~40%)
# ═══════════════════════════════════════════════════════════════════════════


class TestSearchAlgorithmProtocol:
    """Tests for SearchAlgorithm Protocol structural conformance."""

    @pytest.mark.unit
    def test_protocol_is_importable(self) -> None:
        """SearchAlgorithm can be imported from kd2.search.protocol."""
        from kd2.search.protocol import SearchAlgorithm as SA

        assert SA is not None

    @pytest.mark.unit
    def test_protocol_is_runtime_checkable(self) -> None:
        """SearchAlgorithm is decorated with @runtime_checkable."""
        # runtime_checkable protocols support isinstance() checks
        assert (
            hasattr(SearchAlgorithm, "__protocol_attrs__")
            or hasattr(SearchAlgorithm, "__abstractmethods__")
            or isinstance(SearchAlgorithm, type)
        )
        # The definitive test: isinstance must not raise TypeError
        obj = _ConformingAlgorithm()
        # If not runtime_checkable, isinstance raises TypeError
        result = isinstance(obj, SearchAlgorithm)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_conforming_class_is_instance(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """A class implementing all methods passes isinstance check."""
        assert isinstance(conforming_algorithm, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_prepare_fails_isinstance(self) -> None:
        """A class missing prepare() fails isinstance check."""
        obj = _MissingPrepare()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_propose_fails_isinstance(self) -> None:
        """A class missing propose() fails isinstance check."""
        obj = _MissingPropose()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_evaluate_fails_isinstance(self) -> None:
        """A class missing evaluate() fails isinstance check."""
        obj = _MissingEvaluate()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_update_fails_isinstance(self) -> None:
        """A class missing update() fails isinstance check."""
        obj = _MissingUpdate()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_best_score_fails_isinstance(self) -> None:
        """A class missing best_score property fails isinstance check."""
        obj = _MissingBestScore()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_best_expression_fails_isinstance(self) -> None:
        """A class missing best_expression property fails isinstance check."""
        obj = _MissingBestExpression()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_state_fails_isinstance(self) -> None:
        """A class missing state property fails isinstance check."""
        obj = _MissingState()
        assert not isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_protocol_has_prepare_method(self) -> None:
        """Protocol defines prepare method."""
        assert hasattr(SearchAlgorithm, "prepare")

    @pytest.mark.unit
    def test_protocol_has_propose_method(self) -> None:
        """Protocol defines propose method."""
        assert hasattr(SearchAlgorithm, "propose")

    @pytest.mark.unit
    def test_protocol_has_evaluate_method(self) -> None:
        """Protocol defines evaluate method."""
        assert hasattr(SearchAlgorithm, "evaluate")

    @pytest.mark.unit
    def test_protocol_has_update_method(self) -> None:
        """Protocol defines update method."""
        assert hasattr(SearchAlgorithm, "update")

    @pytest.mark.unit
    def test_protocol_has_best_score(self) -> None:
        """Protocol defines best_score property."""
        assert hasattr(SearchAlgorithm, "best_score")

    @pytest.mark.unit
    def test_protocol_has_best_expression(self) -> None:
        """Protocol defines best_expression property."""
        assert hasattr(SearchAlgorithm, "best_expression")

    @pytest.mark.unit
    def test_protocol_has_state(self) -> None:
        """Protocol defines state property."""
        assert hasattr(SearchAlgorithm, "state")


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 2: PlatformComponents (~20%)
# ═══════════════════════════════════════════════════════════════════════════


class TestPlatformComponents:
    """Tests for PlatformComponents dataclass."""

    @pytest.mark.unit
    def test_importable(self) -> None:
        """PlatformComponents can be imported from kd2.search.protocol."""
        from kd2.search.protocol import PlatformComponents as PC

        assert PC is not None

    @pytest.mark.unit
    def test_is_dataclass(self) -> None:
        """PlatformComponents is a dataclass."""
        assert dataclasses.is_dataclass(PlatformComponents)

    @pytest.mark.unit
    def test_has_dataset_field(self) -> None:
        """PlatformComponents has a 'dataset' field."""
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "dataset" in field_names

    @pytest.mark.unit
    def test_has_executor_field(self) -> None:
        """PlatformComponents has an 'executor' field."""
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "executor" in field_names

    @pytest.mark.unit
    def test_has_evaluator_field(self) -> None:
        """PlatformComponents has an 'evaluator' field."""
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "evaluator" in field_names

    @pytest.mark.unit
    def test_has_context_field(self) -> None:
        """PlatformComponents has a 'context' field."""
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "context" in field_names

    @pytest.mark.unit
    def test_has_registry_field(self) -> None:
        """PlatformComponents has a 'registry' field."""
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "registry" in field_names

    @pytest.mark.unit
    def test_exactly_six_fields(self) -> None:
        """PlatformComponents has exactly 6 fields (including recorder)."""
        fields = dataclasses.fields(PlatformComponents)
        assert len(fields) == 6, (
            f"Expected 6 fields, got {len(fields)}: {[f.name for f in fields]}"
        )

    @pytest.mark.unit
    def test_no_solver_field(self) -> None:
        """PlatformComponents does NOT contain a solver field.

        Per /016: plugins bring their own solver.
        """
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "solver" not in field_names

    @pytest.mark.unit
    def test_instantiation_with_mocks(
        self, mock_components: PlatformComponents
    ) -> None:
        """PlatformComponents can be instantiated with all 5 fields."""
        assert mock_components.dataset is not None
        assert mock_components.executor is not None
        assert mock_components.evaluator is not None
        assert mock_components.context is not None
        assert mock_components.registry is not None

    @pytest.mark.unit
    def test_field_access(self, mock_components: PlatformComponents) -> None:
        """Required fields are accessible as attributes."""
        # Required fields (no default) should be non-None
        for field in dataclasses.fields(mock_components):
            value = getattr(mock_components, field.name)
            if (
                field.default is dataclasses.MISSING
                and field.default_factory is dataclasses.MISSING
            ):
                assert value is not None, (
                    f"Required field {field.name} should not be None"
                )


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 3: Behavioral / Property-based (~20%)
# ═══════════════════════════════════════════════════════════════════════════


class TestProtocolBehavior:
    """Tests for behavioral contracts of the SearchAlgorithm protocol."""

    @pytest.mark.unit
    def test_lifecycle_prepare_propose_evaluate_update(
        self,
        conforming_algorithm: _ConformingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """Full lifecycle: prepare -> propose -> evaluate -> update."""
        algo = conforming_algorithm

        # Step 1: prepare
        algo.prepare(mock_components)
        assert algo._prepared

        # Step 2: propose
        candidates = algo.propose(3)
        assert isinstance(candidates, list)
        assert len(candidates) == 3
        assert all(isinstance(c, str) for c in candidates)

        # Step 3: evaluate
        results = algo.evaluate(candidates)
        assert isinstance(results, list)
        assert len(results) == len(candidates)
        assert all(isinstance(r, EvaluationResult) for r in results)

        # Step 4: update
        algo.update(results)

    @pytest.mark.unit
    def test_propose_returns_list_of_strings(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """propose(n) must return a list of strings."""
        result = conforming_algorithm.propose(5)
        assert isinstance(result, list)
        assert all(isinstance(s, str) for s in result)

    @pytest.mark.unit
    def test_propose_n_controls_count(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """propose(n) returns a list; length is up to the algorithm.

        algorithms (e.g. SGA ) to return their full evaluated
        frontier — which can exceed ``n``. ``n`` is the requested batch
        size, not a hard cap. This test only locks the type contract;
        size is intentionally unconstrained.
        """
        for n in [1, 5, 10]:
            result = conforming_algorithm.propose(n)
            assert isinstance(result, list)
            assert len(result) >= 0

    @pytest.mark.unit
    def test_evaluate_returns_list_of_evaluation_result(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """evaluate() must return a list of EvaluationResult."""
        candidates = ["expr_a", "expr_b"]
        results = conforming_algorithm.evaluate(candidates)
        assert isinstance(results, list)
        assert all(isinstance(r, EvaluationResult) for r in results)

    @pytest.mark.unit
    def test_evaluate_result_count_matches_candidates(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """evaluate() returns one result per candidate."""
        for n in [1, 3, 5]:
            candidates = [f"e{i}" for i in range(n)]
            results = conforming_algorithm.evaluate(candidates)
            assert len(results) == len(candidates)

    @pytest.mark.unit
    def test_best_score_returns_float(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """best_score property returns a float."""
        score = conforming_algorithm.best_score
        assert isinstance(score, float)

    @pytest.mark.unit
    def test_best_expression_returns_str(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """best_expression property returns a string."""
        expr = conforming_algorithm.best_expression
        assert isinstance(expr, str)

    @pytest.mark.unit
    def test_state_getter_returns_dict(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """state getter must return a dict."""
        state = conforming_algorithm.state
        assert isinstance(state, dict)

    @pytest.mark.unit
    def test_state_roundtrip(self, conforming_algorithm: _ConformingAlgorithm) -> None:
        """State set then get must preserve values."""
        test_state = {
            "generation": 42,
            "population": ["a", "b", "c"],
            "scores": [0.1, 0.2, 0.3],
            "nested": {"key": "value"},
        }
        conforming_algorithm.state = test_state
        retrieved = conforming_algorithm.state
        assert retrieved == test_state

    @pytest.mark.unit
    def test_state_is_pickle_serializable(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """State dict must be pickle-serializable (for torch.save checkpoint)."""
        test_state = {
            "generation": 10,
            "best_mse": 0.001,
            "population": ["mul(u, u_x)", "u_xx"],
        }
        conforming_algorithm.state = test_state

        # Must survive pickle roundtrip
        state = conforming_algorithm.state
        pickled = pickle.dumps(state)
        restored = pickle.loads(pickled)
        assert restored == state

    @pytest.mark.unit
    def test_multiple_update_cycles(
        self,
        conforming_algorithm: _ConformingAlgorithm,
        mock_components: PlatformComponents,
    ) -> None:
        """Algorithm can handle multiple propose-evaluate-update cycles."""
        algo = conforming_algorithm
        algo.prepare(mock_components)

        for _ in range(3):
            candidates = algo.propose(2)
            results = algo.evaluate(candidates)
            algo.update(results)

        # After multiple cycles, best_score and best_expression should still
        # be valid types
        assert isinstance(algo.best_score, float)
        assert isinstance(algo.best_expression, str)


# ═══════════════════════════════════════════════════════════════════════════
# Test Group 4: Negative / Edge Cases (~20%)
# ═══════════════════════════════════════════════════════════════════════════


class TestNegativeCases:
    """Negative tests: missing members, wrong types, edge cases."""

    @pytest.mark.unit
    def test_empty_class_not_instance(self) -> None:
        """A completely empty class is not a SearchAlgorithm."""

        class Empty:
            pass

        assert not isinstance(Empty(), SearchAlgorithm)

    @pytest.mark.unit
    def test_partial_implementation_not_instance(self) -> None:
        """A class with only some methods is not a SearchAlgorithm."""

        class Partial:
            def prepare(self, components: PlatformComponents) -> None:
                pass

            def propose(self, n: int) -> list[str]:
                return []

        assert not isinstance(Partial(), SearchAlgorithm)

    @pytest.mark.unit
    def test_missing_state_setter_behavior(self) -> None:
        """A class without state setter has limited conformance.

        Note: runtime_checkable Protocol checks method/attribute existence
        but may not check setter. This test documents the expected behavior.
        """
        obj = _NoStateSetter()
        # The object has a state getter but no setter.
        # Whether this passes isinstance depends on Protocol implementation.
        # We document the contract: state must be settable.
        has_state_getter = hasattr(obj, "state")
        assert has_state_getter # getter exists

        # Setting state should raise if no setter
        with pytest.raises(AttributeError):
            obj.state = {"key": "value"} # type: ignore[misc]

    @pytest.mark.unit
    def test_propose_zero_returns_empty(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """propose(0) should return an empty list."""
        result = conforming_algorithm.propose(0)
        assert result == []

    @pytest.mark.unit
    def test_evaluate_empty_list(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """evaluate([]) should return an empty list."""
        result = conforming_algorithm.evaluate([])
        assert result == []

    @pytest.mark.unit
    def test_update_empty_results(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """update([]) should not raise."""
        # Should handle gracefully
        conforming_algorithm.update([])

    @pytest.mark.unit
    def test_platform_components_missing_field_raises(self) -> None:
        """PlatformComponents with missing fields raises TypeError."""
        with pytest.raises(TypeError):
            PlatformComponents( # type: ignore[call-arg]
                dataset=MagicMock(),
                executor=MagicMock(),
                # missing evaluator, context, registry
            )

    @pytest.mark.unit
    def test_platform_components_extra_field_raises(self) -> None:
        """PlatformComponents does not accept extra keyword arguments."""
        with pytest.raises(TypeError):
            PlatformComponents( # type: ignore[call-arg]
                dataset=MagicMock(),
                executor=MagicMock(),
                evaluator=MagicMock(),
                context=MagicMock(),
                registry=MagicMock(),
                recorder=None,
                solver=MagicMock(), # Not part of the spec!
            )

    @pytest.mark.unit
    def test_protocol_is_not_instantiable_directly(self) -> None:
        """SearchAlgorithm Protocol should not be directly instantiated.

        Protocols are abstract contracts. Direct instantiation is an error.
        """
        with pytest.raises(TypeError):
            SearchAlgorithm() # type: ignore[call-arg]

    @pytest.mark.unit
    def test_state_set_empty_dict(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """Setting state to empty dict should work."""
        conforming_algorithm.state = {}
        assert conforming_algorithm.state == {}

    @pytest.mark.unit
    def test_best_score_default_is_finite_or_inf(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """best_score should be a numeric float (possibly inf for initial)."""
        score = conforming_algorithm.best_score
        assert isinstance(score, float)
        # inf is valid for initial "no best yet" state

    @pytest.mark.unit
    def test_best_expression_default_is_string(
        self, conforming_algorithm: _ConformingAlgorithm
    ) -> None:
        """best_expression should be a string even before any update."""
        expr = conforming_algorithm.best_expression
        assert isinstance(expr, str)


# =========================================================================
# Test Group 5: PlatformComponents.recorder (P4-T1b)
# =========================================================================


class TestPlatformComponentsRecorder:
    """Tests for the new recorder field on PlatformComponents."""

    @pytest.mark.unit
    def test_has_recorder_field(self) -> None:
        """PlatformComponents has a 'recorder' field."""
        field_names = {f.name for f in dataclasses.fields(PlatformComponents)}
        assert "recorder" in field_names

    @pytest.mark.unit
    def test_recorder_defaults_to_none(self) -> None:
        """PlatformComponents.recorder defaults to None (backward compat).

        Existing code that constructs PlatformComponents without a recorder
        argument must still work.
        """
        pc = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        assert pc.recorder is None

    @pytest.mark.unit
    def test_recorder_accepts_viz_recorder(self) -> None:
        """PlatformComponents accepts a VizRecorder instance."""
        from kd2.search.recorder import VizRecorder

        recorder = VizRecorder()
        pc = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
            recorder=recorder,
        )
        assert pc.recorder is recorder

    @pytest.mark.unit
    def test_recorder_accepts_none_explicitly(self) -> None:
        """PlatformComponents(recorder=None) works explicitly."""
        pc = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
            recorder=None,
        )
        assert pc.recorder is None

    @pytest.mark.unit
    def test_backward_compat_no_recorder_arg(self) -> None:
        """Existing 5-field construction is backward compatible.

        All current call sites use PlatformComponents without recorder.
        This must not break.
        """
        # Positional construction of the 5 required fields
        pc = PlatformComponents(
            dataset=MagicMock(),
            executor=MagicMock(),
            evaluator=MagicMock(),
            context=MagicMock(),
            registry=MagicMock(),
        )
        assert pc.dataset is not None
        assert pc.recorder is None


# =========================================================================
# Test Group 6: SearchAlgorithm.config property (P4-T1b)
# =========================================================================


class _ConformingAlgorithmWithConfig(_ConformingAlgorithm):
    """Extended conforming algorithm that also implements config property."""

    @property
    def config(self) -> dict:
        return {"algorithm": "test", "param": 42}


class _MissingConfig:
    """Algorithm with all SearchAlgorithm methods EXCEPT config.

    Does NOT inherit from _ConformingAlgorithm to avoid inheriting config.
    """

    def prepare(self, components: Any) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    def evaluate(self, candidates: list[str]) -> list[Any]:
        return []

    def update(self, results: list[Any]) -> None:
        pass

    @property
    def best_score(self) -> float:
        return 0.0

    @property
    def best_expression(self) -> str:
        return ""

    @property
    def state(self) -> dict[str, Any]:
        return {}

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        pass


class TestSearchAlgorithmConfig:
    """Tests for the new config property on SearchAlgorithm protocol."""

    @pytest.mark.unit
    def test_protocol_has_config(self) -> None:
        """SearchAlgorithm Protocol defines a 'config' property."""
        assert hasattr(SearchAlgorithm, "config")

    @pytest.mark.unit
    def test_conforming_with_config_is_instance(self) -> None:
        """A class with config property passes isinstance(SearchAlgorithm)."""
        obj = _ConformingAlgorithmWithConfig()
        assert isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_config_returns_dict(self) -> None:
        """config property returns a dict."""
        obj = _ConformingAlgorithmWithConfig()
        cfg = obj.config
        assert isinstance(cfg, dict)

    @pytest.mark.unit
    def test_config_contains_keys(self) -> None:
        """config dict should contain recognizable configuration keys.

        The specific keys depend on the algorithm, but the dict must
        not be empty.
        """
        obj = _ConformingAlgorithmWithConfig()
        cfg = obj.config
        assert len(cfg) > 0

    @pytest.mark.unit
    def test_missing_config_fails_isinstance(self) -> None:
        """A class missing config property fails isinstance check.

        This verifies that config is part of the protocol contract.
        """
        obj = _MissingConfig()
        assert not isinstance(obj, SearchAlgorithm)


# =========================================================================
# Test Group 7: IterativeSearchAlgorithm sub-protocol ()
# =========================================================================


class _SearchOnlyAlgorithm(_ConformingAlgorithm):
    """Satisfies SearchAlgorithm but NOT IterativeSearchAlgorithm.

    Does not have a between_iterations() method.
    """

    pass


class _IterativeAlgorithm(_ConformingAlgorithm):
    """Satisfies both SearchAlgorithm and IterativeSearchAlgorithm.

    Has between_iterations() in addition to all SearchAlgorithm methods.
    """

    def __init__(self) -> None:
        super().__init__()
        self.between_calls: int = 0

    def between_iterations(self) -> None:
        self.between_calls += 1


class _BetweenOnlyAlgorithm:
    """Has between_iterations() but missing some SearchAlgorithm methods.

    This should NOT satisfy IterativeSearchAlgorithm since it doesn't
    satisfy the parent protocol (SearchAlgorithm).
    """

    def between_iterations(self) -> None:
        pass

    def prepare(self, components: PlatformComponents) -> None:
        pass

    def propose(self, n: int) -> list[str]:
        return []

    # Missing: evaluate, update, best_score, best_expression, config, state


class TestIterativeSearchAlgorithmProtocol:
    """Tests for IterativeSearchAlgorithm sub-protocol conformance.

    IterativeSearchAlgorithm extends SearchAlgorithm with a single
    additional method: between_iterations() -> None.
    It must be @runtime_checkable.
    """

    @pytest.mark.smoke
    def test_importable(self) -> None:
        """IterativeSearchAlgorithm can be imported from kd2.search.protocol."""
        from kd2.search.protocol import IterativeSearchAlgorithm

        assert IterativeSearchAlgorithm is not None

    @pytest.mark.smoke
    def test_importable_from_package(self) -> None:
        """IterativeSearchAlgorithm can be imported from kd2.search."""
        from kd2.search import IterativeSearchAlgorithm

        assert IterativeSearchAlgorithm is not None

    @pytest.mark.unit
    def test_is_runtime_checkable(self) -> None:
        """IterativeSearchAlgorithm supports isinstance() checks.

        If not @runtime_checkable, isinstance() raises TypeError.
        """
        from kd2.search.protocol import IterativeSearchAlgorithm

        obj = _IterativeAlgorithm()
        # Must not raise TypeError
        result = isinstance(obj, IterativeSearchAlgorithm)
        assert isinstance(result, bool)

    @pytest.mark.unit
    def test_search_only_not_iterative(self) -> None:
        """A class with only SearchAlgorithm methods is NOT IterativeSearchAlgorithm.

        This is the core discriminator: lacking between_iterations()
        means the algorithm should NOT match the sub-protocol.
        """
        from kd2.search.protocol import IterativeSearchAlgorithm

        obj = _SearchOnlyAlgorithm()
        assert isinstance(obj, SearchAlgorithm)
        assert not isinstance(obj, IterativeSearchAlgorithm)

    @pytest.mark.unit
    def test_iterative_algorithm_is_iterative(self) -> None:
        """A class implementing between_iterations() IS IterativeSearchAlgorithm."""
        from kd2.search.protocol import IterativeSearchAlgorithm

        obj = _IterativeAlgorithm()
        assert isinstance(obj, IterativeSearchAlgorithm)

    @pytest.mark.unit
    def test_iterative_algorithm_is_also_search(self) -> None:
        """IterativeSearchAlgorithm instances are also SearchAlgorithm instances.

        The sub-protocol inherits from SearchAlgorithm.
        """
        obj = _IterativeAlgorithm()
        assert isinstance(obj, SearchAlgorithm)

    @pytest.mark.unit
    def test_between_only_not_iterative(self) -> None:
        """Having between_iterations() without full SearchAlgorithm is not enough.

        IterativeSearchAlgorithm requires all SearchAlgorithm methods PLUS
        between_iterations(). Partial conformance should fail.
        """
        from kd2.search.protocol import IterativeSearchAlgorithm

        obj = _BetweenOnlyAlgorithm()
        # Missing evaluate, update, best_score, etc.
        assert not isinstance(obj, SearchAlgorithm)
        assert not isinstance(obj, IterativeSearchAlgorithm)

    @pytest.mark.unit
    def test_protocol_has_between_iterations_method(self) -> None:
        """The protocol class itself defines between_iterations."""
        from kd2.search.protocol import IterativeSearchAlgorithm

        assert hasattr(IterativeSearchAlgorithm, "between_iterations")

    @pytest.mark.unit
    def test_between_iterations_callable(self) -> None:
        """between_iterations() is callable on a conforming instance."""
        obj = _IterativeAlgorithm()
        assert callable(getattr(obj, "between_iterations", None))

    @pytest.mark.unit
    def test_between_iterations_returns_none(self) -> None:
        """between_iterations() returns None (side-effect only method)."""
        obj = _IterativeAlgorithm()
        result = obj.between_iterations()
        assert result is None

    @pytest.mark.unit
    def test_existing_conforming_algorithm_not_iterative(self) -> None:
        """The existing _ConformingAlgorithm is NOT IterativeSearchAlgorithm.

        Ensures backward compatibility: existing algorithms without
        between_iterations() are not affected.
        """
        from kd2.search.protocol import IterativeSearchAlgorithm

        obj = _ConformingAlgorithm()
        assert isinstance(obj, SearchAlgorithm)
        assert not isinstance(obj, IterativeSearchAlgorithm)

    @pytest.mark.unit
    def test_protocol_not_directly_instantiable(self) -> None:
        """IterativeSearchAlgorithm Protocol cannot be directly instantiated."""
        from kd2.search.protocol import IterativeSearchAlgorithm

        with pytest.raises(TypeError):
            IterativeSearchAlgorithm() # type: ignore[call-arg]
