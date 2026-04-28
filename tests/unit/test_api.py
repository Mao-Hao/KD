"""Tests for the high-level ``kd2.api.Model`` facade."""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import pytest
import torch

from kd2.api import Model
from kd2.data.schema import AxisInfo, FieldData, PDEDataset, TaskType
from kd2.data.synthetic import generate_burgers_data
from kd2.search.result import ExperimentResult
from kd2.search.sga import SGAConfig

# Test fixtures

# Tiny dataset: keeps each test under a few seconds.
_NX = 32
_NT = 16
_NU = 0.1
_FAST_GENERATIONS = 3
_FAST_POPULATION = 5


@pytest.fixture
def small_burgers_dataset():
    """Small Burgers dataset for fast smoke tests."""
    return generate_burgers_data(nx=_NX, nt=_NT, nu=_NU, seed=0)


def _fast_model(**overrides) -> Model:
    """Construct a Model with fast defaults for tests."""
    kwargs = {
        "algorithm": "sga",
        "generations": _FAST_GENERATIONS,
        "population": _FAST_POPULATION,
        "depth": 3,
        "width": 3,
        "seed": 0,
        "verbose": False,
    }
    kwargs.update(overrides)
    return Model(**kwargs)


# 1. Construction defaults


def test_model_init_defaults() -> None:
    """Constructing with defaults yields an unfit Model with friendly repr."""
    m = Model()
    rep = repr(m)
    assert "fitted=False" in rep
    assert "algorithm='sga'" in rep
    assert "generations=50" in rep


# 2. Pre-fit attribute access raises


def test_model_unfit_attribute_access() -> None:
    """Accessing ``best_expr_`` before ``.fit()`` raises a helpful error."""
    m = Model()
    with pytest.raises(RuntimeError) as exc_info:
        _ = m.best_expr_
    assert ".fit" in str(exc_info.value)

    with pytest.raises(RuntimeError):
        _ = m.best_score_

    with pytest.raises(RuntimeError):
        _ = m.result_

    with pytest.raises(RuntimeError):
        _ = m.algorithm_


# 3. fit returns self (chainable)


def test_model_fit_returns_self(small_burgers_dataset) -> None:
    """``.fit()`` returns ``self`` for sklearn-style chaining."""
    m = _fast_model()
    result = m.fit(small_burgers_dataset)
    assert result is m


# 4. End-to-end fit smoke test


def test_model_fit_burgers_smoke(small_burgers_dataset) -> None:
    """Fitting completes and exposes sane post-fit attributes."""
    m = _fast_model()
    m.fit(small_burgers_dataset)

    assert isinstance(m.best_expr_, str)
    assert m.best_expr_ # non-empty
    assert isinstance(m.best_score_, float)
    assert math.isfinite(m.best_score_)
    assert isinstance(m.result_, ExperimentResult)


# 5. repr after fit


def test_model_repr_after_fit(small_burgers_dataset) -> None:
    """After fit, repr shows ``fitted=True``, the discovered expression, and
    the user-facing algorithm name (``"sga"``) — not the plugin class name.
    """
    m = _fast_model()
    m.fit(small_burgers_dataset)
    rep = repr(m)
    assert "fitted=True" in rep
    assert m.best_expr_ in rep
    # repr must surface the user-facing algorithm name (from plugin.config),
    # not the plugin class name (e.g. "SGAPlugin") that lives on
    # ExperimentResult.algorithm_name.
    assert "'sga'" in rep
    assert "SGAPlugin" not in rep


# 6. Verbose stdout behavior


def test_model_verbose_silent(
    small_burgers_dataset, capsys: pytest.CaptureFixture[str]
) -> None:
    """``verbose=False`` is silent; ``verbose=True`` emits ``[kd2]`` lines."""
    m_silent = _fast_model(verbose=False)
    m_silent.fit(small_burgers_dataset)
    captured = capsys.readouterr()
    assert captured.out == ""

    m_loud = _fast_model(verbose=True)
    m_loud.fit(small_burgers_dataset)
    captured = capsys.readouterr()
    lines = [line for line in captured.out.splitlines() if "[kd2]" in line]
    assert len(lines) >= 1


# 7. Unsupported algorithm


def test_model_unsupported_algorithm(small_burgers_dataset) -> None:
    """Unknown algorithm names raise ``NotImplementedError`` mentioning the name."""
    m = Model(algorithm="dlga", verbose=False)
    with pytest.raises(NotImplementedError) as exc_info:
        m.fit(small_burgers_dataset)
    assert "dlga" in str(exc_info.value)


# 8. Config override


def test_model_config_override(small_burgers_dataset) -> None:
    """Passing a complete ``SGAConfig`` bypasses the builder and is used as-is."""
    custom_pop = 7
    cfg = SGAConfig(num=custom_pop, depth=3, width=3, seed=0)
    m = Model(algorithm="sga", generations=2, verbose=False, config=cfg)
    m.fit(small_burgers_dataset)
    # Verify the runner used the override config (population=7), not the
    # facade default of 20.
    assert m.result_.config["num"] == custom_pop


# 9. Autograd derivatives mode


def test_model_algorithm_attribute(small_burgers_dataset) -> None:
    """``model.algorithm_`` exposes the live plugin instance after fit."""
    from kd2.search.protocol import SearchAlgorithm

    m = _fast_model()
    m.fit(small_burgers_dataset)
    assert isinstance(m.algorithm_, SearchAlgorithm)
    # Plugin instance carries the same best-expression as the result.
    assert m.algorithm_.best_expression == m.best_expr_


# 10. Autograd derivatives mode


def test_model_autograd_derivatives(small_burgers_dataset) -> None:
    """``derivatives='autograd'`` forwards ``use_autograd=True`` to SGAConfig."""
    m = Model(
        algorithm="sga",
        generations=2,
        population=3,
        depth=3,
        width=3,
        seed=0,
        verbose=False,
        derivatives="autograd",
        autograd_train_epochs=10, # keep training tiny
    )
    m.fit(small_burgers_dataset)
    assert m.result_.config["use_autograd"] is True


def test_model_autograd_still_requires_uniform_grid() -> None:
    """``derivatives='autograd'`` does NOT bypass the uniform-grid check.

    The Model facade always builds a ``FiniteDiffProvider`` (Layer 3 tree
    FD + the Method-of-Lines integrator both need uniform spacing even
    when Layer 2 derivatives go through autograd). A non-uniform dataset
    must therefore raise at construction-time regardless of
    ``derivatives`` mode. This locks the docs/loading_your_data.md F2 fix:
    the docs no longer claim autograd lifts the uniform-grid requirement.
    """
    nx, nt = 32, 16
    # Quadratic spacing → strictly increasing but non-uniform.
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64) ** 2
    t = torch.linspace(0.0, 1.0, nt, dtype=torch.float64)
    u = torch.randn(nx, nt, dtype=torch.float64)
    nonuniform_dataset = PDEDataset(
        name="nonuniform_for_autograd_test",
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

    m = Model(
        algorithm="sga",
        generations=2,
        population=3,
        depth=3,
        width=3,
        seed=0,
        verbose=False,
        derivatives="autograd",
        autograd_train_epochs=5,
    )
    with pytest.raises(ValueError, match="non-uniform|uniform"):
        m.fit(nonuniform_dataset)


# 11. Strict kwargs validation (typo)


def test_unknown_kwarg_raises_typeerror() -> None:
    """Unknown kwargs (typos) raise ``TypeError`` with the bad name + valid set."""
    with pytest.raises(TypeError) as exc_info:
        Model(populaton=10, verbose=False) # typo: missing 'i'
    msg = str(exc_info.value)
    assert "populaton" in msg
    # Mentions at least one valid forwardable kwarg as guidance.
    assert "p_var" in msg or "p_mute" in msg or "lam" in msg


# 12. Strict kwargs validation (collision with explicit param)


def test_collision_kwarg_raises_typeerror() -> None:
    """Passing ``num=`` and ``population=`` together raises ``TypeError``."""
    with pytest.raises(TypeError) as exc_info:
        Model(population=20, num=99, verbose=False)
    msg = str(exc_info.value)
    assert "num" in msg
    assert "population" in msg


# 13. config= cannot mix with non-default facade params


def test_config_override_with_facade_param_raises() -> None:
    """``config=`` plus a non-default SGA-related facade param raises."""
    cfg = SGAConfig(num=7, depth=3, width=3, seed=0)
    with pytest.raises(ValueError) as exc_info:
        Model(config=cfg, depth=8, verbose=False)
    msg = str(exc_info.value)
    assert "depth" in msg
    assert "config" in msg


# 14. config= is deep-copied (no aliasing)


def test_config_deepcopy_isolates_mutation(small_burgers_dataset) -> None:
    """Mutating user's config after construction does not affect the fit."""
    cfg = SGAConfig(num=5, depth=3, width=3, seed=0)
    m = Model(algorithm="sga", generations=2, verbose=False, config=cfg)
    # Mutate the user-provided config BEFORE fit to verify deepcopy isolation.
    cfg.num = 999
    m.fit(small_burgers_dataset)
    # The runner should have used the value snapshotted at construction time
    # (5), not the mutated one (999).
    assert m.result_.config["num"] == 5


# 15. LHS field validation


def _make_velocity_dataset() -> PDEDataset:
    """Build a tiny dataset whose only field is ``velocity`` (not ``u``)."""
    nx, nt = 8, 6
    x = torch.linspace(0.0, 1.0, nx)
    t = torch.linspace(0.0, 1.0, nt)
    values = torch.zeros(nx, nt) + 0.1 # nonzero, finite
    return PDEDataset(
        name="velocity_test",
        task_type=TaskType.PDE,
        axes={"x": AxisInfo(name="x", values=x), "t": AxisInfo(name="t", values=t)},
        axis_order=["x", "t"],
        fields={"velocity": FieldData(name="velocity", values=values)},
        # lhs_field deliberately left empty so the facade default ("u") kicks in.
        lhs_field="",
        lhs_axis="t",
    )


def test_lhs_field_not_in_dataset_raises() -> None:
    """A dataset without field 'u' and unset ``lhs_field`` raises a clear error."""
    dataset = _make_velocity_dataset()
    m = Model(
        algorithm="sga",
        generations=2,
        population=3,
        depth=3,
        width=3,
        seed=0,
        verbose=False,
    )
    with pytest.raises(ValueError) as exc_info:
        m.fit(dataset)
    msg = str(exc_info.value)
    assert "velocity" in msg
    assert "u" in msg # mentions the missing default field


# 16. Invalid derivatives string


def test_invalid_derivatives_raises() -> None:
    """A typo in the ``derivatives`` value raises ``ValueError``."""
    with pytest.raises(ValueError) as exc_info:
        Model(derivatives="autogard", verbose=False) # typo
    msg = str(exc_info.value)
    assert "autogard" in msg
    assert "finite_diff" in msg
    assert "autograd" in msg


# 17. field_model without autograd is rejected


def test_field_model_without_autograd_raises() -> None:
    """Providing ``field_model`` while ``derivatives='finite_diff'`` is rejected."""

    class _DummyFieldModel:
        """Minimal stand-in; only its non-None identity matters."""

    with pytest.raises(ValueError) as exc_info:
        Model(
            derivatives="finite_diff",
            field_model=_DummyFieldModel(),
            verbose=False,
        )
    msg = str(exc_info.value)
    assert "field_model" in msg
    assert "autograd" in msg


# 18. callbacks parameter is forwarded to the runner


class _RecordingCallback:
    """Test-only callback that counts lifecycle invocations."""

    def __init__(self) -> None:
        self.start_calls = 0
        self.end_calls = 0
        self.iter_end_calls = 0

    @property
    def should_stop(self) -> bool:
        return False

    def on_experiment_start(self, algorithm: Any) -> None:
        self.start_calls += 1

    def on_iteration_start(self, iteration: int, algorithm: Any) -> None:
        pass

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: Any,
        candidates: list[str],
        results: list[Any],
    ) -> None:
        self.iter_end_calls += 1

    def on_experiment_end(self, algorithm: Any) -> None:
        self.end_calls += 1


def test_callbacks_parameter_appended(small_burgers_dataset) -> None:
    """Custom callbacks are invoked during fit alongside built-in ones."""
    cb = _RecordingCallback()
    m = _fast_model(callbacks=[cb])
    m.fit(small_burgers_dataset)
    assert cb.start_calls == 1
    assert cb.end_calls == 1
    assert cb.iter_end_calls >= 1


# 19. Failed fit resets state (no stale results)


def test_failed_fit_resets_state(small_burgers_dataset) -> None:
    """A second fit that fails clears stale results from a prior successful fit."""
    m = _fast_model()
    m.fit(small_burgers_dataset)
    assert isinstance(m.best_expr_, str) # first fit succeeded

    bad_dataset = _make_velocity_dataset() # triggers LHS validation error
    with pytest.raises(ValueError):
        m.fit(bad_dataset)

    # State must be reset; the OLD result from the successful fit must NOT
    # be silently returned.
    with pytest.raises(RuntimeError):
        _ = m.best_expr_
    with pytest.raises(RuntimeError):
        _ = m.result_
    with pytest.raises(RuntimeError):
        _ = m.algorithm_


# 20. repr after fit reads result.algorithm_name (immutable)


def test_repr_after_fit_uses_result_algorithm_name(small_burgers_dataset) -> None:
    """Post-fit repr survives mutation of ``model.algorithm`` attribute."""
    m = _fast_model()
    m.fit(small_burgers_dataset)
    # User accidentally clobbers the attribute (it's a public name).
    m.algorithm = "lying_value" # type: ignore[assignment]
    rep = repr(m)
    # repr should reflect what was actually run (sourced from the plugin's
    # frozen config via ExperimentResult.config["algorithm"]), not the lying
    # mutated attribute.
    assert "fitted=True" in rep
    assert "lying_value" not in rep
    expected_algo = m.result_.config.get("algorithm", m.result_.algorithm_name)
    assert f"'{expected_algo}'" in rep


# 21. config= conflict detection uses sentinels (not equality)


def test_explicit_default_with_config_still_raises() -> None:
    """``Model(config=cfg, population=20)`` rejects even though 20 == default.

    The exclusivity check distinguishes "user did not pass anything" from
    "user explicitly typed the literal default value". Equality-based
    detection (``population != _DEFAULT_POPULATION``) would silently use
    the config's value here; the sentinel-based check rejects it.
    """
    cfg = SGAConfig(num=99, depth=3, width=3, seed=0)
    # 20 is the literal default of the ``population`` facade param. The user
    # is being explicit, which conflicts with passing ``config=``.
    with pytest.raises(ValueError) as exc_info:
        Model(config=cfg, population=20, verbose=False)
    msg = str(exc_info.value)
    assert "population" in msg
    assert "config" in msg


# 22. callbacks shape is validated at construction


def test_invalid_callback_raises_typeerror() -> None:
    """Non-``RunnerCallback`` items in ``callbacks=`` raise at construction.

    Catching the misuse early avoids an opaque ``AttributeError`` mid-fit.
    """
    with pytest.raises(TypeError) as exc_info:
        Model(callbacks=[42], verbose=False) # type: ignore[list-item]
    msg = str(exc_info.value)
    assert "callbacks[0]" in msg
    assert "RunnerCallback" in msg


# 23. LHS fallback propagates to SGA
#
# Regression: previously the ``("u", "t")`` fallback in ``_build_components``
# was applied only to the local Evaluator; the original (empty) lhs metadata
# still reached ``SGAPlugin.prepare`` which raised
# ``ValueError("LHS field and axis must both be set for SGA")``. The fix
# rewrites the dataset via ``dataclasses.replace`` so all downstream
# consumers see the resolved values.


def _strip_lhs(dataset: PDEDataset, *, field: bool, axis: bool) -> PDEDataset:
    """Return a copy of ``dataset`` with selected lhs metadata cleared."""
    return dataclasses.replace(
        dataset,
        lhs_field="" if field else dataset.lhs_field,
        lhs_axis="" if axis else dataset.lhs_axis,
    )


def test_lhs_fallback_succeeds_when_both_empty(small_burgers_dataset) -> None:
    """Empty ``lhs_field`` AND ``lhs_axis`` falls back to ('u', 't') end-to-end.

    The dataset has field 'u' and axis 't', so the convention defaults
    apply. Fit must succeed all the way through SGA, not crash in
    ``SGAPlugin._extract_lhs_target``.
    """
    dataset = _strip_lhs(small_burgers_dataset, field=True, axis=True)
    assert dataset.lhs_field == ""
    assert dataset.lhs_axis == ""
    m = _fast_model()
    m.fit(dataset)
    assert m._fitted is True
    assert m.best_expr_ # non-empty discovered expression


def test_lhs_fallback_succeeds_when_only_field_empty(small_burgers_dataset) -> None:
    """Only ``lhs_field`` empty: fallback to 'u', existing axis kept."""
    dataset = _strip_lhs(small_burgers_dataset, field=True, axis=False)
    assert dataset.lhs_field == ""
    assert dataset.lhs_axis == "t"
    m = _fast_model()
    m.fit(dataset)
    assert m._fitted is True


def test_lhs_fallback_succeeds_when_only_axis_empty(small_burgers_dataset) -> None:
    """Only ``lhs_axis`` empty: fallback to 't', existing field kept."""
    dataset = _strip_lhs(small_burgers_dataset, field=False, axis=True)
    assert dataset.lhs_field == "u"
    assert dataset.lhs_axis == ""
    m = _fast_model()
    m.fit(dataset)
    assert m._fitted is True


def test_lhs_fallback_does_not_mutate_user_dataset(small_burgers_dataset) -> None:
    """The fallback must use ``dataclasses.replace`` (not in-place mutation)
    so the user's dataset reference keeps its original (empty) lhs metadata.
    """
    dataset = _strip_lhs(small_burgers_dataset, field=True, axis=True)
    m = _fast_model()
    m.fit(dataset)
    # User-side reference is untouched: the writeback created a new instance
    # for downstream, the original argument retains the empty fields.
    assert dataset.lhs_field == ""
    assert dataset.lhs_axis == ""


def test_lhs_fallback_writeback_reaches_plugin(small_burgers_dataset) -> None:
    """The plugin instance held on the Model sees the resolved LHS metadata.

    Verifies the fix's intent: the dataset propagated through
    ``PlatformComponents`` → ``SGAPlugin.prepare`` carries the resolved
    ``lhs_field='u'`` / ``lhs_axis='t'``, not the original empty strings.
    """
    dataset = _strip_lhs(small_burgers_dataset, field=True, axis=True)
    m = _fast_model()
    m.fit(dataset)
    plugin = m.algorithm_ # post-fit SGAPlugin
    # SGAPlugin keeps the dataset on the diff context after prepare.
    diff_ctx = plugin._diff_ctx # type: ignore[attr-defined]
    assert diff_ctx is not None
    assert diff_ctx.lhs_axis == "t"


# 24. Float32 dataset end-to-end via Model.fit
#
# Regression: ``FiniteDiffProvider`` rejected float32 grids at rtol=1e-6
# (rel deviation ~6e-5 at n=1000). The public ``Model.fit()`` instantiates
# the provider for any dataset, so float32 input — natural on Apple
# Silicon MPS where ``generate_burgers_data`` forces float32 — would
# crash before SGA ever ran. Now rtol=1e-4 matches integrator's relaxed
# check.


def _to_float32(dataset: PDEDataset) -> PDEDataset:
    """Return a copy of ``dataset`` with all axis/field tensors as float32."""
    assert dataset.axes is not None
    assert dataset.fields is not None
    new_axes = {
        name: dataclasses.replace(axis, values=axis.values.to(torch.float32))
        for name, axis in dataset.axes.items()
    }
    new_fields = {
        name: dataclasses.replace(field, values=field.values.to(torch.float32))
        for name, field in dataset.fields.items()
    }
    return dataclasses.replace(dataset, axes=new_axes, fields=new_fields)


def test_model_fit_with_float32_dataset(small_burgers_dataset) -> None:
    """``Model.fit`` succeeds on a float32 burgers dataset (Apple Silicon path)."""
    dataset = _to_float32(small_burgers_dataset)
    # Sanity check: every coord / field tensor really is float32 now.
    assert dataset.axes is not None
    for axis in dataset.axes.values():
        assert axis.values.dtype == torch.float32
    assert dataset.fields is not None
    for field in dataset.fields.values():
        assert field.values.dtype == torch.float32

    m = _fast_model()
    m.fit(dataset)
    assert m._fitted is True
    assert m.best_expr_


# 25. Top-level API exports


def test_vizengine_is_top_level_exported() -> None:
    """``kd2.VizEngine`` is the same class as ``kd2.viz.engine.VizEngine``."""
    import kd2
    from kd2.viz.engine import VizEngine

    assert kd2.VizEngine is VizEngine


def test_top_level_byod_exports_present() -> None:
    """BYOD ergonomics surface: PDEDataset, AxisInfo, FieldData, TaskType,
    DataTopology, preview, VizEngine and the four built-in loaders are all
    reachable from ``import kd2``.
    """
    import kd2

    expected = (
        "Model",
        "PDEDataset",
        "AxisInfo",
        "FieldData",
        "TaskType",
        "DataTopology",
        "VizEngine",
        "preview",
        "load_chafee_infante",
        "load_kdv",
        "load_pde_compound",
        "load_pde_divide",
        "generate_advection_data",
        "generate_burgers_data",
        "generate_diffusion_data",
        "ExperimentResult",
        "SGAConfig",
    )
    for symbol in expected:
        assert hasattr(kd2, symbol), f"kd2.{symbol} is missing"
        assert symbol in kd2.__all__, f"kd2.__all__ missing '{symbol}'"
