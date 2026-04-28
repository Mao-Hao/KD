"""High-level PySR-style facade for kd2.

Single-class entry point that wraps the standard wiring (executor +
evaluator + derivative provider + plugin + runner) into an sklearn-style
``Model`` with ``.fit(dataset)`` and post-fit ``best_expr_`` /
``best_score_`` / ``result_`` attributes.

Example:
    >>> import kd2
    >>> m = kd2.Model(algorithm="sga", generations=50, population=20)
    >>> m.fit(dataset)
    >>> print(m.best_expr_, m.best_score_)
"""

from __future__ import annotations

import copy
import dataclasses
from typing import TYPE_CHECKING, Any

from kd2.core.evaluator import Evaluator
from kd2.core.executor.context import ExecutionContext
from kd2.core.expr import FunctionRegistry, PythonExecutor
from kd2.core.linear_solve.least_squares import LeastSquaresSolver
from kd2.data.derivatives.finite_diff import FiniteDiffProvider
from kd2.search.callbacks import RunnerCallback
from kd2.search.protocol import PlatformComponents
from kd2.search.runner import ExperimentRunner
from kd2.search.sga import SGAConfig, SGAPlugin

if TYPE_CHECKING:
    from kd2.core.evaluator import EvaluationResult
    from kd2.data.schema import PDEDataset
    from kd2.search.protocol import SearchAlgorithm
    from kd2.search.result import ExperimentResult

__all__ = ["Model"]

# Constants

_SUPPORTED_ALGORITHMS = ("sga",)
_DEFAULT_LHS_FIELD = "u"
_DEFAULT_LHS_AXIS = "t"
# Order of the LHS time derivative the facade wires into the Evaluator.
# Hard-coded to 1 because SGAPlugin._extract_lhs_target also takes order=1
# from the dataset; changing one without the other would silently fit the
# wrong target. Higher-order LHS PDE (e.g. wave equation u_tt = ...) is
# NOT supported through this facade — see ``Model`` docstring "Limitations".
_FACADE_LHS_ORDER = 1
_PROGRESS_PREFIX = "[kd2]"
_FIT_REQUIRED_MSG = "Model has not been fit. Call .fit(dataset) first."

# Sentinel used to distinguish "user explicitly passed the default value" from
# "user did not pass anything" in ``__init__``. Equality-based detection (e.g.
# ``population == 20``) cannot tell those cases apart, so ``Model(config=cfg,
# population=20)`` would silently use the config's value while the user
# expected an error.
_UNSET: Any = object()

# Defaults for facade parameters; used to populate the resolved value when the
# corresponding ``__init__`` argument is left as ``_UNSET``. Keep in sync with
# the docstring (signatures use ``_UNSET`` so collision detection works).
_DEFAULT_POPULATION = 20
_DEFAULT_DEPTH = 4
_DEFAULT_WIDTH = 5
_DEFAULT_AIC_RATIO = 1.0
_DEFAULT_DERIVATIVES = "finite_diff"
_DEFAULT_SEED = 0

_VALID_DERIVATIVES = frozenset({"finite_diff", "autograd"})

# kwargs are forwarded to ``SGAConfig`` if the field exists. Some SGAConfig
# fields are explicitly mapped from facade parameters and would collide if
# also passed via kwargs.
_SGA_FIELDS = frozenset(f.name for f in dataclasses.fields(SGAConfig))
_EXPLICITLY_MAPPED = frozenset(
    {"num", "depth", "width", "aic_ratio", "seed", "use_autograd"}
)
_ALLOWED_KWARGS = _SGA_FIELDS - _EXPLICITLY_MAPPED

# Pretty-print mapping from SGAConfig field name → facade parameter name. Used
# in collision error messages so the user knows what facade parameter to use
# instead of the underlying config field.
_PRETTY_MAPPED = {"num": "population", "use_autograd": "derivatives"}


# Verbose progress callback


class _ProgressPrinter:
    """RunnerCallback that prints per-iteration progress to stdout.

    Matches the ``[kd2] Generation N/M | best AIC=... | expr=...`` format.
    Emits a final ``[kd2] Done.`` line on experiment end.
    """

    def __init__(self, total_generations: int) -> None:
        self._total = total_generations

    @property
    def should_stop(self) -> bool: # noqa: D401 - protocol property
        """Never requests stopping."""
        return False

    def on_experiment_start(self, algorithm: SearchAlgorithm) -> None:
        """No-op."""

    def on_iteration_start(self, iteration: int, algorithm: SearchAlgorithm) -> None:
        """No-op."""

    def on_iteration_end(
        self,
        iteration: int,
        algorithm: SearchAlgorithm,
        candidates: list[str],
        results: list[EvaluationResult],
    ) -> None:
        """Print one progress line per iteration."""
        gen = iteration + 1
        print(
            f"{_PROGRESS_PREFIX} Generation {gen:>3}/{self._total} | "
            f"best AIC={algorithm.best_score:.4g} | "
            f"expr={algorithm.best_expression}"
        )

    def on_experiment_end(self, algorithm: SearchAlgorithm) -> None:
        """Print final summary line."""
        print(
            f"{_PROGRESS_PREFIX} Done. Best: {algorithm.best_expression} "
            f"(AIC={algorithm.best_score:.4g})"
        )


# Public Model facade


class Model:
    """High-level facade for PDE discovery (PySR-style API).

    The facade wraps the ``ExperimentRunner`` + plugin + components stack
    into a single class with sklearn-style ``.fit()`` and trailing-underscore
    post-fit attributes (``best_expr_``, ``best_score_``, ``result_``).

    Limitations:
        Only **first-order LHS** PDE is supported (e.g., ``u_t = f(u, u_x, ...)``).
        Higher-order LHS such as the wave equation ``u_tt = c**2 * u_xx`` is
        NOT supported by this facade — both the LHS target inside SGA and the
        Evaluator are wired with ``order=1`` (see ``_FACADE_LHS_ORDER``). To
        discover a second-order-in-time PDE, reduce it to a first-order
        system manually (introduce ``v = u_t``, then discover ``u_t = v`` and
        ``v_t = ...`` separately).

    Args:
        algorithm: Search algorithm name. Only ``"sga"`` is supported in v0.1.
        generations: Maximum number of search iterations.
        population: SGA population size (number of PDE candidates).
        depth: Maximum tree depth per term.
        width: Maximum number of terms per PDE.
        aic_ratio: AIC penalty ratio.
        derivatives: Derivative provider mode: ``"finite_diff"`` or
            ``"autograd"`` (forwards ``use_autograd=True`` to ``SGAConfig``).
            When ``"autograd"``, the SGA terminal derivatives (u_x, u_t) are
            routed through an autograd-trained surrogate, while the dataset's
            finite-difference provider remains in place for tree-internal
            derivative operators.
        seed: Random seed for reproducibility.
        verbose: When True, print per-iteration progress to stdout.
        config: Optional pre-built ``SGAConfig``. When provided, it is the
            single source of SGA settings; passing any non-default
            SGA-related facade parameters (``population``, ``depth``,
            ``width``, ``aic_ratio``, ``derivatives``, ``seed``) or extra
            ``kwargs`` raises ``ValueError``. Only ``algorithm``,
            ``generations``, ``verbose``, and ``callbacks`` remain effective
            on the facade. The config is deep-copied to prevent aliasing.
        callbacks: Optional list of additional ``RunnerCallback`` instances
            to attach to the runner. The verbose progress printer is appended
            automatically when ``verbose=True``. The list is shallow-copied
            so later ``.append()`` on the user's list cannot leak in, but
            the callback instances themselves are NOT deep-copied: reusing
            the same callback across fits will retain state from previous
            fits. This is intentional (some callbacks integrate over runs)
            but means stateful callbacks should be reset between fits if a
            clean slate is desired.
        **kwargs: Forwarded to ``SGAConfig`` if the field name matches an
            allowed (non-explicitly-mapped) field. Unknown or colliding
            kwargs raise ``TypeError``.
    """

    def __init__(
        self,
        algorithm: str = "sga",
        generations: int = 50,
        population: int = _UNSET,
        depth: int = _UNSET,
        width: int = _UNSET,
        aic_ratio: float = _UNSET,
        derivatives: str = _UNSET,
        seed: int = _UNSET,
        verbose: bool = True,
        config: SGAConfig | None = None,
        callbacks: list[RunnerCallback] | None = None,
        **kwargs: Any,
    ) -> None:
        # Resolve sentinels to defaults BEFORE the value-based validators so
        # they see real values, while keeping the original ``_UNSET`` markers
        # for ``_validate_config_exclusivity`` (which must distinguish "user
        # passed the literal default" from "user did not pass anything").
        population_resolved = (
            _DEFAULT_POPULATION if population is _UNSET else population
        )
        depth_resolved = _DEFAULT_DEPTH if depth is _UNSET else depth
        width_resolved = _DEFAULT_WIDTH if width is _UNSET else width
        aic_ratio_resolved = _DEFAULT_AIC_RATIO if aic_ratio is _UNSET else aic_ratio
        derivatives_resolved = (
            _DEFAULT_DERIVATIVES if derivatives is _UNSET else derivatives
        )
        seed_resolved = _DEFAULT_SEED if seed is _UNSET else seed

        self._validate_derivatives(derivatives_resolved)
        self._validate_kwargs(kwargs)
        self._validate_field_model(kwargs, derivatives_resolved)
        # Pass the SENTINEL-bearing values (not the resolved ones) so the
        # exclusivity check can distinguish "default" from "explicit default".
        self._validate_config_exclusivity(
            config, population, depth, width, aic_ratio, derivatives, seed, kwargs
        )
        self._validate_callbacks(callbacks)

        self.algorithm = algorithm
        self.generations = generations
        self.population = population_resolved
        self.depth = depth_resolved
        self.width = width_resolved
        self.aic_ratio = aic_ratio_resolved
        self.derivatives = derivatives_resolved
        self.seed = seed_resolved
        self.verbose = verbose
        # Deep-copy the user's config so that later mutations on the user's
        # object cannot leak into this Model's fit. The deepcopy at fit-time
        # in ``_build_config`` is a second line of defense; snapshotting here
        # makes the contract "value taken at construction" instead of
        # "value taken at fit".
        self._config_override = copy.deepcopy(config) if config is not None else None
        # Shallow-copy the callbacks list so external ``.append`` on the
        # user's list cannot leak into future fits. Individual callback
        # instances are still aliased — see class docstring.
        self._user_callbacks: list[RunnerCallback] | None = (
            list(callbacks) if callbacks is not None else None
        )
        self._extra_kwargs: dict[str, Any] = kwargs

        # Post-fit state (sklearn convention: trailing _ set after fit).
        self._fitted: bool = False
        self._result: ExperimentResult | None = None
        self._algorithm: SearchAlgorithm | None = None

    # -- Construction-time validators ----------------------------------------

    @staticmethod
    def _validate_derivatives(derivatives: str) -> None:
        """Reject ``derivatives`` values outside the supported whitelist."""
        if derivatives not in _VALID_DERIVATIVES:
            raise ValueError(
                f"Invalid derivatives='{derivatives}'. "
                f"Must be one of {sorted(_VALID_DERIVATIVES)}."
            )

    @staticmethod
    def _validate_kwargs(kwargs: dict[str, Any]) -> None:
        """Reject unknown kwargs and kwargs that collide with facade params."""
        unknown = set(kwargs) - _SGA_FIELDS
        if unknown:
            raise TypeError(
                f"Unknown keyword arguments for kd2.Model: {sorted(unknown)}. "
                f"Valid extra kwargs (forwarded to SGAConfig): "
                f"{sorted(_ALLOWED_KWARGS)}"
            )
        collision = set(kwargs) & _EXPLICITLY_MAPPED
        if collision:
            hints = ", ".join(
                f"{k} (use '{_PRETTY_MAPPED.get(k, k)}=' instead)"
                for k in sorted(collision)
            )
            raise TypeError(
                f"Keyword arguments collide with explicit facade parameters: {hints}"
            )

    @staticmethod
    def _validate_field_model(kwargs: dict[str, Any], derivatives: str) -> None:
        """Require ``derivatives='autograd'`` when a ``field_model`` is given."""
        if kwargs.get("field_model") is not None and derivatives != "autograd":
            raise ValueError(
                "Pre-trained 'field_model' was provided but "
                "derivatives='finite_diff'. Set derivatives='autograd' to "
                "use the field model."
            )

    @staticmethod
    def _validate_config_exclusivity(
        config: SGAConfig | None,
        population: Any,
        depth: Any,
        width: Any,
        aic_ratio: Any,
        derivatives: Any,
        seed: Any,
        kwargs: dict[str, Any],
    ) -> None:
        """Reject mixing ``config=`` with overlapping SGA-related parameters.

        Receives the SENTINEL-bearing values from ``__init__`` so that
        ``Model(config=cfg, population=20)`` (where 20 happens to equal the
        default) is rejected — the user-set check uses ``is _UNSET`` rather
        than equality to the default value.
        """
        if config is None:
            return
        overrides: list[str] = []
        if population is not _UNSET:
            overrides.append("population")
        if depth is not _UNSET:
            overrides.append("depth")
        if width is not _UNSET:
            overrides.append("width")
        if aic_ratio is not _UNSET:
            overrides.append("aic_ratio")
        if derivatives is not _UNSET:
            overrides.append("derivatives")
        if seed is not _UNSET:
            overrides.append("seed")
        if kwargs:
            overrides.append(f"kwargs={sorted(kwargs)}")
        if overrides:
            raise ValueError(
                f"Cannot pass both 'config=' and SGA-related parameters: "
                f"{overrides}. When 'config' is provided, use it as the "
                f"single source of SGA settings; only 'generations', "
                f"'verbose', 'callbacks' remain effective on the facade."
            )

    @staticmethod
    def _validate_callbacks(callbacks: list[RunnerCallback] | None) -> None:
        """Reject callbacks that are not ``RunnerCallback`` instances.

        Catches the misuse early at construction time instead of crashing
        mid-fit with an opaque ``AttributeError`` from inside the runner.
        """
        if callbacks is None:
            return
        for i, cb in enumerate(callbacks):
            if not isinstance(cb, RunnerCallback):
                raise TypeError(
                    f"callbacks[{i}] is not a RunnerCallback (got {type(cb).__name__})"
                )

    # -- Public API ----------------------------------------------------------

    def fit(self, dataset: PDEDataset) -> Model:
        """Run the search and populate post-fit attributes.

        Args:
            dataset: The PDE dataset to discover an equation for.

        Returns:
            ``self`` for sklearn-style chaining.

        Raises:
            NotImplementedError: If ``algorithm`` is not supported.
            ValueError: If the dataset is missing the required LHS field
                or axis.
        """
        # Reset post-fit state up-front so that a failed fit cannot leave
        # stale results from a prior successful fit accessible.
        self._fitted = False
        self._result = None
        self._algorithm = None

        if self.algorithm not in _SUPPORTED_ALGORITHMS:
            raise NotImplementedError(
                f"Algorithm '{self.algorithm}' is not implemented. "
                f"Supported algorithms: {list(_SUPPORTED_ALGORITHMS)}"
            )

        config = self._build_config()
        components = self._build_components(dataset)
        plugin = SGAPlugin(config)
        runner = ExperimentRunner(
            algorithm=plugin,
            max_iterations=self.generations,
            batch_size=config.num,
            callbacks=self._build_callbacks(),
        )
        self._result = runner.run(components)
        self._algorithm = plugin
        self._fitted = True
        return self

    @property
    def best_expr_(self) -> str:
        """Best discovered expression string (post-fit only)."""
        return self._require_result().best_expression

    @property
    def best_score_(self) -> float:
        """Best score (AIC, lower is better) (post-fit only)."""
        return self._require_result().best_score

    @property
    def result_(self) -> ExperimentResult:
        """Full ``ExperimentResult`` from the last fit (post-fit only)."""
        return self._require_result()

    @property
    def algorithm_(self) -> SearchAlgorithm:
        """The fitted ``SearchAlgorithm`` plugin instance (post-fit only).

        Exposed for advanced workflows such as ``VizEngine.render_all``,
        which needs the live plugin to render plugin-specific viz.
        """
        self._check_fitted()
        if self._algorithm is None:
            raise RuntimeError(_FIT_REQUIRED_MSG)
        return self._algorithm

    def __repr__(self) -> str:
        """Friendly representation of the model state."""
        if not self._fitted or self._result is None:
            return (
                f"Model(algorithm={self.algorithm!r}, "
                f"generations={self.generations}, fitted=False)"
            )
        # Read algorithm name from the plugin's frozen config so post-fit
        # mutation of ``self.algorithm`` cannot lie about what was actually
        # run. Prefer the user-facing name (e.g. ``"sga"``) baked into the
        # plugin config; fall back to the class-name in ``algorithm_name``
        # (e.g. ``"SGAPlugin"``) when a plugin omits the key.
        algo_label = self._result.config.get("algorithm", self._result.algorithm_name)
        return (
            f"Model(algorithm={algo_label!r}, fitted=True, "
            f"best_expr={self._result.best_expression!r}, "
            f"best_score={self._result.best_score:.4g})"
        )

    # -- Private helpers -----------------------------------------------------

    def _check_fitted(self) -> None:
        """Raise ``RuntimeError`` if ``.fit()`` has not been called."""
        if not self._fitted:
            raise RuntimeError(_FIT_REQUIRED_MSG)

    def _require_result(self) -> ExperimentResult:
        """Return ``self._result`` after asserting fitted state.

        Used by post-fit properties so mypy can narrow ``Optional`` and so the
        ``None`` check also survives ``python -O`` (unlike a bare ``assert``).
        """
        self._check_fitted()
        if self._result is None: # defensive — _check_fitted guards this
            raise RuntimeError(_FIT_REQUIRED_MSG)
        return self._result

    def _build_config(self) -> SGAConfig:
        """Construct an ``SGAConfig`` from the constructor arguments.

        If a ``config`` override was provided, return a deep copy so that
        later mutations of the user's config object do not affect the fit.
        Otherwise, forward only kwargs that match real ``SGAConfig`` fields
        (validated up-front in ``__init__``).
        """
        if self._config_override is not None:
            return copy.deepcopy(self._config_override)

        # All kwargs were validated in __init__ to be valid SGAConfig fields
        # and to not collide with explicitly-mapped facade params.
        return SGAConfig(
            num=self.population,
            depth=self.depth,
            width=self.width,
            aic_ratio=self.aic_ratio,
            seed=self.seed,
            use_autograd=(self.derivatives == "autograd"),
            **self._extra_kwargs,
        )

    def _build_components(self, dataset: PDEDataset) -> PlatformComponents:
        """Wire up the platform stack for the given dataset.

        Auto-detects the LHS target from ``dataset.lhs_field`` /
        ``dataset.lhs_axis``, falling back to ``("u", "t")`` when either is
        unset. The resolved values are written back to a new dataset
        instance via ``dataclasses.replace`` so downstream consumers
        (notably ``SGAPlugin.prepare``) see the same LHS target as the
        local ``Evaluator`` — without the writeback, SGA receives the
        original empty fields and raises ``ValueError``.

        The fallback is a convenience for the common single-field
        time-evolution case (field ``u``, time axis ``t``). For datasets
        with non-conventional naming (e.g. field ``v`` or axis ``tau``),
        set ``lhs_field`` / ``lhs_axis`` explicitly on the dataset.
        """
        lhs_field = dataset.lhs_field or _DEFAULT_LHS_FIELD
        lhs_axis = dataset.lhs_axis or _DEFAULT_LHS_AXIS

        # Validate the LHS resolves to real entries on the dataset, with a
        # clear error message instead of a cryptic KeyError downstream.
        field_names = set(dataset.fields.keys()) if dataset.fields else set()
        axis_names = set(dataset.axes.keys()) if dataset.axes else set()
        if lhs_field not in field_names:
            raise ValueError(
                f"LHS field '{lhs_field}' not found in dataset (available "
                f"fields: {sorted(field_names)}). Set dataset.lhs_field "
                f"explicitly."
            )
        if lhs_axis not in axis_names:
            raise ValueError(
                f"LHS axis '{lhs_axis}' not found in dataset (available "
                f"axes: {sorted(axis_names)}). Set dataset.lhs_axis "
                f"explicitly."
            )

        # Propagate resolved LHS to all downstream consumers by snapshotting
        # the dataset with the filled-in metadata. SGAPlugin.prepare reads
        # dataset.lhs_field / dataset.lhs_axis directly and rejects empty
        # values; the local resolution above only services the Evaluator.
        if dataset.lhs_field != lhs_field or dataset.lhs_axis != lhs_axis:
            dataset = dataclasses.replace(
                dataset, lhs_field=lhs_field, lhs_axis=lhs_axis
            )

        provider = FiniteDiffProvider(dataset, max_order=2)
        context = ExecutionContext(dataset=dataset, derivative_provider=provider)
        registry = FunctionRegistry.create_default()
        executor = PythonExecutor(registry)
        solver = LeastSquaresSolver()
        # Facade contract: first-order LHS only. Must match
        # SGAPlugin._extract_lhs_target which also uses order=1; see the
        # ``Limitations`` section of ``Model``'s docstring.
        lhs = provider.get_derivative(lhs_field, lhs_axis, _FACADE_LHS_ORDER).flatten()
        evaluator = Evaluator(
            executor=executor,
            solver=solver,
            context=context,
            lhs=lhs,
        )
        return PlatformComponents(
            dataset=dataset,
            executor=executor,
            evaluator=evaluator,
            context=context,
            registry=registry,
        )

    def _build_callbacks(self) -> list[RunnerCallback]:
        """Return the runner callback list.

        Always starts with any user-provided callbacks (in order). Appends
        ``_ProgressPrinter`` when ``verbose=True``.
        """
        cbs: list[RunnerCallback] = list(self._user_callbacks or [])
        if self.verbose:
            cbs.append(_ProgressPrinter(total_generations=self.generations))
        return cbs
