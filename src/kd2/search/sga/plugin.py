"""SGA search algorithm plugin implementing the SearchAlgorithm protocol.

Maps one SGA generation to one Runner iteration:
- prepare(): extract data from PlatformComponents, init population
- propose(n): genetic ops (crossover -> mutate -> replace), return candidates
- evaluate(candidates): score each using internal pipeline
- update(results): merge + sort + truncate population
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import Tensor

from kd2.core.evaluator import EvaluationResult
from kd2.core.metrics import nmse as metrics_nmse
from kd2.data.derivatives.autograd import AutogradProvider
from kd2.data.derivatives.finite_diff import DX_ZERO_FLOOR, UNIFORM_GRID_RTOL
from kd2.models.field_model import FieldModel
from kd2.models.trainer import FieldModelTrainer
from kd2.search.protocol import PlatformComponents
from kd2.search.sga.config import OPS, ROOT, SGAConfig, build_den
from kd2.search.sga.convert import pde_to_kd2_expr, tree_to_kd2_expr
from kd2.search.sga.evaluate import DiffContext, build_theta, execute_pde
from kd2.search.sga.pde import PDE
from kd2.search.sga.train import CandidateResult, TrainResult, evaluate_candidate

logger = logging.getLogger(__name__)

_INVALID_AIC = float("inf")
"""AIC penalty for invalid results."""

_AIC_LOWER_BOUND = -100.0
"""AIC values below this threshold are rejected as pathological (the predecessor alignment)."""

_MAX_RESAMPLE_PER_INDIVIDUAL = 50
"""Maximum resample attempts per pathological individual during init."""

_R2_EPS = 1e-15
"""Threshold for treating the target variance as zero in R² computation."""


@dataclass
class _ScoredPDE:
    """A PDE after internal SGA evaluation, ready for selection."""

    pde: PDE
    score: float
    result: EvaluationResult


def _is_valid_aic(aic: float) -> bool:
    """Check whether an AIC score is valid (finite and above lower bound).

    the predecessor rejects ``a_err < -100`` as numerical artifacts from overfitting
    or degenerate data. Uses strict less-than: -100.0 exactly is valid.
    """
    return math.isfinite(aic) and aic >= _AIC_LOWER_BOUND


def _product(shape: tuple[int, ...]) -> int:
    """Compute product of shape dimensions."""
    result = 1
    for s in shape:
        result *= s
    return result


def _rand_float(rng: torch.Generator) -> float:
    """Return a random float in [0, 1) using the generator."""
    return float(torch.rand(1, generator=rng).item())


def _numeric_flatten(tensor: Tensor) -> Tensor:
    """Return a detached flat numeric view for SGA storage.

    SGA's non-autograd search must not retain autograd graphs across
    iterations (causes memory blow-up + accidental in-place mutation
    hazards). All numeric storage sites in SGAPlugin route through this
    helper to ensure detachment.
    """
    return tensor.detach().flatten()


def _safe_evaluate_aic(
    pde: PDE,
    data_dict: dict[str, Tensor],
    default_terms: Tensor | None,
    y: Tensor | None,
    config: SGAConfig,
    diff_ctx: DiffContext | None,
) -> tuple[float, PDE]:
    """Evaluate a PDE candidate, returning (AIC, pruned_pde) or (inf, pde) on error."""
    try:
        cr = evaluate_candidate(
            pde,
            data_dict,
            default_terms,
            y if y is not None else torch.zeros(1),
            config,
            diff_ctx=diff_ctx,
        )
        return cr.aic_score, cr.pruned_pde
    except Exception:
        logger.debug("Evaluation failed for PDE, assigning inf AIC")
        return _INVALID_AIC, pde


class SGAPlugin:
    """SGA search algorithm implementing SearchAlgorithm protocol."""

    def __init__(self, config: SGAConfig | None = None) -> None:
        self._config = config or SGAConfig()
        self._population: list[PDE] | None = None
        self._scores: list[float] | None = None
        self._best_score: float = float("inf")
        self._best_expression: str = ""
        self._best_formatted_cache: str | None = None
        """Lazy-rendered ``best_expression`` (with coefficients). Reset on
        every best-of-generation update so the next read recomputes."""
        self._vars: list[str] = []
        self._data_dict: dict[str, Tensor] = {}
        self._den: tuple[tuple[str, int], ...] = ()
        self._diff_ctx: DiffContext | None = None
        self._default_terms: Tensor | None = None
        self._default_term_name: str | None = None
        self._y: Tensor | None = None
        self._rng: torch.Generator = torch.Generator()
        self._prepared: bool = False
        self._offspring: list[PDE] | None = None
        self._offspring_results: list[EvaluationResult] | None = None
        self._pending_population: list[PDE] | None = None
        self._pending_scores: list[float] | None = None
        self._recorder: Any | None = None
        self._autograd_provider: AutogradProvider | None = None
        """Internal AutogradProvider for Layer 2 terminals when use_autograd=True.
        Does NOT replace components.context.derivative_provider"""

    @property
    def _delta(self) -> dict[str, float]:
        """Compatibility view of the prepared grid spacing map."""
        if self._diff_ctx is None:
            return {}
        return self._diff_ctx.delta

    @property
    def _lhs_axis(self) -> str | None:
        """Compatibility view of the prepared lhs axis."""
        if self._diff_ctx is None:
            return None
        return self._diff_ctx.lhs_axis

    @property
    def config(self) -> dict[str, Any]:
        """Return the plugin configuration as a plain dictionary."""
        return {
            "algorithm": "sga",
            **asdict(self._config),
        }

    # -- Protocol methods -------------------------------------------------------

    def prepare(self, components: PlatformComponents) -> None:
        """Initialize algorithm with platform components."""
        self._clear_pending_generation()
        dataset = components.dataset
        context = components.context
        self._recorder = components.recorder

        # T5: fail-fast naming guardrails (before building data_dict)
        self._validate_naming(dataset)

        # Determine field shape for broadcasting coordinates
        field_shape: tuple[int, ...] | None = None
        if dataset.fields is not None:
            for fd in dataset.fields.values():
                field_shape = tuple(fd.values.shape)
                break

        # Build data_dict -- all entries must have same flat size
        data_dict: dict[str, Tensor] = {}

        if dataset.fields is not None:
            for field_name in dataset.fields:
                data_dict[field_name] = _numeric_flatten(
                    context.get_variable(field_name)
                )

        axis_names = self._ordered_axes(dataset)
        self._den = build_den(axis_names, dataset.lhs_axis or "")
        delta = self._build_delta_map(dataset, axis_names)
        if field_shape is not None:
            axis_map = {axis_name: idx for idx, axis_name in enumerate(axis_names)}
            self._diff_ctx = DiffContext(
                field_shape=field_shape,
                axis_map=axis_map,
                delta=delta,
                lhs_axis=dataset.lhs_axis,
            )
        else:
            self._diff_ctx = None

        # Coordinates: broadcast to field shape before flattening
        # Exclude lhs_axis (e.g. "t") -- only spatial coordinates belong in VARS
        # (the predecessor: _build_vars skips axis == lhs_axis + _assert_no_lhs_in_vars)
        n_flat = _product(field_shape) if field_shape is not None else 0
        if dataset.axes is not None and field_shape is not None:
            for axis_name in dataset.axes:
                if axis_name == dataset.lhs_axis:
                    continue
                coord = context.get_variable(axis_name)
                if coord.numel() == n_flat:
                    data_dict[axis_name] = _numeric_flatten(coord)
                else:
                    broadcast = self._broadcast_coord(
                        coord, axis_name, dataset, field_shape
                    )
                    data_dict[axis_name] = _numeric_flatten(broadcast)

        # build internal AutogradProvider for Layer 2 if requested.
        # Reset on every prepare() so re-prepare doesn't reuse stale surrogate.
        self._autograd_provider = None
        if self._config.use_autograd:
            if field_shape is None:
                raise ValueError(
                    "use_autograd=True requires dataset.fields with grid shape."
                )
            self._autograd_provider = self._build_autograd_provider(
                dataset, field_shape
            )

        self._add_derivatives(dataset, context, data_dict)
        self._data_dict = data_dict

        # VARS = all data keys (lhs_axis derivatives already excluded above)
        self._vars = sorted(data_dict.keys())

        # y = LHS derivative (e.g. u_t), computed separately from VARS
        # the predecessor: self.ut = target_axis_grads[self.lhs_axis] (stored outside VARS)
        self._y = self._extract_lhs_target(dataset, context)

        # Default terms: the field itself as a single column
        if dataset.lhs_field and dataset.lhs_field in data_dict:
            self._default_terms = data_dict[dataset.lhs_field].flatten().unsqueeze(1)
            self._default_term_name = dataset.lhs_field
        else:
            self._default_terms = None
            self._default_term_name = None

        # Only seed RNG for fresh runs; checkpoint restore already set RNG state
        if self._population is None:
            if not self._vars:
                raise ValueError(
                    "No variables available for SGA. "
                    "Check dataset fields, axes, and lhs configuration."
                )
            self._rng.manual_seed(self._config.seed)
            self._init_population()

        self._prepared = True

    def propose(self, n: int) -> list[str]:
        """Propose candidate expressions via one staged SGA generation.

        Internally this mirrors the predecessor:
        crossover -> evaluate -> truncate -> mutate/replace -> evaluate -> truncate.
        The returned list is the complete evaluated offspring frontier for the
        protocol/callback surface; :meth:`update` commits the already-selected
        pending generation.

        Raises:
            RuntimeError: If prepare() has not been called.
            ValueError: If n < 1.
        """
        if not self._prepared:
            raise RuntimeError("prepare() must be called before propose()")
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")

        offspring = self._apply_genetic_ops()
        # Expose every evaluated offspring to the protocol surface. Internal
        # stage selection has already happened on local pending population.
        self._offspring = offspring
        return [pde_to_kd2_expr(pde) for pde in offspring]

    def evaluate(self, candidates: list[str]) -> list[EvaluationResult]:
        """Return evaluation results for proposed candidate expressions.

        Normal staged generations are evaluated inside :meth:`propose`, so this
        method returns cached results. The direct evaluation path remains as a
        compatibility fallback for tests and legacy private callers.

        Caller contract: ``candidates`` must be the list returned by the most
        recent :meth:`propose` call (same length, same order). Mismatched or
        post-:meth:`update` candidates fall back to ``invalid_result`` per
        position rather than raising — this is a lenient fallback for
        debug/REPL use, not a sanctioned reuse pattern.

        Raises:
            RuntimeError: If prepare() has not been called.
        """
        if not self._prepared:
            raise RuntimeError("prepare() must be called before evaluate()")
        if not candidates:
            return []

        if self._offspring_results is not None:
            return [
                self._offspring_results[i]
                if i < len(self._offspring_results)
                else self._invalid_result(expr_str)
                for i, expr_str in enumerate(candidates)
            ]

        offspring = self._offspring or []
        results: list[EvaluationResult] = []
        for i, expr_str in enumerate(candidates):
            if i < len(offspring):
                try:
                    cr = evaluate_candidate(
                        offspring[i],
                        self._data_dict,
                        self._default_terms,
                        self._y if self._y is not None else torch.zeros(1),
                        self._config,
                        diff_ctx=self._diff_ctx,
                    )
                    # Sync genotype: replace offspring with pruned version
                    offspring[i] = cr.pruned_pde
                    results.append(self._to_eval_result(cr, expr_str))
                except Exception:
                    logger.debug("Evaluation failed for candidate %d", i)
                    results.append(self._invalid_result(expr_str))
            else:
                results.append(self._invalid_result(expr_str))
        return results

    def update(self, results: list[EvaluationResult]) -> None:
        """Commit the staged generation, or merge legacy direct results."""
        if self._pending_population is not None and self._pending_scores is not None:
            self._commit_pending_generation()
            return

        if not results:
            return

        offspring = self._offspring or []
        population = list(self._population or [])
        scores = list(self._scores or [])

        for i, result in enumerate(results):
            if i < len(offspring):
                aic = result.aic if result.aic is not None else _INVALID_AIC
                if not result.is_valid or not math.isfinite(aic):
                    aic = _INVALID_AIC
                population.append(offspring[i])
                scores.append(aic)

        if population:
            paired = list(zip(scores, population, strict=True))
            paired.sort(key=lambda x: x[0])
            scores = [s for s, _ in paired]
            population = [p for _, p in paired]

            num = self._config.num
            self._population = population[:num]
            self._scores = scores[:num]

            if self._scores and self._scores[0] < self._best_score:
                self._best_score = self._scores[0]
                self._best_expression = pde_to_kd2_expr(self._population[0])
                self._best_formatted_cache = None

        if self._recorder is not None:
            self._recorder.log("best_aic", self._best_score)

        self._offspring = None
        self._offspring_results = None

    @property
    def best_score(self) -> float:
        """Return the best AIC score found so far."""
        return self._best_score

    @property
    def best_expression(self) -> str:
        """Return the best expression with coefficients (lazy, cached).

        Falls back gracefully when the regression selects only the default
        term (genotype-empty case): returns ``"<lhs_field>"`` instead of the
        empty string.
        """
        if self._best_formatted_cache is None:
            self._best_formatted_cache = self._format_best_expression()
        return self._best_formatted_cache

    def build_final_result(self) -> EvaluationResult:
        """Re-evaluate the best PDE and return a fully populated result."""
        best_pde = self._best_pde()
        if best_pde is None or self._y is None:
            return self._invalid_final_result("No best PDE available for final result")

        try:
            candidate = evaluate_candidate(
                best_pde,
                self._data_dict,
                self._default_terms,
                self._y,
                self._config,
                diff_ctx=self._diff_ctx,
            )
            predicted = self._predict_rhs(candidate.pruned_pde, candidate.coefficients)
        except Exception as exc:
            logger.debug("Failed to build final result", exc_info=exc)
            return self._invalid_final_result(f"Final result evaluation failed: {exc}")

        residuals = (predicted - self._y).detach()
        is_valid = _is_valid_aic(candidate.aic_score) and math.isfinite(candidate.mse)
        return EvaluationResult(
            mse=candidate.mse,
            nmse=metrics_nmse(candidate.mse, self._target_variance()),
            r2=self._compute_r2(predicted),
            aic=candidate.aic_score,
            complexity=len(candidate.selected_indices),
            coefficients=candidate.coefficients.detach(),
            is_valid=is_valid,
            error_message="" if is_valid else "Invalid AIC or MSE",
            selected_indices=list(candidate.selected_indices),
            residuals=residuals,
            terms=self._build_term_list(candidate.pruned_pde),
            expression=self._best_expression,
        )

    def build_result_target(self) -> Tensor:
        """Return the LHS target used for the final SGA evaluation.

        Why clone: callers receive an independent tensor so in-place
        mutation cannot poison ``self._y`` or subsequent calls.
        """
        if self._y is None:
            return torch.zeros(0)
        return self._y.detach().clone()

    @property
    def state(self) -> dict[str, Any]:
        """Algorithm state for checkpointing (pickle-serializable)."""
        return {
            "population": self._population,
            "scores": self._scores,
            "best_score": self._best_score,
            "best_expression": self._best_expression,
            "vars": self._vars,
            "rng_state": self._rng.get_state().numpy().tobytes(),
        }

    @state.setter
    def state(self, value: dict[str, Any]) -> None:
        """Restore algorithm state from checkpoint."""
        if not value:
            self._clear_pending_generation()
            return
        if "population" in value:
            self._population = value["population"]
        if "scores" in value:
            self._scores = value["scores"]
        if "best_score" in value:
            self._best_score = value["best_score"]
        if "best_expression" in value:
            self._best_expression = value["best_expression"]
            self._best_formatted_cache = None
        if "vars" in value:
            self._vars = value["vars"]
        if "rng_state" in value:
            rng_state = value["rng_state"]
            if isinstance(rng_state, bytes):
                import numpy as np

                rng_state = torch.from_numpy(
                    np.frombuffer(rng_state, dtype=np.uint8).copy()
                )
            self._rng.set_state(rng_state)
        self._clear_pending_generation()

    # -- Internal helpers -------------------------------------------------------

    def _clear_pending_generation(self) -> None:
        """Clear transient proposal state that must not survive restore/prepare."""
        self._offspring = None
        self._offspring_results = None
        self._pending_population = None
        self._pending_scores = None

    def _best_pde(self) -> PDE | None:
        """Return the current best PDE if one exists."""
        population = self._population or []
        if not population:
            return None
        return population[0]

    def _predict_rhs(self, pde: PDE, coefficients: Tensor) -> Tensor:
        """Rebuild the RHS prediction for a pruned PDE."""
        valid_terms, _ = execute_pde(pde, self._data_dict, self._diff_ctx)
        theta = build_theta(valid_terms, self._default_terms)
        if theta.shape[1] == 0:
            if self._y is None:
                return torch.zeros(0)
            return torch.zeros_like(self._y)
        return (theta @ coefficients).detach()

    def _build_term_list(self, pde: PDE) -> list[str]:
        """Return theta-column term names aligned with selected_indices."""
        terms = [tree_to_kd2_expr(tree) for tree in pde.terms]
        if self._default_term_name is not None:
            return [self._default_term_name, *terms]
        return terms

    def _target_variance(self) -> float:
        """Return biased Var(y) for NMSE / R² normalization.

        Uses ``correction=0`` (population variance, divides by N) so the
        denominator matches the ``ss_tot/N`` form implicit in
        ``_compute_r2`` — without that, a mean predictor reports
        ``r2 = 1/N`` from ``_r2_from_mse`` while ``_compute_r2``
        correctly reports ``0``.

        Returns 0.0 when y is unset, has fewer than two samples, or
        ``torch.var`` yields a non-finite value (PyTorch returns NaN
        for single-element tensors). Pairs with ``kd2.core.metrics.nmse``
        which falls back to raw MSE when target variance is below its
        eps guard, giving consistent behavior across the platform.
        """
        if self._y is None or self._y.numel() < 2:
            return 0.0
        var = float(torch.var(self._y, correction=0).item())
        return var if math.isfinite(var) else 0.0

    def _format_best_expression(self) -> str:
        """Render best expression with coefficients.

        Strategy:
        1. Try ``build_final_result`` for coefficients + term names.
        2. If valid, return ``"<lhs> = c0*term0 + c1*term1 + ..."``.
        3. On failure or zero active terms, fall back to genotype string.
        4. If genotype is empty (default-only fit), return the LHS field
           name (e.g. ``"u"``) instead of the empty string.
        """
        try:
            res = self.build_final_result()
        except Exception: # build_final_result is defensive but be safe
            res = None

        if (
            res is not None
            and res.is_valid
            and res.coefficients is not None
            and res.terms
        ):
            rhs = self._format_rhs_with_coefficients(res.coefficients, res.terms)
            if rhs:
                lhs = self._lhs_label_str()
                return f"{lhs} = {rhs}" if lhs else rhs

        if self._best_expression:
            return self._best_expression
        if self._default_term_name:
            return self._default_term_name
        return ""

    def _lhs_label_str(self) -> str:
        """Return ``<field>_<axis>`` LHS label (e.g. ``"u_t"``) or empty."""
        if self._default_term_name and self._lhs_axis:
            return f"{self._default_term_name}_{self._lhs_axis}"
        return ""

    @staticmethod
    def _format_rhs_with_coefficients(coefficients: Tensor, terms: list[str]) -> str:
        """Build ``c0*term0 + c1*term1 + ...`` skipping near-zero coefficients."""
        if coefficients.numel() == 0 or not terms:
            return ""
        n = min(coefficients.numel(), len(terms))
        active: list[tuple[float, str]] = []
        for i in range(n):
            c = float(coefficients[i].item())
            if abs(c) < 1e-10:
                continue
            active.append((c, terms[i]))
        if not active:
            return ""
        parts: list[str] = []
        for i, (c, name) in enumerate(active):
            mag = f"{abs(c):.4g}"
            if i == 0:
                parts.append(f"-{mag}*{name}" if c < 0 else f"{mag}*{name}")
            else:
                parts.append(f"{'-' if c < 0 else '+'} {mag}*{name}")
        return " ".join(parts)

    def _compute_r2(self, predicted: Tensor) -> float:
        """Compute R² for the rebuilt prediction.

        Follows sklearn convention: R² has upper bound 1 (perfect fit)
        but no lower bound — a catastrophic predictor (mse ≫ var)
        produces a large negative value. For a constant target
        (``ss_tot ≈ 0``) sklearn returns 1.0 when the prediction is
        also perfect (``ss_res ≈ 0``) and 0.0 otherwise; we mirror that.
        Returns ``-inf`` for non-finite ss_res.
        """
        if self._y is None:
            return -float("inf")
        ss_res = ((self._y - predicted) ** 2).sum().item()
        if not math.isfinite(ss_res):
            return -float("inf")
        ss_tot = ((self._y - self._y.mean()) ** 2).sum().item()
        if ss_tot < _R2_EPS:
            return 1.0 if ss_res < _R2_EPS else 0.0
        return 1.0 - ss_res / ss_tot

    def _invalid_final_result(self, error_message: str) -> EvaluationResult:
        """Create an invalid final result with a zero residual fallback."""
        residuals = torch.zeros_like(self._y) if self._y is not None else torch.zeros(0)
        return EvaluationResult(
            mse=float("inf"),
            nmse=float("inf"),
            r2=-float("inf"),
            aic=float("inf"),
            complexity=0,
            coefficients=None,
            is_valid=False,
            error_message=error_message,
            selected_indices=[],
            residuals=residuals,
            terms=[],
            expression=self._best_expression,
        )

    def _broadcast_coord(
        self,
        coord: Tensor,
        axis_name: str,
        dataset: Any,
        field_shape: tuple[int, ...],
    ) -> Tensor:
        """Broadcast 1D coordinate to field shape using axis_order."""
        if dataset.axis_order is None:
            return coord.flatten()
        axis_idx = dataset.axis_order.index(axis_name)
        shape = [1] * len(field_shape)
        shape[axis_idx] = coord.shape[0]
        return coord.view(*shape).expand(field_shape).contiguous()

    def _extract_lhs_target(
        self,
        dataset: Any,
        context: Any,
    ) -> Tensor:
        """Extract LHS derivative (e.g. u_t) as regression target y.

        The LHS derivative is NOT in data_dict/VARS (excluded to avoid
        trivial solutions). the predecessor: self.ut stored separately.

        Raises ValueError when the LHS derivative is unavailable instead
        of silently falling back to zeros
        the predecessor: raises ValueError("Structured grid missing lhs_axis ...").
        """
        if dataset.lhs_field and dataset.lhs_axis:
            try:
                get_deriv = self._resolve_get_derivative(context)
                deriv: Tensor = get_deriv(dataset.lhs_field, dataset.lhs_axis, 1)
                return _numeric_flatten(deriv)
            except (KeyError, ValueError) as exc:
                raise ValueError(
                    f"LHS derivative {dataset.lhs_field}_{dataset.lhs_axis} "
                    f"not available. SGA requires the LHS target derivative "
                    f"to be computable from the dataset."
                ) from exc
        raise ValueError(
            "LHS field and axis must both be set for SGA. "
            f"Got lhs_field={dataset.lhs_field!r}, lhs_axis={dataset.lhs_axis!r}."
        )

    def _validate_naming(self, dataset: Any) -> None:
        """Fail-fast guardrails for field/axis naming conflicts

        Checks ported from the predecessor ``_validate_registry_names``:
        1. Field names must not overlap with axis names.
        2. Field names must not collide with derivative keys (e.g. ``u_x``).
        3. Legacy LHS derivative aliases (e.g. ``ut`` for ``u_t``) must not
           appear as field names.
        """
        field_names = list(dataset.fields.keys()) if dataset.fields else []
        axis_names = list(dataset.axes.keys()) if dataset.axes else []

        # 1. Field vs axis overlap
        overlap = sorted(set(field_names) & set(axis_names))
        if overlap:
            raise ValueError(
                f"Field names conflict with axis names: {overlap}. "
                f"Fields and axes must have distinct names."
            )

        # 2. Field vs derivative key overlap
        deriv_keys = {
            f"{field}_{axis * order}"
            for field in field_names
            for axis in axis_names
            for order in (1, 2)
        }
        deriv_overlap = sorted(set(field_names) & deriv_keys)
        if deriv_overlap:
            raise ValueError(
                f"Field names conflict with derivative keys: {deriv_overlap}. "
                f"Rename the field to avoid ambiguity."
            )

        # 3. Legacy LHS derivative alias (e.g. "ut" when lhs_field="u", lhs_axis="t")
        lhs_field = dataset.lhs_field
        lhs_axis = dataset.lhs_axis
        if lhs_field and lhs_axis and len(lhs_axis) == 1:
            legacy_alias = f"{lhs_field}{lhs_axis}"
            if legacy_alias in field_names:
                raise ValueError(
                    f"Field name '{legacy_alias}' conflicts with LHS derivative "
                    f"alias ({lhs_field}_{lhs_axis}). This would allow the LHS "
                    f"derivative to leak into the RHS variable pool."
                )

    def _add_derivatives(
        self,
        dataset: Any,
        context: Any,
        data_dict: dict[str, Tensor],
    ) -> None:
        """Add first-order spatial derivative tensors to data_dict.

        Only first-order derivatives are precomputed as VARS terminals. Higher-order derivatives
        are reached through tree composition (e.g. ``d(u_x, x)``) or the
        ``d^2`` operator, avoiding Theta column redundancy

        Excludes lhs_axis derivatives (e.g. u_t for lhs_axis="t")
        to avoid trivial solutions. the predecessor: _assert_no_lhs_in_grad_fields.
        """
        if dataset.fields is None or dataset.axes is None:
            return
        for field_name in dataset.fields:
            for axis_name in dataset.axes:
                if axis_name == dataset.lhs_axis:
                    continue
                key = f"{field_name}_{axis_name}"
                try:
                    get_deriv = self._resolve_get_derivative(context)
                    deriv = get_deriv(field_name, axis_name, 1)
                    data_dict[key] = _numeric_flatten(deriv)
                except (KeyError, ValueError):
                    logger.debug("Derivative %s not available", key)

    def _resolve_get_derivative(
        self,
        context: Any,
    ) -> Callable[[str, str, int], Tensor]:
        """Return the active ``get_derivative`` callable for SGA Layer 2

        When ``use_autograd=True`` and an internal ``AutogradProvider`` has
        been built in :meth:`prepare`, route through it. Otherwise call
        through the platform's ``context.get_derivative`` (which itself
        delegates to ``context.derivative_provider`` — preserves the public
        ExecutionContext interface for any test or wrapper that mocks it).

        Layer 1 (raw ``u`` leaf via ``context.get_variable``) and Layer 3
        (tree ``d`` / ``d^2`` via ``_finite_diff_torch``) are unaffected.
        """
        if self._autograd_provider is not None:
            return self._autograd_provider.get_derivative
        return context.get_derivative # type: ignore[no-any-return]

    def _build_autograd_provider(
        self,
        dataset: Any,
        field_shape: tuple[int, ...],
    ) -> AutogradProvider:
        """Build an internal ``AutogradProvider`` for SGA Layer 2

        Trains a default :class:`FieldModel` from the dataset (or reuses
        ``self._config.field_model`` if provided) and wraps it. The returned
        provider lives on the plugin instance only — the platform-shared
        ``components.context.derivative_provider`` is left untouched.
        """
        if dataset.fields is None or dataset.axes is None:
            raise ValueError(
                "use_autograd=True requires dataset.fields and dataset.axes."
            )

        field_names = list(dataset.fields.keys())
        coord_names = self._ordered_axes(dataset)
        if not coord_names:
            raise ValueError("use_autograd=True requires at least one ordered axis.")

        flat_coords: dict[str, Tensor] = {}
        for axis_name in coord_names:
            coord_1d = dataset.axes[axis_name].values
            broadcast = self._broadcast_coord(coord_1d, axis_name, dataset, field_shape)
            flat_coords[axis_name] = broadcast.flatten()

        flat_targets: dict[str, Tensor] = {
            name: fdata.values.flatten() for name, fdata in dataset.fields.items()
        }

        if self._config.field_model is not None:
            field_model: FieldModel = self._config.field_model
            self._validate_field_model(field_model, coord_names, field_names)
        else:
            field_model = FieldModel(
                coord_names=coord_names,
                field_names=field_names,
            )
            trainer = FieldModelTrainer(field_model, lr=self._config.autograd_train_lr)
            trainer.fit(
                coords=flat_coords,
                targets=flat_targets,
                max_epochs=self._config.autograd_train_epochs,
                seed=self._config.seed,
            )

        grad_coords = {
            name: c.detach().clone().requires_grad_(True)
            for name, c in flat_coords.items()
        }
        return AutogradProvider(
            model=field_model,
            coords=grad_coords,
            dataset=dataset,
            max_order=1,
        )

    @staticmethod
    def _validate_field_model(
        field_model: FieldModel,
        coord_names: list[str],
        field_names: list[str],
    ) -> None:
        """Reject pre-trained FieldModels whose I/O names don't match dataset."""
        if list(field_model.coord_names) != list(coord_names):
            raise ValueError(
                f"field_model.coord_names {field_model.coord_names} != "
                f"dataset axes {coord_names}"
            )
        if list(field_model.field_names) != list(field_names):
            raise ValueError(
                f"field_model.field_names {field_model.field_names} != "
                f"dataset fields {field_names}"
            )

    def _ordered_axes(self, dataset: Any) -> list[str]:
        """Return dataset axes in a stable order."""
        if dataset.axis_order is not None:
            return list(dataset.axis_order)
        if dataset.axes is None:
            return []
        return list(dataset.axes.keys())

    def _build_delta_map(
        self,
        dataset: Any,
        axis_names: list[str],
    ) -> dict[str, float]:
        """Compute uniform grid spacings from dataset axis coordinates."""
        if dataset.axes is None:
            return {}
        delta: dict[str, float] = {}
        for axis_name in axis_names:
            if axis_name not in dataset.axes:
                continue
            coord = dataset.axes[axis_name].values.reshape(-1)
            if coord.numel() < 2:
                continue
            diffs = torch.diff(coord)
            first_diff = float(diffs[0].item())
            if abs(first_diff) < DX_ZERO_FLOOR:
                raise ValueError(
                    f"Structured grid axis '{axis_name}' has degenerate spacing "
                    f"dx={first_diff:.6g}; finite-difference stencils require "
                    "nonzero dx."
                )
            max_diff = float(diffs.max().item())
            min_diff = float(diffs.min().item())
            if abs(max_diff - min_diff) > abs(first_diff) * UNIFORM_GRID_RTOL:
                raise ValueError(
                    f"Structured grid requires uniform spacing for axis '{axis_name}' "
                    f"(min spacing = {min_diff:.6e}, max spacing = {max_diff:.6e})."
                )
            if not (torch.all(diffs > 0) or torch.all(diffs < 0)):
                raise ValueError(
                    f"Structured grid axis '{axis_name}' must be strictly monotonic."
                )
            delta[axis_name] = first_diff
        return delta

    def _init_population(self) -> None:
        """Create initial random population, sorted by AIC.

        Pathological individuals (AIC=inf) are rejected and resampled
        up to ``_MAX_RESAMPLE_PER_INDIVIDUAL`` times per slot. If all
        retries are exhausted for any slot, a ``RuntimeError`` is raised
        to fail fast rather than starting search with a degenerate pool.

        the predecessor reference: ``SGA.__init__`` uses
        ``while a_err < -100 or a_err == np.inf``
        with no upper bound; kd2 adds a finite retry limit.
        """
        from kd2.search.sga.genetic import random_pde

        population: list[PDE] = []
        scores: list[float] = []
        for i in range(self._config.num):
            pde = random_pde(self._config, self._vars, OPS, ROOT, self._den, self._rng)
            aic, pruned = _safe_evaluate_aic(
                pde,
                self._data_dict,
                self._default_terms,
                self._y,
                self._config,
                self._diff_ctx,
            )

            retries = 0
            while not _is_valid_aic(aic) and retries < _MAX_RESAMPLE_PER_INDIVIDUAL:
                logger.debug(
                    "Init individual %d: AIC=%s, resampling (retry %d/%d)",
                    i,
                    aic,
                    retries + 1,
                    _MAX_RESAMPLE_PER_INDIVIDUAL,
                )
                pde = random_pde(
                    self._config,
                    self._vars,
                    OPS,
                    ROOT,
                    self._den,
                    self._rng,
                )
                aic, pruned = _safe_evaluate_aic(
                    pde,
                    self._data_dict,
                    self._default_terms,
                    self._y,
                    self._config,
                    self._diff_ctx,
                )
                retries += 1

            if not _is_valid_aic(aic):
                raise RuntimeError(
                    f"Init population failed: individual {i} still has "
                    f"invalid AIC after {_MAX_RESAMPLE_PER_INDIVIDUAL} resample "
                    f"retries. All random candidates are pathological — "
                    f"check data, derivative quality, or STRidge config."
                )

            # Use pruned PDE: genotype synced with evaluation
            population.append(pruned)
            scores.append(aic)

        paired = list(zip(scores, population, strict=True))
        paired.sort(key=lambda x: x[0])
        self._scores = [s for s, _ in paired]
        self._population = [p for _, p in paired]

        # Track initial best from population
        if self._scores and math.isfinite(self._scores[0]):
            self._best_score = self._scores[0]
            self._best_expression = pde_to_kd2_expr(self._population[0])
            self._best_formatted_cache = None

    def _apply_genetic_ops(self) -> list[PDE]:
        """Run one v1-style generation and return evaluated offspring.

        v1 performs two selection stages per generation:
        crossover -> evaluate -> truncate -> mutate/replace -> evaluate -> truncate.

        SearchAlgorithm still calls propose/evaluate/update externally, so this
        method computes the staged generation on local copies and stores the
        final population in ``_pending_population`` for ``update()`` to commit.
        """
        from kd2.search.sga.config import OP1, OP2
        from kd2.search.sga.genetic import crossover, mutate, replace

        population = list(self._population or [])
        if not population:
            self._pending_population = []
            self._pending_scores = []
            self._offspring_results = []
            return []

        cfg = self._config
        scores = self._current_scores(len(population))
        population, scores = self._sort_truncate_population(population, scores)
        evaluated_offspring: list[_ScoredPDE] = []

        # Crossover: pair top individuals via shuffle pairing
        # p_cro=0 must produce zero crossover offspring (the predecessor semantics).
        # Do NOT use max(1, ...) — that forces crossover even when disabled.
        # fix: shuffle pairing (v1 semantics)
        # Each top individual participates exactly once as original,
        # paired with a shuffled partner. Produces num_cross pairs
        # -> 2 * num_cross crossover offspring.
        num_cross = int(len(population) * cfg.p_cro)
        if num_cross > 0:
            top = population[:num_cross]
            perm = torch.randperm(len(top), generator=self._rng).tolist()
            shuffled = [top[i] for i in perm]
            for orig, partner in zip(top, shuffled, strict=True):
                c1, c2 = crossover(orig, partner, self._rng)
                evaluated_offspring.extend(
                    [self._score_offspring(c1), self._score_offspring(c2)]
                )

            population, scores = self._merge_sort_truncate(
                population,
                scores,
                evaluated_offspring,
            )

        # Mutate+replace non-elite population members
        mutation_offspring: list[_ScoredPDE] = []
        for i in range(1, len(population)):
            m = mutate(
                population[i],
                self._vars,
                OP1,
                OP2,
                self._den,
                cfg.p_mute,
                self._rng,
            )
            if _rand_float(self._rng) < cfg.p_rep:
                m = replace(
                    m,
                    self._vars,
                    OPS,
                    ROOT,
                    self._den,
                    cfg.depth,
                    cfg.p_var,
                    self._rng,
                )
            mutation_offspring.append(self._score_offspring(m))

        if mutation_offspring:
            evaluated_offspring.extend(mutation_offspring)
            population, scores = self._merge_sort_truncate(
                population,
                scores,
                mutation_offspring,
            )

        self._pending_population = population
        self._pending_scores = scores
        self._offspring_results = [item.result for item in evaluated_offspring]
        return [item.pde for item in evaluated_offspring]

    def _current_scores(self, population_len: int) -> list[float]:
        """Return scores aligned with current population, defaulting to penalties."""
        if self._scores is None or len(self._scores) != population_len:
            return [_INVALID_AIC] * population_len
        return list(self._scores)

    def _merge_sort_truncate(
        self,
        population: list[PDE],
        scores: list[float],
        offspring: list[_ScoredPDE],
    ) -> tuple[list[PDE], list[float]]:
        """Merge evaluated offspring into a population and truncate by AIC."""
        merged_population = [*population, *(item.pde for item in offspring)]
        merged_scores = [*scores, *(item.score for item in offspring)]
        return self._sort_truncate_population(merged_population, merged_scores)

    def _sort_truncate_population(
        self,
        population: list[PDE],
        scores: list[float],
    ) -> tuple[list[PDE], list[float]]:
        """Sort population by score and keep at most ``config.num`` individuals."""
        if not population:
            return [], []
        paired = list(zip(scores, population, strict=True))
        paired.sort(key=lambda x: x[0])
        paired = paired[: self._config.num]
        return [p for _, p in paired], [s for s, _ in paired]

    def _score_offspring(self, pde: PDE) -> _ScoredPDE:
        """Evaluate one offspring with the internal SGA pipeline."""
        expression = pde_to_kd2_expr(pde)
        if not self._prepared:
            return _ScoredPDE(
                pde,
                _INVALID_AIC,
                self._evaluation_failed_result(expression),
            )
        try:
            candidate = evaluate_candidate(
                pde,
                self._data_dict,
                self._default_terms,
                self._y if self._y is not None else torch.zeros(1),
                self._config,
                diff_ctx=self._diff_ctx,
            )
            pruned = candidate.pruned_pde
            result = self._to_eval_result(candidate, pde_to_kd2_expr(pruned))
            aic = result.aic if result.aic is not None else _INVALID_AIC
            score = aic if result.is_valid and math.isfinite(aic) else _INVALID_AIC
            return _ScoredPDE(pruned, score, result)
        except Exception:
            logger.debug("Evaluation failed for staged offspring")
            return _ScoredPDE(
                pde,
                _INVALID_AIC,
                self._evaluation_failed_result(expression),
            )

    def _commit_pending_generation(self) -> None:
        """Commit the staged generation computed during ``propose()``."""
        self._population = self._pending_population
        self._scores = self._pending_scores

        if self._scores and self._scores[0] < self._best_score:
            self._best_score = self._scores[0]
            if self._population:
                self._best_expression = pde_to_kd2_expr(self._population[0])
                self._best_formatted_cache = None

        if self._recorder is not None:
            self._recorder.log("best_aic", self._best_score)

        self._clear_pending_generation()

    def _r2_from_mse(self, mse: float) -> float:
        """Approximate R² from MSE + target variance

        Used by ``_to_eval_result`` where only the scalar MSE survives
        the train/candidate pipeline (the prediction tensor is not kept
        on every offspring).

        Uses the biased variance from ``_target_variance`` (divisor N),
        which matches the ``ss_tot/N`` denominator implicit in
        ``_compute_r2``. Both helpers therefore agree on the baseline
        (mean predictor → 0.0).

        sklearn convention: for a constant target (``var ≈ 0``) sklearn
        returns 1.0 when the prediction is perfect (``mse ≈ 0``) and 0.0
        otherwise; we mirror that. R² has upper bound 1 (perfect fit)
        but no lower bound — preserve large negative values so reports
        and plots can flag catastrophic predictors
        """
        if not math.isfinite(mse):
            return -math.inf
        target_var = self._target_variance()
        if target_var < _R2_EPS:
            return 1.0 if mse < _R2_EPS else 0.0
        return 1.0 - mse / target_var

    def _to_eval_result(
        self, result: CandidateResult | TrainResult, expression: str
    ) -> EvaluationResult:
        """Convert a CandidateResult or TrainResult to an EvaluationResult."""
        # CandidateResult delegates properties, so access is uniform
        aic = result.aic_score
        mse = result.mse
        is_valid = _is_valid_aic(aic) and math.isfinite(mse)
        return EvaluationResult(
            mse=mse,
            nmse=metrics_nmse(mse, self._target_variance()),
            r2=self._r2_from_mse(mse),
            aic=aic,
            complexity=len(result.selected_indices),
            coefficients=result.coefficients,
            is_valid=is_valid,
            error_message="" if is_valid else "Invalid AIC or MSE",
            selected_indices=result.selected_indices,
            residuals=None,
            terms=None,
            expression=expression,
        )

    def _invalid_result(self, expression: str) -> EvaluationResult:
        """Create an invalid EvaluationResult with penalty values."""
        return EvaluationResult(
            mse=float("inf"),
            nmse=float("inf"),
            r2=-float("inf"),
            aic=float("inf"),
            complexity=0,
            coefficients=None,
            is_valid=False,
            error_message="No corresponding PDE for evaluation",
            selected_indices=None,
            residuals=None,
            terms=None,
            expression=expression,
        )

    def _evaluation_failed_result(self, expression: str) -> EvaluationResult:
        """Create an invalid result for a candidate that failed during scoring."""
        result = self._invalid_result(expression)
        result.error_message = "Candidate evaluation failed"
        return result
