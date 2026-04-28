"""FieldModelTrainer — train FieldModel on observed data.

Supports:
- Automatic z-score normalization (computed from data, set on model)
- MSE loss with Adam optimizer
- Early stopping on validation loss
- RNG isolation via torch.random.fork_rng()
- Reproducible training with seed control
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

from kd2.models.field_model import FieldModel

logger = logging.getLogger(__name__)


@dataclass
class TrainingResult:
    """Result returned by :meth:`FieldModelTrainer.fit`.

    Attributes:
        final_loss: Training loss at the last epoch.
        epochs_run: Total number of epochs executed.
        early_stopped: Whether training stopped early due to patience.
        val_loss: Validation loss at the last epoch, or ``None`` if
            no validation split was used (``val_ratio=0``).
    """

    final_loss: float
    epochs_run: int
    early_stopped: bool
    val_loss: float | None


class FieldModelTrainer:
    """Trains a :class:`FieldModel` on observed coordinate/field data.

    Usage::

        model = FieldModel(coord_names=["x", "t"], field_names=["u"])
        trainer = FieldModelTrainer(model, lr=1e-3)
        result = trainer.fit(
            coords={"x": x, "t": t},
            targets={"u": u},
            max_epochs=5000,
            patience=100,
        )
    """

    def __init__(
        self,
        model: FieldModel,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
    ) -> None:
        self._model = model
        self._lr = lr
        self._weight_decay = weight_decay

    def fit(
        self,
        coords: dict[str, Tensor],
        targets: dict[str, Tensor],
        max_epochs: int = 10000,
        patience: int | None = 100,
        val_ratio: float = 0.2,
        seed: int = 0,
    ) -> TrainingResult:
        """Train the model on the given data.

        Args:
            coords: Input coordinates ``{name: tensor}``.
            targets: Target field values ``{name: tensor}``.
            max_epochs: Maximum number of training epochs.
            patience: Early stopping patience (epochs without val_loss
                improvement). ``None`` disables early stopping.
            val_ratio: Fraction of data reserved for validation.
                ``0`` means all data is used for training.
            seed: Random seed for weight init and data splitting.

        Returns:
            :class:`TrainingResult` with final metrics.

        Raises:
            ValueError: If ``val_ratio`` is not in ``[0, 1)``.
        """
        _validate_val_ratio(val_ratio)

        # Match model dtype to input data (e.g. float64 grid data)
        _data_dtype = _infer_dtype(coords, targets)
        _model_dtype = next(self._model.parameters()).dtype
        if _data_dtype != _model_dtype:
            logger.debug(
                "Casting model from %s to %s to match input data",
                _model_dtype,
                _data_dtype,
            )
            self._model = self._model.to(dtype=_data_dtype)

        # Collect CUDA device ids for fork_rng
        cuda_devices: list[int] = []
        if torch.cuda.is_available():
            cuda_devices = list(range(torch.cuda.device_count()))

        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(seed)
            return self._fit_inner(
                coords, targets, max_epochs, patience, val_ratio, seed
            )

    # -- private -----------------------------------------------------------

    def _fit_inner(
        self,
        coords: dict[str, Tensor],
        targets: dict[str, Tensor],
        max_epochs: int,
        patience: int | None,
        val_ratio: float,
        seed: int,
    ) -> TrainingResult:
        """Core training loop (called inside fork_rng).

        NOTE: manual_seed has already been called before this method,
        so re-initializing weights here yields reproducible init.
        """
        # 0. Re-initialize model weights for reproducibility
        _reinit_parameters(self._model)

        # 1. Compute and set normalization
        self._set_normalization(coords, targets)

        # 2. Split data
        n_samples = _get_n_samples(coords)
        train_idx, val_idx = _split_indices(n_samples, val_ratio, seed)

        train_coords = _index_dict(coords, train_idx)
        train_targets = _index_dict(targets, train_idx)
        val_coords = _index_dict(coords, val_idx) if val_idx is not None else None
        val_targets = _index_dict(targets, val_idx) if val_idx is not None else None

        # Detach training coords (no gradient needed during training)
        train_coords = {k: v.detach() for k, v in train_coords.items()}

        # 3. Optimizer + loss
        self._model.train()
        optimizer = torch.optim.Adam(
            self._model.parameters(),
            lr=self._lr,
            weight_decay=self._weight_decay,
        )
        criterion = nn.MSELoss()

        # Warn if patience is set but no validation split
        if patience is not None and val_ratio == 0.0:
            logger.warning(
                "patience=%d has no effect when val_ratio=0 "
                "(no validation data for early stopping)",
                patience,
            )

        # 4. Training loop
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        final_loss = float("inf")
        final_val_loss: float | None = None
        early_stopped = False
        epoch = 0

        for epoch in range(1, max_epochs + 1):
            final_loss = _train_step(
                self._model, optimizer, criterion, train_coords, train_targets
            )

            # Validate
            if val_coords is not None and val_targets is not None:
                final_val_loss = _eval_loss(
                    self._model, criterion, val_coords, val_targets
                )

                # Early stopping check
                if patience is not None:
                    if final_val_loss < best_val_loss:
                        best_val_loss = final_val_loss
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= patience:
                            early_stopped = True
                            logger.debug(
                                "Early stop at epoch %d (patience=%d)",
                                epoch,
                                patience,
                            )
                            break

        # 5. Finalize
        self._model.eval()
        return TrainingResult(
            final_loss=final_loss,
            epochs_run=epoch,
            early_stopped=early_stopped,
            val_loss=final_val_loss,
        )

    def _set_normalization(
        self,
        coords: dict[str, Tensor],
        targets: dict[str, Tensor],
    ) -> None:
        """Compute z-score stats from data and set on model."""
        coord_stats: dict[str, tuple[Tensor, Tensor]] = {}
        for name in self._model.coord_names:
            data = coords[name]
            coord_stats[name] = (data.mean(), data.std())

        field_stats: dict[str, tuple[Tensor, Tensor]] = {}
        for name in self._model.field_names:
            data = targets[name]
            field_stats[name] = (data.mean(), data.std())

        self._model.set_normalization(coord_stats, field_stats)


# ======================================================================
# Pure helpers (module-level for testability)
# ======================================================================


def _reinit_parameters(model: nn.Module) -> None:
    """Re-initialize all model parameters using their default init.

    Called inside fork_rng after manual_seed to ensure reproducible
    weight initialization regardless of external RNG state.
    """
    for module in model.modules():
        reset_fn = getattr(module, "reset_parameters", None)
        if reset_fn is not None:
            reset_fn()


def _infer_dtype(coords: dict[str, Tensor], targets: dict[str, Tensor]) -> torch.dtype:
    """Infer dtype from input data (first tensor found)."""
    for tensors in (coords, targets):
        for v in tensors.values():
            return v.dtype
    return torch.float32


def _validate_val_ratio(val_ratio: float) -> None:
    """Raise ValueError if val_ratio is out of [0, 1)."""
    if val_ratio < 0.0 or val_ratio >= 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")


def _get_n_samples(data: dict[str, Tensor]) -> int:
    """Return the number of samples (length of first tensor)."""
    return next(iter(data.values())).shape[0]


def _split_indices(n: int, val_ratio: float, seed: int) -> tuple[Tensor, Tensor | None]:
    """Split sample indices into train/val sets.

    Returns:
        (train_indices, val_indices) where val_indices is None
        when val_ratio == 0.
    """
    if val_ratio == 0.0:
        return torch.arange(n), None

    # Use a separate generator for reproducible splitting
    gen = torch.Generator()
    gen.manual_seed(seed)
    perm = torch.randperm(n, generator=gen)

    n_val = max(1, int(n * val_ratio))
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    return train_idx, val_idx


def _index_dict(data: dict[str, Tensor], idx: Tensor) -> dict[str, Tensor]:
    """Index every tensor in a dict by the given indices."""
    return {k: v[idx] for k, v in data.items()}


def _train_step(
    model: FieldModel,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    coords: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> float:
    """Single training step. Returns detached loss value."""
    model.train()
    optimizer.zero_grad()
    preds = model(**coords)
    loss = _compute_mse(criterion, preds, targets)
    loss.backward() # type: ignore[no-untyped-call]
    optimizer.step()
    return loss.detach().item()


@torch.no_grad()
def _eval_loss(
    model: FieldModel,
    criterion: nn.Module,
    coords: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> float:
    """Evaluate loss without gradient tracking."""
    model.eval()
    preds = model(**coords)
    loss = _compute_mse(criterion, preds, targets)
    return loss.item()


def _compute_mse(
    criterion: nn.Module,
    preds: dict[str, Tensor],
    targets: dict[str, Tensor],
) -> Tensor:
    """Compute mean MSE across all fields."""
    losses: list[Tensor] = []
    for name, pred in preds.items():
        losses.append(criterion(pred, targets[name]))
    return torch.stack(losses).mean()
