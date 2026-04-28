"""Tests for FieldModelTrainer."""

from __future__ import annotations

import math

import pytest
import torch
from torch import Tensor

from kd2.models.field_model import FieldModel
from kd2.models.trainer import FieldModelTrainer, TrainingResult

# -- helpers ---------------------------------------------------------------


def _make_sin_data(n: int = 100) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """1D: u = sin(x), x in [0, 2*pi]."""
    x = torch.linspace(0, 2 * math.pi, n)
    u = torch.sin(x)
    return {"x": x}, {"u": u}


def _make_2d_data(
    nx: int = 50, nt: int = 30
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    """2D: u(x,t) = sin(x)*cos(t)."""
    x_1d = torch.linspace(0, 2 * math.pi, nx)
    t_1d = torch.linspace(0, math.pi, nt)
    xx, tt = torch.meshgrid(x_1d, t_1d, indexing="ij")
    x_flat = xx.reshape(-1)
    t_flat = tt.reshape(-1)
    u_flat = torch.sin(x_flat) * torch.cos(t_flat)
    return {"x": x_flat, "t": t_flat}, {"u": u_flat}


# -- basic interface -------------------------------------------------------


class TestTrainerInterface:
    """Test basic construction and return type."""

    def test_creates_trainer(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model)
        assert trainer is not None

    def test_fit_returns_training_result(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(coords, targets, max_epochs=10, patience=None)
        assert isinstance(result, TrainingResult)
        assert isinstance(result.final_loss, float)
        assert isinstance(result.epochs_run, int)
        assert isinstance(result.early_stopped, bool)


# -- fitting quality -------------------------------------------------------


class TestFittingQuality:
    """Acceptance: model must actually learn the target function."""

    def test_fit_sin_1d(self) -> None:
        """sin(x) should reach loss < 1e-4."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[64, 64])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(100)
        result = trainer.fit(
            coords, targets, max_epochs=5000, patience=None, val_ratio=0.0, seed=42
        )
        assert result.final_loss < 1e-4, f"sin(x) loss too high: {result.final_loss}"

    def test_fit_2d_sin_cos(self) -> None:
        """sin(x)*cos(t) should reach loss < 1e-3."""
        model = FieldModel(
            coord_names=["x", "t"], field_names=["u"], hidden_sizes=[64, 64]
        )
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_2d_data(50, 30)
        result = trainer.fit(
            coords, targets, max_epochs=5000, patience=None, val_ratio=0.0, seed=42
        )
        assert result.final_loss < 1e-3, f"2D loss too high: {result.final_loss}"


# -- early stopping --------------------------------------------------------


class TestEarlyStopping:
    """Test early stopping behaviour."""

    def test_early_stop_triggers(self) -> None:
        """With patience, training should stop before max_epochs once converged."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[64, 64])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(100)
        result = trainer.fit(
            coords, targets, max_epochs=10000, patience=200, val_ratio=0.2, seed=42
        )
        assert result.early_stopped is True
        assert result.epochs_run < 10000

    def test_patience_none_runs_full(self) -> None:
        """patience=None → run all max_epochs."""
        max_ep = 200
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=max_ep, patience=None, val_ratio=0.0
        )
        assert result.epochs_run == max_ep
        assert result.early_stopped is False

    def test_val_loss_present_when_val_split(self) -> None:
        """val_ratio > 0 → result.val_loss is not None."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=50, patience=None, val_ratio=0.2
        )
        assert result.val_loss is not None

    def test_val_loss_none_when_no_split(self) -> None:
        """val_ratio=0 → result.val_loss is None."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=50, patience=None, val_ratio=0.0
        )
        assert result.val_loss is None


# -- RNG isolation ---------------------------------------------------------


class TestRNGIsolation:
    """RNG state must be unchanged after fit()."""

    def test_rng_state_preserved(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)

        state_before = torch.get_rng_state().clone()
        trainer.fit(coords, targets, max_epochs=50, patience=None, val_ratio=0.0)
        state_after = torch.get_rng_state()

        assert torch.equal(state_before, state_after), "RNG state leaked!"


# -- reproducibility -------------------------------------------------------


class TestReproducibility:
    """Same seed → same final_loss."""

    def test_same_seed_same_loss(self) -> None:
        coords, targets = _make_sin_data(50)

        def _run(seed: int) -> float:
            model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
            trainer = FieldModelTrainer(model, lr=1e-3)
            result = trainer.fit(
                coords, targets, max_epochs=100, patience=None, val_ratio=0.0, seed=seed
            )
            return result.final_loss

        loss_a = _run(seed=123)
        loss_b = _run(seed=123)
        assert loss_a == loss_b, f"Reproducibility failed: {loss_a} vs {loss_b}"

    def test_different_seed_different_loss(self) -> None:
        coords, targets = _make_sin_data(50)

        def _run(seed: int) -> float:
            model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
            trainer = FieldModelTrainer(model, lr=1e-3)
            result = trainer.fit(
                coords, targets, max_epochs=100, patience=None, val_ratio=0.0, seed=seed
            )
            return result.final_loss

        loss_a = _run(seed=0)
        loss_b = _run(seed=999)
        # Very unlikely to be equal with different seeds
        assert loss_a != loss_b


# -- normalization ---------------------------------------------------------


class TestAutoNormalization:
    """fit() should set normalization buffers on the model."""

    def test_normalization_buffers_set(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(100)
        trainer.fit(coords, targets, max_epochs=10, patience=None, val_ratio=0.0)

        # Check that coord mean/std are set (not identity 0/1)
        x_mean = model.coord_x_mean
        x_std = model.coord_x_std
        assert x_mean.item() != 0.0, "coord mean should be set"
        assert x_std.item() != 1.0, "coord std should be set"

        # Check field stats
        u_mean = model.field_u_mean
        u_std = model.field_u_std
        # sin(x) on [0, 2pi] has mean ≈ 0, std ≈ 0.707 — std should differ from 1
        assert u_std.item() != 1.0, "field std should be set"

    def test_normalization_values_correct(self) -> None:
        """Normalization stats should match data statistics."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        x = torch.linspace(0, 10, 100)
        u = x * 2 # mean=10, std known
        coords = {"x": x}
        targets = {"u": u}
        trainer.fit(coords, targets, max_epochs=10, patience=None, val_ratio=0.0)

        x_mean = model.coord_x_mean.item()
        x_std = model.coord_x_std.item()
        assert abs(x_mean - x.mean().item()) < 1e-6
        assert abs(x_std - x.std().item()) < 1e-6

        u_mean = model.field_u_mean.item()
        u_std = model.field_u_std.item()
        assert abs(u_mean - u.mean().item()) < 1e-6
        assert abs(u_std - u.std().item()) < 1e-6


# -- validation ------------------------------------------------------------


class TestValidation:
    """Input validation tests."""

    def test_val_ratio_1_raises(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        with pytest.raises(ValueError, match="val_ratio"):
            trainer.fit(coords, targets, max_epochs=10, val_ratio=1.0)

    def test_val_ratio_negative_raises(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        with pytest.raises(ValueError, match="val_ratio"):
            trainer.fit(coords, targets, max_epochs=10, val_ratio=-0.1)


# -- model state after training -------------------------------------------


class TestModelState:
    """Model should be in eval mode after training."""

    def test_model_eval_after_fit(self) -> None:
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        trainer.fit(coords, targets, max_epochs=10, patience=None, val_ratio=0.0)
        assert not model.training, "Model should be in eval mode after fit()"


# -- TrainingResult completeness -------------------------------------------


class TestTrainingResultCompleteness:
    """Verify TrainingResult fields are correctly populated."""

    def test_epochs_run_leq_max_epochs(self) -> None:
        """epochs_run should never exceed max_epochs."""
        max_ep = 150
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=max_ep, patience=None, val_ratio=0.0
        )
        assert result.epochs_run <= max_ep

    def test_epochs_run_equals_max_when_no_early_stop(self) -> None:
        """Without early stopping, epochs_run == max_epochs."""
        max_ep = 80
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=max_ep, patience=None, val_ratio=0.0
        )
        assert result.epochs_run == max_ep
        assert result.early_stopped is False

    def test_final_loss_is_finite(self) -> None:
        """final_loss must be a finite float."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=50, patience=None, val_ratio=0.0
        )
        assert math.isfinite(result.final_loss)
        assert result.final_loss >= 0.0

    def test_val_loss_is_finite_when_present(self) -> None:
        """val_loss, if present, must be finite and non-negative."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=50, patience=None, val_ratio=0.2
        )
        assert result.val_loss is not None
        assert math.isfinite(result.val_loss)
        assert result.val_loss >= 0.0

    def test_early_stopped_false_when_patience_none(self) -> None:
        """early_stopped is always False when patience=None."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)
        result = trainer.fit(
            coords, targets, max_epochs=50, patience=None, val_ratio=0.2
        )
        assert result.early_stopped is False


# -- early stopping supplemental ------------------------------------------


class TestEarlyStoppingSupplemental:
    """Additional early stopping tests."""

    def test_early_stopped_flag_consistent_with_epochs(self) -> None:
        """If early_stopped is True, epochs_run < max_epochs."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[64, 64])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(100)
        result = trainer.fit(
            coords, targets, max_epochs=10000, patience=200, val_ratio=0.2, seed=42
        )
        if result.early_stopped:
            assert result.epochs_run < 10000
        else:
            assert result.epochs_run == 10000


# -- multi-field training --------------------------------------------------


class TestMultiFieldTraining:
    """Training with multiple output fields."""

    def test_two_field_training(self) -> None:
        """Train model with two output fields simultaneously."""
        n = 100
        x = torch.linspace(0, 2 * math.pi, n)
        u = torch.sin(x)
        v = torch.cos(x)
        coords = {"x": x}
        targets = {"u": u, "v": v}

        model = FieldModel(
            coord_names=["x"], field_names=["u", "v"], hidden_sizes=[64, 64]
        )
        trainer = FieldModelTrainer(model, lr=1e-3)
        result = trainer.fit(
            coords, targets, max_epochs=5000, patience=None, val_ratio=0.0, seed=42
        )
        # Both fields should be learned reasonably well
        assert result.final_loss < 1e-3, (
            f"Multi-field loss too high: {result.final_loss}"
        )

    def test_multi_field_normalization_buffers(self) -> None:
        """fit() should set normalization for each field."""
        n = 50
        x = torch.linspace(0, 2 * math.pi, n)
        u = torch.sin(x)
        v = torch.cos(x)
        coords = {"x": x}
        targets = {"u": u, "v": v}

        model = FieldModel(coord_names=["x"], field_names=["u", "v"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        trainer.fit(coords, targets, max_epochs=10, patience=None, val_ratio=0.0)
        # Both fields should have normalization set
        assert hasattr(model, "field_u_mean")
        assert hasattr(model, "field_v_mean")
        assert hasattr(model, "field_u_std")
        assert hasattr(model, "field_v_std")
        # u and v have different distributions, so their stats differ
        assert model.field_u_mean.item() != model.field_v_mean.item()


# -- weight_decay ---------------------------------------------------------


class TestWeightDecay:
    """Test that weight_decay parameter has an effect."""

    def test_weight_decay_changes_training(self) -> None:
        """Different weight_decay values lead to different loss trajectories."""
        coords, targets = _make_sin_data(50)

        def _run(wd: float) -> float:
            model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
            trainer = FieldModelTrainer(model, lr=1e-3, weight_decay=wd)
            result = trainer.fit(
                coords, targets, max_epochs=200, patience=None, val_ratio=0.0, seed=42
            )
            return result.final_loss

        loss_no_wd = _run(0.0)
        loss_high_wd = _run(0.1)
        # With high weight_decay, optimizer penalizes large weights,
        # so loss should differ from no weight_decay
        assert loss_no_wd != loss_high_wd


# -- numpy RNG isolation --------------------------------------------------


class TestNumpyRNGIsolation:
    """Ensure numpy RNG (if used) is not affected by training."""

    def test_numpy_rng_state_preserved(self) -> None:
        """Training should not change numpy random state."""
        import numpy as np

        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        coords, targets = _make_sin_data(50)

        state_before = np.random.get_state()
        trainer.fit(coords, targets, max_epochs=50, patience=None, val_ratio=0.0)
        state_after = np.random.get_state()

        # Element [1] is the uint32 array of internal state
        assert np.array_equal(state_before[1], state_after[1]) # type: ignore[index]


# -- refit (training same model twice) ------------------------------------


class TestRefit:
    """Training the same model twice with different seeds."""

    def test_refit_resets_weights(self) -> None:
        """fit() re-initializes weights, so two calls with same seed give same loss."""
        coords, targets = _make_sin_data(50)
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)

        result1 = trainer.fit(
            coords, targets, max_epochs=100, patience=None, val_ratio=0.0, seed=42
        )
        result2 = trainer.fit(
            coords, targets, max_epochs=100, patience=None, val_ratio=0.0, seed=42
        )
        assert result1.final_loss == result2.final_loss


class TestDtypeAutoMatch:
    """Trainer should auto-cast model to match input data dtype."""

    def test_float64_input_casts_model(self) -> None:
        """float64 data → model auto-cast to float64."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        assert next(model.parameters()).dtype == torch.float32

        trainer = FieldModelTrainer(model, lr=1e-3)
        x = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float64)
        u = torch.sin(x)
        result = trainer.fit(
            {"x": x}, {"u": u}, max_epochs=50, patience=None, val_ratio=0.0, seed=0
        )
        # Model should now be float64
        assert next(model.parameters()).dtype == torch.float64
        assert result.final_loss < 1.0 # sanity: training ran

    def test_float32_input_keeps_model(self) -> None:
        """float32 data → model stays float32 (no cast)."""
        model = FieldModel(coord_names=["x"], field_names=["u"], hidden_sizes=[16])
        trainer = FieldModelTrainer(model, lr=1e-3)
        x = torch.linspace(0, 2 * torch.pi, 50, dtype=torch.float32)
        u = torch.sin(x)
        result = trainer.fit(
            {"x": x}, {"u": u}, max_epochs=50, patience=None, val_ratio=0.0, seed=0
        )
        assert next(model.parameters()).dtype == torch.float32
        assert result.final_loss < 1.0
