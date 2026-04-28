"""FieldModel — neural network surrogate for PDE field data.

Fits continuous, differentiable functions to observed field data.
Normalization is handled internally — autograd derivatives are
automatically correct in the original coordinate space.

Architecture:
    coords → normalize → concat → shared MLP trunk → multi-head → denormalize

Usage:
    model = FieldModel(coord_names=["x", "t"], field_names=["u"])
    model.set_normalization(coord_stats, field_stats)
    output = model(x=x_tensor, t=t_tensor) # → {"u": u_tensor}
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

logger = logging.getLogger(__name__)

# Supported activation functions
_ACTIVATIONS: dict[str, Callable[[], Callable[[Tensor], Tensor]]] = {
    "tanh": lambda: nn.Tanh(),
    "relu": lambda: nn.ReLU(),
    "sin": lambda: torch.sin,
}

# Fallback std when the data has zero variance
_STD_FLOOR = 1.0


class FieldModel(nn.Module):
    """Neural network surrogate for field data.

    Accepts raw coordinate tensors, normalizes internally, runs through
    a shared MLP trunk, and denormalizes the per-field head outputs.
    Because normalization/denormalization live inside the forward graph,
    ``torch.autograd.grad`` produces correct derivatives in the original
    coordinate space without manual chain-rule corrections.

    Attributes:
        coord_names: Names of input coordinate dimensions.
        field_names: Names of output field variables.
        n_coords: Number of coordinate dimensions.
        n_fields: Number of output fields.
    """

    def __init__(
        self,
        coord_names: list[str],
        field_names: list[str],
        hidden_sizes: list[int] | None = None,
        activation: str = "tanh",
    ) -> None:
        """Initialize FieldModel.

        Args:
            coord_names: Ordered list of coordinate dimension names
                (e.g. ``["x", "t"]``).
            field_names: Ordered list of field variable names
                (e.g. ``["u"]`` or ``["u", "v"]``).
            hidden_sizes: Widths of hidden layers. Defaults to ``[64, 64]``.
            activation: Activation function name — one of
                ``"tanh"``, ``"sin"``, ``"relu"``.

        Raises:
            ValueError: If *coord_names* is empty.
            ValueError: If *field_names* is empty.
            ValueError: If *hidden_sizes* is empty.
            ValueError: If *activation* is not a recognised name.
        """
        super().__init__()

        # --- validation ---------------------------------------------------
        if not coord_names:
            raise ValueError("coord_names must not be empty")
        if not field_names:
            raise ValueError("field_names must not be empty")
        if hidden_sizes is None:
            hidden_sizes = [64, 64]
        if not hidden_sizes:
            raise ValueError("hidden_sizes must not be empty")
        if activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{activation}'. Supported: {sorted(_ACTIVATIONS)}"
            )

        # --- metadata (not parameters) ------------------------------------
        self.coord_names = list(coord_names)
        self.field_names = list(field_names)
        self.n_coords = len(coord_names)
        self.n_fields = len(field_names)

        # --- build shared trunk -------------------------------------------
        act_factory = _ACTIVATIONS[activation]
        layers: list[nn.Module | Callable[[Tensor], Tensor]] = []
        in_size = self.n_coords
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(act_factory())
            in_size = h

        # Wrap in Sequential — sin (a plain function) needs FuncModule
        wrapped: list[nn.Module] = []
        for layer in layers:
            if isinstance(layer, nn.Module):
                wrapped.append(layer)
            else:
                wrapped.append(_FuncModule(layer))
        self.trunk = nn.Sequential(*wrapped)

        # --- multi-head output --------------------------------------------
        self.head = nn.Linear(in_size, self.n_fields)

        # --- normalization buffers (initially identity: mean=0, std=1) -----
        self._register_norm_buffers()

    # Normalization

    def _register_norm_buffers(self) -> None:
        """Register default (identity) normalization buffers."""
        for name in self.coord_names:
            self.register_buffer(f"coord_{name}_mean", torch.tensor(0.0))
            self.register_buffer(f"coord_{name}_std", torch.tensor(1.0))
        for name in self.field_names:
            self.register_buffer(f"field_{name}_mean", torch.tensor(0.0))
            self.register_buffer(f"field_{name}_std", torch.tensor(1.0))

    def set_normalization(
        self,
        coord_stats: dict[str, tuple[Tensor, Tensor]],
        field_stats: dict[str, tuple[Tensor, Tensor]],
    ) -> None:
        """Set normalization parameters as registered buffers.

        After calling this method, :meth:`forward` will normalize inputs
        and denormalize outputs using the provided statistics.

        Args:
            coord_stats: ``{name: (mean, std)}`` for each coordinate.
            field_stats: ``{name: (mean, std)}`` for each field.

        Note:
            If *std* is zero for any entry, it is silently replaced
            with 1.0 to avoid division by zero.
        """
        for name in self.coord_names:
            mean, std = coord_stats[name]
            std = _floor_std(std)
            # Copy into existing buffers (preserves device)
            getattr(self, f"coord_{name}_mean").copy_(mean)
            getattr(self, f"coord_{name}_std").copy_(std)

        for name in self.field_names:
            mean, std = field_stats[name]
            std = _floor_std(std)
            getattr(self, f"field_{name}_mean").copy_(mean)
            getattr(self, f"field_{name}_std").copy_(std)

        # Buffers are now updated; forward() uses them automatically.

    # Forward

    def forward(self, **coords: Tensor) -> dict[str, Tensor]:
        """Run forward pass: raw coords in, raw field values out.

        All normalization and denormalization happen inside the
        computation graph so autograd derivatives are correct.

        Args:
            **coords: Keyword tensors keyed by *coord_names*.
                Each tensor should be 1-D with shape ``(N,)``.

        Returns:
            Dict mapping each field name to an output tensor of
            shape ``(N,)``.
        """
        # --- normalize coords & stack ------------------------------------
        normed: list[Tensor] = []
        for name in self.coord_names:
            c = coords[name]
            mean: Tensor = getattr(self, f"coord_{name}_mean")
            std: Tensor = getattr(self, f"coord_{name}_std")
            normed.append((c - mean) / std)

        inp = torch.stack(normed, dim=-1) # (N, n_coords)

        # --- shared trunk → head -----------------------------------------
        hidden = self.trunk(inp) # (N, last_hidden)
        raw_out = self.head(hidden) # (N, n_fields)

        # --- denormalize per field & split --------------------------------
        result: dict[str, Tensor] = {}
        for i, name in enumerate(self.field_names):
            f_mean: Tensor = getattr(self, f"field_{name}_mean")
            f_std: Tensor = getattr(self, f"field_{name}_std")
            result[name] = raw_out[..., i] * f_std + f_mean

        return result


# ======================================================================
# Helpers
# ======================================================================


class _FuncModule(nn.Module):
    """Wrap a plain callable (e.g. ``torch.sin``) as an ``nn.Module``."""

    def __init__(self, func: Callable[[Tensor], Tensor]) -> None:
        super().__init__()
        self._func = func

    def forward(self, x: Tensor) -> Tensor:
        return self._func(x)


def _floor_std(std: Tensor) -> Tensor:
    """Replace zero std with ``_STD_FLOOR`` to avoid division by zero."""
    if std.item() == 0.0:
        return torch.tensor(_STD_FLOOR, dtype=std.dtype, device=std.device)
    return std
