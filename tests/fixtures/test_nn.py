"""Neural network fixtures for integration tests.

Provides:
- SimpleMLP: Simple MLP for approximating PDE fields (2-input, 1-output)
- train_test_nn: Training helper with deterministic seeding

Usage:
    model = SimpleMLP(hidden_size=64)
    train_test_nn(model, x, t, u_target, epochs=500, lr=1e-3)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class SimpleMLP(nn.Module):
    """Simple MLP for PDE field approximation.

    Architecture: (x, t) -> Linear -> Tanh -> Linear -> Tanh -> Linear -> u
    Two inputs (x, t), one output (u), with configurable hidden size.

    Accepts keyword arguments x and t to match AutogradProvider conventions.
    """

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, *, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass.

        Args:
            x: Spatial coordinate tensor (1D, flattened).
            t: Temporal coordinate tensor (1D, flattened).

        Returns:
            Predicted field values, same shape as x.
        """
        inp = torch.stack([x, t], dim=-1)
        out: Tensor = self.net(inp).squeeze(-1)
        return out


def train_test_nn(
    model: nn.Module,
    x: Tensor,
    t: Tensor,
    u_target: Tensor,
    epochs: int = 500,
    lr: float = 1e-3,
    seed: int = 42,
) -> float:
    """Train a test neural network on target data.

    Uses MSE loss and Adam optimizer. Sets torch.manual_seed for
    reproducibility before training.

    Args:
        model: Neural network model to train.
        x: Spatial coordinates (1D, requires_grad will be set).
        t: Temporal coordinates (1D, requires_grad will be set).
        u_target: Target field values (1D, same length as x).
        epochs: Number of training epochs.
        lr: Learning rate for Adam optimizer.
        seed: Random seed for reproducibility.

    Returns:
        Final training loss value.
    """
    torch.manual_seed(seed)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Detach inputs for training (we don't need grad w.r.t. coords during training)
    x_train = x.detach()
    t_train = t.detach()

    model.train()
    final_loss = float("inf")

    for _epoch in range(epochs):
        optimizer.zero_grad()
        u_pred = model(x=x_train, t=t_train)
        loss = loss_fn(u_pred, u_target.detach())
        loss.backward()
        optimizer.step()
        final_loss = loss.detach().item()

    model.eval()
    return final_loss
