"""
Q-Network module.

This module will define the neural network used to estimate action values.
"""

from __future__ import annotations

import torch
from torch import nn


class QNetwork(nn.Module):
    """Small multilayer perceptron for estimating action values."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [128, 128]

        layers: list[nn.Module] = []
        input_dim = state_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, action_dim))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Q-values for each action."""
        return self.network(x)
