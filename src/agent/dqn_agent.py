"""
DQN Agent module.

This module will coordinate action selection, learning updates, and target
network synchronization.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from src.agent.q_network import QNetwork
from src.agent.replay_buffer import ReplayBuffer


class DQNAgent:
    """DQN agent with masked epsilon-greedy action selection."""

    def __init__(
        self,
        state_dim: int = 15,
        action_dim: int = 9,
        hidden_dims: list[int] | None = None,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 1000,
        buffer_capacity: int = 10000,
        batch_size: int = 64,
        target_update_interval: int = 100,
        device: str | None = None,
        use_action_mask: bool = True,
        random_state: int = 42,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dims = hidden_dims or [128, 128]
        self.lr = lr
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.batch_size = batch_size
        self.target_update_interval = max(1, target_update_interval)
        self.use_action_mask = use_action_mask
        self.random_state = random_state

        random.seed(random_state)
        np.random.seed(random_state)
        torch.manual_seed(random_state)

        resolved_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(resolved_device)

        self.q_network = QNetwork(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim, self.hidden_dims).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        self.training_steps = 0
        self.optimization_steps = 0

    def select_action(
        self,
        state: Any,
        action_mask: Any = None,
        training: bool = True,
    ) -> int:
        """Select an integer action with optional action masking."""
        epsilon = self.get_epsilon() if training else 0.0

        if training and random.random() < epsilon:
            action = self._sample_random_action(action_mask)
        else:
            action = self._select_greedy_action(state, action_mask)

        if training:
            self.training_steps += 1
        return int(action)

    def update(self) -> dict[str, float | bool | None]:
        """Run one DQN optimization step if enough replay data exists."""
        if len(self.replay_buffer) < self.batch_size:
            return {"loss": None, "updated": False}

        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch["states"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch["rewards"], dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(batch["next_states"], dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch["dones"], dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_network(states).gather(1, actions)

        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_action_masks = batch["next_action_masks"]
            if self.use_action_mask and next_action_masks is not None:
                mask = torch.as_tensor(
                    next_action_masks,
                    dtype=torch.float32,
                    device=self.device,
                )
                next_q_values = next_q_values.masked_fill(mask <= 0, -1e9)
            max_next_q = next_q_values.max(dim=1, keepdim=True).values
            target_q = rewards + self.gamma * max_next_q * (1.0 - dones)

        loss = nn.functional.mse_loss(q_values, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.optimization_steps += 1
        if self.optimization_steps % self.target_update_interval == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return {"loss": float(loss.item()), "updated": True}

    def get_epsilon(self) -> float:
        """Return current linearly decayed epsilon."""
        progress = min(1.0, self.training_steps / float(self.epsilon_decay_steps))
        return float(
            self.epsilon_start
            + progress * (self.epsilon_end - self.epsilon_start)
        )

    def save(self, path: str) -> None:
        """Save model, optimizer, and training counters."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self._config(),
                "training_steps": self.training_steps,
                "optimization_steps": self.optimization_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load model, optimizer, and training counters when present."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint["q_network"])
        if "target_network" in checkpoint:
            self.target_network.load_state_dict(checkpoint["target_network"])
        else:
            self.target_network.load_state_dict(self.q_network.state_dict())
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.training_steps = int(checkpoint.get("training_steps", self.training_steps))
        self.optimization_steps = int(
            checkpoint.get("optimization_steps", self.optimization_steps)
        )

    def _sample_random_action(self, action_mask: Any = None) -> int:
        """Sample a random action, respecting the mask when possible."""
        valid_actions = self._valid_actions_from_mask(action_mask)
        if valid_actions:
            return int(random.choice(valid_actions))
        return int(random.randrange(self.action_dim))

    def _select_greedy_action(self, state: Any, action_mask: Any = None) -> int:
        """Select the highest-Q action, masking invalid actions when available."""
        state_tensor = torch.as_tensor(
            np.asarray(state, dtype=np.float32),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor).squeeze(0)
            if self.use_action_mask and action_mask is not None:
                mask_tensor = torch.as_tensor(
                    np.asarray(action_mask, dtype=np.float32),
                    dtype=torch.float32,
                    device=self.device,
                )
                if torch.sum(mask_tensor > 0).item() > 0:
                    q_values = q_values.masked_fill(mask_tensor <= 0, -1e9)
                else:
                    return int(random.randrange(self.action_dim))
            return int(torch.argmax(q_values).item())

    def _valid_actions_from_mask(self, action_mask: Any = None) -> list[int]:
        """Extract valid action indices from a mask."""
        if not self.use_action_mask or action_mask is None:
            return list(range(self.action_dim))
        mask = np.asarray(action_mask, dtype=np.float32)
        return [int(index) for index, value in enumerate(mask) if value > 0]

    def _config(self) -> dict[str, Any]:
        """Return checkpoint configuration metadata."""
        return {
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "hidden_dims": self.hidden_dims,
            "lr": self.lr,
            "gamma": self.gamma,
            "epsilon_start": self.epsilon_start,
            "epsilon_end": self.epsilon_end,
            "epsilon_decay_steps": self.epsilon_decay_steps,
            "batch_size": self.batch_size,
            "target_update_interval": self.target_update_interval,
            "use_action_mask": self.use_action_mask,
            "random_state": self.random_state,
        }
