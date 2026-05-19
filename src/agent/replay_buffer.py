"""
Replay Buffer module.

This module will store transitions sampled during agent interaction with the
environment.
"""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np


class ReplayBuffer:
    """Fixed-size replay buffer storing transitions as numpy-compatible values."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: deque[dict[str, Any]] = deque(maxlen=capacity)

    def push(
        self,
        state: Any,
        action: int,
        reward: float,
        next_state: Any,
        done: bool,
        action_mask: Any = None,
        next_action_mask: Any = None,
    ) -> None:
        """Store one transition."""
        self.buffer.append(
            {
                "state": np.asarray(state, dtype=np.float32),
                "action": int(action),
                "reward": float(reward),
                "next_state": np.asarray(next_state, dtype=np.float32),
                "done": bool(done),
                "action_mask": None
                if action_mask is None
                else np.asarray(action_mask, dtype=np.float32),
                "next_action_mask": None
                if next_action_mask is None
                else np.asarray(next_action_mask, dtype=np.float32),
            }
        )

    def sample(self, batch_size: int) -> dict[str, np.ndarray | None]:
        """Randomly sample transitions and return stacked numpy arrays."""
        if batch_size > len(self.buffer):
            raise ValueError("Cannot sample more transitions than the buffer contains.")

        batch = random.sample(list(self.buffer), batch_size)
        action_masks = [item["action_mask"] for item in batch]
        next_action_masks = [item["next_action_mask"] for item in batch]

        return {
            "states": np.stack([item["state"] for item in batch]),
            "actions": np.asarray([item["action"] for item in batch], dtype=np.int64),
            "rewards": np.asarray([item["reward"] for item in batch], dtype=np.float32),
            "next_states": np.stack([item["next_state"] for item in batch]),
            "dones": np.asarray([item["done"] for item in batch], dtype=np.float32),
            "action_masks": None
            if any(mask is None for mask in action_masks)
            else np.stack(action_masks),
            "next_action_masks": None
            if any(mask is None for mask in next_action_masks)
            else np.stack(next_action_masks),
        }

    def __len__(self) -> int:
        """Return the number of stored transitions."""
        return len(self.buffer)
