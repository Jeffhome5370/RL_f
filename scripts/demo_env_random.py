"""
Demo script for random interaction with the tool-selection environment.

The script samples only valid actions from the action mask and stops when the
environment reaches a terminal evaluation or max-step auto termination.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ACTIONS
from src.data.dataset_manager import DatasetManager
from src.env.tool_selection_env import ToolSelectionEnv


def main() -> None:
    """Run one short random environment episode."""
    random.seed(42)
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    env = ToolSelectionEnv(max_steps=6, random_state=42)

    state = env.reset(dataset)
    done = False
    print(f"Dataset: {dataset['name']}")
    print(f"Initial state shape: {state.shape}")

    while not done:
        valid_actions = env.get_valid_actions()
        action = random.choice(valid_actions)
        next_state, reward, done, info = env.step(action)

        print(f"\nStep: {info['current_step']}")
        print(f"Action: {info['action']} ({ACTIONS.get(info['action'], 'unknown')})")
        print(f"Reward: {reward:.4f}")
        print(f"Done: {done}")
        print(f"Pipeline actions: {info['pipeline_actions']}")
        print(f"Action mask: {env.get_action_mask()}")
        if info["evaluation_result"] is not None:
            print(f"Evaluation status: {info['evaluation_result']['status']}")
            print(f"Macro F1: {info['evaluation_result']['macro_f1']:.4f}")

        state = next_state

    print(f"\nFinal state shape: {state.shape}")


if __name__ == "__main__":
    main()
