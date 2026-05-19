"""
Training script for the DQN Tool Selection Agent.

This script will eventually launch DQN training from configuration settings.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.dqn_agent import DQNAgent
from src.agent.trainer import DQNTrainer
from src.data.dataset_manager import DatasetManager
from src.env.tool_selection_env import ToolSelectionEnv
from src.evaluator.evaluator import Evaluator
from src.tools.pipeline_cache import PipelineCache


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train the DQN Tool Selection Agent.")
    parser.add_argument("--episodes", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default="outputs")
    return parser.parse_args()


def main() -> None:
    """Load datasets, train DQN, and save logs/models."""
    args = parse_args()
    set_seed(args.seed)

    datasets = DatasetManager(random_state=args.seed).load_all()
    cache = PipelineCache()
    evaluator = Evaluator(cache=cache, random_state=args.seed)
    env = ToolSelectionEnv(evaluator=evaluator, max_steps=6, random_state=args.seed)
    agent = DQNAgent(
        state_dim=15,
        action_dim=9,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        batch_size=args.batch_size,
        random_state=args.seed,
    )
    trainer = DQNTrainer(
        agent=agent,
        env=env,
        datasets=datasets,
        output_dir=args.output_dir,
        num_episodes=args.episodes,
        max_steps=6,
        save_interval=max(1, args.episodes // 5),
        log_interval=max(1, args.episodes // 10),
        random_state=args.seed,
    )

    summary = trainer.train()
    print("\nDQN Training Summary")
    print("====================")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"Cache stats: {cache.stats()}")


def set_seed(seed: int) -> None:
    """Set common random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


if __name__ == "__main__":
    main()
