"""
Tests for DQN network, replay buffer, agent, and trainer modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.agent.dqn_agent import DQNAgent
from src.agent.q_network import QNetwork
from src.agent.replay_buffer import ReplayBuffer
from src.agent.trainer import DQNTrainer
from src.data.dataset_manager import DatasetManager
from src.env.tool_selection_env import ToolSelectionEnv


def test_q_network_output_shape() -> None:
    """QNetwork should produce one Q-value per action."""
    network = QNetwork(state_dim=15, action_dim=9)
    q_values = network(torch.randn(4, 15))
    assert q_values.shape == (4, 9)


def test_replay_buffer_push_increases_length() -> None:
    """ReplayBuffer.push should add one transition."""
    buffer = ReplayBuffer(capacity=10)
    buffer.push(
        state=np.zeros(15, dtype=np.float32),
        action=1,
        reward=-0.05,
        next_state=np.ones(15, dtype=np.float32),
        done=False,
    )
    assert len(buffer) == 1


def test_replay_buffer_sample_returns_required_keys() -> None:
    """ReplayBuffer.sample should return the expected batch dictionary."""
    buffer = ReplayBuffer(capacity=10)
    buffer.push(
        state=np.zeros(15, dtype=np.float32),
        action=1,
        reward=-0.05,
        next_state=np.ones(15, dtype=np.float32),
        done=False,
        action_mask=np.ones(9, dtype=np.float32),
        next_action_mask=np.ones(9, dtype=np.float32),
    )
    batch = buffer.sample(batch_size=1)
    assert {
        "states",
        "actions",
        "rewards",
        "next_states",
        "dones",
        "action_masks",
        "next_action_masks",
    }.issubset(batch.keys())


def test_dqn_agent_select_action_returns_integer() -> None:
    """DQNAgent.select_action should return an integer action."""
    agent = DQNAgent(state_dim=15, action_dim=9, random_state=42)
    action = agent.select_action(
        np.zeros(15, dtype=np.float32),
        action_mask=np.ones(9, dtype=np.float32),
        training=True,
    )
    assert isinstance(action, int)


def test_dqn_agent_greedy_selection_respects_action_mask() -> None:
    """Greedy selection should only return valid masked actions."""
    agent = DQNAgent(state_dim=15, action_dim=9, epsilon_start=0.0, random_state=42)
    action_mask = np.asarray([0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=np.float32)
    action = agent.select_action(
        np.zeros(15, dtype=np.float32),
        action_mask=action_mask,
        training=False,
    )
    assert action == 6


def test_dqn_agent_update_returns_not_updated_when_buffer_small() -> None:
    """DQNAgent.update should wait until enough samples exist."""
    agent = DQNAgent(state_dim=15, action_dim=9, batch_size=4, random_state=42)
    result = agent.update()
    assert result == {"loss": None, "updated": False}


def test_dqn_agent_update_returns_loss_after_buffer_fill() -> None:
    """DQNAgent.update should optimize once the replay buffer has enough data."""
    agent = DQNAgent(state_dim=15, action_dim=9, batch_size=2, random_state=42)
    for _ in range(2):
        agent.replay_buffer.push(
            state=np.zeros(15, dtype=np.float32),
            action=1,
            reward=1.0,
            next_state=np.ones(15, dtype=np.float32),
            done=False,
            action_mask=np.ones(9, dtype=np.float32),
            next_action_mask=np.ones(9, dtype=np.float32),
        )

    result = agent.update()
    assert result["updated"] is True
    assert isinstance(result["loss"], float)


def test_dqn_agent_save_creates_model_file(tmp_path: Path) -> None:
    """DQNAgent.save should create a checkpoint file."""
    model_path = tmp_path / "dqn_test.pt"
    DQNAgent(state_dim=15, action_dim=9, random_state=42).save(str(model_path))
    assert model_path.exists()


def test_dqn_agent_load_restores_saved_model(tmp_path: Path) -> None:
    """DQNAgent.load should read a saved checkpoint file."""
    model_path = tmp_path / "dqn_test.pt"
    agent = DQNAgent(state_dim=15, action_dim=9, random_state=42)
    agent.save(str(model_path))

    loaded_agent = DQNAgent(state_dim=15, action_dim=9, random_state=7)
    loaded_agent.load(str(model_path))
    assert loaded_agent.training_steps == agent.training_steps


def test_dqn_trainer_runs_small_training_and_saves_outputs(tmp_path: Path) -> None:
    """DQNTrainer should run a tiny training job and save log/model files."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    env = ToolSelectionEnv(max_steps=3, random_state=42)
    agent = DQNAgent(
        state_dim=15,
        action_dim=9,
        batch_size=2,
        epsilon_decay_steps=10,
        random_state=42,
    )
    trainer = DQNTrainer(
        agent=agent,
        env=env,
        datasets=[dataset],
        output_dir=str(tmp_path),
        num_episodes=3,
        save_interval=10,
        log_interval=10,
        random_state=42,
    )

    summary = trainer.train()
    assert summary["episodes"] == 3
    assert Path(summary["log_path"]).exists()
    assert Path(summary["final_model_path"]).exists()
