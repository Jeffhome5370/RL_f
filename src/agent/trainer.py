"""
Trainer module.

This module will coordinate training episodes and evaluation checkpoints.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any


class DQNTrainer:
    """Train a DQNAgent through ToolSelectionEnv interaction."""

    LOG_FIELDS = [
        "episode",
        "dataset_name",
        "total_reward",
        "macro_f1",
        "pipeline_actions",
        "pipeline_length",
        "invalid_action_count",
        "evaluate_success",
        "runtime_sec",
        "steps",
        "epsilon",
        "loss",
        "status",
    ]

    def __init__(
        self,
        agent: Any,
        env: Any,
        datasets: list[dict[str, Any]] | dict[str, dict[str, Any]],
        output_dir: str = "outputs",
        num_episodes: int = 500,
        max_steps: int = 6,
        save_interval: int = 100,
        log_interval: int = 10,
        random_state: int = 42,
    ) -> None:
        self.agent = agent
        self.env = env
        self.datasets = list(datasets.values()) if isinstance(datasets, dict) else list(datasets)
        self.output_dir = Path(output_dir)
        self.num_episodes = num_episodes
        self.max_steps = max_steps
        self.save_interval = max(1, save_interval)
        self.log_interval = max(1, log_interval)
        self.rng = random.Random(random_state)

        self.log_dir = self.output_dir / "logs"
        self.model_dir = self.output_dir / "models"
        self.log_path = self.log_dir / "dqn_train_log.csv"
        self.final_model_path = self.model_dir / "dqn_final.pt"
        self.best_model_path = self.model_dir / "dqn_best.pt"

    def train(self) -> dict[str, Any]:
        """Run DQN training and save logs plus model checkpoints."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        best_reward = float("-inf")
        best_episode = -1

        for episode in range(1, self.num_episodes + 1):
            dataset = self.rng.choice(self.datasets)
            row = self._run_episode(episode, dataset)
            rows.append(row)

            if row["total_reward"] > best_reward:
                best_reward = float(row["total_reward"])
                best_episode = episode
                self.agent.save(str(self.best_model_path))

            if episode % self.save_interval == 0:
                self.agent.save(str(self.model_dir / f"dqn_episode_{episode}.pt"))
            if episode % self.log_interval == 0:
                print(
                    f"Episode {episode}/{self.num_episodes}: "
                    f"reward={row['total_reward']:.4f}, "
                    f"epsilon={row['epsilon']:.4f}, status={row['status']}"
                )

        self._write_log(rows)
        self.agent.save(str(self.final_model_path))

        return {
            "episodes": self.num_episodes,
            "best_reward": best_reward,
            "best_episode": best_episode,
            "final_epsilon": self.agent.get_epsilon(),
            "log_path": str(self.log_path),
            "final_model_path": str(self.final_model_path),
            "best_model_path": str(self.best_model_path),
        }

    def _run_episode(self, episode: int, dataset: dict[str, Any]) -> dict[str, Any]:
        """Run one environment episode and collect training metrics."""
        state = self.env.reset(dataset)
        action_mask = self.env.get_action_mask()
        done = False
        total_reward = 0.0
        steps = 0
        last_loss: float | None = None
        final_info: dict[str, Any] | None = None

        while not done:
            action = self.agent.select_action(state, action_mask=action_mask, training=True)
            next_state, reward, done, info = self.env.step(action)
            next_action_mask = self.env.get_action_mask()

            self.agent.replay_buffer.push(
                state=state,
                action=action,
                reward=reward,
                next_state=next_state,
                done=done,
                action_mask=action_mask,
                next_action_mask=next_action_mask,
            )
            update_info = self.agent.update()
            if update_info["loss"] is not None:
                last_loss = float(update_info["loss"])

            state = next_state
            action_mask = next_action_mask
            total_reward += float(reward)
            steps += 1
            final_info = info

        return self._build_log_row(
            episode=episode,
            dataset=dataset,
            total_reward=total_reward,
            steps=steps,
            final_info=final_info or {},
            loss=last_loss,
        )

    def _build_log_row(
        self,
        episode: int,
        dataset: dict[str, Any],
        total_reward: float,
        steps: int,
        final_info: dict[str, Any],
        loss: float | None,
    ) -> dict[str, Any]:
        """Convert final episode info into one CSV-ready row."""
        evaluation_result = final_info.get("evaluation_result")
        evaluate_success = bool(
            evaluation_result is not None and evaluation_result.get("status") == "success"
        )
        pipeline_actions = list(final_info.get("pipeline_actions", []))

        return {
            "episode": episode,
            "dataset_name": dataset.get("name", "unknown"),
            "total_reward": total_reward,
            "macro_f1": evaluation_result.get("macro_f1") if evaluate_success else "",
            "pipeline_actions": "|".join(pipeline_actions),
            "pipeline_length": len(pipeline_actions),
            "invalid_action_count": int(final_info.get("invalid_action_count", 0)),
            "evaluate_success": evaluate_success,
            "runtime_sec": evaluation_result.get("runtime_sec", 0.0)
            if evaluation_result
            else 0.0,
            "steps": steps,
            "epsilon": self.agent.get_epsilon(),
            "loss": "" if loss is None else loss,
            "status": "success" if evaluate_success else "failed",
        }

    def _write_log(self, rows: list[dict[str, Any]]) -> None:
        """Write the training log CSV."""
        with self.log_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=self.LOG_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
