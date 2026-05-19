"""
Random Agent baseline module.

This module will provide a random tool-selection baseline.
"""

from __future__ import annotations

import random
import time
from typing import Any

from src.constants import ACTIONS


class RandomAgent:
    """Random baseline that interacts with ToolSelectionEnv using integer actions."""

    def __init__(
        self,
        use_action_mask: bool = True,
        random_state: int = 42,
    ) -> None:
        self.use_action_mask = use_action_mask
        self.rng = random.Random(random_state)

    def select_action(self, env: Any) -> int:
        """Select a random action, optionally restricted to valid actions."""
        if self.use_action_mask:
            valid_actions = env.get_valid_actions()
            if not valid_actions:
                raise ValueError("No valid actions available.")
            return int(self.rng.choice(valid_actions))
        return int(self.rng.choice(list(ACTIONS.keys())))

    def run_episode(self, env: Any, dataset: dict[str, Any]) -> dict[str, Any]:
        """Run one random episode until the environment terminates."""
        start_time = time.perf_counter()
        total_reward = 0.0
        steps = 0
        final_info: dict[str, Any] | None = None

        try:
            env.reset(dataset)
            done = False
            while not done:
                action = self.select_action(env)
                _, reward, done, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                final_info = info

            return build_episode_result(
                method="random_agent",
                dataset=dataset,
                total_reward=total_reward,
                steps=steps,
                final_info=final_info,
                elapsed_runtime_sec=time.perf_counter() - start_time,
            )
        except Exception as exc:
            return failed_episode_result(
                method="random_agent",
                dataset=dataset,
                total_reward=total_reward,
                steps=steps,
                final_info=final_info,
                elapsed_runtime_sec=time.perf_counter() - start_time,
                error=str(exc),
            )


def build_episode_result(
    method: str,
    dataset: dict[str, Any],
    total_reward: float,
    steps: int,
    final_info: dict[str, Any] | None,
    elapsed_runtime_sec: float,
) -> dict[str, Any]:
    """Create a standard baseline result dictionary from final environment info."""
    final_info = final_info or {}
    evaluation_result = final_info.get("evaluation_result")
    evaluate_success = bool(
        evaluation_result is not None and evaluation_result.get("status") == "success"
    )

    return {
        "method": method,
        "dataset_name": dataset.get("name", "unknown"),
        "total_reward": float(total_reward),
        "macro_f1": evaluation_result.get("macro_f1") if evaluate_success else None,
        "pipeline_actions": list(final_info.get("pipeline_actions", [])),
        "pipeline_length": len(final_info.get("pipeline_actions", [])),
        "invalid_action_count": int(final_info.get("invalid_action_count", 0)),
        "evaluate_success": evaluate_success,
        "runtime_sec": float(
            evaluation_result.get("runtime_sec", elapsed_runtime_sec)
            if evaluation_result
            else elapsed_runtime_sec
        ),
        "steps": int(steps),
        "status": "success" if evaluate_success else "failed",
        "error": None if evaluate_success else _extract_error(evaluation_result),
    }


def failed_episode_result(
    method: str,
    dataset: dict[str, Any],
    total_reward: float,
    steps: int,
    final_info: dict[str, Any] | None,
    elapsed_runtime_sec: float,
    error: str,
) -> dict[str, Any]:
    """Create a standard failed baseline result dictionary."""
    result = build_episode_result(
        method=method,
        dataset=dataset,
        total_reward=total_reward,
        steps=steps,
        final_info=final_info,
        elapsed_runtime_sec=elapsed_runtime_sec,
    )
    result["status"] = "failed"
    result["error"] = error
    result["evaluate_success"] = False
    result["macro_f1"] = None
    return result


def _extract_error(evaluation_result: dict[str, Any] | None) -> str | None:
    """Return a useful error string from an evaluation result."""
    if evaluation_result is None:
        return "Episode ended without a successful evaluation."
    return evaluation_result.get("error")
