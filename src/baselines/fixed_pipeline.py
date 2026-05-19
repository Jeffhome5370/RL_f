"""
Fixed Pipeline baseline module.

This module will provide predefined pipeline baselines for comparison.
"""

from __future__ import annotations

import time
from typing import Any

from src.baselines.random_agent import build_episode_result, failed_episode_result


FIXED_PIPELINES = {
    "fixed_scaler_svm": [1, 6, 8],
    "fixed_feature_rf": [4, 5, 8],
    "fixed_minmax_knn": [2, 7, 8],
    "fixed_rf_only": [5, 8],
}


class FixedPipelineAgent:
    """Baseline that executes a predefined integer action sequence."""

    def __init__(
        self,
        name: str,
        actions: list[int],
    ) -> None:
        self.name = name
        self.actions = list(actions)

    def run_episode(self, env: Any, dataset: dict[str, Any]) -> dict[str, Any]:
        """Run the fixed action sequence through the environment."""
        start_time = time.perf_counter()
        total_reward = 0.0
        steps = 0
        final_info: dict[str, Any] | None = None

        try:
            env.reset(dataset)
            done = False
            for action in self.actions:
                if done:
                    break
                _, reward, done, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                final_info = info

            return build_episode_result(
                method=self.name,
                dataset=dataset,
                total_reward=total_reward,
                steps=steps,
                final_info=final_info,
                elapsed_runtime_sec=time.perf_counter() - start_time,
            )
        except Exception as exc:
            return failed_episode_result(
                method=self.name,
                dataset=dataset,
                total_reward=total_reward,
                steps=steps,
                final_info=final_info,
                elapsed_runtime_sec=time.perf_counter() - start_time,
                error=str(exc),
            )
