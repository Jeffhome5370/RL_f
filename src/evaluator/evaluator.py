"""
Evaluator module.

This module will evaluate candidate pipelines on validation or test data.
"""

from __future__ import annotations

import time
from typing import Any

from src.evaluator.metrics import compute_macro_f1
from src.tools.pipeline_builder import PipelineBuilder
from src.tools.pipeline_cache import PipelineCache


class Evaluator:
    """Build, train, and validate sklearn pipelines from action sequences."""

    def __init__(
        self,
        pipeline_builder: PipelineBuilder | None = None,
        cache: PipelineCache | None = None,
        random_state: int = 42,
        use_cache: bool = True,
    ) -> None:
        self.pipeline_builder = pipeline_builder or PipelineBuilder(random_state=random_state)
        self.cache = cache
        self.use_cache = use_cache

    def evaluate(self, dataset: dict[str, Any], actions: list[str]) -> dict[str, Any]:
        """
        Evaluate an action list on a dataset dictionary.

        Failures are captured in the returned result so batch evaluation scripts
        can continue after one invalid or unsuccessful pipeline.
        """
        start_time = time.perf_counter()
        result = self._base_result(dataset, actions)

        try:
            build_result = self.pipeline_builder.build_pipeline(
                actions,
                n_features=dataset.get("n_features"),
            )
            effective_actions = build_result["effective_actions"]

            result.update(
                {
                    "effective_actions": effective_actions,
                    "pipeline_length": build_result["pipeline_length"],
                    "model_action": build_result["model_action"],
                }
            )

            if self.use_cache and self.cache is not None:
                if self.cache.has(dataset["name"], effective_actions):
                    cached_result = self.cache.get(dataset["name"], effective_actions)
                    cached_result["from_cache"] = True
                    return cached_result

            pipeline = build_result["pipeline"]
            pipeline.fit(dataset["X_train"], dataset["y_train"])
            y_pred = pipeline.predict(dataset["X_val"])

            result.update(
                {
                    "macro_f1": compute_macro_f1(dataset["y_val"], y_pred),
                    "runtime_sec": time.perf_counter() - start_time,
                    "from_cache": False,
                    "status": "success",
                    "error": None,
                }
            )

            if self.use_cache and self.cache is not None:
                self.cache.set(dataset["name"], effective_actions, result)

            return result
        except Exception as exc:
            result.update(
                {
                    "runtime_sec": time.perf_counter() - start_time,
                    "status": "failed",
                    "error": str(exc),
                }
            )
            return result

    @staticmethod
    def _base_result(dataset: dict[str, Any], actions: list[str]) -> dict[str, Any]:
        """Create the standard evaluator result structure."""
        return {
            "dataset_name": dataset.get("name", "unknown"),
            "actions": list(actions),
            "effective_actions": [],
            "pipeline_length": 0,
            "model_action": None,
            "macro_f1": 0.0,
            "runtime_sec": 0.0,
            "from_cache": False,
            "status": "failed",
            "error": None,
        }
