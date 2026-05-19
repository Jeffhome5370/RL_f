"""
Grid Search baseline module.

This module will provide a conventional search baseline for selected pipeline
configurations.
"""

from __future__ import annotations

from typing import Any

from src.evaluator.evaluator import Evaluator
from src.tools.pipeline_cache import PipelineCache


class GridSearchBaseline:
    """Enumerate and evaluate a small fixed sklearn pipeline search space."""

    def __init__(
        self,
        evaluator: Evaluator | None = None,
        cache: PipelineCache | None = None,
        random_state: int = 42,
        max_pipeline_length: int = 4,
    ) -> None:
        self.cache = cache or PipelineCache()
        self.evaluator = evaluator or Evaluator(
            cache=self.cache,
            random_state=random_state,
        )
        self.random_state = random_state
        self.max_pipeline_length = max_pipeline_length

    def generate_action_sequences(self) -> list[list[str]]:
        """Generate 27 valid action-name sequences for grid search."""
        scalers = [None, "standard_scaler", "minmax_scaler"]
        feature_steps = [None, "pca", "feature_selection"]
        models = ["random_forest", "svm", "knn"]

        sequences = []
        for scaler in scalers:
            for feature_step in feature_steps:
                for model in models:
                    actions = []
                    if scaler is not None:
                        actions.append(scaler)
                    if feature_step is not None:
                        actions.append(feature_step)
                    actions.extend([model, "evaluate"])
                    if len(actions) <= self.max_pipeline_length:
                        sequences.append(actions)
        return sequences

    def run_dataset(self, dataset: dict[str, Any]) -> dict[str, Any]:
        """Evaluate all grid pipelines on one dataset and return the best result."""
        try:
            all_results = []
            for actions in self.generate_action_sequences():
                result = self.evaluator.evaluate(dataset, actions)
                result = dict(result)
                result["method"] = "grid_search"
                if result["status"] != "success":
                    result["macro_f1"] = None
                all_results.append(result)

            best_result = self.select_best_result(all_results)
            num_success = sum(result["status"] == "success" for result in all_results)
            num_failed = len(all_results) - num_success

            return {
                "method": "grid_search",
                "dataset_name": dataset.get("name", "unknown"),
                "best_actions": None if best_result is None else best_result["actions"],
                "best_macro_f1": None if best_result is None else best_result["macro_f1"],
                "best_pipeline_length": None
                if best_result is None
                else best_result["pipeline_length"],
                "best_runtime_sec": None if best_result is None else best_result["runtime_sec"],
                "num_pipelines": len(all_results),
                "num_success": num_success,
                "num_failed": num_failed,
                "all_results": all_results,
                "cache_stats": self.cache.stats(),
                "status": "success" if best_result is not None else "failed",
                "error": None if best_result is not None else "No successful pipelines.",
            }
        except Exception as exc:
            return {
                "method": "grid_search",
                "dataset_name": dataset.get("name", "unknown"),
                "best_actions": None,
                "best_macro_f1": None,
                "best_pipeline_length": None,
                "best_runtime_sec": None,
                "num_pipelines": 0,
                "num_success": 0,
                "num_failed": 0,
                "all_results": [],
                "cache_stats": self.cache.stats(),
                "status": "failed",
                "error": str(exc),
            }

    def run_all(self, datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run grid search for each dataset."""
        return [self.run_dataset(dataset) for dataset in datasets]

    def select_best_result(self, results: list[dict[str, Any]]) -> dict[str, Any] | None:
        """
        Select the best successful pipeline.

        Sort by highest macro F1, then shortest pipeline, then lowest runtime.
        """
        successful_results = [
            result for result in results if result.get("status") == "success"
        ]
        if not successful_results:
            return None
        return sorted(
            successful_results,
            key=lambda result: (
                -float(result.get("macro_f1", 0.0)),
                int(result.get("pipeline_length", 0)),
                float(result.get("runtime_sec", 0.0)),
            ),
        )[0]
