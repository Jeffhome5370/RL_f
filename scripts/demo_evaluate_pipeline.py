"""
Demo script for building and evaluating fixed sklearn pipelines.

This script exercises the ToolExecutor, PipelineBuilder, PipelineCache, and
Evaluator without using any RL or DQN components.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_manager import DatasetManager
from src.evaluator.evaluator import Evaluator
from src.tools.pipeline_cache import PipelineCache


def main() -> None:
    """Evaluate several manually defined action lists."""
    manager = DatasetManager(random_state=42)
    dataset = manager.load_dataset("iris")
    cache = PipelineCache()
    evaluator = Evaluator(cache=cache, random_state=42)

    action_lists = [
        ["standard_scaler", "svm", "evaluate"],
        ["minmax_scaler", "knn", "evaluate"],
        ["feature_selection", "random_forest", "evaluate"],
        ["standard_scaler", "pca", "svm", "evaluate"],
        ["random_forest", "evaluate"],
    ]

    print(f"Dataset: {dataset['name']}")
    for actions in action_lists:
        result = evaluator.evaluate(dataset, actions)
        print_result(actions, result)

    print("\nRunning same pipeline again to demonstrate cache...")
    repeated_actions = action_lists[0]
    cached_result = evaluator.evaluate(dataset, repeated_actions)
    print_result(repeated_actions, cached_result)
    print(f"Cache stats: {cache.stats()}")


def print_result(actions: list[str], result: dict) -> None:
    """Print a compact evaluation result."""
    print(f"\nActions: {actions}")
    print(f"Status: {result['status']}")
    print(f"Macro F1: {result['macro_f1']:.4f}")
    print(f"Runtime: {result['runtime_sec']:.4f} sec")
    print(f"From cache: {result['from_cache']}")
    if result["error"]:
        print(f"Error: {result['error']}")


if __name__ == "__main__":
    main()
