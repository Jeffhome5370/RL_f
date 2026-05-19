"""
Baseline runner script.

This script will eventually run baseline tool-selection strategies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.fixed_pipeline import FIXED_PIPELINES, FixedPipelineAgent
from src.baselines.random_agent import RandomAgent
from src.data.dataset_manager import DatasetManager
from src.env.tool_selection_env import ToolSelectionEnv
from src.evaluator.evaluator import Evaluator
from src.tools.pipeline_cache import PipelineCache


def main() -> None:
    """Run random and fixed pipeline baselines on all supported datasets."""
    random_episodes_per_dataset = 5
    random_state = 42

    output_logs = PROJECT_ROOT / "outputs" / "logs"
    output_reports = PROJECT_ROOT / "outputs" / "reports"
    output_logs.mkdir(parents=True, exist_ok=True)
    output_reports.mkdir(parents=True, exist_ok=True)

    manager = DatasetManager(random_state=random_state)
    datasets = manager.load_all()
    shared_cache = PipelineCache()
    evaluator = Evaluator(cache=shared_cache, random_state=random_state)

    results = []
    for dataset_name, dataset in datasets.items():
        print(f"\nDataset: {dataset_name}")

        for episode_index in range(random_episodes_per_dataset):
            env = ToolSelectionEnv(evaluator=evaluator, max_steps=6, random_state=random_state)
            agent = RandomAgent(use_action_mask=True, random_state=random_state + episode_index)
            result = agent.run_episode(env, dataset)
            results.append(result)
            print_result(result)

        for name, actions in FIXED_PIPELINES.items():
            env = ToolSelectionEnv(evaluator=evaluator, max_steps=6, random_state=random_state)
            agent = FixedPipelineAgent(name=name, actions=actions)
            result = agent.run_episode(env, dataset)
            results.append(result)
            print_result(result)

    results_df = pd.DataFrame(results)
    summary_df = summarize_results(results_df)

    raw_path = output_logs / "baseline_results.csv"
    summary_path = output_reports / "baseline_summary.csv"
    results_df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)

    print("\nBaseline Summary")
    print("================")
    print(summary_df.to_string(index=False))
    print(f"\nSaved raw results to: {raw_path}")
    print(f"Saved summary results to: {summary_path}")
    print(f"Cache stats: {shared_cache.stats()}")


def summarize_results(results_df: pd.DataFrame) -> pd.DataFrame:
    """Compute grouped baseline summary metrics."""
    grouped = results_df.groupby("method", dropna=False)
    return grouped.agg(
        avg_macro_f1=("macro_f1", "mean"),
        avg_total_reward=("total_reward", "mean"),
        avg_pipeline_length=("pipeline_length", "mean"),
        avg_invalid_action_count=("invalid_action_count", "mean"),
        evaluate_success_rate=("evaluate_success", "mean"),
        avg_runtime_sec=("runtime_sec", "mean"),
        avg_steps=("steps", "mean"),
    ).reset_index()


def print_result(result: dict) -> None:
    """Print one compact baseline result row."""
    macro_f1 = result["macro_f1"]
    macro_text = "None" if macro_f1 is None else f"{macro_f1:.4f}"
    print(
        f"{result['method']}: status={result['status']}, "
        f"macro_f1={macro_text}, reward={result['total_reward']:.4f}, "
        f"steps={result['steps']}, pipeline={result['pipeline_actions']}"
    )


if __name__ == "__main__":
    main()
