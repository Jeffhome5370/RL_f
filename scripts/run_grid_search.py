"""
Grid Search runner script.

This script evaluates a small fixed search space of sklearn pipeline action
sequences using the existing Evaluator and PipelineCache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.grid_search import GridSearchBaseline
from src.data.dataset_manager import DatasetManager
from src.evaluator.evaluator import Evaluator
from src.tools.pipeline_cache import PipelineCache


def main() -> None:
    """Run Grid Search on all supported datasets and save CSV outputs."""
    output_logs = PROJECT_ROOT / "outputs" / "logs"
    output_reports = PROJECT_ROOT / "outputs" / "reports"
    output_logs.mkdir(parents=True, exist_ok=True)
    output_reports.mkdir(parents=True, exist_ok=True)

    manager = DatasetManager(random_state=42)
    datasets = list(manager.load_all().values())
    cache = PipelineCache()
    evaluator = Evaluator(cache=cache, random_state=42)
    grid_search = GridSearchBaseline(evaluator=evaluator, cache=cache, random_state=42)

    dataset_results = grid_search.run_all(datasets)
    detailed_rows = flatten_pipeline_results(dataset_results)
    best_rows = build_best_rows(dataset_results)
    summary_row = build_summary(dataset_results, cache.stats())

    detailed_path = output_logs / "grid_search_results.csv"
    best_path = output_reports / "grid_search_best_results.csv"
    summary_path = output_reports / "grid_search_summary.csv"

    pd.DataFrame(detailed_rows).to_csv(detailed_path, index=False)
    pd.DataFrame(best_rows).to_csv(best_path, index=False)
    pd.DataFrame([summary_row]).to_csv(summary_path, index=False)

    for result in dataset_results:
        print_best_result(result)

    print("\nGrid Search Summary")
    print("===================")
    print(f"Datasets: {summary_row['num_datasets']}")
    print(f"Average best macro F1: {summary_row['avg_best_macro_f1']:.4f}")
    print(f"Average best pipeline length: {summary_row['avg_best_pipeline_length']:.2f}")
    print(f"Average best runtime: {summary_row['avg_best_runtime_sec']:.4f} sec")
    print(f"Total pipelines evaluated: {summary_row['total_pipelines_evaluated']}")
    print(f"Total success: {summary_row['total_success']}")
    print(f"Total failed: {summary_row['total_failed']}")
    print(f"Cache hit rate: {summary_row['cache_hit_rate']:.4f}")
    print(f"\nSaved detailed results to: {detailed_path}")
    print(f"Saved best results to: {best_path}")
    print(f"Saved summary results to: {summary_path}")


def flatten_pipeline_results(dataset_results: list[dict]) -> list[dict]:
    """Flatten all per-pipeline results into CSV-friendly rows."""
    rows = []
    for dataset_result in dataset_results:
        for result in dataset_result["all_results"]:
            row = dict(result)
            row["actions"] = "|".join(row.get("actions", []))
            row["effective_actions"] = "|".join(row.get("effective_actions", []))
            rows.append(row)
    return rows


def build_best_rows(dataset_results: list[dict]) -> list[dict]:
    """Build one best-result row per dataset."""
    rows = []
    for result in dataset_results:
        rows.append(
            {
                "method": result["method"],
                "dataset_name": result["dataset_name"],
                "best_actions": ""
                if result["best_actions"] is None
                else "|".join(result["best_actions"]),
                "best_macro_f1": result["best_macro_f1"],
                "best_pipeline_length": result["best_pipeline_length"],
                "best_runtime_sec": result["best_runtime_sec"],
                "num_pipelines": result["num_pipelines"],
                "num_success": result["num_success"],
                "num_failed": result["num_failed"],
                "status": result["status"],
                "error": result["error"],
            }
        )
    return rows


def build_summary(dataset_results: list[dict], cache_stats: dict) -> dict:
    """Create overall Grid Search summary metrics."""
    best_macro_f1 = [
        result["best_macro_f1"]
        for result in dataset_results
        if result["best_macro_f1"] is not None
    ]
    best_lengths = [
        result["best_pipeline_length"]
        for result in dataset_results
        if result["best_pipeline_length"] is not None
    ]
    best_runtimes = [
        result["best_runtime_sec"]
        for result in dataset_results
        if result["best_runtime_sec"] is not None
    ]

    return {
        "method": "grid_search",
        "num_datasets": len(dataset_results),
        "avg_best_macro_f1": mean_or_zero(best_macro_f1),
        "avg_best_pipeline_length": mean_or_zero(best_lengths),
        "avg_best_runtime_sec": mean_or_zero(best_runtimes),
        "total_pipelines_evaluated": sum(
            result["num_pipelines"] for result in dataset_results
        ),
        "total_success": sum(result["num_success"] for result in dataset_results),
        "total_failed": sum(result["num_failed"] for result in dataset_results),
        "cache_hit_rate": cache_stats["hit_rate"],
    }


def print_best_result(result: dict) -> None:
    """Print the best Grid Search result for one dataset."""
    print(f"\nDataset: {result['dataset_name']}")
    print(f"Status: {result['status']}")
    print(f"Best actions: {result['best_actions']}")
    print(f"Best macro F1: {result['best_macro_f1']}")
    print(f"Best pipeline length: {result['best_pipeline_length']}")
    print(f"Pipelines evaluated: {result['num_pipelines']}")


def mean_or_zero(values: list[float | int]) -> float:
    """Return the mean of values, or zero for an empty list."""
    return float(sum(values) / len(values)) if values else 0.0


if __name__ == "__main__":
    main()
