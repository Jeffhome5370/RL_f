"""
Tests for result analysis, plotting, and report generation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.analyze_results import (
    build_final_comparison_summary,
    generate_analysis_outputs,
    load_all_results,
    summarize_baselines,
    summarize_dqn_training,
    summarize_grid_search,
)
from src.utils.plot import (
    plot_dqn_f1_curve,
    plot_dqn_reward_curve,
    plot_method_bar_chart,
)
from src.utils.report import generate_markdown_report


def fake_dqn_df() -> pd.DataFrame:
    """Create a small fake DQN training log."""
    return pd.DataFrame(
        {
            "episode": [1, 2, 3],
            "total_reward": [0.1, 0.5, 0.9],
            "macro_f1": [0.2, 0.6, 0.8],
            "pipeline_length": [3, 2, 2],
            "invalid_action_count": [2, 1, 0],
            "evaluate_success": [False, True, True],
            "runtime_sec": [0.3, 0.2, 0.1],
            "steps": [6, 4, 3],
        }
    )


def fake_baseline_df() -> pd.DataFrame:
    """Create a small fake baseline result table."""
    return pd.DataFrame(
        {
            "method": ["random_agent", "random_agent", "fixed_scaler_svm"],
            "macro_f1": [0.4, 0.6, 0.9],
            "total_reward": [0.1, 0.3, 0.8],
            "pipeline_length": [3, 2, 2],
            "invalid_action_count": [1, 0, 0],
            "evaluate_success": [True, True, True],
            "runtime_sec": [0.2, 0.3, 0.1],
            "steps": [5, 4, 3],
        }
    )


def fake_grid_best_df() -> pd.DataFrame:
    """Create a small fake Grid Search best-result table."""
    return pd.DataFrame(
        {
            "dataset_name": ["iris", "wine"],
            "best_actions": ["svm|evaluate", "random_forest|evaluate"],
            "best_macro_f1": [0.95, 1.0],
            "best_pipeline_length": [1, 1],
            "best_runtime_sec": [0.02, 0.03],
            "status": ["success", "success"],
        }
    )


def test_summarize_dqn_training_works_on_fake_dataframe() -> None:
    """DQN summary should compute best and final-window metrics."""
    summary = summarize_dqn_training(fake_dqn_df(), last_n=2)
    assert summary["method"] == "dqn_agent"
    assert summary["episodes"] == 3
    assert summary["best_total_reward"] == 0.9
    assert summary["final_avg_reward_last_20"] == 0.7


def test_summarize_baselines_works_on_fake_dataframe() -> None:
    """Baseline summary should group by method."""
    summary = summarize_baselines(fake_baseline_df())
    assert set(summary["method"]) == {"random_agent", "fixed_scaler_svm"}
    assert "avg_macro_f1" in summary.columns


def test_summarize_grid_search_works_on_fake_dataframe() -> None:
    """Grid Search summary should average best-result metrics."""
    summary = summarize_grid_search(fake_grid_best_df())
    assert summary["method"] == "grid_search"
    assert summary["num_datasets"] == 2
    assert summary["avg_macro_f1"] == 0.975


def test_build_final_comparison_summary_returns_dataframe() -> None:
    """Final comparison summary should combine available sources."""
    summary = build_final_comparison_summary(
        {
            "dqn_train": fake_dqn_df(),
            "baseline_results": fake_baseline_df(),
            "grid_search_results": None,
            "grid_search_best": fake_grid_best_df(),
        },
        last_n=2,
    )
    assert isinstance(summary, pd.DataFrame)
    assert "method" in summary.columns
    assert set(summary["method"]).issuperset({"dqn_agent", "grid_search"})


def test_plot_functions_create_png_files(tmp_path: Path) -> None:
    """Plot utilities should write PNG files."""
    dqn_reward_path = tmp_path / "dqn_reward_curve.png"
    dqn_f1_path = tmp_path / "dqn_f1_curve.png"
    method_path = tmp_path / "method_f1_comparison.png"
    comparison_df = build_final_comparison_summary(
        {
            "dqn_train": fake_dqn_df(),
            "baseline_results": fake_baseline_df(),
            "grid_search_results": None,
            "grid_search_best": fake_grid_best_df(),
        },
        last_n=2,
    )

    plot_dqn_reward_curve(fake_dqn_df(), str(dqn_reward_path))
    plot_dqn_f1_curve(fake_dqn_df(), str(dqn_f1_path))
    plot_method_bar_chart(
        comparison_df,
        "avg_macro_f1",
        str(method_path),
        "Method F1",
        "Macro F1",
    )

    assert dqn_reward_path.exists()
    assert dqn_f1_path.exists()
    assert method_path.exists()


def test_markdown_report_generation_creates_file(tmp_path: Path) -> None:
    """Markdown report generation should write a .md file."""
    report_path = tmp_path / "final_report.md"
    comparison_df = build_final_comparison_summary(
        {
            "dqn_train": fake_dqn_df(),
            "baseline_results": fake_baseline_df(),
            "grid_search_results": None,
            "grid_search_best": fake_grid_best_df(),
        },
        last_n=2,
    )
    generate_markdown_report(
        output_path=str(report_path),
        comparison_df=comparison_df,
        summaries={
            "dqn": summarize_dqn_training(fake_dqn_df(), last_n=2),
            "baselines": summarize_baselines(fake_baseline_df()),
            "grid_search": summarize_grid_search(fake_grid_best_df()),
        },
        generated_figures=["method_f1_comparison.png"],
        skipped_plots=[],
        missing_files=[],
    )

    assert report_path.exists()
    assert "# DQN Tool Selection Agent - Experiment Report" in report_path.read_text(
        encoding="utf-8"
    )


def test_missing_optional_input_files_do_not_crash(tmp_path: Path) -> None:
    """Loading missing result files should return None values without raising."""
    results = load_all_results(str(tmp_path))
    assert set(results.keys()) == {
        "dqn_train",
        "baseline_results",
        "grid_search_results",
        "grid_search_best",
    }
    assert all(value is None for value in results.values())


def test_final_comparison_summary_contains_method_column() -> None:
    """The final comparison table should expose a method column."""
    summary = build_final_comparison_summary(
        {
            "dqn_train": None,
            "baseline_results": fake_baseline_df(),
            "grid_search_results": None,
            "grid_search_best": None,
        }
    )
    assert "method" in summary.columns


def test_dqn_summary_uses_only_last_n_episodes() -> None:
    """DQN summary should use the requested final window."""
    summary = summarize_dqn_training(fake_dqn_df(), last_n=1)
    assert summary["final_avg_reward_last_20"] == 0.9
    assert summary["avg_invalid_action_count_last_20"] == 0.0


def test_generate_analysis_outputs_with_partial_files(tmp_path: Path) -> None:
    """Full analysis should work when only some input CSVs are present."""
    logs_dir = tmp_path / "logs"
    reports_dir = tmp_path / "reports"
    logs_dir.mkdir()
    reports_dir.mkdir()
    fake_baseline_df().to_csv(logs_dir / "baseline_results.csv", index=False)

    outputs = generate_analysis_outputs(str(tmp_path), last_n=2)
    assert Path(outputs["final_summary_path"]).exists()
    assert Path(outputs["report_path"]).exists()
    assert "method" in outputs["comparison_df"].columns
