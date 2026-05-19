"""
Markdown report generation utilities.

This module writes a lightweight experiment report from already-computed CSV
summaries and generated figures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


REPORT_SECTIONS = [
    "1. Project Goal",
    "2. Compared Methods",
    "3. Result Summary",
    "4. DQN Training Behavior",
    "5. Baseline Comparison",
    "6. Grid Search Comparison",
    "7. Agent Behavior Analysis",
    "8. Generated Figures",
    "9. Limitations",
    "10. Conclusion",
]


def generate_markdown_report(
    output_path: str,
    comparison_df: pd.DataFrame,
    summaries: dict[str, Any],
    generated_figures: list[str],
    skipped_plots: list[str],
    missing_files: list[str],
) -> None:
    """Generate the final Markdown experiment report."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    best_f1_method = _best_method(comparison_df, "avg_macro_f1")
    best_reward_method = _best_method(comparison_df, "avg_total_reward")

    lines = [
        "# DQN Tool Selection Agent - Experiment Report",
        "",
        "## 1. Project Goal",
        "",
        (
            "This project treats tabular machine learning pipeline construction "
            "as a sequential decision-making problem. The agent or baseline "
            "method selects preprocessing tools, a classifier, and an evaluation "
            "action based on dataset meta-features and the current pipeline state."
        ),
        "",
        "## 2. Compared Methods",
        "",
        "- Random Agent",
        "- Fixed Pipeline baselines",
        "- Grid Search",
        "- DQN Agent",
        "",
        "## 3. Result Summary",
        "",
        _summary_table(comparison_df),
        "",
        f"Best average macro F1: {_format_best(best_f1_method, 'avg_macro_f1')}.",
        f"Best average reward: {_format_best(best_reward_method, 'avg_total_reward')}.",
        "",
        "## 4. DQN Training Behavior",
        "",
        _dqn_training_text(summaries.get("dqn")),
        "",
        "## 5. Baseline Comparison",
        "",
        _baseline_text(summaries.get("baselines")),
        "",
        "## 6. Grid Search Comparison",
        "",
        _grid_search_text(summaries.get("grid_search")),
        "",
        "## 7. Agent Behavior Analysis",
        "",
        _agent_behavior_text(comparison_df),
        "",
        "## 8. Generated Figures",
        "",
    ]

    if generated_figures:
        for figure in generated_figures:
            lines.append(f"![{_figure_title(figure)}](../figures/{Path(figure).name})")
            lines.append("")
    else:
        lines.append("No figures were generated because the required metrics were unavailable.")
        lines.append("")

    if skipped_plots:
        lines.append("Skipped plots:")
        for skipped in skipped_plots:
            lines.append(f"- {skipped}")
        lines.append("")

    if missing_files:
        lines.append("Missing input files:")
        for missing in missing_files:
            lines.append(f"- `{missing}`")
        lines.append("")

    lines.extend(
        [
            "## 9. Limitations",
            "",
            "- The dataset pool is small and limited to built-in scikit-learn datasets.",
            "- The action space is intentionally small and discrete.",
            "- No hyperparameter tuning is performed for individual sklearn tools.",
            "- Results use a validation split instead of cross-validation.",
            "- DQN behavior may require more episodes to become stable.",
            "",
            "## 10. Conclusion",
            "",
            (
                "The analysis summarizes the available experiment CSV files and "
                "compares methods only on metrics present in those files. Grid "
                "Search provides a useful upper-reference baseline over the small "
                "enumerated action space, while DQN performance should be judged "
                "with care because training stability depends on episode count and "
                "reward design."
            ),
            "",
        ]
    )

    output.write_text("\n".join(lines), encoding="utf-8")


def _summary_table(comparison_df: pd.DataFrame) -> str:
    """Return a Markdown table for the final comparison summary."""
    if comparison_df.empty:
        return "No comparison summary is available."
    display_cols = [
        column
        for column in [
            "method",
            "avg_macro_f1",
            "avg_total_reward",
            "avg_pipeline_length",
            "evaluate_success_rate",
            "source",
        ]
        if column in comparison_df.columns
    ]
    table_df = comparison_df[display_cols].copy()
    header = "| " + " | ".join(display_cols) + " |"
    separator = "| " + " | ".join(["---"] * len(display_cols)) + " |"
    rows = []
    for _, row in table_df.iterrows():
        rows.append(
            "| "
            + " | ".join(_table_value(row[column]) for column in display_cols)
            + " |"
        )
    return "\n".join([header, separator, *rows])


def _best_method(comparison_df: pd.DataFrame, metric: str) -> dict[str, Any] | None:
    """Return the row with the highest numeric metric."""
    if comparison_df.empty or metric not in comparison_df.columns:
        return None
    temp = comparison_df.copy()
    temp[metric] = pd.to_numeric(temp[metric], errors="coerce")
    temp = temp.dropna(subset=[metric])
    if temp.empty:
        return None
    return temp.sort_values(metric, ascending=False).iloc[0].to_dict()


def _format_best(row: dict[str, Any] | None, metric: str) -> str:
    """Format a best-method row for prose."""
    if row is None:
        return "unavailable"
    metric_value = row.get(metric)
    if metric_value is None or pd.isna(metric_value):
        return "unavailable"
    return f"{row.get('method')} ({metric_value:.4f})"


def _dqn_training_text(summary: dict[str, Any] | None) -> str:
    """Describe DQN training metrics without overclaiming."""
    if not summary:
        return "DQN training data is unavailable, so training behavior cannot be assessed."
    return (
        f"The DQN log contains {summary.get('episodes')} episodes. "
        f"Best total reward was {_fmt(summary.get('best_total_reward'))}; "
        f"best macro F1 was {_fmt(summary.get('best_macro_f1'))}. "
        f"Over the final window, average reward was "
        f"{_fmt(summary.get('final_avg_reward_last_20'))}, average macro F1 was "
        f"{_fmt(summary.get('final_avg_f1_last_20'))}, average pipeline length was "
        f"{_fmt(summary.get('avg_pipeline_length_last_20'))}, and invalid action "
        f"count averaged {_fmt(summary.get('avg_invalid_action_count_last_20'))}."
    )


def _baseline_text(summary_df: pd.DataFrame | None) -> str:
    """Describe available baseline summary results."""
    if summary_df is None or summary_df.empty:
        return "Baseline result data is unavailable."
    best = _best_method(summary_df.rename(columns={"avg_macro_f1": "avg_macro_f1"}), "avg_macro_f1")
    if best is None:
        return "Baseline runs are available, but macro F1 is unavailable."
    return f"Among available baseline rows, `{best['method']}` has the highest average macro F1."


def _grid_search_text(summary: dict[str, Any] | None) -> str:
    """Describe Grid Search results."""
    if not summary:
        return "Grid Search best-result data is unavailable."
    return (
        f"Grid Search evaluated best pipelines for {summary.get('num_datasets')} datasets. "
        f"Average best macro F1 was {_fmt(summary.get('avg_macro_f1'))}, "
        f"with average best pipeline length {_fmt(summary.get('avg_pipeline_length'))}."
    )


def _agent_behavior_text(comparison_df: pd.DataFrame) -> str:
    """Describe behavior-level comparisons from available summary metrics."""
    if comparison_df.empty:
        return "No method-level behavior summary is available."
    parts = []
    if "avg_invalid_action_count" in comparison_df.columns:
        parts.append(
            "Invalid-action behavior is summarized by average invalid action count where available."
        )
    if "avg_pipeline_length" in comparison_df.columns:
        parts.append(
            "Pipeline compactness is summarized by average pipeline length where available."
        )
    if not parts:
        return "Behavior metrics are unavailable in the current CSV files."
    return " ".join(parts)


def _figure_title(path: str) -> str:
    """Convert a figure file name to a readable title."""
    return Path(path).stem.replace("_", " ").title()


def _fmt(value: Any) -> str:
    """Format numeric values while preserving unavailable metrics."""
    if value is None or pd.isna(value):
        return "unavailable"
    return f"{float(value):.4f}"


def _table_value(value: Any) -> str:
    """Format values for Markdown tables."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
