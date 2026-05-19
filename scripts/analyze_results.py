"""
Results analysis script.

This script will eventually summarize and visualize experiment outputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.plot import (
    plot_dqn_f1_curve,
    plot_dqn_reward_curve,
    plot_method_bar_chart,
)
from src.utils.report import generate_markdown_report


INPUT_FILES = {
    "dqn_train": ("logs", "dqn_train_log.csv"),
    "baseline_results": ("logs", "baseline_results.csv"),
    "grid_search_results": ("logs", "grid_search_results.csv"),
    "grid_search_best": ("reports", "grid_search_best_results.csv"),
}


def load_csv_if_exists(path: str) -> pd.DataFrame | None:
    """Load a CSV if it exists; otherwise warn and return None."""
    csv_path = Path(path)
    if not csv_path.exists():
        print(f"Warning: missing input file: {csv_path}")
        return None
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded: {csv_path}")
        return df
    except Exception as exc:
        print(f"Warning: could not load {csv_path}: {exc}")
        return None


def load_all_results(output_dir: str = "outputs") -> dict[str, pd.DataFrame | None]:
    """Load all known result CSVs, returning None for missing files."""
    root = Path(output_dir)
    return {
        key: load_csv_if_exists(str(root / folder / filename))
        for key, (folder, filename) in INPUT_FILES.items()
    }


def summarize_dqn_training(dqn_df: pd.DataFrame, last_n: int = 20) -> dict[str, Any]:
    """Summarize DQN training, using the final window for comparison metrics."""
    if dqn_df.empty:
        return _empty_dqn_summary()

    df = dqn_df.copy()
    for column in [
        "episode",
        "total_reward",
        "macro_f1",
        "pipeline_length",
        "invalid_action_count",
        "runtime_sec",
    ]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "evaluate_success" in df.columns:
        df["evaluate_success"] = _to_bool_series(df["evaluate_success"])

    tail_df = df.tail(last_n) if last_n > 0 else df
    return {
        "method": "dqn_agent",
        "episodes": int(len(df)),
        "best_total_reward": _max_or_none(df, "total_reward"),
        "best_macro_f1": _max_or_none(df, "macro_f1"),
        "final_avg_reward_last_20": _mean_or_none(tail_df, "total_reward"),
        "final_avg_f1_last_20": _mean_or_none(tail_df, "macro_f1"),
        "avg_pipeline_length_last_20": _mean_or_none(tail_df, "pipeline_length"),
        "avg_invalid_action_count_last_20": _mean_or_none(
            tail_df,
            "invalid_action_count",
        ),
        "evaluate_success_rate_last_20": _mean_or_none(tail_df, "evaluate_success"),
        "avg_runtime_sec_last_20": _mean_or_none(tail_df, "runtime_sec"),
    }


def summarize_baselines(baseline_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize baseline results by method."""
    if baseline_df.empty or "method" not in baseline_df.columns:
        return pd.DataFrame()

    df = baseline_df.copy()
    numeric_columns = [
        "macro_f1",
        "total_reward",
        "pipeline_length",
        "invalid_action_count",
        "runtime_sec",
        "steps",
    ]
    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    if "evaluate_success" in df.columns:
        df["evaluate_success"] = _to_bool_series(df["evaluate_success"])

    aggregations = {"num_runs": ("method", "size")}
    column_map = {
        "macro_f1": "avg_macro_f1",
        "total_reward": "avg_total_reward",
        "pipeline_length": "avg_pipeline_length",
        "invalid_action_count": "avg_invalid_action_count",
        "evaluate_success": "evaluate_success_rate",
        "runtime_sec": "avg_runtime_sec",
        "steps": "avg_steps",
    }
    for source_col, output_col in column_map.items():
        if source_col in df.columns:
            aggregations[output_col] = (source_col, "mean")

    return df.groupby("method", dropna=False).agg(**aggregations).reset_index()


def summarize_grid_search(grid_best_df: pd.DataFrame) -> dict[str, Any]:
    """Summarize Grid Search best-pipeline results."""
    if grid_best_df.empty:
        return _empty_grid_summary()

    df = grid_best_df.copy()
    macro_col = _first_present(df, ["best_macro_f1", "macro_f1", "avg_macro_f1"])
    length_col = _first_present(df, ["best_pipeline_length", "pipeline_length"])
    runtime_col = _first_present(df, ["best_runtime_sec", "runtime_sec"])
    status_col = _first_present(df, ["status"])

    for column in [macro_col, length_col, runtime_col]:
        if column is not None:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    success_rate = None
    if status_col is not None:
        success_rate = float((df[status_col].astype(str) == "success").mean())

    return {
        "method": "grid_search",
        "num_datasets": int(len(df)),
        "avg_macro_f1": _mean_or_none(df, macro_col),
        "avg_pipeline_length": _mean_or_none(df, length_col),
        "avg_runtime_sec": _mean_or_none(df, runtime_col),
        "avg_total_reward": None,
        "avg_invalid_action_count": None,
        "evaluate_success_rate": success_rate,
    }


def build_final_comparison_summary(
    results: dict[str, pd.DataFrame | None],
    last_n: int = 20,
) -> pd.DataFrame:
    """Combine DQN, baseline, and Grid Search summaries into one table."""
    rows: list[dict[str, Any]] = []

    dqn_df = results.get("dqn_train")
    if dqn_df is not None and not dqn_df.empty:
        dqn_summary = summarize_dqn_training(dqn_df, last_n=last_n)
        rows.append(
            {
                "method": "dqn_agent",
                "avg_macro_f1": dqn_summary["final_avg_f1_last_20"],
                "avg_total_reward": dqn_summary["final_avg_reward_last_20"],
                "avg_pipeline_length": dqn_summary["avg_pipeline_length_last_20"],
                "avg_invalid_action_count": dqn_summary[
                    "avg_invalid_action_count_last_20"
                ],
                "evaluate_success_rate": dqn_summary["evaluate_success_rate_last_20"],
                "avg_runtime_sec": dqn_summary["avg_runtime_sec_last_20"],
                "num_runs_or_datasets": dqn_summary["episodes"],
                "source": "dqn_train_log_last_20",
            }
        )

    baseline_df = results.get("baseline_results")
    if baseline_df is not None and not baseline_df.empty:
        baseline_summary = summarize_baselines(baseline_df)
        for _, row in baseline_summary.iterrows():
            rows.append(
                {
                    "method": row.get("method"),
                    "avg_macro_f1": row.get("avg_macro_f1"),
                    "avg_total_reward": row.get("avg_total_reward"),
                    "avg_pipeline_length": row.get("avg_pipeline_length"),
                    "avg_invalid_action_count": row.get("avg_invalid_action_count"),
                    "evaluate_success_rate": row.get("evaluate_success_rate"),
                    "avg_runtime_sec": row.get("avg_runtime_sec"),
                    "num_runs_or_datasets": row.get("num_runs"),
                    "source": "baseline_results",
                }
            )

    grid_best_df = results.get("grid_search_best")
    if grid_best_df is not None and not grid_best_df.empty:
        grid_summary = summarize_grid_search(grid_best_df)
        rows.append(
            {
                "method": "grid_search",
                "avg_macro_f1": grid_summary["avg_macro_f1"],
                "avg_total_reward": grid_summary["avg_total_reward"],
                "avg_pipeline_length": grid_summary["avg_pipeline_length"],
                "avg_invalid_action_count": grid_summary["avg_invalid_action_count"],
                "evaluate_success_rate": grid_summary["evaluate_success_rate"],
                "avg_runtime_sec": grid_summary["avg_runtime_sec"],
                "num_runs_or_datasets": grid_summary["num_datasets"],
                "source": "grid_search_best_results",
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "method",
            "avg_macro_f1",
            "avg_total_reward",
            "avg_pipeline_length",
            "avg_invalid_action_count",
            "evaluate_success_rate",
            "avg_runtime_sec",
            "num_runs_or_datasets",
            "source",
        ],
    )


def generate_figures(
    results: dict[str, pd.DataFrame | None],
    comparison_df: pd.DataFrame,
    figures_dir: Path,
) -> tuple[list[str], list[str]]:
    """Generate available plots and return generated/skipped figure names."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    generated: list[str] = []
    skipped: list[str] = []

    dqn_df = results.get("dqn_train")
    if dqn_df is not None:
        _try_plot(
            lambda: plot_dqn_reward_curve(
                dqn_df,
                str(figures_dir / "dqn_reward_curve.png"),
            ),
            "dqn_reward_curve.png",
            generated,
            skipped,
        )
        _try_plot(
            lambda: plot_dqn_f1_curve(
                dqn_df,
                str(figures_dir / "dqn_f1_curve.png"),
            ),
            "dqn_f1_curve.png",
            generated,
            skipped,
        )
    else:
        skipped.extend(["dqn_reward_curve.png: DQN log unavailable", "dqn_f1_curve.png: DQN log unavailable"])

    metric_plots = [
        (
            "avg_macro_f1",
            "method_f1_comparison.png",
            "Method Macro F1 Comparison",
            "Average Macro F1",
        ),
        (
            "avg_total_reward",
            "method_reward_comparison.png",
            "Method Reward Comparison",
            "Average Total Reward",
        ),
        (
            "avg_pipeline_length",
            "method_pipeline_length_comparison.png",
            "Method Pipeline Length Comparison",
            "Average Pipeline Length",
        ),
        (
            "avg_invalid_action_count",
            "method_invalid_action_comparison.png",
            "Method Invalid Action Comparison",
            "Average Invalid Action Count",
        ),
        (
            "evaluate_success_rate",
            "method_evaluate_success_comparison.png",
            "Method Evaluation Success Comparison",
            "Evaluation Success Rate",
        ),
        (
            "avg_runtime_sec",
            "method_runtime_comparison.png",
            "Method Runtime Comparison",
            "Average Runtime (sec)",
        ),
    ]
    for metric, filename, title, ylabel in metric_plots:
        _try_plot(
            lambda metric=metric, filename=filename, title=title, ylabel=ylabel: plot_method_bar_chart(
                comparison_df,
                metric,
                str(figures_dir / filename),
                title,
                ylabel,
            ),
            filename,
            generated,
            skipped,
        )

    return generated, skipped


def generate_analysis_outputs(output_dir: str = "outputs", last_n: int = 20) -> dict[str, Any]:
    """Run the full analysis pipeline without launching experiments."""
    root = Path(output_dir)
    reports_dir = root / "reports"
    figures_dir = root / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    results = load_all_results(output_dir)
    usable_inputs = [df for df in results.values() if df is not None and not df.empty]
    if not usable_inputs:
        raise RuntimeError("No usable input CSV files found for analysis.")

    comparison_df = build_final_comparison_summary(results, last_n=last_n)
    final_summary_path = reports_dir / "final_comparison_summary.csv"
    comparison_df.to_csv(final_summary_path, index=False)

    generated_figures, skipped_plots = generate_figures(results, comparison_df, figures_dir)
    summaries = {
        "dqn": summarize_dqn_training(results["dqn_train"], last_n=last_n)
        if results.get("dqn_train") is not None
        else None,
        "baselines": summarize_baselines(results["baseline_results"])
        if results.get("baseline_results") is not None
        else None,
        "grid_search": summarize_grid_search(results["grid_search_best"])
        if results.get("grid_search_best") is not None
        else None,
    }
    missing_files = [
        str(root / folder / filename)
        for key, (folder, filename) in INPUT_FILES.items()
        if results.get(key) is None
    ]

    report_path = reports_dir / "final_report.md"
    generate_markdown_report(
        output_path=str(report_path),
        comparison_df=comparison_df,
        summaries=summaries,
        generated_figures=generated_figures,
        skipped_plots=skipped_plots,
        missing_files=missing_files,
    )

    return {
        "results": results,
        "comparison_df": comparison_df,
        "final_summary_path": str(final_summary_path),
        "report_path": str(report_path),
        "generated_figures": [str(figures_dir / figure) for figure in generated_figures],
        "skipped_plots": skipped_plots,
    }


def parse_args() -> argparse.Namespace:
    """Parse analysis CLI arguments."""
    parser = argparse.ArgumentParser(description="Analyze existing experiment CSV outputs.")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--last-n", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Run result analysis and report generation."""
    args = parse_args()
    outputs = generate_analysis_outputs(args.output_dir, last_n=args.last_n)

    print("\nSaved final summary:")
    print(outputs["final_summary_path"])
    print("\nSaved report:")
    print(outputs["report_path"])
    print("\nSaved figures:")
    for figure in outputs["generated_figures"]:
        print(figure)
    if outputs["skipped_plots"]:
        print("\nSkipped plots:")
        for skipped in outputs["skipped_plots"]:
            print(skipped)


def _try_plot(plot_func, filename: str, generated: list[str], skipped: list[str]) -> None:
    """Attempt a plot and record whether it was generated or skipped."""
    try:
        plot_func()
        generated.append(filename)
    except Exception as exc:
        skipped.append(f"{filename}: {exc}")


def _to_bool_series(series: pd.Series) -> pd.Series:
    """Convert common boolean-like values to floats for averaging."""
    if series.dtype == bool:
        return series.astype(float)
    return series.map(
        lambda value: str(value).strip().lower() in {"true", "1", "yes", "success"}
    ).astype(float)


def _mean_or_none(df: pd.DataFrame, column: str | None) -> float | None:
    """Return a numeric column mean or None when unavailable."""
    if column is None or column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.mean())


def _max_or_none(df: pd.DataFrame, column: str) -> float | None:
    """Return a numeric column max or None when unavailable."""
    if column not in df.columns:
        return None
    series = pd.to_numeric(df[column], errors="coerce").dropna()
    if series.empty:
        return None
    return float(series.max())


def _first_present(df: pd.DataFrame, columns: list[str]) -> str | None:
    """Return the first present column from a list of candidates."""
    for column in columns:
        if column in df.columns:
            return column
    return None


def _empty_dqn_summary() -> dict[str, Any]:
    """Return an empty DQN summary structure."""
    return {
        "method": "dqn_agent",
        "episodes": 0,
        "best_total_reward": None,
        "best_macro_f1": None,
        "final_avg_reward_last_20": None,
        "final_avg_f1_last_20": None,
        "avg_pipeline_length_last_20": None,
        "avg_invalid_action_count_last_20": None,
        "evaluate_success_rate_last_20": None,
        "avg_runtime_sec_last_20": None,
    }


def _empty_grid_summary() -> dict[str, Any]:
    """Return an empty Grid Search summary structure."""
    return {
        "method": "grid_search",
        "num_datasets": 0,
        "avg_macro_f1": None,
        "avg_pipeline_length": None,
        "avg_runtime_sec": None,
        "avg_total_reward": None,
        "avg_invalid_action_count": None,
        "evaluate_success_rate": None,
    }


if __name__ == "__main__":
    main()
