"""
Plot utility module.

This module will contain plotting helpers for training curves and evaluation
results.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def plot_dqn_reward_curve(df: pd.DataFrame, output_path: str) -> None:
    """Plot DQN total reward by episode."""
    _require_columns(df, ["episode", "total_reward"])
    output = _prepare_output_path(output_path)

    plot_df = df.copy()
    plot_df["episode"] = pd.to_numeric(plot_df["episode"], errors="coerce")
    plot_df["total_reward"] = pd.to_numeric(plot_df["total_reward"], errors="coerce")
    plot_df = plot_df.dropna(subset=["episode", "total_reward"])
    if plot_df.empty:
        raise ValueError("No numeric DQN reward data available for plotting.")

    plt.figure(figsize=(8, 4.5))
    plt.plot(plot_df["episode"], plot_df["total_reward"], marker="o", linewidth=1.5)
    plt.title("DQN Training Reward")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_dqn_f1_curve(df: pd.DataFrame, output_path: str) -> None:
    """Plot DQN macro F1 by episode."""
    _require_columns(df, ["episode", "macro_f1"])
    output = _prepare_output_path(output_path)

    plot_df = df.copy()
    plot_df["episode"] = pd.to_numeric(plot_df["episode"], errors="coerce")
    plot_df["macro_f1"] = pd.to_numeric(plot_df["macro_f1"], errors="coerce")
    plot_df = plot_df.dropna(subset=["episode", "macro_f1"])
    if plot_df.empty:
        raise ValueError("No numeric DQN macro F1 data available for plotting.")

    plt.figure(figsize=(8, 4.5))
    plt.plot(plot_df["episode"], plot_df["macro_f1"], marker="o", linewidth=1.5)
    plt.title("DQN Training Macro F1")
    plt.xlabel("Episode")
    plt.ylabel("Macro F1")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def plot_method_bar_chart(
    summary_df: pd.DataFrame,
    metric: str,
    output_path: str,
    title: str,
    ylabel: str,
) -> None:
    """Plot a method-level metric comparison as a bar chart."""
    save_bar_plot(
        df=summary_df,
        x_col="method",
        y_col=metric,
        output_path=output_path,
        title=title,
        ylabel=ylabel,
    )


def save_bar_plot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    output_path: str,
    title: str,
    ylabel: str,
) -> None:
    """Save a simple matplotlib bar chart."""
    _require_columns(df, [x_col, y_col])
    output = _prepare_output_path(output_path)

    plot_df = df[[x_col, y_col]].copy()
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors="coerce")
    plot_df = plot_df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        raise ValueError(f"No numeric data available for metric '{y_col}'.")

    plt.figure(figsize=(8, 4.8))
    plt.bar(plot_df[x_col].astype(str), plot_df[y_col])
    plt.title(title)
    plt.xlabel("Method")
    plt.ylabel(ylabel)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output, dpi=150)
    plt.close()


def _prepare_output_path(output_path: str) -> Path:
    """Create parent folders and return a Path for figure output."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise a clear error if required plot columns are absent."""
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required plotting columns: {missing}")
