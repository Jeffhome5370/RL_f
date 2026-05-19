"""
Metrics module.

This module will provide metric calculation helpers for classification tasks.
"""

from __future__ import annotations

from typing import Any

from sklearn.metrics import f1_score


def compute_macro_f1(y_true: Any, y_pred: Any) -> float:
    """Compute macro-averaged F1 with safe zero-division handling."""
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def safe_score(y_true: Any, y_pred: Any) -> dict[str, float]:
    """Return supported metric scores in a dictionary."""
    return {"macro_f1": compute_macro_f1(y_true, y_pred)}
