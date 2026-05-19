"""
Evaluator package.

This package will contain model evaluation and metric helpers.
"""

from src.evaluator.evaluator import Evaluator
from src.evaluator.metrics import compute_macro_f1, safe_score

__all__ = ["Evaluator", "compute_macro_f1", "safe_score"]
