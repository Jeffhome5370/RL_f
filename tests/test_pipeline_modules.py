"""
Tests for sklearn pipeline construction, caching, and evaluation modules.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_manager import DatasetManager
from src.evaluator.evaluator import Evaluator
from src.tools.pipeline_builder import PipelineBuilder
from src.tools.pipeline_cache import PipelineCache
from src.tools.tool_executor import ToolExecutor


REQUIRED_RESULT_KEYS = {
    "dataset_name",
    "actions",
    "effective_actions",
    "pipeline_length",
    "model_action",
    "macro_f1",
    "runtime_sec",
    "from_cache",
    "status",
    "error",
}


def test_tool_executor_creates_scaler() -> None:
    """standard_scaler should map to a sklearn scaler."""
    tool = ToolExecutor(random_state=42).create_tool("standard_scaler")
    assert isinstance(tool, StandardScaler)


def test_tool_executor_creates_classifier() -> None:
    """random_forest should map to a sklearn classifier."""
    tool = ToolExecutor(random_state=42).create_tool("random_forest")
    assert isinstance(tool, RandomForestClassifier)


def test_pipeline_builder_returns_valid_pipeline() -> None:
    """A valid action list should produce a sklearn Pipeline."""
    result = PipelineBuilder(random_state=42).build_pipeline(
        ["standard_scaler", "svm", "evaluate"],
        n_features=4,
    )

    assert isinstance(result["pipeline"], Pipeline)
    assert result["effective_actions"] == ["standard_scaler", "svm"]
    assert result["pipeline_length"] == 2
    assert result["model_action"] == "svm"


def test_pipeline_builder_ignores_evaluate() -> None:
    """evaluate should be treated as terminal and excluded from the pipeline."""
    result = PipelineBuilder(random_state=42).build_pipeline(
        ["standard_scaler", "svm", "evaluate"],
        n_features=4,
    )
    assert "evaluate" not in result["pipeline"].named_steps


def test_pipeline_builder_ignores_do_nothing() -> None:
    """do_nothing should not become a sklearn Pipeline step."""
    result = PipelineBuilder(random_state=42).build_pipeline(
        ["do_nothing", "standard_scaler", "svm", "evaluate"],
        n_features=4,
    )
    assert result["effective_actions"] == ["standard_scaler", "svm"]
    assert "do_nothing" not in result["pipeline"].named_steps


def test_pipeline_builder_raises_without_model() -> None:
    """A pipeline without a model should be rejected."""
    with pytest.raises(ValueError, match="No model action"):
        PipelineBuilder(random_state=42).build_pipeline(
            ["standard_scaler", "pca", "evaluate"],
            n_features=4,
        )


def test_pipeline_builder_raises_with_multiple_models() -> None:
    """A pipeline with multiple model actions should be rejected."""
    with pytest.raises(ValueError):
        PipelineBuilder(random_state=42).build_pipeline(
            ["standard_scaler", "svm", "random_forest", "evaluate"],
            n_features=4,
        )


def test_evaluator_returns_required_result_keys() -> None:
    """Evaluator.evaluate should return the expected result dictionary."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    result = Evaluator(random_state=42).evaluate(
        dataset,
        ["standard_scaler", "svm", "evaluate"],
    )

    assert REQUIRED_RESULT_KEYS.issubset(result.keys())


def test_valid_pipeline_evaluation_succeeds() -> None:
    """A valid fixed pipeline should train and evaluate successfully."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    result = Evaluator(random_state=42).evaluate(
        dataset,
        ["standard_scaler", "svm", "evaluate"],
    )

    assert result["status"] == "success"
    assert result["error"] is None
    assert 0.0 <= result["macro_f1"] <= 1.0


def test_cache_returns_from_cache_on_second_evaluation() -> None:
    """The second identical evaluation should return the cached result."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    cache = PipelineCache()
    evaluator = Evaluator(cache=cache, random_state=42)
    actions = ["standard_scaler", "svm", "evaluate"]

    first_result = evaluator.evaluate(dataset, actions)
    second_result = evaluator.evaluate(dataset, actions)

    assert first_result["from_cache"] is False
    assert second_result["from_cache"] is True
    assert second_result["status"] == "success"
    assert cache.stats()["hits"] == 1
