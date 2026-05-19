"""
Tests for Grid Search baseline generation, selection, and execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.grid_search import GridSearchBaseline
from src.data.dataset_manager import DatasetManager


MODEL_ACTIONS = {"random_forest", "svm", "knn"}
REQUIRED_RESULT_KEYS = {
    "method",
    "dataset_name",
    "best_actions",
    "best_macro_f1",
    "best_pipeline_length",
    "best_runtime_sec",
    "num_pipelines",
    "num_success",
    "num_failed",
    "all_results",
    "cache_stats",
    "status",
    "error",
}


def test_generate_action_sequences_returns_27_pipelines() -> None:
    """Grid Search should generate the expected 27 combinations."""
    pipelines = GridSearchBaseline().generate_action_sequences()
    assert len(pipelines) == 27


def test_every_pipeline_ends_with_evaluate() -> None:
    """All generated pipelines should end with evaluate."""
    pipelines = GridSearchBaseline().generate_action_sequences()
    assert all(pipeline[-1] == "evaluate" for pipeline in pipelines)


def test_every_pipeline_contains_exactly_one_model() -> None:
    """Each generated pipeline should contain exactly one model action."""
    pipelines = GridSearchBaseline().generate_action_sequences()
    for pipeline in pipelines:
        model_count = sum(action in MODEL_ACTIONS for action in pipeline)
        assert model_count == 1


def test_no_pipeline_contains_do_nothing() -> None:
    """Grid Search should not include do_nothing actions."""
    pipelines = GridSearchBaseline().generate_action_sequences()
    assert all("do_nothing" not in pipeline for pipeline in pipelines)


def test_no_actions_after_model_except_evaluate() -> None:
    """Generated pipelines should place evaluate immediately after the model."""
    pipelines = GridSearchBaseline().generate_action_sequences()
    for pipeline in pipelines:
        model_index = next(
            index for index, action in enumerate(pipeline) if action in MODEL_ACTIONS
        )
        assert pipeline[model_index + 1 :] == ["evaluate"]


def test_select_best_result_chooses_highest_macro_f1() -> None:
    """Best selection should prefer highest macro F1."""
    best = GridSearchBaseline().select_best_result(
        [
            {"status": "success", "macro_f1": 0.8, "pipeline_length": 2, "runtime_sec": 0.1},
            {"status": "success", "macro_f1": 0.9, "pipeline_length": 3, "runtime_sec": 0.2},
        ]
    )
    assert best["macro_f1"] == 0.9


def test_select_best_result_uses_shorter_pipeline_tie_breaker() -> None:
    """If macro F1 ties, shorter pipeline should win."""
    best = GridSearchBaseline().select_best_result(
        [
            {"status": "success", "macro_f1": 0.9, "pipeline_length": 3, "runtime_sec": 0.1},
            {"status": "success", "macro_f1": 0.9, "pipeline_length": 2, "runtime_sec": 0.2},
        ]
    )
    assert best["pipeline_length"] == 2


def test_select_best_result_uses_lower_runtime_second_tie_breaker() -> None:
    """If macro F1 and length tie, lower runtime should win."""
    best = GridSearchBaseline().select_best_result(
        [
            {"status": "success", "macro_f1": 0.9, "pipeline_length": 2, "runtime_sec": 0.2},
            {"status": "success", "macro_f1": 0.9, "pipeline_length": 2, "runtime_sec": 0.1},
        ]
    )
    assert best["runtime_sec"] == 0.1


def test_run_dataset_returns_required_keys() -> None:
    """run_dataset should return the expected summary dictionary."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    result = GridSearchBaseline(random_state=42).run_dataset(dataset)
    assert REQUIRED_RESULT_KEYS.issubset(result.keys())


def test_run_dataset_status_is_success_or_failed() -> None:
    """run_dataset status should be one of the expected values."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    result = GridSearchBaseline(random_state=42).run_dataset(dataset)
    assert result["status"] in {"success", "failed"}


def test_grid_search_on_iris_does_not_crash() -> None:
    """Running Grid Search on iris should evaluate all 27 pipelines."""
    dataset = DatasetManager(random_state=42).load_dataset("iris")
    result = GridSearchBaseline(random_state=42).run_dataset(dataset)
    assert result["num_pipelines"] == 27
    assert len(result["all_results"]) == 27
