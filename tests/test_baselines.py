"""
Tests for random and fixed-pipeline baseline agents.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.baselines.fixed_pipeline import FixedPipelineAgent
from src.baselines.random_agent import RandomAgent
from src.constants import ACTIONS
from src.data.dataset_manager import DatasetManager
from src.env.tool_selection_env import ToolSelectionEnv


REQUIRED_RESULT_KEYS = {
    "method",
    "dataset_name",
    "total_reward",
    "macro_f1",
    "pipeline_actions",
    "pipeline_length",
    "invalid_action_count",
    "evaluate_success",
    "runtime_sec",
    "steps",
    "status",
    "error",
}


def load_iris_dataset() -> dict:
    """Load the small iris dataset for baseline tests."""
    return DatasetManager(random_state=42).load_dataset("iris")


def make_env() -> ToolSelectionEnv:
    """Create a deterministic environment for baseline tests."""
    return ToolSelectionEnv(max_steps=6, random_state=42)


def test_random_agent_select_action_returns_integer() -> None:
    """RandomAgent.select_action should return an integer action index."""
    env = make_env()
    env.reset(load_iris_dataset())
    action = RandomAgent(use_action_mask=True, random_state=42).select_action(env)

    assert isinstance(action, int)
    assert action in ACTIONS


def test_random_agent_with_mask_selects_valid_action() -> None:
    """Masked random selection should choose only currently valid actions."""
    env = make_env()
    env.reset(load_iris_dataset())
    agent = RandomAgent(use_action_mask=True, random_state=42)

    for _ in range(20):
        action = agent.select_action(env)
        assert action in env.get_valid_actions()


def test_random_agent_run_episode_returns_required_keys() -> None:
    """RandomAgent.run_episode should return the standard result format."""
    result = RandomAgent(use_action_mask=True, random_state=42).run_episode(
        make_env(),
        load_iris_dataset(),
    )

    assert REQUIRED_RESULT_KEYS.issubset(result.keys())


def test_fixed_pipeline_agent_can_run_scaler_svm() -> None:
    """FixedPipelineAgent should execute standard_scaler -> svm -> evaluate."""
    result = FixedPipelineAgent(
        name="fixed_scaler_svm",
        actions=[1, 6, 8],
    ).run_episode(make_env(), load_iris_dataset())

    assert result["status"] == "success"
    assert result["evaluate_success"] is True
    assert result["pipeline_actions"] == ["standard_scaler", "svm"]


def test_fixed_pipeline_run_episode_returns_required_keys() -> None:
    """FixedPipelineAgent.run_episode should return the standard result format."""
    result = FixedPipelineAgent(
        name="fixed_scaler_svm",
        actions=[1, 6, 8],
    ).run_episode(make_env(), load_iris_dataset())

    assert REQUIRED_RESULT_KEYS.issubset(result.keys())


def test_fixed_pipeline_result_method_matches_name() -> None:
    """Fixed pipeline result method should equal the configured agent name."""
    result = FixedPipelineAgent(
        name="fixed_scaler_svm",
        actions=[1, 6, 8],
    ).run_episode(make_env(), load_iris_dataset())

    assert result["method"] == "fixed_scaler_svm"


def test_fixed_pipeline_episode_eventually_ends() -> None:
    """A valid fixed pipeline should end through evaluation."""
    result = FixedPipelineAgent(
        name="fixed_scaler_svm",
        actions=[1, 6, 8],
    ).run_episode(make_env(), load_iris_dataset())

    assert result["steps"] <= 6
    assert result["evaluate_success"] is True


def test_pipeline_length_is_non_negative() -> None:
    """Baseline pipeline length should be non-negative."""
    result = RandomAgent(use_action_mask=True, random_state=42).run_episode(
        make_env(),
        load_iris_dataset(),
    )

    assert result["pipeline_length"] >= 0


def test_invalid_action_count_is_non_negative() -> None:
    """Baseline invalid action count should be non-negative."""
    result = RandomAgent(use_action_mask=False, random_state=42).run_episode(
        make_env(),
        load_iris_dataset(),
    )

    assert result["invalid_action_count"] >= 0


def test_baseline_status_is_success_or_failed() -> None:
    """Baseline result status should be one of the expected strings."""
    result = RandomAgent(use_action_mask=True, random_state=42).run_episode(
        make_env(),
        load_iris_dataset(),
    )

    assert result["status"] in {"success", "failed"}
