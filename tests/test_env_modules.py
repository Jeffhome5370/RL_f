"""
Tests for environment state, action masking, rewards, and step behavior.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset_manager import DatasetManager
from src.env.action_mask import ActionMask
from src.env.state_builder import StateBuilder
from src.env.tool_selection_env import ToolSelectionEnv


REQUIRED_INFO_KEYS = {
    "action",
    "action_name",
    "valid_action",
    "invalid_reason",
    "pipeline_actions",
    "current_step",
    "invalid_action_count",
    "estimated_cost",
    "evaluation_result",
    "auto_terminated",
}


def load_iris_dataset() -> dict:
    """Load the small iris dataset for environment tests."""
    return DatasetManager(random_state=42).load_dataset("iris")


def test_state_builder_returns_shape_15() -> None:
    """StateBuilder should return a float32 vector with 15 features."""
    state = StateBuilder().build_state(
        dataset_profile={
            "n_samples": 150,
            "n_features": 4,
            "n_classes": 3,
            "missing_ratio": 0.0,
            "class_imbalance_ratio": 0.33,
            "numeric_ratio": 1.0,
            "mean_abs_correlation": 0.4,
        },
        pipeline_actions=["standard_scaler", "svm"],
        current_step=2,
        invalid_action_count=0,
        estimated_cost=0.8,
    )

    assert isinstance(state, np.ndarray)
    assert state.shape == (15,)
    assert state.dtype == np.float32


def test_reset_returns_initial_state_shape_15() -> None:
    """Environment reset should return the initial state vector."""
    state = ToolSelectionEnv(max_steps=6, random_state=42).reset(load_iris_dataset())
    assert state.shape == (15,)


def test_action_mask_shape() -> None:
    """Action masks should align with the 9-action space."""
    mask = ActionMask().get_action_mask([])
    assert mask.shape == (9,)
    assert mask.dtype == np.float32


def test_evaluate_invalid_before_model() -> None:
    """Evaluate should be invalid until a model action is selected."""
    masker = ActionMask()
    assert not masker.is_valid_action(8, [])
    assert masker.get_action_mask([])[8] == 0.0


def test_scaler_selection_invalidates_both_scalers() -> None:
    """Selecting one scaler should invalidate both scaler actions."""
    masker = ActionMask()
    mask = masker.get_action_mask(["standard_scaler"])
    assert mask[1] == 0.0
    assert mask[2] == 0.0


def test_model_selection_invalidates_other_models() -> None:
    """Selecting a model should invalidate all model actions."""
    masker = ActionMask()
    mask = masker.get_action_mask(["svm"])
    assert mask[5] == 0.0
    assert mask[6] == 0.0
    assert mask[7] == 0.0
    assert mask[8] == 1.0


def test_valid_non_terminal_action_updates_pipeline_actions() -> None:
    """A valid preprocessing action should be appended to pipeline state."""
    env = ToolSelectionEnv(max_steps=6, random_state=42)
    env.reset(load_iris_dataset())
    next_state, reward, done, info = env.step(1)

    assert next_state.shape == (15,)
    assert reward < 0
    assert done is False
    assert info["pipeline_actions"] == ["standard_scaler"]
    assert env.get_pipeline_actions() == ["standard_scaler"]


def test_invalid_action_increases_invalid_action_count() -> None:
    """Invalid actions should increment invalid_action_count."""
    env = ToolSelectionEnv(max_steps=6, random_state=42)
    env.reset(load_iris_dataset())
    env.step(1)
    _, _, done, info = env.step(2)

    assert done is False
    assert info["valid_action"] is False
    assert info["invalid_action_count"] == 1
    assert info["invalid_reason"] is not None


def test_evaluate_after_model_ends_episode() -> None:
    """Evaluate should end the episode after a model has been selected."""
    env = ToolSelectionEnv(max_steps=6, random_state=42)
    env.reset(load_iris_dataset())
    env.step(6)
    next_state, reward, done, info = env.step(8)

    assert next_state.shape == (15,)
    assert done is True
    assert reward > -1.0
    assert info["evaluation_result"] is not None
    assert info["evaluation_result"]["status"] == "success"


def test_step_returns_expected_tuple_types() -> None:
    """step() should return next_state, reward, done, and info."""
    env = ToolSelectionEnv(max_steps=6, random_state=42)
    env.reset(load_iris_dataset())
    result = env.step(0)

    assert isinstance(result, tuple)
    assert len(result) == 4
    assert isinstance(result[0], np.ndarray)
    assert isinstance(result[1], float)
    assert isinstance(result[2], bool)
    assert isinstance(result[3], dict)


def test_environment_auto_terminates_at_max_steps() -> None:
    """Repeated non-terminal actions should auto terminate at max_steps."""
    env = ToolSelectionEnv(max_steps=2, random_state=42)
    env.reset(load_iris_dataset())
    env.step(0)
    _, reward, done, info = env.step(0)

    assert done is True
    assert reward == -1.0
    assert info["auto_terminated"] is True
    assert info["evaluation_result"]["status"] == "failed"


def test_info_dictionary_contains_required_keys() -> None:
    """Every step should return a useful debugging info dictionary."""
    env = ToolSelectionEnv(max_steps=6, random_state=42)
    env.reset(load_iris_dataset())
    _, _, _, info = env.step(0)

    assert REQUIRED_INFO_KEYS.issubset(info.keys())
