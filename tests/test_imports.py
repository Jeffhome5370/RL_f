"""
Import smoke tests for the project skeleton.

This file can be run with plain Python or with pytest.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.constants import ACTIONS, MODEL_ACTIONS, PREPROCESS_ACTIONS, TERMINAL_ACTION


def test_constants_importable():
    """Verify that action constants are importable and minimally consistent."""
    assert ACTIONS[0] == "do_nothing"
    assert ACTIONS[TERMINAL_ACTION] == "evaluate"
    assert PREPROCESS_ACTIONS == {1, 2, 3, 4}
    assert MODEL_ACTIONS == {5, 6, 7}


def test_main_modules_importable():
    """Verify that placeholder modules can be imported successfully."""
    import src.agent.dqn_agent
    import src.agent.q_network
    import src.agent.replay_buffer
    import src.agent.trainer
    import src.baselines.fixed_pipeline
    import src.baselines.grid_search
    import src.baselines.random_agent
    import src.data.data_profiler
    import src.data.dataset_manager
    import src.env.action_mask
    import src.env.reward
    import src.env.state_builder
    import src.env.tool_selection_env
    import src.evaluator.evaluator
    import src.evaluator.metrics
    import src.tools.pipeline_builder
    import src.tools.pipeline_cache
    import src.tools.tool_executor
    import src.utils.logger
    import src.utils.plot
    import src.utils.seed


if __name__ == "__main__":
    test_constants_importable()
    test_main_modules_importable()
    print("Import smoke tests passed.")
