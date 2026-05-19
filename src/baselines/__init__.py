"""
Baselines package.

This package will contain simple baseline strategies for comparison with the
RL agent.
"""

from src.baselines.fixed_pipeline import FIXED_PIPELINES, FixedPipelineAgent
from src.baselines.grid_search import GridSearchBaseline
from src.baselines.random_agent import RandomAgent

__all__ = ["FIXED_PIPELINES", "FixedPipelineAgent", "GridSearchBaseline", "RandomAgent"]
