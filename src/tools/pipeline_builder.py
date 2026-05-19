"""
Pipeline Builder module.

This module will translate selected actions into a machine learning pipeline
representation.
"""

from __future__ import annotations

from typing import Any

from sklearn.pipeline import Pipeline

from src.tools.tool_executor import ToolExecutor


class PipelineBuilder:
    """Build sklearn Pipelines from ordered action-name lists."""

    def __init__(self, random_state: int = 42) -> None:
        self.executor = ToolExecutor(random_state=random_state)

    def build_pipeline(
        self,
        actions: list[str],
        n_features: int | None = None,
    ) -> dict[str, Any]:
        """
        Convert an action sequence into a sklearn Pipeline and metadata.

        The resulting effective action list excludes ``do_nothing`` and
        ``evaluate``. Exactly one model action must appear, and it must be the
        final effective action.
        """
        steps = []
        effective_actions = []
        model_action: str | None = None

        for raw_action in actions:
            action = raw_action.lower()

            if self.executor.is_terminal_action(action):
                break
            if action == "do_nothing":
                continue

            if model_action is not None and self.executor.is_model_action(action):
                raise ValueError("Multiple model actions selected. Use exactly one model action.")
            if model_action is not None:
                raise ValueError(
                    "Actions after a model are invalid before the evaluate action."
                )

            if self.executor.is_model_action(action):
                model_action = action
            elif not self.executor.is_preprocess_action(action):
                self.executor.create_tool(action, n_features=n_features)
                raise ValueError(f"Unsupported pipeline action '{action}'.")

            tool = self.executor.create_tool(action, n_features=n_features)
            if tool is not None:
                steps.append((self._unique_step_name(action, steps), tool))
                effective_actions.append(action)

        model_actions = [
            action for action in effective_actions if self.executor.is_model_action(action)
        ]
        if not model_actions:
            raise ValueError("No model action selected. Choose one of: random_forest, svm, knn.")
        if len(model_actions) > 1:
            raise ValueError("Multiple model actions selected. Use exactly one model action.")
        if effective_actions[-1] != model_actions[0]:
            raise ValueError("The model action must be the final effective pipeline action.")

        return {
            "pipeline": Pipeline(steps),
            "effective_actions": effective_actions,
            "pipeline_length": len(effective_actions),
            "model_action": model_actions[0],
        }

    @staticmethod
    def _unique_step_name(action: str, steps: list[tuple[str, Any]]) -> str:
        """Create a unique sklearn Pipeline step name."""
        existing_names = {name for name, _ in steps}
        if action not in existing_names:
            return action

        suffix = 2
        while f"{action}_{suffix}" in existing_names:
            suffix += 1
        return f"{action}_{suffix}"
