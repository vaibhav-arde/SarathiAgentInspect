"""Agent Reasoning Evaluation.

Evaluates an agent's planning capabilities and the efficiency
of its multi-step reasoning chains.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sarathi_agent_inspect.metrics.deepeval_wrappers import WrappedGEvalMetric

if TYPE_CHECKING:
    from sarathi_agent_inspect.providers.base import BaseProvider


class PlanningEvaluator:
    """Evaluates the quality of agent plans."""

    def __init__(self, provider: BaseProvider | None = None) -> None:
        # Reusing GEval for qualitative planning assessment
        self.geval = WrappedGEvalMetric(
            name="Planning Quality",
            criteria="Evaluate if the plan is logical, complete, and avoids redundant steps.",
            evaluation_params=[], # Standard params will be added during execution
            provider=provider,
        )

    async def evaluate_plan(self, task: str, plan: str) -> float:
        """Score the agent's plan for a given task."""
        result = await self.geval.evaluate(
            input_text=task,
            actual_output=plan,
        )
        return result.score


class ReasoningEvaluator:
    """Evaluates the internal consistency of multi-step reasoning."""

    @staticmethod
    def detect_redundancy(thoughts: list[str]) -> float:
        """Simple heuristic to detect repetitive thoughts.

        Returns a score from 0.0 (high redundancy) to 1.0 (no redundancy).
        """
        if not thoughts:
            return 1.0

        unique_thoughts = set(thoughts)
        return len(unique_thoughts) / len(thoughts)
