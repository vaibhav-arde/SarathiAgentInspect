"""Agent Memory Evaluation.

Tests if an agent can retain and utilize information across
long-running multi-step traces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sarathi_agent_inspect.evaluators.agent.trace import AgentTrace


class MemoryRetentionEvaluator:
    """Evaluates how well the agent remembers facts from earlier steps."""

    def __init__(self, trace: AgentTrace) -> None:
        self.trace = trace

    def check_fact_usage(self, fact: str, start_step_idx: int = 0) -> bool:
        """Check if a specific fact from early steps is used in later thoughts or actions.

        Args:
            fact: The string/fact to look for.
            start_step_idx: The step from which to start searching (usually after the fact was introduced).
        """
        all_steps = []
        for span in self.trace.spans:
            all_steps.extend(span.steps)

        if start_step_idx >= len(all_steps):
            return False

        fact_lower = fact.lower()
        return any(fact_lower in step.content.lower() for step in all_steps[start_step_idx:])

    def calculate_memory_score(self, golden_facts: list[str]) -> float:
        """Calculate the ratio of correctly retained facts."""
        if not golden_facts:
            return 1.0

        retained_count = sum(1 for fact in golden_facts if self.check_fact_usage(fact))
        return retained_count / len(golden_facts)
