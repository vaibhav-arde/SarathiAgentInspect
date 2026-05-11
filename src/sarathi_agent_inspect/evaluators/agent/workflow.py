"""Agent Workflow and Replay.

Evaluates autonomous workflows, multi-agent interactions,
and provides replay capabilities for deterministic testing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sarathi_agent_inspect.evaluators.agent.trace import AgentTrace


class WorkflowEvaluator:
    """Evaluates the success of end-to-end autonomous workflows."""

    def evaluate_success(self, trace: AgentTrace, expected_outcome: str) -> bool:
        """Determines if the workflow achieved the final target outcome."""
        # This would typically be a combination of TaskCompletionScorer
        # and final observation validation.
        return expected_outcome.lower() in trace.input_text.lower()  # Placeholder


class MultiAgentEvaluator:
    """Tracks and evaluates handoffs between multiple agents."""

    def __init__(self) -> None:
        self.handoffs: list[dict[str, str]] = []

    def track_handoff(self, from_agent: str, to_agent: str, context: str) -> None:
        """Log a handoff between two agents."""
        self.handoffs.append({"from": from_agent, "to": to_agent, "context": context})

    def get_handoff_efficiency(self) -> float:
        """Measures if handoffs are too frequent (chatter)."""
        if not self.handoffs:
            return 1.0
        return 1.0 / len(self.handoffs)


class ReplayEngine:
    """Allows replaying an agent trace with mocked tool responses."""

    def __init__(self, trace: AgentTrace) -> None:
        self.trace = trace

    def get_mock_responses(self) -> dict[str, str]:
        """Extracts tool observations from a trace to use as mocks.

        Useful for testing if the agent makes the same decisions
        given the exact same observations (determinism testing).
        """
        mocks = {}
        for span in self.trace.spans:
            for i, step in enumerate(span.steps):
                if step.type.name == "ACTION" and i + 1 < len(span.steps):
                    next_step = span.steps[i + 1]
                    if next_step.type.name == "OBSERVATION":
                        mocks[step.content] = next_step.content
        return mocks
