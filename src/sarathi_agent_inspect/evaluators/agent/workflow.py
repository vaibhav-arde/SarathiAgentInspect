"""Agent Workflow and Replay.

Evaluates autonomous workflows, multi-agent interactions,
and provides replay capabilities for deterministic testing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from sarathi_agent_inspect.evaluators.agent.trace import StepType, TraceScorer

if TYPE_CHECKING:
    from sarathi_agent_inspect.evaluators.agent.trace import AgentTrace


@dataclass(frozen=True)
class WorkflowEvaluationResult:
    """Structured outcome of an end-to-end workflow evaluation."""

    success: bool
    score: float
    outcome_match_score: float
    completion_signal_score: float
    tool_coverage_score: float
    error_penalty: float
    details: dict[str, Any] = field(default_factory=dict)


class WorkflowEvaluator:
    """Evaluates the success of end-to-end autonomous workflows."""

    def evaluate(
        self,
        trace: AgentTrace,
        expected_outcome: str,
        *,
        required_tools: list[str] | None = None,
        success_threshold: float = 0.7,
    ) -> WorkflowEvaluationResult:
        """Score a workflow using its terminal state, tool usage, and errors."""
        steps = trace.iter_steps()
        final_content = trace.final_content()
        completion_signal = self._completion_signal_score(trace)
        outcome_match = self._outcome_match_score(expected_outcome, final_content, steps)
        tool_coverage = self._tool_coverage_score(steps, required_tools or [])
        error_penalty = self._error_penalty(steps)
        efficiency = TraceScorer.calculate_efficiency(trace)

        raw_score = (
            (outcome_match * 0.45)
            + (completion_signal * 0.25)
            + (tool_coverage * 0.15)
            + (efficiency * 0.15)
            - error_penalty
        )
        score = max(0.0, min(1.0, raw_score))

        return WorkflowEvaluationResult(
            success=score >= success_threshold,
            score=score,
            outcome_match_score=outcome_match,
            completion_signal_score=completion_signal,
            tool_coverage_score=tool_coverage,
            error_penalty=error_penalty,
            details={
                "final_content": final_content,
                "required_tools": required_tools or [],
                "used_tools": [step.content for step in steps if step.type == StepType.ACTION],
                "efficiency_score": efficiency,
                "step_count": len(steps),
            },
        )

    def evaluate_success(
        self,
        trace: AgentTrace,
        expected_outcome: str,
        *,
        required_tools: list[str] | None = None,
        success_threshold: float = 0.7,
    ) -> bool:
        """Determine whether the workflow achieved its outcome."""
        return self.evaluate(
            trace,
            expected_outcome,
            required_tools=required_tools,
            success_threshold=success_threshold,
        ).success

    @staticmethod
    def _normalize_tokens(text: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    def _outcome_match_score(self, expected_outcome: str, final_content: str, steps: list[Any]) -> float:
        """Measure how well the final state reflects the expected outcome."""
        expected_tokens = set(self._normalize_tokens(expected_outcome))
        if not expected_tokens:
            return 1.0

        final_tokens = set(self._normalize_tokens(final_content))
        final_coverage = len(expected_tokens & final_tokens) / len(expected_tokens) if final_tokens else 0.0

        history_tokens = set()
        for step in steps:
            history_tokens.update(self._normalize_tokens(step.content))
        history_coverage = len(expected_tokens & history_tokens) / len(expected_tokens) if history_tokens else 0.0

        return min(1.0, (final_coverage * 0.7) + (history_coverage * 0.3))

    @staticmethod
    def _completion_signal_score(trace: AgentTrace) -> float:
        """Look for explicit completion metadata or successful terminal observations."""
        if trace.metadata.get("task_completed") is True or trace.metadata.get("status") == "success":
            return 1.0

        final_step = trace.final_step()
        if final_step is None:
            return 0.0

        if final_step.metadata.get("task_completed") is True or final_step.metadata.get("status") == "success":
            return 1.0

        terminal_type_bonus = 1.0 if final_step.type in {StepType.OBSERVATION, StepType.SYSTEM} else 0.6
        return terminal_type_bonus if final_step.type != StepType.ERROR else 0.0

    @staticmethod
    def _tool_coverage_score(steps: list[Any], required_tools: list[str]) -> float:
        """Score tool usage against expected critical actions."""
        if not required_tools:
            return 1.0

        action_text = " ".join(step.content.lower() for step in steps if step.type == StepType.ACTION)
        matched = sum(1 for tool in required_tools if tool.lower() in action_text)
        return matched / len(required_tools)

    @staticmethod
    def _error_penalty(steps: list[Any]) -> float:
        """Penalize explicit errors in the trace."""
        error_steps = sum(1 for step in steps if step.type == StepType.ERROR)
        return min(0.4, error_steps * 0.1)


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
