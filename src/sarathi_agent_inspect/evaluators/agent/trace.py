"""Agent Tracing Architecture.

Provides structures for capturing and scoring multi-step agent executions.
Includes step-level and span-level evaluation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepType(Enum):
    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    SYSTEM = "system"
    ERROR = "error"


@dataclass
class AgentStep:
    """A single atomic unit of agent execution."""
    step_id: str
    type: StepType
    content: str
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    cost_usd: float = 0.0


@dataclass
class AgentSpan:
    """A group of related steps representing a sub-task."""
    span_id: str
    name: str
    steps: list[AgentStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    def add_step(self, step: AgentStep) -> None:
        self.steps.append(step)

    def complete(self) -> None:
        self.end_time = time.time()

    @property
    def duration_ms(self) -> float:
        if not self.end_time:
            return 0.0
        return (self.end_time - self.start_time) * 1000.0


@dataclass
class AgentTrace:
    """Full execution history of an agent's task."""
    trace_id: str
    task_input: str
    spans: list[AgentSpan] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    total_cost_usd: float = 0.0

    def add_span(self, span: AgentSpan) -> None:
        self.spans.append(span)


class TraceScorer:
    """Evaluates agent traces at multiple levels."""

    @staticmethod
    def calculate_efficiency(trace: AgentTrace) -> float:
        """Score based on the ratio of productive steps vs total steps.

        Productive steps are Thoughts and Actions.
        Too many thoughts without actions might indicate looping or indecision.
        """
        total_steps = sum(len(span.steps) for span in trace.spans)
        if total_steps == 0:
            return 1.0

        thought_count = 0
        action_count = 0
        for span in trace.spans:
            for step in span.steps:
                if step.type == StepType.THOUGHT:
                    thought_count += 1
                elif step.type == StepType.ACTION:
                    action_count += 1

        if action_count == 0:
            return 0.0 if thought_count > 0 else 1.0

        # Heuristic: 1 Thought per Action is ideal (1.0).
        # More thoughts decrease the efficiency score.
        ratio = action_count / thought_count if thought_count > 0 else 1.0
        return min(1.0, ratio)
