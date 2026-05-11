"""Agent Tool Evaluation.

Provides utilities for tracking tool invocations, validating correctness,
and detecting tool hallucinations.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sarathi_agent_inspect.providers.base import BaseProvider


from pydantic import BaseModel, ValidationError


class ToolEvaluator:
    """Evaluates the correctness of agent tool calls."""

    def __init__(self, provider: BaseProvider | None = None) -> None:
        self.provider = provider

    def validate_schema(self, arguments: str | dict[str, Any], schema: type[BaseModel] | dict[str, Any]) -> bool:
        """Strict schema validation for tool arguments.

        Supports both Pydantic models (recommended for enterprise) or simple dict schemas.
        """
        try:
            args = json.loads(arguments) if isinstance(arguments, str) else arguments

            if isinstance(schema, type) and issubclass(schema, BaseModel):
                schema.model_validate(args)
                return True

            # Fallback to simple dict key check
            if isinstance(schema, dict):
                return all(key in args for key in schema)

            return False
        except (json.JSONDecodeError, TypeError, ValidationError):
            return False

    async def evaluate_semantic_correctness(self, tool_name: str, arguments: str, context: str) -> float:
        """Semantic evaluation of tool calls via LLM judge.

        Checks if the chosen tool and arguments make sense given the agent's current task.
        """
        if not self.provider:
            return 1.0  # Cannot evaluate without provider

        prompt = (
            f"Context: {context}\n"
            f"Agent chose tool: {tool_name}\n"
            f"With arguments: {arguments}\n\n"
            "Is this tool call logical and necessary to complete the task? "
            "Respond ONLY with a score between 0.0 and 1.0."
        )

        response = await self.provider.generate(prompt=prompt, temperature=0.0)
        try:
            return float(response.content.strip())
        except ValueError:
            return 0.0

    def detect_hallucination(self, tool_name: str, allowed_tools: list[str]) -> bool:
        """Detect if the agent called a tool that doesn't exist."""
        return tool_name not in allowed_tools


class ToolTracker:
    """Tracks frequency and performance of tool usage."""

    def __init__(self) -> None:
        self.stats: dict[str, dict[str, Any]] = {}

    def track_call(self, tool_name: str, latency_ms: float, cost_usd: float) -> None:
        """Update tool statistics."""
        if tool_name not in self.stats:
            self.stats[tool_name] = {"calls": 0, "total_latency": 0.0, "total_cost": 0.0}

        self.stats[tool_name]["calls"] += 1
        self.stats[tool_name]["total_latency"] += latency_ms
        self.stats[tool_name]["total_cost"] += cost_usd

    def get_report(self) -> dict[str, Any]:
        """Generate a summary of tool performance."""
        return self.stats
