"""Token accounting and tracking utilities.

Provides mechanisms to track token usage and calculate
aggregate costs across multiple LLM calls or entire evaluation sessions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sarathi_agent_inspect.providers.base import ProviderResponse


@dataclass
class TokenUsage:
    """Snapshot of token usage for a single request or aggregated session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: TokenUsage | ProviderResponse) -> None:
        """Add tokens and cost from another usage record or provider response.

        Args:
            other: The usage record or response to add to this instance.
        """
        self.prompt_tokens += getattr(other, "prompt_tokens", 0) or 0
        self.completion_tokens += getattr(other, "completion_tokens", 0) or 0
        self.total_tokens += getattr(other, "total_tokens", 0) or 0
        self.cost_usd += getattr(other, "cost_usd", 0.0) or 0.0


@dataclass
class TokenTracker:
    """Session-level tracker for token usage across multiple models.

    Maintains total aggregate usage as well as per-model breakdowns.
    """

    total_usage: TokenUsage = field(default_factory=TokenUsage)
    model_usage: dict[str, TokenUsage] = field(default_factory=dict)
    calls_tracked: int = 0

    def track(self, response: ProviderResponse) -> None:
        """Track usage from a single provider response.

        Args:
            response: The completed provider response.
        """
        self.total_usage.add(response)

        model_name = response.model
        if model_name not in self.model_usage:
            self.model_usage[model_name] = TokenUsage()
        self.model_usage[model_name].add(response)

        self.calls_tracked += 1


def format_token_report(tracker: TokenTracker) -> str:
    """Format a human-readable report of token usage.

    Args:
        tracker: The populated token tracker.

    Returns:
        Formatted string report.
    """
    lines = [
        "=== Token Usage Report ===",
        f"Total Calls: {tracker.calls_tracked}",
        f"Total Tokens: {tracker.total_usage.total_tokens:,} "
        f"(Prompt: {tracker.total_usage.prompt_tokens:,} | "
        f"Completion: {tracker.total_usage.completion_tokens:,})",
        f"Estimated Cost: ${tracker.total_usage.cost_usd:.6f}",
    ]

    if tracker.model_usage:
        lines.append("\nBreakdown by Model:")
        for model, usage in tracker.model_usage.items():
            lines.append(f"  - {model}: {usage.total_tokens:,} tokens (${usage.cost_usd:.6f})")

    return "\n".join(lines)
