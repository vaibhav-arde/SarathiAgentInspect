"""Metric configuration and threshold management.

Provides the configuration architecture for metrics, enforcing
thresholds, strict vs. soft pass/fail strategies, and
scoring boundaries for the framework.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from sarathi_agent_inspect.core.types import Score, Threshold  # noqa: TC001


class MetricConfig(BaseModel):
    """Configuration for a single metric execution.

    Enterprise considerations:
        - Centralized threshold management prevents arbitrary threshold lowering.
        - strict_mode enforces exact boundaries.
        - allow_async enables/disables concurrent execution per metric.
    """

    threshold: Threshold = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="The minimum score required to pass.",
    )
    strict_mode: bool = Field(
        default=True,
        description="If True, exact threshold matches pass. If False, soft pass allowed via tolerance.",
    )
    tolerance: float = Field(
        default=0.0,
        ge=0.0,
        description="Allowed tolerance below the threshold for a soft pass (only if strict_mode is False).",
    )
    weight: float = Field(
        default=1.0,
        gt=0.0,
        description="Weight of this metric when used inside a composite metric.",
    )
    allow_async: bool = Field(
        default=True,
        description="Whether this metric can be executed asynchronously.",
    )
    extra_params: dict[str, Any] = Field(
        default_factory=dict,
        description="Metric-specific custom parameters (e.g., specific evaluation criteria).",
    )
    normalization_strategy: Literal["min_max", "z_score", "none"] = Field(
        default="none",
        description="How to normalize the raw score to the 0.0-1.0 range.",
    )

    def is_passing(self, score: Score) -> bool:
        """Determine if a score passes based on the threshold and tolerance."""
        if self.strict_mode:
            return score >= self.threshold
        return score >= (self.threshold - self.tolerance)
