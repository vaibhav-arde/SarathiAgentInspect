"""Abstract base metric interface.

Defines the contract for all evaluation metrics, including
built-in DeepEval metrics, custom metrics, composite metrics,
and async metrics.

A metric:
    1. Computes a numeric score for a test case
    2. Validates the score against a threshold
    3. Returns a structured MetricResult
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sarathi_agent_inspect.core.types import JsonDict, Score, Threshold


@dataclass(frozen=True)
class MetricResult:
    """Result of a single metric computation.

    Attributes:
        metric_name: Name of the metric.
        score: Computed score (typically 0.0 to 1.0).
        passed: Whether the score meets the threshold.
        threshold: The threshold used for pass/fail.
        reason: Human-readable explanation of the score.
        metadata: Additional metric-specific data.
    """

    metric_name: str
    score: Score
    passed: bool
    threshold: Threshold
    reason: str = ""
    metadata: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "metric_name": self.metric_name,
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "reason": self.reason,
            "metadata": self.metadata,
        }


class BaseMetric(ABC):
    """Abstract base class for evaluation metrics.

    All metric implementations (DeepEval wrappers, custom metrics,
    composite metrics) must inherit from this class.
    """

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return the metric identifier."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Return a human-readable description of what the metric measures."""
        ...

    @property
    def score_range(self) -> tuple[float, float]:
        """Return the valid score range (min, max).

        Override if the metric uses a non-standard range.
        """
        return (0.0, 1.0)

    @abstractmethod
    async def compute(
        self,
        *,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        context: list[str] | None = None,
        retrieval_context: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Compute the metric score for a test case.

        Args:
            input_text: The input/query text.
            actual_output: The actual LLM/system output.
            expected_output: The expected/golden output.
            context: Ground truth context documents.
            retrieval_context: Retrieved context documents.
            **kwargs: Additional metric-specific parameters.

        Returns:
            MetricResult with the computed score.

        Raises:
            MetricComputationError: If the metric fails to compute.
        """
        ...

    def validate_threshold(self, score: Score, threshold: Threshold) -> bool:
        """Check if a score meets the threshold.

        Args:
            score: The computed metric score.
            threshold: The pass/fail threshold.

        Returns:
            True if score >= threshold.
        """
        return score >= threshold
