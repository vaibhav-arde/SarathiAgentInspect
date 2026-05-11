"""Base reporting abstractions.

Provides the foundational classes for generating reports across
different formats (JSON, HTML, Allure).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any


@dataclass
class ReportMetadata:
    """Metadata about an evaluation run."""

    run_id: str
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    environment: str = "local"
    framework_version: str = "0.1.0"
    tags: list[str] = field(default_factory=list)
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0


@dataclass
class EvaluationSummary:
    """Aggregated summary of an evaluation run."""

    total_records: int
    passed_count: int
    failed_count: int
    pass_rate: float
    average_score: float
    metadata: ReportMetadata


class BaseReporter(ABC):
    """Abstract base class for all reporters."""

    @abstractmethod
    def generate(self, results: list[Any], summary: EvaluationSummary) -> Any:
        """Generate a report from evaluation results.

        Args:
            results: List of detailed evaluation results.
            summary: Aggregated summary of the run.
        """
        ...
