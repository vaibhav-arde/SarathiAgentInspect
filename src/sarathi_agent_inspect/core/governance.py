"""Governance and Quality Gates for AI Evaluations.

Enforces performance thresholds and regression blocking to ensure
high-quality deployments.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from sarathi_agent_inspect.reporting.base import EvaluationSummary

logger = logging.getLogger(__name__)


class GateConfig(BaseModel):
    """Configuration for a quality gate."""

    min_pass_rate: float = 0.8
    min_average_score: float = 0.7
    max_cost_usd: float | None = None
    regression_threshold: float = -0.05  # Block if pass rate drops > 5%


class GateResult(BaseModel):
    """Result of a quality gate check."""

    passed: bool
    reason: str
    metrics: dict[str, Any] = Field(default_factory=dict)


class QualityGate:
    """Enforces thresholds on evaluation summaries."""

    @staticmethod
    def check(summary: EvaluationSummary, config: GateConfig) -> GateResult:
        """Check if a summary passes the defined quality gate."""
        if summary.pass_rate < config.min_pass_rate:
            return GateResult(
                passed=False,
                reason=(
                    f"Pass rate {(summary.pass_rate * 100):.1f}% is below threshold {(config.min_pass_rate * 100):.1f}%"
                ),
                metrics={"pass_rate": summary.pass_rate},
            )

        if summary.average_score < config.min_average_score:
            return GateResult(
                passed=False,
                reason=f"Average score {summary.average_score:.2f} is below threshold {config.min_average_score:.2f}",
                metrics={"average_score": summary.average_score},
            )

        if config.max_cost_usd and summary.metadata.total_cost_usd > config.max_cost_usd:
            return GateResult(
                passed=False,
                reason=f"Cost ${summary.metadata.total_cost_usd:.4f} exceeds limit ${config.max_cost_usd:.4f}",
                metrics={"cost": summary.metadata.total_cost_usd},
            )

        return GateResult(passed=True, reason="All thresholds met.", metrics={"pass_rate": summary.pass_rate})


class RegressionBlocker:
    """Blocks deployments based on performance drift compared to history."""

    @staticmethod
    def evaluate_drift(current: EvaluationSummary, history: list[EvaluationSummary], config: GateConfig) -> GateResult:
        """Evaluate performance drift against historical baseline."""
        if not history:
            return GateResult(passed=True, reason="No history available for comparison.")

        # Baseline is the latest run from main branch
        baseline = history[-1]
        pass_rate_delta = current.pass_rate - baseline.pass_rate

        if pass_rate_delta < config.regression_threshold:
            return GateResult(
                passed=False,
                reason=(
                    f"Regression detected! Pass rate dropped by {(abs(pass_rate_delta) * 100):.1f}% "
                    f"compared to baseline ({(baseline.pass_rate * 100):.1f}%)"
                ),
                metrics={"delta": pass_rate_delta, "baseline": baseline.pass_rate},
            )

        return GateResult(
            passed=True,
            reason=f"Performance stable. Drift: {(pass_rate_delta * 100):+.1f}%",
            metrics={"delta": pass_rate_delta},
        )
