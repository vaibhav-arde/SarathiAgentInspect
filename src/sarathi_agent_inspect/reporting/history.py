"""Historical tracking and trend analysis.

Enables persistence of evaluation results over time to detect
regressions and track performance improvements.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from sarathi_agent_inspect.reporting.base import EvaluationSummary

logger = logging.getLogger(__name__)


class HistoricalTracker:
    """Manages the storage and retrieval of historical evaluation runs."""

    def __init__(self, history_file: str | Path = ".sarathi/history.jsonl") -> None:
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(parents=True, exist_ok=True)

    def record_run(self, summary: EvaluationSummary) -> None:
        """Append a run summary to the historical log."""
        try:
            with open(self.history_file, "a") as f:
                data = summary.model_dump()
                f.write(json.dumps(data, default=str) + "\n")
            logger.info(f"Recorded evaluation run {summary.metadata.run_id} to history.")
        except Exception as e:
            logger.error(f"Failed to record run to history: {e}")

    def load_history(self, limit: int = 50) -> list[EvaluationSummary]:
        """Load the most recent historical runs."""
        if not self.history_file.exists():
            return []

        runs: list[EvaluationSummary] = []
        try:
            with open(self.history_file) as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        runs.append(EvaluationSummary.model_validate(data))
            # Return last N runs
            return runs[-limit:]
        except Exception as e:
            logger.error(f"Failed to load history: {e}")
            return []


class TrendAnalyzer:
    """Calculates performance deltas and trends from historical data."""

    @staticmethod
    def calculate_trends(current: EvaluationSummary, history: list[EvaluationSummary]) -> dict[str, Any]:
        """Compare current run against historical average or baseline."""
        if not history:
            return {"status": "first_run", "pass_rate_delta": 0.0, "cost_delta": 0.0}

        # Use the most recent run as baseline for now
        baseline = history[-1]

        baseline_pass_rate = baseline.pass_rate
        baseline_cost = baseline.metadata.total_cost_usd

        pass_rate_delta = current.pass_rate - baseline_pass_rate
        cost_delta = current.metadata.total_cost_usd - baseline_cost

        return {
            "status": "regression" if pass_rate_delta < -0.05 else "stable",
            "pass_rate_delta": round(pass_rate_delta, 4),
            "cost_delta": round(cost_delta, 4),
            "trend_direction": "up" if pass_rate_delta > 0 else "down",
        }
