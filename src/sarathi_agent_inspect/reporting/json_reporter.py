"""JSON and JSONL reporting.

Provides machine-readable output for integration with other tools
and for historical tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sarathi_agent_inspect.reporting.base import BaseReporter, EvaluationSummary


class JSONReporter(BaseReporter):
    """Generates machine-readable JSON reports."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def generate(self, results: list[Any], summary: EvaluationSummary) -> Path:
        """Generate a JSON report file."""
        report_data = {
            "summary": {
                "total_records": summary.total_records,
                "passed_count": summary.passed_count,
                "failed_count": summary.failed_count,
                "pass_rate": summary.pass_rate,
                "average_score": summary.average_score,
                "metadata": {
                    "run_id": summary.metadata.run_id,
                    "timestamp": summary.metadata.timestamp,
                    "environment": summary.metadata.environment,
                    "total_cost_usd": summary.metadata.total_cost_usd,
                    "total_latency_ms": summary.metadata.total_latency_ms,
                },
            },
            "results": results,
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(report_data, f, indent=2, default=str)

        return self.output_path
