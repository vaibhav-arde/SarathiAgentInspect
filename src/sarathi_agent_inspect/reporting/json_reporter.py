"""JSON and JSONL reporting.

Provides machine-readable output for integration with other tools
and for historical tracking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sarathi_agent_inspect.core.observability import ObservabilityContext
from sarathi_agent_inspect.core.sanitizer import InputSanitizer
from sarathi_agent_inspect.reporting.base import BaseReporter, EvaluationSummary


class JSONReporter(BaseReporter):
    """Generates machine-readable JSON reports."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)

    def generate(self, results: list[Any], summary: EvaluationSummary) -> Path:
        """Generate a JSON report file."""
        report_data = {
            "observability": ObservabilityContext(
                run_id=summary.metadata.run_id,
                trace_id=summary.metadata.trace_id,
            ).to_dict(),
            "summary": summary.model_dump(),
            "results": InputSanitizer.sanitize_for_export(results),
        }

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            json.dump(InputSanitizer.sanitize_for_export(report_data), f, indent=2, default=str)

        return self.output_path
