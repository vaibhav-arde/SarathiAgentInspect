"""HTML reporting implementation.

Generates beautiful, interactive reports with visualizations
using Jinja2 and Chart.js.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader

from sarathi_agent_inspect.reporting.base import BaseReporter, EvaluationSummary


class HTMLReporter(BaseReporter):
    """Generates interactive HTML reports."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        # Assuming template is in the same package
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )

    def generate(self, results: list[Any], summary: EvaluationSummary) -> Path:
        """Generate an HTML report file."""
        template = self.env.get_template("report_template.html")

        # Ensure results have defaults for missing keys to prevent template errors
        normalized_results = []
        for r in results:
            if isinstance(r, dict):
                normalized_results.append(r)
            elif hasattr(r, "to_dict"):
                normalized_results.append(r.to_dict())
            else:
                # Fallback for arbitrary objects
                normalized_results.append(vars(r))

        html_content = template.render(
            results=normalized_results,
            summary=summary,
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            f.write(html_content)

        return self.output_path
