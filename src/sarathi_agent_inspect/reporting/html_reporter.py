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
        from sarathi_agent_inspect.core.sanitizer import InputSanitizer

        template = self.env.get_template("report_template.html")

        # Ensure results have defaults and are sanitized
        normalized_results = []
        for r in results:
            if isinstance(r, dict):
                res_dict = r
            elif hasattr(r, "to_dict"):
                res_dict = r.to_dict()
            else:
                res_dict = vars(r)

            # Sanitize sensitive fields if present
            for field in ["input_text", "actual_output", "input", "output"]:
                if field in res_dict and isinstance(res_dict[field], str):
                    res_dict[field] = InputSanitizer.sanitize(res_dict[field])

            normalized_results.append(res_dict)

        html_content = template.render(
            results=normalized_results,
            summary=summary,
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            f.write(html_content)

        return self.output_path
