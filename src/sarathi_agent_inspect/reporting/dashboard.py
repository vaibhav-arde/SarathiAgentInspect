"""Historical dashboard generation.

Generates a static HTML dashboard summarizing trends across multiple
evaluation runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader


class DashboardGenerator:
    """Generates a historical trend dashboard."""

    def __init__(self, output_path: str | Path) -> None:
        self.output_path = Path(output_path)
        template_dir = Path(__file__).parent / "templates"
        self.env = Environment(
            loader=FileSystemLoader(template_dir),
            autoescape=True
        )

    def generate(self, history: list[Any]) -> Path:
        """Generate the dashboard from historical data."""
        template = self.env.get_template("dashboard_template.html")

        # history is a list of dicts from HistoricalTracker
        html_content = template.render(
            history_json=json.dumps(history, default=str),
        )

        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.output_path, "w") as f:
            f.write(html_content)

        return self.output_path
