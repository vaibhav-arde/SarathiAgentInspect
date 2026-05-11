"""Enterprise Observability and Session Management.

Provides high-level session tracking and common tracing base classes
for consolidated reporting and cost analysis.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class BaseTrace:
    """Common logic for RAG and Agent traces."""
    trace_id: str
    input_text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(UTC))
    end_time: datetime | None = None
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0

    def complete(self) -> None:
        """Mark the trace as finished."""
        self.end_time = datetime.now(UTC)
        if self.start_time:
            self.total_latency_ms = (self.end_time - self.start_time).total_seconds() * 1000.0

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for persistence."""
        return asdict(self)


class EvaluationSession:
    """Tracks a complete batch evaluation session.
    
    Collects results, costs, and environmental metadata for a full run
    and exports a manifest for CI/CD integration.
    """

    def __init__(self, session_id: str, environment: str = "local") -> None:
        self.session_id = session_id
        self.environment = environment
        self.start_time = datetime.now(UTC)
        self.results: list[dict[str, Any]] = []
        self.total_cost_usd: float = 0.0
        self.metadata: dict[str, Any] = {
            "environment": environment,
            "git_commit": "unknown", # Can be injected
            "user": "vaibhav-arde",
        }

    def record_result(self, result: dict[str, Any], cost: float = 0.0) -> None:
        """Add a result to the session."""
        self.results.append(result)
        self.total_cost_usd += cost

    def export_manifest(self, output_dir: Path | str) -> Path:
        """Save the session manifest as JSON."""
        path = Path(output_dir) / f"session_{self.session_id}.json"
        manifest = {
            "session_id": self.session_id,
            "metadata": self.metadata,
            "summary": {
                "total_records": len(self.results),
                "passed": sum(1 for r in self.results if r.get("passed", False)),
                "total_cost_usd": self.total_cost_usd,
                "duration_seconds": (datetime.now(UTC) - self.start_time).total_seconds(),
            },
            "results": self.results,
        }
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        return path
