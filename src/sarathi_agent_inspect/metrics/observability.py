"""Metric observability and score persistence.

Provides tracing capabilities for metric execution and handles
persisting scores and reports to form an enterprise audit trail.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sarathi_agent_inspect.core.observability import ObservabilityContext, merge_observability_metadata
from sarathi_agent_inspect.core.sanitizer import InputSanitizer

if TYPE_CHECKING:
    from sarathi_agent_inspect.metrics.base import MetricResult

logger = logging.getLogger(__name__)


class MetricObserver:
    """Handles tracing and persistence of metric evaluations."""

    def __init__(self, storage_dir: str | Path = ".sarathi/metrics") -> None:
        """Initialize the observer.

        Args:
            storage_dir: Directory to persist metric results.
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def trace_execution(
        self,
        test_id: str,
        results: list[MetricResult],
        *,
        run_id: str | None = None,
        trace_id: str | None = None,
    ) -> None:
        """Log trace information for a metric execution.

        In a full enterprise deployment, this could send data to
        LangSmith, DataDog, or an internal telemetry service.
        """
        passed_count = sum(1 for r in results if r.passed)
        context = ObservabilityContext(
            run_id=run_id or "unknown_run",
            test_id=test_id,
            trace_id=trace_id,
        ).to_dict()
        logger.info(
            f"Run '{context['run_id']}' / Test '{test_id}': {passed_count}/{len(results)} metrics passed. "
            f"Scores: {[f'{r.metric_name}={r.score:.2f}' for r in results]}"
        )
        for r in results:
            if not r.passed:
                logger.warning(f"Metric '{r.metric_name}' failed for '{test_id}': {r.reason}")

    def persist_results(
        self,
        run_id: str,
        test_id: str,
        results: list[MetricResult],
        metadata: dict[str, Any] | None = None,
        trace_id: str | None = None,
    ) -> str:
        """Persist metric results to local storage for auditing.

        Args:
            run_id: Identifier for the entire evaluation run.
            test_id: Identifier for the specific test case.
            results: List of computed metric results.
            metadata: Optional additional context.

        Returns:
            The path to the persisted file.
        """
        run_dir = self.storage_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "test_id": test_id,
            "run_id": run_id,
            "timestamp": time.time(),
            "metadata": merge_observability_metadata(
                metadata,
                run_id=run_id,
                test_id=test_id,
                trace_id=trace_id,
            ),
            "observability": ObservabilityContext(
                run_id=run_id,
                test_id=test_id,
                trace_id=trace_id,
            ).to_dict(),
            "metrics": [r.to_dict() if hasattr(r, "to_dict") else asdict(r) for r in results],
        }

        # Save as JSON
        file_path = run_dir / f"{test_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(InputSanitizer.sanitize_for_export(payload), f, indent=2)

        logger.debug(f"Persisted metric results to {file_path}")
        return str(file_path)
