"""Allure reporting adapter.

Enriches Allure reports with LLM-specific observability data,
including traces, token usage, and judge reasoning.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from sarathi_agent_inspect.core.sanitizer import InputSanitizer

try:
    import allure
except ImportError:
    allure = None  # type: ignore

logger = logging.getLogger(__name__)


class AllureAdapter:
    """Helper to attach Sarathi data to Allure reports."""

    @staticmethod
    def attach_trace(trace: Any) -> None:
        """Attach an agent or RAG trace to the current Allure test."""
        if not allure:
            return

        try:
            name = getattr(trace, "trace_id", "Evaluation Trace")
            if isinstance(trace, dict):
                payload = trace
            elif hasattr(trace, "to_dict"):
                payload = trace.to_dict()
            else:
                payload = vars(trace)
            content = json.dumps(InputSanitizer.sanitize_for_export(payload), indent=2, default=str)
            allure.attach(
                content,
                name=f"Sarathi Trace: {name}",
                attachment_type=allure.attachment_type.JSON,
            )
        except Exception as e:
            logger.warning(f"Failed to attach trace to Allure: {e}")

    @staticmethod
    def attach_metrics(metrics: list[Any]) -> None:
        """Attach evaluation metrics and reasoning to Allure."""
        if not allure:
            return

        for metric in metrics:
            with allure.step(f"Metric: {metric.metric_name} (Score: {metric.score})"):
                allure.attach(
                    InputSanitizer.sanitize(metric.reason),
                    name="Judge Reasoning",
                    attachment_type=allure.attachment_type.TEXT,
                )
                if hasattr(metric, "metadata") and metric.metadata:
                    allure.attach(
                        json.dumps(InputSanitizer.sanitize_for_export(metric.metadata), indent=2, default=str),
                        name="Metric Metadata",
                        attachment_type=allure.attachment_type.JSON,
                    )

    @staticmethod
    def log_cost(cost: float, tokens: int) -> None:
        """Log cost and token usage as Allure tags/parameters."""
        if not allure:
            return

        allure.dynamic.parameter("Cost USD", f"${cost:.5f}")
        allure.dynamic.parameter("Tokens", str(tokens))
