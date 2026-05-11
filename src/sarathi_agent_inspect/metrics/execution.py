"""Metric execution engine.

Handles async execution, parallel evaluation of multiple metrics,
batch processing, retry strategies for flaky LLM calls, and
scoring normalization.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from sarathi_agent_inspect.metrics.base import MetricResult

if TYPE_CHECKING:
    from sarathi_agent_inspect.metrics.base import BaseMetric
    from sarathi_agent_inspect.metrics.config import MetricConfig

logger = logging.getLogger(__name__)


class MetricExecutor:
    """Executes metrics with concurrency, retry, and normalization."""

    def __init__(
        self,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        concurrency: int = 10,
    ) -> None:
        """Initialize the metric executor.

        Args:
            max_retries: Number of times to retry a failed metric.
            retry_delay: Base delay between retries (exponential backoff applied).
            concurrency: Max concurrent metric executions.
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.semaphore = asyncio.Semaphore(concurrency)

    async def execute_metric(
        self,
        metric: BaseMetric,
        config: MetricConfig,
        *,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        context: list[str] | None = None,
        retrieval_context: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Execute a single metric with retries and normalization."""
        async with self.semaphore:
            attempt = 0
            while attempt <= self.max_retries:
                try:
                    result = await metric.compute(
                        input_text=input_text,
                        actual_output=actual_output,
                        expected_output=expected_output,
                        context=context,
                        retrieval_context=retrieval_context,
                        **kwargs,
                        **config.extra_params,
                    )

                    # Normalize score
                    normalized_score = self._normalize_score(
                        result.score,
                        metric.score_range,
                        config.normalization_strategy,
                    )

                    # Validate threshold
                    passed = config.is_passing(normalized_score)

                    return MetricResult(
                        metric_name=metric.metric_name,
                        score=normalized_score,
                        passed=passed,
                        threshold=config.threshold,
                        reason=result.reason,
                        metadata=result.metadata,
                    )
                except Exception as e:
                    attempt += 1
                    if attempt > self.max_retries:
                        logger.error(f"Metric '{metric.metric_name}' failed after {self.max_retries} retries: {e}")
                        # Return a failed result rather than crashing the pipeline
                        return MetricResult(
                            metric_name=metric.metric_name,
                            score=0.0,
                            passed=False,
                            threshold=config.threshold,
                            reason=f"Execution failed: {e}",
                            metadata={"error": str(e)},
                        )

                    # Exponential backoff
                    delay = self.retry_delay * (2 ** (attempt - 1))
                    logger.warning(f"Metric '{metric.metric_name}' failed (attempt {attempt}). Retrying in {delay}s...")
                    await asyncio.sleep(delay)

            # Should never reach here due to the return in the loop
            raise RuntimeError("Unreachable")

    async def execute_batch(
        self,
        metrics: list[tuple[BaseMetric, MetricConfig]],
        *,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        context: list[str] | None = None,
        retrieval_context: list[str] | None = None,
        **kwargs: Any,
    ) -> list[MetricResult]:
        """Execute multiple metrics in parallel for a single test case."""
        tasks = [
            self.execute_metric(
                metric,
                config,
                input_text=input_text,
                actual_output=actual_output,
                expected_output=expected_output,
                context=context,
                retrieval_context=retrieval_context,
                **kwargs,
            )
            for metric, config in metrics
        ]
        return await asyncio.gather(*tasks)

    def _normalize_score(
        self,
        raw_score: float,
        score_range: tuple[float, float],
        strategy: str,
    ) -> float:
        """Normalize a score to the 0.0-1.0 range."""
        if strategy == "none":
            return raw_score

        min_val, max_val = score_range

        if strategy == "min_max":
            if max_val == min_val:
                return 0.0
            # Clamp the raw score to the expected range first
            clamped = max(min_val, min(max_val, raw_score))
            return (clamped - min_val) / (max_val - min_val)

        # Other strategies (e.g. z_score) could be implemented here
        return raw_score
