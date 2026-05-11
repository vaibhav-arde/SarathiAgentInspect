"""Composite and weighted metric implementations.

Allows combining multiple metrics into a single aggregated score.
Crucial for balancing LLM-as-a-judge (subjective) metrics with
deterministic (objective) metrics to mitigate bias.
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal

from sarathi_agent_inspect.metrics.base import BaseMetric, MetricResult
from sarathi_agent_inspect.metrics.registry import MetricRegistry


@MetricRegistry.register("composite")
class CompositeMetric(BaseMetric):
    """Combines multiple metrics into a single score.

    Supports different aggregation strategies (mean, min, max, weighted).
    """

    def __init__(
        self,
        name: str,
        metrics: list[BaseMetric],
        weights: list[float] | None = None,
        aggregation: Literal["mean", "min", "max", "weighted"] = "mean",
        threshold: float = 0.5,
    ) -> None:
        """Initialize a composite metric.

        Args:
            name: Custom name for this composite metric.
            metrics: List of metric instances to combine.
            weights: Optional list of weights (must match metrics length if 'weighted').
            aggregation: Strategy to combine scores.
            threshold: Pass/fail threshold for the final composite score.
        """
        self._name = name
        self.metrics = metrics
        self.aggregation = aggregation
        self.threshold = threshold

        if aggregation == "weighted":
            if not weights or len(weights) != len(metrics):
                raise ValueError("Weights must be provided and match the number of metrics for 'weighted' aggregation.")
            self.weights = weights
        else:
            self.weights = [1.0] * len(metrics)

    @property
    def metric_name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return f"Composite metric combining: {[m.metric_name for m in self.metrics]}"

    async def compute(
        self,
        *,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        context: list[str] | None = None,
        retrieval_context: list[str] | None = None,
        **kwargs: Any,
    ) -> MetricResult:
        """Execute all sub-metrics and aggregate their scores."""
        # Execute sub-metrics concurrently
        tasks = [
            m.compute(
                input_text=input_text,
                actual_output=actual_output,
                expected_output=expected_output,
                context=context,
                retrieval_context=retrieval_context,
                **kwargs,
            )
            for m in self.metrics
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results: list[MetricResult] = []
        errors: list[Exception] = []

        for r in results:
            if isinstance(r, Exception):
                errors.append(r)
            else:
                valid_results.append(r)

        if not valid_results:
            raise RuntimeError(f"All sub-metrics failed in composite metric '{self.metric_name}': {errors}")

        scores = [r.score for r in valid_results]

        # Apply aggregation strategy
        if self.aggregation == "mean":
            final_score = sum(scores) / len(scores)
        elif self.aggregation == "min":
            final_score = min(scores)
        elif self.aggregation == "max":
            final_score = max(scores)
        elif self.aggregation == "weighted":
            total_weight = sum(self.weights[: len(valid_results)])
            if total_weight == 0:
                final_score = 0.0
            else:
                weighted_sum = sum(s * w for s, w in zip(scores, self.weights[: len(valid_results)], strict=False))
                final_score = weighted_sum / total_weight
        else:
            final_score = 0.0

        passed = self.validate_threshold(final_score, self.threshold)

        return MetricResult(
            metric_name=self.metric_name,
            score=final_score,
            passed=passed,
            threshold=self.threshold,
            reason=f"Aggregated via {self.aggregation} strategy from {len(valid_results)} sub-metrics.",
            metadata={
                "sub_results": [r.to_dict() for r in valid_results],
                "errors": [str(e) for e in errors],
            },
        )
