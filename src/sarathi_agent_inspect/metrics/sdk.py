"""Custom Metric SDK.

Provides clean abstractions and decorators for users to easily
build, register, and integrate their own custom metrics (trace-based,
agent, RAG, safety, etc.) into the framework.
"""

from __future__ import annotations

import typing
from typing import Any

from sarathi_agent_inspect.metrics.base import BaseMetric, MetricResult
from sarathi_agent_inspect.metrics.registry import MetricRegistry

if typing.TYPE_CHECKING:
    from collections.abc import Callable


def metric(name: str | None = None) -> Callable[[type[BaseMetric]], type[BaseMetric]]:
    """Decorator to register a custom metric class.

    Example:
        @metric("my_custom_metric")
        class MyMetric(BaseMetric):
            ...
    """
    return MetricRegistry.register(name)


class SimpleAsyncMetric(BaseMetric):
    """Helper class to quickly create metrics from an async function.

    Example:
        async def evaluate_length(input_text: str, actual_output: str, **kwargs) -> float:
            return min(len(actual_output) / 100, 1.0)

        my_metric = SimpleAsyncMetric("length_check", evaluate_length, description="Checks string length")
    """

    def __init__(
        self,
        name: str,
        eval_fn: Callable[..., Any],  # Should return an awaitable float
        description: str = "Custom async metric",
        score_range: tuple[float, float] = (0.0, 1.0),
    ) -> None:
        self._name = name
        self._eval_fn = eval_fn
        self._description = description
        self._score_range = score_range

    @property
    def metric_name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def score_range(self) -> tuple[float, float]:
        return self._score_range

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
        try:
            score = await self._eval_fn(
                input_text=input_text,
                actual_output=actual_output,
                expected_output=expected_output,
                context=context,
                retrieval_context=retrieval_context,
                **kwargs,
            )
            # Default threshold handling relies on the caller or MetricConfig
            # Here we just pass it blindly as successful since we don't have the config
            # But the MetricExecutor will override `passed` based on the config.
            return MetricResult(
                metric_name=self.metric_name,
                score=float(score),
                passed=True,
                threshold=0.0,
                reason="Computed via custom function",
            )
        except Exception as e:
            raise RuntimeError(f"Custom metric '{self.metric_name}' failed: {e}") from e
