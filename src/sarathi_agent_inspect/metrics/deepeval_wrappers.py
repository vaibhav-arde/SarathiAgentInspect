"""DeepEval metric wrappers for the Sarathi framework.

Wraps built-in DeepEval metrics to fit within the Sarathi BaseMetric
architecture, allowing uniform execution, threshold management,
and composite scoring.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sarathi_agent_inspect.metrics.base import BaseMetric, MetricResult
from sarathi_agent_inspect.metrics.registry import MetricRegistry

try:
    from deepeval.metrics import (
        AnswerRelevancyMetric,
        ContextualPrecisionMetric,
        ContextualRecallMetric,
        FaithfulnessMetric,
        GEval,
        HallucinationMetric,
        ToxicityMetric,
    )
    from deepeval.test_case import LLMTestCase, SingleTurnParams
except ImportError:
    pass

if TYPE_CHECKING:
    from sarathi_agent_inspect.providers.base import BaseProvider


class DeepEvalWrapperBase(BaseMetric):
    """Base class for wrapping DeepEval metrics."""

    def __init__(self, provider: BaseProvider | None = None, threshold: float = 0.5) -> None:
        """Initialize the wrapper.

        Args:
            provider: Optional Sarathi provider. If provided, overrides DeepEval's default.
            threshold: Pass/fail threshold.
        """
        self.provider = provider
        self.threshold = threshold
        self._deepeval_model = None

        if self.provider:
            from sarathi_agent_inspect.metrics.deepeval_adapter import ProviderAdapter

            self._deepeval_model = ProviderAdapter(self.provider)

    async def _measure_deepeval(self, test_case: LLMTestCase, metric: Any) -> MetricResult:
        """Helper to run the DeepEval metric and convert the result."""
        try:
            await metric.a_measure(test_case)

            # DeepEval stores results on the metric object after measurement
            score = metric.score
            passed = metric.is_successful()
            reason = metric.reason

            # Override DeepEval's pass/fail with our strict threshold validation if needed
            # (DeepEval applies its own threshold, but we ensure consistency)
            our_passed = self.validate_threshold(score, self.threshold)

            return MetricResult(
                metric_name=self.metric_name,
                score=score,
                passed=our_passed,
                threshold=self.threshold,
                reason=reason,
                metadata={
                    "deepeval_passed": passed,
                },
            )
        except Exception as e:
            raise RuntimeError(f"DeepEval metric {self.metric_name} failed: {e}") from e


@MetricRegistry.register("geval")
class WrappedGEvalMetric(DeepEvalWrapperBase):
    """Wraps DeepEval's GEval metric for custom LLM-as-a-judge scoring."""

    def __init__(
        self,
        name: str,
        criteria: str,
        evaluation_params: list[SingleTurnParams],
        provider: BaseProvider | None = None,
        threshold: float = 0.5,
    ) -> None:
        super().__init__(provider, threshold)
        self._custom_name = name
        self.criteria = criteria
        self.evaluation_params = evaluation_params

        self._metric = GEval(
            name=name,
            criteria=criteria,
            evaluation_params=evaluation_params,
            threshold=threshold,
            model=self._deepeval_model,
        )

    @property
    def metric_name(self) -> str:
        return f"geval_{self._custom_name}"

    @property
    def description(self) -> str:
        return f"GEval: {self.criteria}"

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
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            context=context,
            retrieval_context=retrieval_context,
        )
        return await self._measure_deepeval(test_case, self._metric)


@MetricRegistry.register("faithfulness")
class WrappedFaithfulnessMetric(DeepEvalWrapperBase):
    """Wraps DeepEval's Faithfulness metric."""

    def __init__(self, provider: BaseProvider | None = None, threshold: float = 0.5) -> None:
        super().__init__(provider, threshold)
        self._metric = FaithfulnessMetric(
            threshold=threshold,
            model=self._deepeval_model,
        )

    @property
    def metric_name(self) -> str:
        return "faithfulness"

    @property
    def description(self) -> str:
        return "Measures if the output is factually consistent with the retrieval context."

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
        if not retrieval_context:
            raise ValueError("Faithfulness metric requires retrieval_context.")

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )
        return await self._measure_deepeval(test_case, self._metric)


@MetricRegistry.register("answer_relevancy")
class WrappedAnswerRelevancyMetric(DeepEvalWrapperBase):
    """Wraps DeepEval's Answer Relevancy metric."""

    def __init__(self, provider: BaseProvider | None = None, threshold: float = 0.5) -> None:
        super().__init__(provider, threshold)
        self._metric = AnswerRelevancyMetric(
            threshold=threshold,
            model=self._deepeval_model,
        )

    @property
    def metric_name(self) -> str:
        return "answer_relevancy"

    @property
    def description(self) -> str:
        return "Measures how relevant the output is to the input."

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
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            retrieval_context=retrieval_context,
        )
        return await self._measure_deepeval(test_case, self._metric)


@MetricRegistry.register("hallucination")
class WrappedHallucinationMetric(DeepEvalWrapperBase):
    """Wraps DeepEval's Hallucination metric."""

    def __init__(self, provider: BaseProvider | None = None, threshold: float = 0.5) -> None:
        super().__init__(provider, threshold)
        self._metric = HallucinationMetric(
            threshold=threshold,
            model=self._deepeval_model,
        )

    @property
    def metric_name(self) -> str:
        return "hallucination"

    @property
    def description(self) -> str:
        return "Measures whether the output contains hallucinated facts not in the context."

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
        if not context:
            raise ValueError("Hallucination metric requires context.")

        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            context=context,
        )
        return await self._measure_deepeval(test_case, self._metric)


@MetricRegistry.register("toxicity")
class WrappedToxicityMetric(DeepEvalWrapperBase):
    """Wraps DeepEval's Toxicity metric."""

    def __init__(self, provider: BaseProvider | None = None, threshold: float = 0.5) -> None:
        super().__init__(provider, threshold)
        self._metric = ToxicityMetric(
            threshold=threshold,
            model=self._deepeval_model,
        )

    @property
    def metric_name(self) -> str:
        return "toxicity"

    @property
    def description(self) -> str:
        return "Measures toxicity of the actual output."

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
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
        )
        return await self._measure_deepeval(test_case, self._metric)


@MetricRegistry.register("contextual_precision")
class WrappedContextualPrecisionMetric(DeepEvalWrapperBase):
    """Wraps DeepEval's Contextual Precision metric."""

    def __init__(self, provider: BaseProvider | None = None, threshold: float = 0.5) -> None:
        super().__init__(provider, threshold)
        self._metric = ContextualPrecisionMetric(
            threshold=threshold,
            model=self._deepeval_model,
        )

    @property
    def metric_name(self) -> str:
        return "contextual_precision"

    @property
    def description(self) -> str:
        return "Measures if relevant context is ranked high in retrieval_context."

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
        if not expected_output or not retrieval_context:
            raise ValueError("Contextual Precision requires expected_output and retrieval_context.")
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        return await self._measure_deepeval(test_case, self._metric)


@MetricRegistry.register("contextual_recall")
class WrappedContextualRecallMetric(DeepEvalWrapperBase):
    """Wraps DeepEval's Contextual Recall metric."""

    def __init__(self, provider: BaseProvider | None = None, threshold: float = 0.5) -> None:
        super().__init__(provider, threshold)
        self._metric = ContextualRecallMetric(
            threshold=threshold,
            model=self._deepeval_model,
        )

    @property
    def metric_name(self) -> str:
        return "contextual_recall"

    @property
    def description(self) -> str:
        return "Measures if all relevant information from expected_output is in retrieval_context."

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
        if not expected_output or not retrieval_context:
            raise ValueError("Contextual Recall requires expected_output and retrieval_context.")
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )
        return await self._measure_deepeval(test_case, self._metric)
