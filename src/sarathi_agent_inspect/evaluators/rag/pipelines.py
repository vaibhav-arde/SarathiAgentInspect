"""RAG Evaluation Pipelines.

Provides isolated evaluation pipelines for retrievers and generators,
as well as an end-to-end RAG evaluator.

Enterprise Challenge Addressed:
Retriever-Generator Separation. If a RAG system gives a wrong answer,
these pipelines isolate whether the retriever failed to find the document
(Contextual Recall/Precision) or the generator hallucinated/ignored context.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sarathi_agent_inspect.metrics.config import MetricConfig
from sarathi_agent_inspect.metrics.deepeval_wrappers import (
    WrappedContextualPrecisionMetric,
    WrappedContextualRecallMetric,
    WrappedFaithfulnessMetric,
    WrappedHallucinationMetric,
)
from sarathi_agent_inspect.metrics.execution import MetricExecutor

if TYPE_CHECKING:
    from sarathi_agent_inspect.datasets.regression import RegressionComparator
    from sarathi_agent_inspect.providers.base import BaseProvider


@dataclass
class RAGEvaluationResult:
    """Consolidated result of a RAG evaluation."""
    retriever_score: float
    generator_score: float
    overall_score: float
    passed: bool
    details: dict[str, Any]


class RetrieverEvaluator:
    """Evaluates the retrieval component in isolation."""

    def __init__(self, provider: BaseProvider | None = None, threshold: float = 0.5) -> None:
        self.precision = WrappedContextualPrecisionMetric(provider, threshold)
        self.recall = WrappedContextualRecallMetric(provider, threshold)
        self.executor = MetricExecutor()
        self.config = MetricConfig(threshold=threshold)

    async def evaluate(
        self, input_text: str, expected_output: str, retrieval_context: list[str]
    ) -> RAGEvaluationResult:
        """Evaluate contextual precision and recall."""
        # Using a batch execution
        results = await self.executor.execute_batch(
            [(self.precision, self.config), (self.recall, self.config)],
            input_text=input_text,
            actual_output="",  # Not needed for retriever eval
            expected_output=expected_output,
            retrieval_context=retrieval_context,
        )

        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return RAGEvaluationResult(
            retriever_score=avg_score,
            generator_score=0.0,
            overall_score=avg_score,
            passed=avg_score >= self.config.threshold,
            details={"precision": results[0].to_dict(), "recall": results[1].to_dict()},
        )


class GeneratorEvaluator:
    """Evaluates the generation component in isolation (given context)."""

    def __init__(self, provider: BaseProvider | None = None, threshold: float = 0.5) -> None:
        self.faithfulness = WrappedFaithfulnessMetric(provider, threshold)
        self.hallucination = WrappedHallucinationMetric(provider, threshold)
        self.executor = MetricExecutor()
        self.config = MetricConfig(threshold=threshold)

    async def evaluate(self, input_text: str, actual_output: str, context: list[str]) -> RAGEvaluationResult:
        """Evaluate faithfulness and hallucination."""
        # Note: Hallucination uses 'context', Faithfulness uses 'retrieval_context'.
        # We pass the same context to both for the generator evaluation.
        results = await self.executor.execute_batch(
            [(self.faithfulness, self.config), (self.hallucination, self.config)],
            input_text=input_text,
            actual_output=actual_output,
            context=context,
            retrieval_context=context,
        )

        scores = [r.score for r in results]
        avg_score = sum(scores) / len(scores) if scores else 0.0

        return RAGEvaluationResult(
            retriever_score=0.0,
            generator_score=avg_score,
            overall_score=avg_score,
            passed=avg_score >= self.config.threshold,
            details={"faithfulness": results[0].to_dict(), "hallucination": results[1].to_dict()},
        )


class RAGEvaluator:
    """End-to-End RAG evaluation pipeline."""

    def __init__(self, provider: BaseProvider | None = None, threshold: float = 0.5) -> None:
        self.retriever_eval = RetrieverEvaluator(provider, threshold)
        self.generator_eval = GeneratorEvaluator(provider, threshold)
        self.threshold = threshold

    async def evaluate(
        self,
        input_text: str,
        actual_output: str,
        expected_output: str,
        retrieval_context: list[str],
    ) -> RAGEvaluationResult:
        """Evaluate both retriever and generator."""
        ret_res = await self.retriever_eval.evaluate(input_text, expected_output, retrieval_context)
        gen_res = await self.generator_eval.evaluate(input_text, actual_output, retrieval_context)

        overall = (ret_res.retriever_score + gen_res.generator_score) / 2.0

        return RAGEvaluationResult(
            retriever_score=ret_res.retriever_score,
            generator_score=gen_res.generator_score,
            overall_score=overall,
            passed=overall >= self.threshold,
            details={
                "retriever": ret_res.details,
                "generator": gen_res.details,
            },
        )


class RAGRegressionPipeline:
    """Detects quality drift in RAG pipelines across version updates."""

    def __init__(self, comparator: RegressionComparator) -> None:
        self.comparator = comparator

    def detect_drift(self, baseline_name: str, new_scores: dict[str, float]) -> bool:
        """Compare new RAG scores against a baseline regression snapshot.

        Args:
            baseline_name: The name of the captured baseline.
            new_scores: Dict mapping test IDs to their overall RAG scores.

        Returns:
            True if drift is acceptable, False if it failed the regression check.
        """
        report = self.comparator.compare(baseline_name, new_scores)
        return report.passed
