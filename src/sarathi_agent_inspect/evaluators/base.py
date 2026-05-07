"""Abstract base evaluator interface.

Defines the contract for evaluation pipelines that combine
metrics, test cases, and judge models to produce evaluation results.

An evaluator:
    1. Accepts test cases (input + expected output + actual output)
    2. Runs one or more metrics against each test case
    3. Aggregates results with pass/fail determination
    4. Returns structured EvaluationResult objects
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sarathi_agent_inspect.core.types import JsonDict, Metadata, Score, Threshold


@dataclass
class EvaluationResult:
    """Result of evaluating a single test case.

    Attributes:
        test_case_id: Unique identifier for the test case.
        passed: Whether the evaluation passed all thresholds.
        overall_score: Aggregated score across all metrics.
        metric_scores: Individual metric name → score mapping.
        threshold: The threshold used for pass/fail determination.
        metadata: Additional context (model, provider, timestamps).
        errors: Any errors encountered during evaluation.
        timestamp: When the evaluation completed.
    """

    test_case_id: str
    passed: bool
    overall_score: Score
    metric_scores: dict[str, Score] = field(default_factory=dict)
    threshold: Threshold = 0.7
    metadata: Metadata = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    raw_results: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "test_case_id": self.test_case_id,
            "passed": self.passed,
            "overall_score": self.overall_score,
            "metric_scores": self.metric_scores,
            "threshold": self.threshold,
            "metadata": self.metadata,
            "errors": self.errors,
            "timestamp": self.timestamp.isoformat(),
        }


class BaseEvaluator(ABC):
    """Abstract base class for evaluation pipelines.

    Subclasses implement domain-specific evaluation logic
    (e.g., RAG evaluation, agent evaluation, chatbot evaluation).
    """

    @property
    @abstractmethod
    def evaluator_name(self) -> str:
        """Return the evaluator identifier."""
        ...

    @abstractmethod
    async def evaluate(
        self,
        *,
        input_text: str,
        actual_output: str,
        expected_output: str | None = None,
        context: list[str] | None = None,
        retrieval_context: list[str] | None = None,
        **kwargs: Any,
    ) -> EvaluationResult:
        """Evaluate a single test case.

        Args:
            input_text: The input/query text.
            actual_output: The actual LLM/system output.
            expected_output: The expected/golden output, if available.
            context: Ground truth context documents.
            retrieval_context: Retrieved context documents.
            **kwargs: Additional evaluator-specific parameters.

        Returns:
            EvaluationResult with scores and pass/fail status.

        Raises:
            EvaluationError: If the evaluation fails.
        """
        ...

    async def evaluate_batch(
        self,
        test_cases: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[EvaluationResult]:
        """Evaluate a batch of test cases.

        Default implementation evaluates sequentially.
        Override for parallel or optimized batch evaluation.

        Args:
            test_cases: List of test case dictionaries.
            **kwargs: Additional evaluator-specific parameters.

        Returns:
            List of EvaluationResult objects.
        """
        results: list[EvaluationResult] = []
        for test_case in test_cases:
            result = await self.evaluate(**test_case, **kwargs)
            results.append(result)
        return results
