"""Embedding benchmark and versioning utilities.

Provides utilities to baseline and track embedding models across versions
to prevent 'Context Drift'.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from sarathi_agent_inspect.providers.base import BaseProvider

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingVersionTracker:
    """Tracks versions of embedding models to prevent silent regressions."""

    model_name: str
    version: str
    dimension: int
    deployment_date: str
    baseline_score: float | None = None

    def check_drift(self, current_score: float, tolerance: float = 0.05) -> bool:
        """Check if the current score has drifted significantly from the baseline."""
        if self.baseline_score is None:
            return True
        drift = self.baseline_score - current_score
        if drift > tolerance:
            logger.warning(
                f"Context Drift Detected! Model {self.model_name} (v{self.version}) "
                f"drifted by {drift:.2f} (tolerance {tolerance:.2f})."
            )
            return False
        return True


class EmbeddingModel(Protocol):
    """Abstract interface for pluggable embedding models."""

    def embed_text(self, text: str) -> list[float]:
        """Convert a single string into a vector embedding."""
        ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Convert a list of strings into a list of vector embeddings."""
        ...


class OllamaEmbeddingModel:
    """Reference implementation using Ollama for embeddings."""

    def __init__(self, provider: BaseProvider, model: str | None = None) -> None:
        self.provider = provider
        self.model = model or provider.model_name

    def embed_text(self, text: str) -> list[float]:
        # This would call the provider's embedding API
        # For now, we stub this as the provider interface primarily handles text generation.
        # In a real enterprise setup, BaseProvider would have an embed() method.
        logger.info(f"Embedding text using Ollama model: {self.model}")
        return [0.1] * 128  # Stubbed vector

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]


class EmbeddingBenchmark:
    """Benchmarks an embedding model against standard or custom datasets."""

    def __init__(self, model: EmbeddingModel, tracker: EmbeddingVersionTracker) -> None:
        self.model = model
        self.tracker = tracker

    def cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Basic cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        magnitude1 = sum(a * a for a in vec1) ** 0.5
        magnitude2 = sum(b * b for b in vec2) ** 0.5

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def evaluate_semantic_similarity(self, pairs: list[tuple[str, str]]) -> float:
        """Evaluate how well the model scores semantic similarity pairs.

        A simple heuristic benchmark that embeds pairs of texts that are
        known to be similar, and averages their cosine similarity.
        """
        if not pairs:
            return 0.0

        total_sim = 0.0
        for text1, text2 in pairs:
            v1 = self.model.embed_text(text1)
            v2 = self.model.embed_text(text2)
            total_sim += self.cosine_similarity(v1, v2)

        avg_score = total_sim / len(pairs)
        self.tracker.check_drift(avg_score)

        return avg_score
