"""RAG Evaluation Engine.

Comprehensive toolkit for testing Retrieval-Augmented Generation systems.
"""

from sarathi_agent_inspect.evaluators.rag.adversarial import RAGAdversary
from sarathi_agent_inspect.evaluators.rag.chunking import ChunkAnalyzer
from sarathi_agent_inspect.evaluators.rag.citations import CitationValidator
from sarathi_agent_inspect.evaluators.rag.embeddings import (
    EmbeddingBenchmark,
    EmbeddingModel,
    EmbeddingVersionTracker,
)
from sarathi_agent_inspect.evaluators.rag.pipelines import (
    GeneratorEvaluator,
    RAGEvaluationResult,
    RAGEvaluator,
    RAGRegressionPipeline,
    RetrieverEvaluator,
)
from sarathi_agent_inspect.evaluators.rag.tracing import (
    RAGTrace,
    RetrievalDebugger,
    RetrievalNode,
)

__all__ = [
    "ChunkAnalyzer",
    "CitationValidator",
    "EmbeddingBenchmark",
    "EmbeddingModel",
    "EmbeddingVersionTracker",
    "GeneratorEvaluator",
    "RAGAdversary",
    "RAGEvaluationResult",
    "RAGEvaluator",
    "RAGRegressionPipeline",
    "RAGTrace",
    "RetrievalDebugger",
    "RetrievalNode",
    "RetrieverEvaluator",
]
