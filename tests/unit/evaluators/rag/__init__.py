"""Tests for the RAG Evaluation Engine."""

from sarathi_agent_inspect.evaluators.rag import ChunkAnalyzer


def test_imports():
    """Ensure everything imports correctly."""
    assert ChunkAnalyzer is not None
