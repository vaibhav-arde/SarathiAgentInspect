"""Unit tests for the RAG Evaluation Engine."""

from sarathi_agent_inspect.evaluators.rag.adversarial import RAGAdversary
from sarathi_agent_inspect.evaluators.rag.chunking import ChunkAnalyzer
from sarathi_agent_inspect.evaluators.rag.citations import CitationValidator
from sarathi_agent_inspect.evaluators.rag.embeddings import (
    EmbeddingBenchmark,
    EmbeddingVersionTracker,
)
from sarathi_agent_inspect.evaluators.rag.tracing import (
    RAGTrace,
    RetrievalDebugger,
    RetrievalNode,
)


def test_chunk_overlap_ratio() -> None:
    """Test token overlap calculation."""
    chunks = [
        "The quick brown fox jumps",
        "brown fox jumps over the",
        "over the lazy dog today",
    ]
    # words:
    # 1: the, quick, brown, fox, jumps
    # 2: brown, fox, jumps, over, the
    # 3: over, the, lazy, dog, today
    # overlap 1-2: brown, fox, jumps (3)
    # overlap 2-3: over, the (2)
    # total overlap words = 5
    # total words = 5 + 5 + 5 = 15
    # ratio = 5/15 = 0.333
    # ratio = 6/15 = 0.4
    ratio = ChunkAnalyzer().calculate_overlap_ratio(chunks)
    assert ratio == 0.4


def test_chunk_overlap_empty() -> None:
    assert ChunkAnalyzer().calculate_overlap_ratio([]) == 0.0
    assert ChunkAnalyzer().calculate_overlap_ratio(["hello world"]) == 0.0


def test_rank_context_distribution() -> None:
    """Test context ranking."""
    retrieved = [
        "Not relevant at all",
        "This is the golden snippet here",
        "Also irrelevant",
    ]
    expected = ["golden snippet"]
    indices = ChunkAnalyzer().rank_context_distribution(retrieved, expected)
    assert indices == [1]


def test_citation_validator_valid() -> None:
    """Test citation validation with valid citations."""
    validator = CitationValidator()
    text = "The sky is blue [1]. Water is wet [2]."
    context = ["Sky is blue", "Water is wet"]

    result = validator.validate_citations(text, context)
    assert result["total_citations"] == 2.0
    assert result["valid_citations"] == 2.0
    assert result["citation_accuracy"] == 1.0


def test_citation_validator_fabricated() -> None:
    """Test citation validation with fabricated citations."""
    validator = CitationValidator()
    text = "The sky is blue [1]. Earth is flat [99]."
    context = ["Sky is blue"]

    result = validator.validate_citations(text, context)
    assert result["total_citations"] == 2.0
    assert result["valid_citations"] == 1.0
    assert result["citation_accuracy"] == 0.5


def test_rag_tracer() -> None:
    """Test RAG trace and debugger."""
    trace = RAGTrace(trace_id="123", input_text="What is AI?")

    node1 = RetrievalNode(query="AI definition", documents_retrieved=5, latency_ms=150.0, cost_usd=0.01)
    node2 = RetrievalNode(query="AI history", documents_retrieved=2, latency_ms=350.0, cost_usd=0.02)

    trace.add_node(node1)
    trace.add_node(node2)
    trace.complete(gen_latency=1000.0, gen_cost=0.05)

    assert trace.total_cost_usd == 0.08

    bottlenecks = RetrievalDebugger.identify_bottlenecks(trace)
    assert bottlenecks["slowest_retrieval_query"] == "AI history"
    assert bottlenecks["slowest_retrieval_ms"] == 350.0
    assert bottlenecks["costliest_query"] == "AI history"


def test_missing_context_analysis() -> None:
    """Test missing context analysis."""
    retrieved = ["doc1", "doc2", "doc3"]
    golden = ["doc2", "doc4"]

    res = RetrievalDebugger.missing_context_analysis(retrieved, golden)
    assert res["missed_documents"] == 1
    assert res["extra_documents"] == 2
    assert res["recall_ratio"] == 0.5


def test_embedding_benchmark() -> None:
    """Test embedding drift detection."""

    class MockEmbeddingModel:
        def embed_text(self, text: str) -> list[float]:
            # mock returning identical vectors for easy 1.0 similarity
            return [0.1, 0.2, 0.3]

        def embed_batch(self, texts: list[str]) -> list[list[float]]:
            return [[0.1, 0.2, 0.3] for _ in texts]

    tracker = EmbeddingVersionTracker(
        model_name="mock-embed",
        version="1.0",
        dimension=3,
        deployment_date="2026-05-11",
        baseline_score=0.95,
    )

    benchmark = EmbeddingBenchmark(MockEmbeddingModel(), tracker)

    # Cosine similarity of [0.1, 0.2, 0.3] with itself is 1.0
    score = benchmark.evaluate_semantic_similarity([("text1", "text2")])
    assert score == 1.0

    # Drift check should pass since score (1.0) > baseline (0.95)
    assert tracker.check_drift(score, tolerance=0.05) is True


def test_rag_adversary_context_poisoning() -> None:
    """Test context poisoning."""
    from unittest.mock import MagicMock

    mock_provider = MagicMock()
    adversary = RAGAdversary(provider=mock_provider, seed=42)
    context = ["Fact 1", "Fact 2"]
    poison = ["Fake Fact 1", "Fake Fact 2"]

    poisoned_start = adversary.inject_poisoned_context(context, poison, position="start")
    assert len(poisoned_start) == 3
    assert poisoned_start[0] in poison
    assert poisoned_start[1] == "Fact 1"

    poisoned_end = adversary.inject_poisoned_context(context, poison, position="end")
    assert len(poisoned_end) == 3
    assert poisoned_end[-1] in poison
