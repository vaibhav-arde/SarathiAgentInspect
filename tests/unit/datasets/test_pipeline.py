"""Unit tests for dataset pipeline."""

from sarathi_agent_inspect.datasets.pipeline import DatasetPipeline


def test_pipeline_filter():
    """Test pipeline filtering."""
    data = [
        {"input": "a", "score": 0.5},
        {"input": "b", "score": 0.9},
        {"input": "c", "score": 0.2},
    ]

    pipeline = DatasetPipeline(data).filter(lambda r: r["score"] > 0.5)
    results = pipeline.to_list()

    assert len(results) == 1
    assert results[0]["input"] == "b"

def test_pipeline_map():
    """Test pipeline mapping."""
    data = [{"input": "hello"}]

    pipeline = DatasetPipeline(data).map(lambda r: {**r, "input": r["input"].upper()})
    results = pipeline.to_list()

    assert results[0]["input"] == "HELLO"

def test_pipeline_deduplicate():
    """Test pipeline deduplication."""
    data = [
        {"input": "dup"},
        {"input": "unique"},
        {"input": "dup"},
    ]

    pipeline = DatasetPipeline(data).deduplicate()
    results = pipeline.to_list()

    assert len(results) == 2
    assert results[0]["input"] == "dup"
    assert results[1]["input"] == "unique"

def test_pipeline_tagging():
    """Test pipeline tagging."""
    data = [{"input": "test"}]

    pipeline = DatasetPipeline(data).tag(["v1", "regression"])
    results = pipeline.to_list()

    assert "v1" in results[0]["metadata"]["tags"]
    assert "regression" in results[0]["metadata"]["tags"]
