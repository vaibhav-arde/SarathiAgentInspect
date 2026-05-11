"""Unit tests for dataset pipeline."""

import pytest

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


# ── Batch Tests ─────────────────────────────────────────────────────


def test_pipeline_batch_exact_division():
    """Test batching where records divide evenly."""
    data = [{"id": i} for i in range(6)]

    batches = list(DatasetPipeline(data).batch(batch_size=3))
    assert len(batches) == 2
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3


def test_pipeline_batch_remainder():
    """Test batching with a partial final batch."""
    data = [{"id": i} for i in range(7)]

    batches = list(DatasetPipeline(data).batch(batch_size=3))
    assert len(batches) == 3
    assert len(batches[0]) == 3
    assert len(batches[1]) == 3
    assert len(batches[2]) == 1  # Remainder


def test_pipeline_batch_single_item():
    """Test batching with batch_size=1."""
    data = [{"id": 0}, {"id": 1}]

    batches = list(DatasetPipeline(data).batch(batch_size=1))
    assert len(batches) == 2
    assert batches[0] == [{"id": 0}]


def test_pipeline_batch_larger_than_dataset():
    """Test batching where batch_size exceeds dataset size."""
    data = [{"id": 0}, {"id": 1}]

    batches = list(DatasetPipeline(data).batch(batch_size=100))
    assert len(batches) == 1
    assert len(batches[0]) == 2


def test_pipeline_batch_invalid_size():
    """Test batching with invalid batch_size raises ValueError."""
    data = [{"id": 0}]

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(DatasetPipeline(data).batch(batch_size=0))

    with pytest.raises(ValueError, match="batch_size must be positive"):
        list(DatasetPipeline(data).batch(batch_size=-1))


def test_pipeline_batch_empty_source():
    """Test batching an empty dataset yields nothing."""
    batches = list(DatasetPipeline([]).batch(batch_size=5))
    assert batches == []


def test_pipeline_batch_with_filter():
    """Test batching after filtering (pipeline chaining)."""
    data = [{"id": i, "keep": i % 2 == 0} for i in range(10)]

    batches = list(DatasetPipeline(data).filter(lambda r: r["keep"]).batch(batch_size=2))
    assert len(batches) == 3  # 5 even numbers / 2 = 2 full + 1 partial
    assert len(batches[0]) == 2
    assert len(batches[2]) == 1


# ── Sample Tests ────────────────────────────────────────────────────


def test_pipeline_sample_subset():
    """Test sampling a subset of records."""
    data = [{"id": i} for i in range(100)]

    result = DatasetPipeline(data).sample(10, seed=42)
    assert len(result) == 10


def test_pipeline_sample_reproducible():
    """Test that sampling with the same seed is reproducible."""
    data = [{"id": i} for i in range(50)]

    result1 = DatasetPipeline(data).sample(5, seed=123)
    result2 = DatasetPipeline(data).sample(5, seed=123)
    assert result1 == result2


def test_pipeline_sample_exceeds_total():
    """Test sampling more than available returns all records."""
    data = [{"id": 0}, {"id": 1}]

    result = DatasetPipeline(data).sample(100)
    assert len(result) == 2


def test_pipeline_sample_different_seeds():
    """Test that different seeds produce different samples."""
    data = [{"id": i} for i in range(100)]

    result1 = DatasetPipeline(data).sample(10, seed=1)
    result2 = DatasetPipeline(data).sample(10, seed=2)
    assert result1 != result2


# ── Head Tests ──────────────────────────────────────────────────────


def test_pipeline_head():
    """Test head returns first N records."""
    data = [{"id": i} for i in range(20)]

    result = DatasetPipeline(data).head(5)
    assert len(result) == 5
    assert result[0]["id"] == 0
    assert result[4]["id"] == 4


def test_pipeline_head_exceeds_total():
    """Test head with N larger than dataset."""
    data = [{"id": 0}]

    result = DatasetPipeline(data).head(100)
    assert len(result) == 1


def test_pipeline_head_default():
    """Test head with default N=10."""
    data = [{"id": i} for i in range(20)]

    result = DatasetPipeline(data).head()
    assert len(result) == 10
