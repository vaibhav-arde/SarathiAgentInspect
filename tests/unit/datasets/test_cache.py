"""Unit tests for dataset caching strategy."""

import json
import time
from typing import Any

import pytest

from sarathi_agent_inspect.datasets.cache import DatasetCache


@pytest.fixture
def sample_json_file(tmp_path: Any) -> Any:
    """Create a sample JSON file for cache testing."""
    data = [{"input": "hello"}, {"input": "world"}]
    path = tmp_path / "cache_test.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return str(path)


@pytest.fixture
def sample_records() -> Any:
    """Sample dataset records."""
    return [{"input": "hello"}, {"input": "world"}]


def test_cache_put_and_get(sample_json_file: Any, sample_records: Any) -> None:
    """Test basic cache put and get."""
    cache = DatasetCache()
    cache.put(sample_json_file, sample_records)

    result = cache.get(sample_json_file)
    assert result is not None
    assert len(result) == 2
    assert result[0]["input"] == "hello"


def test_cache_miss() -> None:
    """Test cache miss returns None."""
    cache = DatasetCache()
    assert cache.get("/nonexistent/file.json") is None


def test_cache_invalidation(sample_json_file: Any, sample_records: Any) -> None:
    """Test explicit cache invalidation."""
    cache = DatasetCache()
    cache.put(sample_json_file, sample_records)

    assert cache.get(sample_json_file) is not None
    removed = cache.invalidate(sample_json_file)
    assert removed is True
    assert cache.get(sample_json_file) is None


def test_cache_invalidate_nonexistent() -> None:
    """Test invalidating a non-cached path returns False."""
    cache = DatasetCache()
    assert cache.invalidate("/nonexistent.json") is False


def test_cache_clear(sample_json_file: Any, sample_records: Any) -> None:
    """Test clearing all cache entries."""
    cache = DatasetCache()
    cache.put(sample_json_file, sample_records)
    assert cache.size == 1

    cache.clear()
    assert cache.size == 0
    assert cache.get(sample_json_file) is None


def test_cache_lru_eviction(tmp_path: Any) -> None:
    """Test LRU eviction when cache is full."""
    cache = DatasetCache(max_entries=2)

    # Create 3 files
    for i in range(3):
        path = tmp_path / f"test_{i}.json"
        with open(path, "w") as f:
            json.dump([{"id": i}], f)

        cache.put(str(path), [{"id": i}])

    # Cache should only have 2 entries (oldest evicted)
    assert cache.size == 2

    # First file should have been evicted
    first_path = str(tmp_path / "test_0.json")
    assert cache.get(first_path) is None

    # Last two should still be cached
    assert cache.get(str(tmp_path / "test_1.json")) is not None
    assert cache.get(str(tmp_path / "test_2.json")) is not None


def test_cache_staleness_detection(sample_json_file: Any, sample_records: Any) -> None:
    """Test that modifying a file invalidates the cache."""
    cache = DatasetCache()
    cache.put(sample_json_file, sample_records)

    assert cache.get(sample_json_file) is not None

    # Modify the file (change mtime)
    time.sleep(0.05)
    with open(sample_json_file, "w") as f:
        json.dump([{"input": "modified"}], f)

    # Cache should detect staleness and return None
    assert cache.get(sample_json_file) is None


def test_cache_ttl_expiry(sample_json_file: Any, sample_records: Any) -> None:
    """Test TTL-based cache expiry."""
    cache = DatasetCache(ttl_seconds=0.1)
    cache.put(sample_json_file, sample_records)

    assert cache.get(sample_json_file) is not None

    # Wait for TTL to expire
    time.sleep(0.15)
    assert cache.get(sample_json_file) is None


def test_cache_stats(sample_json_file: Any, sample_records: Any) -> None:
    """Test cache statistics."""
    cache = DatasetCache(max_entries=10, ttl_seconds=3600)
    cache.put(sample_json_file, sample_records)
    cache.get(sample_json_file)  # access_count = 1

    stats = cache.stats
    assert stats["size"] == 1
    assert stats["max_entries"] == 10
    assert stats["ttl_seconds"] == 3600


def test_cache_access_count(sample_json_file: Any, sample_records: Any) -> None:
    """Test that access count is tracked."""
    cache = DatasetCache()
    cache.put(sample_json_file, sample_records)

    # Access 3 times
    for _ in range(3):
        cache.get(sample_json_file)

    stats = cache.stats
    entry_stats = next(iter(stats["entries"].values()))
    assert entry_stats["access_count"] == 3
