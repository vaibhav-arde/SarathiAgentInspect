"""Unit tests for dataset loaders."""

import csv
import json
from typing import Any

import pytest

from sarathi_agent_inspect.datasets.loaders.csv_loader import CSVLoader
from sarathi_agent_inspect.datasets.loaders.json_loader import JSONLoader
from sarathi_agent_inspect.datasets.loaders.parquet_loader import ParquetLoader
from sarathi_agent_inspect.datasets.schemas import ChatbotRecord


@pytest.fixture
def temp_json_file(tmp_path: Any) -> Any:
    """Create a temporary JSON file."""
    data = [{"input": "Hi", "expected_output": "Hello"}, {"input": "Bye", "expected_output": "Goodbye"}]
    path = tmp_path / "test.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


@pytest.fixture
def temp_jsonl_file(tmp_path: Any) -> Any:
    """Create a temporary JSONL file."""
    path = tmp_path / "test.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"input": "Hi", "expected_output": "Hello"}) + "\n")
        f.write(json.dumps({"input": "Bye", "expected_output": "Goodbye"}) + "\n")
    return path


@pytest.fixture
def temp_csv_file(tmp_path: Any) -> Any:
    """Create a temporary CSV file."""
    path = tmp_path / "test.csv"
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["input", "expected_output"])
        writer.writeheader()
        writer.writerow({"input": "Hi", "expected_output": "Hello"})
        writer.writerow({"input": "Bye", "expected_output": "Goodbye"})
    return path


@pytest.fixture
def temp_parquet_file(tmp_path: Any) -> Any:
    """Create a temporary Parquet file."""
    import polars as pl

    df = pl.DataFrame({"input": ["Hi", "Bye"], "expected_output": ["Hello", "Goodbye"]})
    path = tmp_path / "test.parquet"
    df.write_parquet(path)
    return path


def test_json_loader(temp_json_file: Any) -> None:
    """Test JSON loader."""
    loader = JSONLoader(schema_class=ChatbotRecord)
    loader.load(temp_json_file)
    assert len(loader) == 2
    records = list(loader)
    assert records[0]["input"] == "Hi"

    validation = loader.validate()
    assert validation.is_valid is True


def test_jsonl_loader(temp_jsonl_file: Any) -> None:
    """Test JSONL loader."""
    loader = JSONLoader(schema_class=ChatbotRecord)
    loader.load(temp_jsonl_file)
    # JSONL length requires a pass if not cached, but our implementation handles it
    assert len(loader) == 2
    records = list(loader)
    assert records[1]["input"] == "Bye"


def test_csv_loader(temp_csv_file: Any) -> None:
    """Test CSV loader."""
    loader = CSVLoader(schema_class=ChatbotRecord)
    loader.load(temp_csv_file)
    assert len(loader) == 2
    records = list(loader)
    assert records[0]["input"] == "Hi"


def test_parquet_loader(temp_parquet_file: Any) -> None:
    """Test Parquet loader."""
    loader = ParquetLoader(schema_class=ChatbotRecord)
    loader.load(temp_parquet_file)
    assert len(loader) == 2
    records = list(loader)
    assert records[0]["input"] == "Hi"
