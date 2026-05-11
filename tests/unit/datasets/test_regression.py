"""Unit tests for regression dataset tooling."""

import json
from typing import Any

import pytest

from sarathi_agent_inspect.datasets.regression import (
    RegressionComparator,
    RegressionReport,
    RegressionSnapshot,
)

# ── RegressionSnapshot Tests ────────────────────────────────────────


def test_snapshot_add_and_count() -> None:
    """Test adding records to a snapshot."""
    snapshot = RegressionSnapshot(version="1.0.0")
    snapshot.add_record("test_001", score=0.95, output="Answer A", input_prompt="Q1")
    snapshot.add_record("test_002", score=0.88, output="Answer B", input_prompt="Q2")

    assert snapshot.record_count == 2
    assert len(snapshot) == 2


def test_snapshot_get_record() -> None:
    """Test retrieving a specific record."""
    snapshot = RegressionSnapshot(version="1.0.0")
    snapshot.add_record("test_001", score=0.95, output="Answer A")

    record = snapshot.get_record("test_001")
    assert record is not None
    assert record["score"] == 0.95
    assert record["output"] == "Answer A"

    assert snapshot.get_record("nonexistent") is None


def test_snapshot_save_and_load(tmp_path: Any) -> None:
    """Test saving and loading a snapshot."""
    path = tmp_path / "baseline.json"

    # Save
    snapshot = RegressionSnapshot(version="2.0.0")
    snapshot.add_record("t1", score=0.9, output="A")
    snapshot.add_record("t2", score=0.85, output="B")
    snapshot.save(path)

    # Load
    loaded = RegressionSnapshot.load(path)
    assert loaded.version == "2.0.0"
    assert loaded.record_count == 2
    record = loaded.get_record("t1")
    assert record is not None
    assert record["score"] == 0.9


def test_snapshot_load_missing_file() -> None:
    """Test loading a non-existent snapshot raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        RegressionSnapshot.load("/nonexistent/baseline.json")


def test_snapshot_load_invalid_format(tmp_path: Any) -> None:
    """Test loading invalid JSON raises ValueError."""
    path = tmp_path / "bad.json"
    with open(path, "w") as f:
        json.dump({"not_a_snapshot": True}, f)

    with pytest.raises(ValueError, match="Invalid snapshot format"):
        RegressionSnapshot.load(path)


def test_snapshot_iteration() -> None:
    """Test iterating over snapshot records."""
    snapshot = RegressionSnapshot(version="1.0.0")
    snapshot.add_record("t1", score=0.9, output="A")
    snapshot.add_record("t2", score=0.85, output="B")

    records = list(snapshot)
    assert len(records) == 2


# ── RegressionComparator Tests ──────────────────────────────────────


def test_comparator_all_pass() -> None:
    """Test comparison where all scores are above tolerance."""
    baseline = RegressionSnapshot(version="1.0.0")
    baseline.add_record("t1", score=0.90, output="A")
    baseline.add_record("t2", score=0.85, output="B")

    comparator = RegressionComparator(baseline, default_tolerance=0.05)
    report = comparator.compare({"t1": 0.92, "t2": 0.83})

    assert report.overall_passed is True
    assert report.passed_count == 2
    assert report.failed_count == 0
    assert report.pass_rate == 100.0


def test_comparator_regression_detected() -> None:
    """Test comparison where a regression is detected."""
    baseline = RegressionSnapshot(version="1.0.0")
    baseline.add_record("t1", score=0.90, output="A")
    baseline.add_record("t2", score=0.85, output="B")

    comparator = RegressionComparator(baseline, default_tolerance=0.05)
    # t2 dropped from 0.85 to 0.70 — that's a 0.15 drop, exceeding 0.05 tolerance
    report = comparator.compare({"t1": 0.88, "t2": 0.70})

    assert report.overall_passed is False
    assert report.passed_count == 1
    assert report.failed_count == 1

    # Check the failing result
    failed = [r for r in report.results if not r.passed]
    assert len(failed) == 1
    assert failed[0].test_id == "t2"
    assert failed[0].score_delta < 0


def test_comparator_new_test() -> None:
    """Test that new tests (not in baseline) auto-pass."""
    baseline = RegressionSnapshot(version="1.0.0")
    baseline.add_record("t1", score=0.90, output="A")

    comparator = RegressionComparator(baseline)
    report = comparator.compare({"t1": 0.88, "new_test": 0.75})

    assert report.overall_passed is True
    assert report.total_records == 2

    new_result = next(r for r in report.results if r.test_id == "new_test")
    assert new_result.passed is True
    assert new_result.details == {"status": "new_test"}


def test_comparator_custom_tolerance() -> None:
    """Test per-test-id tolerance overrides."""
    baseline = RegressionSnapshot(version="1.0.0")
    baseline.add_record("strict_test", score=0.90, output="A")
    baseline.add_record("relaxed_test", score=0.90, output="B")

    comparator = RegressionComparator(baseline, default_tolerance=0.05)
    report = comparator.compare(
        {"strict_test": 0.80, "relaxed_test": 0.80},
        tolerances={"relaxed_test": 0.15},  # Allow up to 15% drop
    )

    results = {r.test_id: r for r in report.results}
    assert results["strict_test"].passed is False  # 0.10 drop > 0.05 tolerance
    assert results["relaxed_test"].passed is True  # 0.10 drop < 0.15 tolerance


# ── RegressionReport Tests ──────────────────────────────────────────


def test_report_to_dict() -> None:
    """Test report serialization."""
    baseline = RegressionSnapshot(version="1.0.0")
    baseline.add_record("t1", score=0.9, output="A")

    comparator = RegressionComparator(baseline)
    report = comparator.compare({"t1": 0.85})

    report_dict = report.to_dict()
    assert "overall_passed" in report_dict
    assert "pass_rate" in report_dict
    assert "results" in report_dict
    assert len(report_dict["results"]) == 1


def test_report_empty() -> None:
    """Test report with no records."""
    report = RegressionReport()
    assert report.overall_passed is True
    assert report.pass_rate == 100.0
