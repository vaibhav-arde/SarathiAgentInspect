"""Unit tests for reporting system."""

import json
from pathlib import Path
from typing import Any

import pytest

from sarathi_agent_inspect.reporting.base import EvaluationSummary, ReportMetadata
from sarathi_agent_inspect.reporting.history import HistoricalTracker, TrendAnalyzer
from sarathi_agent_inspect.reporting.html_reporter import HTMLReporter
from sarathi_agent_inspect.reporting.json_reporter import JSONReporter


@pytest.fixture
def sample_summary() -> EvaluationSummary:
    metadata = ReportMetadata(run_id="run_123", total_cost_usd=0.05, total_latency_ms=1200.0)
    return EvaluationSummary(
        total_records=10,
        passed_count=8,
        failed_count=2,
        pass_rate=0.8,
        average_score=0.85,
        metadata=metadata,
    )


@pytest.fixture
def sample_results() -> list[dict[str, Any]]:
    return [
        {"test_id": "t1", "score": 1.0, "passed": True, "latency": 100.0, "cost": 0.001},
        {"test_id": "t2", "score": 0.0, "passed": False, "latency": 200.0, "cost": 0.002, "reason": "Failed test"},
    ]


def test_json_reporter(tmp_path: Path, sample_results: list[dict[str, Any]], sample_summary: EvaluationSummary) -> None:
    output_path = tmp_path / "report.json"
    reporter = JSONReporter(output_path)

    reporter.generate(sample_results, sample_summary)

    assert output_path.exists()
    with open(output_path) as f:
        data = json.load(f)
        assert data["summary"]["total_records"] == 10
        assert len(data["results"]) == 2


def test_html_reporter(tmp_path: Path, sample_results: list[dict[str, Any]], sample_summary: EvaluationSummary) -> None:
    output_path = tmp_path / "report.html"
    reporter = HTMLReporter(output_path)

    reporter.generate(sample_results, sample_summary)

    assert output_path.exists()
    content = output_path.read_text()
    assert "Sarathi Evaluation Report" in content
    assert "run_123" in content
    assert "80.0%" in content


def test_historical_tracker(tmp_path: Path, sample_summary: EvaluationSummary) -> None:
    history_file = tmp_path / "history.jsonl"
    tracker = HistoricalTracker(history_file)

    tracker.record_run(sample_summary)
    tracker.record_run(sample_summary)

    history = tracker.load_history()
    assert len(history) == 2
    assert history[0].metadata.run_id == "run_123"


def test_trend_analyzer(sample_summary: EvaluationSummary) -> None:
    history = [
        EvaluationSummary(
            total_records=10,
            passed_count=7,
            failed_count=3,
            pass_rate=0.7,
            average_score=0.75,
            metadata=ReportMetadata(run_id="old_run", total_cost_usd=0.04),
        )
    ]

    trends = TrendAnalyzer.calculate_trends(sample_summary, history)

    assert trends["status"] == "stable"
    assert trends["pass_rate_delta"] == 0.1
    assert trends["cost_delta"] == 0.01
