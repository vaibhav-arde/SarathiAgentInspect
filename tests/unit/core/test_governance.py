"""Unit tests for governance logic."""

import pytest

from sarathi_agent_inspect.core.governance import GateConfig, QualityGate, RegressionBlocker
from sarathi_agent_inspect.reporting.base import EvaluationSummary, ReportMetadata


@pytest.fixture
def sample_summary() -> EvaluationSummary:
    return EvaluationSummary(
        total_records=10,
        passed_count=8,
        failed_count=2,
        pass_rate=0.8,
        average_score=0.85,
        metadata=ReportMetadata(run_id="run_1", total_cost_usd=0.05),
    )


@pytest.mark.smoke
def test_quality_gate_pass(sample_summary: EvaluationSummary) -> None:
    config = GateConfig(min_pass_rate=0.75, min_average_score=0.8)
    result = QualityGate.check(sample_summary, config)
    assert result.passed is True


@pytest.mark.smoke
def test_quality_gate_fail_pass_rate(sample_summary: EvaluationSummary) -> None:
    config = GateConfig(min_pass_rate=0.9)
    result = QualityGate.check(sample_summary, config)
    assert result.passed is False
    assert "Pass rate 80.0% is below threshold 90.0%" in result.reason


@pytest.mark.smoke
def test_regression_blocker_pass(sample_summary: EvaluationSummary) -> None:
    history = [
        EvaluationSummary(
            total_records=10,
            passed_count=7,
            failed_count=3,
            pass_rate=0.7,
            average_score=0.75,
            metadata=ReportMetadata(run_id="old_run"),
        )
    ]
    config = GateConfig(regression_threshold=-0.05)
    result = RegressionBlocker.evaluate_drift(sample_summary, history, config)
    assert result.passed is True
    assert "Performance stable" in result.reason


@pytest.mark.smoke
def test_regression_blocker_fail(sample_summary: EvaluationSummary) -> None:
    history = [
        EvaluationSummary(
            total_records=10,
            passed_count=9,
            failed_count=1,
            pass_rate=0.9,  # Drop from 0.9 to 0.8 is 10%
            average_score=0.95,
            metadata=ReportMetadata(run_id="old_run"),
        )
    ]
    config = GateConfig(regression_threshold=-0.05)
    result = RegressionBlocker.evaluate_drift(sample_summary, history, config)
    assert result.passed is False
    assert "Regression detected" in result.reason
