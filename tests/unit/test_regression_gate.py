"""Unit tests for the CI regression gate utility."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any


def _load_gate_module() -> Any:
    root = Path(__file__).resolve().parents[2]
    module_path = root / "ci" / "regression_gate.py"
    spec = importlib.util.spec_from_file_location("sarathi_regression_gate", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_summary_from_pytest_json_report(tmp_path: Path) -> None:
    module = _load_gate_module()
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "created": 1778496193.0,
                "duration": 12.5,
                "summary": {"passed": 3, "failed": 1, "total": 4, "collected": 4},
                "tests": [],
            }
        ),
        encoding="utf-8",
    )

    summary = module.load_summary_from_report(report_path)

    assert summary.total_records == 4
    assert summary.passed_count == 3
    assert summary.failed_count == 1
    assert summary.pass_rate == 0.75
    assert summary.average_score == 0.75
    assert summary.metadata.total_latency_ms == 12500.0


def test_load_summary_from_sarathi_report(tmp_path: Path) -> None:
    module = _load_gate_module()
    report_path = tmp_path / "sarathi-report.json"
    report_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total_records": 5,
                    "passed_count": 4,
                    "failed_count": 1,
                    "pass_rate": 0.8,
                    "average_score": 0.82,
                    "metadata": {"run_id": "run-123"},
                }
            }
        ),
        encoding="utf-8",
    )

    summary = module.load_summary_from_report(report_path)

    assert summary.total_records == 5
    assert summary.pass_rate == 0.8
    assert summary.average_score == 0.82
    assert summary.metadata.run_id == "run-123"
