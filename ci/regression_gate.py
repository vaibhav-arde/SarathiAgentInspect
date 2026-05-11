"""CI Gate Utility.

Used in GitHub Actions to enforce quality gates and regression blocking.
Supports both Sarathi-native reports and pytest-json-report output.
"""
# ruff: noqa: T201

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from sarathi_agent_inspect.core.governance import GateConfig, QualityGate, RegressionBlocker
from sarathi_agent_inspect.reporting.base import EvaluationSummary, ReportMetadata
from sarathi_agent_inspect.reporting.history import HistoricalTracker


def _metadata_from_pytest_report(data: dict[str, Any]) -> ReportMetadata:
    created = float(data.get("created", datetime.now(UTC).timestamp()))
    duration_seconds = float(data.get("duration", 0.0))
    created_at = datetime.fromtimestamp(created, tz=UTC)

    return ReportMetadata(
        run_id=f"pytest-{int(created)}",
        timestamp=created_at.isoformat(),
        environment=os.getenv("SARATHI_ENV", "ci"),
        total_cost_usd=0.0,
        total_latency_ms=duration_seconds * 1000.0,
        tags=["pytest-json-report"],
    )


def _summary_from_pytest_report(data: dict[str, Any]) -> EvaluationSummary:
    pytest_summary = data.get("summary")
    if not isinstance(pytest_summary, dict):
        raise ValueError("pytest JSON report is missing a summary object")

    total = int(pytest_summary.get("total", pytest_summary.get("collected", 0)))
    passed = int(pytest_summary.get("passed", 0))
    failed = sum(int(pytest_summary.get(key, 0)) for key in ("failed", "error", "xfailed", "xpassed"))
    if failed == 0 and total >= passed:
        failed = total - passed

    pass_rate = (passed / total) if total else 1.0

    return EvaluationSummary(
        total_records=total,
        passed_count=passed,
        failed_count=failed,
        # Binary pytest outcomes do not expose a richer per-test score.
        average_score=pass_rate,
        pass_rate=pass_rate,
        metadata=_metadata_from_pytest_report(data),
    )


def load_summary_from_report(report_path: Path) -> EvaluationSummary:
    """Load an EvaluationSummary from a supported report format."""
    with report_path.open(encoding="utf-8") as file_obj:
        data = json.load(file_obj)

    summary = data.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Report is missing a top-level 'summary' object")

    try:
        return EvaluationSummary.model_validate(summary)
    except ValidationError:
        if "tests" not in data:
            raise
        return _summary_from_pytest_report(data)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sarathi CI Quality Gate")
    parser.add_argument("--report", type=str, required=True, help="Path to the current JSON report")
    parser.add_argument(
        "--action",
        choices=["check", "snapshot"],
        default="check",
        help="Action to perform: 'check' against baseline or 'snapshot' current run as new baseline",
    )
    parser.add_argument("--branch", type=str, default="main", help="Target branch name")
    parser.add_argument("--baseline-dir", type=str, default=".sarathi/baselines", help="Baseline storage dir")
    parser.add_argument("--history", type=str, default=".sarathi/history.jsonl", help="Path to history file")
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    parser.add_argument("--min-average-score", type=float, default=0.7)
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        help="Max allowed drop in pass rate (e.g. 0.05 for 5%)",
    )
    parser.add_argument("--version", type=str, default="1.0.0", help="Baseline version for snapshot")
    parser.add_argument("--label", type=str, help="Optional label for snapshot")

    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: Report not found at {report_path}")
        sys.exit(1)

    try:
        summary = load_summary_from_report(report_path)
    except Exception as exc:
        print(f"Error: Failed to parse report at {report_path}: {exc}")
        sys.exit(1)

    # Initialize store
    from sarathi_agent_inspect.datasets.regression import (
        RegressionBaselineStore,
        RegressionComparator,
        RegressionSnapshot,
    )

    store = RegressionBaselineStore(base_dir=args.baseline_dir, branch=args.branch)

    if args.action == "snapshot":
        print(f"Creating new baseline snapshot for branch '{args.branch}'...")
        # Create snapshot from report data
        with report_path.open(encoding="utf-8") as f:
            data = json.load(f)

        snapshot = RegressionSnapshot(version=args.version)
        # If it's a pytest report, we extract tests
        if "tests" in data:
            for test in data["tests"]:
                test_id = test.get("nodeid", "unknown")
                outcome = test.get("outcome", "failed")
                score = 1.0 if outcome == "passed" else 0.0
                snapshot.add_record(test_id, score=score, output=outcome)
        else:
            # Native report - assuming we have 'results'
            results = data.get("results", [])
            for res in results:
                test_id = res.get("test_id", "unknown")
                snapshot.add_record(
                    test_id,
                    score=res.get("score", 0.0),
                    output=res.get("output", ""),
                )

        info = store.save_snapshot(snapshot, label=args.label)
        print(f"✅ Baseline snapshot '{info.snapshot_id}' saved to {info.path}")
        sys.exit(0)

    # Workflow: Check for regressions
    config = GateConfig(
        min_pass_rate=args.min_pass_rate,
        min_average_score=args.min_average_score,
        regression_threshold=-args.max_regression,
    )

    # 1. Threshold check
    threshold_result = QualityGate.check(summary, config)
    if not threshold_result.passed:
        print(f"Quality Gate Failed: {threshold_result.reason}")
        sys.exit(1)
    print(f"Thresholds Passed: {threshold_result.reason}")

    # 2. Historical Regression Check (using History tracker)
    tracker = HistoricalTracker(args.history)
    history = tracker.load_history(limit=5)
    historical_regression = RegressionBlocker.evaluate_drift(summary, history, config)
    if not historical_regression.passed:
        print(f"Historical Regression Gate Failed: {historical_regression.reason}")
        sys.exit(1)
    print(f"Historical Regression Check Passed: {historical_regression.reason}")

    # 3. Fine-grained Baseline Comparison (using BaselineStore)
    baseline = store.load_latest()
    if baseline is not None:
        print(f"Comparing against latest baseline for branch '{args.branch}'...")
        comparator = RegressionComparator(baseline, default_tolerance=args.max_regression)

        # Extract current scores from report
        current_scores = {}
        with report_path.open(encoding="utf-8") as f:
            data = json.load(f)
        if "tests" in data:
            for test in data["tests"]:
                current_scores[test.get("nodeid", "unknown")] = 1.0 if test.get("outcome") == "passed" else 0.0
        else:
            for res in data.get("results", []):
                current_scores[res.get("test_id", "unknown")] = res.get("score", 0.0)

        report = comparator.compare(current_scores)
        if not report.overall_passed:
            print(f"❌ Fine-grained Regression Failed! {report.failed_count} tests regressed.")
            # Print regressions
            for res in report.results:
                if not res.passed:
                    print(f"  - {res.test_id}: {res.baseline_score} -> {res.current_score} (delta: {res.score_delta})")
            sys.exit(1)
        print(f"✅ Fine-grained Regression Check Passed: All {report.total_records} tests stable.")
    else:
        print(f"No baseline found for branch '{args.branch}'. Skipping fine-grained comparison.")

    tracker.record_run(summary)
    print("🚀 All quality gates passed successfully!")


if __name__ == "__main__":
    main()
