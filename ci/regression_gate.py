"""CI Gate Utility.

Used in GitHub Actions to enforce quality gates and regression blocking.
"""
# ruff: noqa: T201

import argparse
import json
import sys
from pathlib import Path

from sarathi_agent_inspect.core.governance import GateConfig, QualityGate, RegressionBlocker
from sarathi_agent_inspect.reporting.history import HistoricalTracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Sarathi CI Quality Gate")
    parser.add_argument("--report", type=str, required=True, help="Path to the current JSON report")
    parser.add_argument("--history", type=str, default=".sarathi/history.jsonl", help="Path to history file")
    parser.add_argument("--min-pass-rate", type=float, default=0.8)
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.05,
        help="Max allowed drop in pass rate (e.g. 0.05 for 5%)"
    )

    args = parser.parse_args()

    report_path = Path(args.report)
    if not report_path.exists():
        print(f"Error: Report not found at {report_path}")
        sys.exit(1)

    with open(report_path) as f:
        data = json.load(f)
        # Assuming the JSON report has a 'summary' key that matches EvaluationSummary schema
        from sarathi_agent_inspect.reporting.base import EvaluationSummary
        summary = EvaluationSummary.model_validate(data["summary"])

    config = GateConfig(
        min_pass_rate=args.min_pass_rate,
        regression_threshold=-args.max_regression
    )

    # 1. Check Thresholds
    threshold_result = QualityGate.check(summary, config)
    if not threshold_result.passed:
        print(f"❌ Quality Gate Failed: {threshold_result.reason}")
        sys.exit(1)
    print(f"✅ Thresholds Passed: {threshold_result.reason}")

    # 2. Check Regression
    tracker = HistoricalTracker(args.history)
    history = tracker.load_history(limit=5)

    regression_result = RegressionBlocker.evaluate_drift(summary, history, config)
    if not regression_result.passed:
        print(f"❌ Regression Gate Failed: {regression_result.reason}")
        sys.exit(1)
    print(f"✅ Regression Check Passed: {regression_result.reason}")

    print("🚀 All quality gates passed successfully!")


if __name__ == "__main__":
    main()
