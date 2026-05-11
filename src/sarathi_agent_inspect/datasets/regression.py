"""Regression dataset management.

Provides tooling for capturing baseline snapshots of model outputs,
comparing new outputs against baselines, and generating regression reports.

Architecture:
    - RegressionSnapshot: Captures and persists model outputs as a versioned baseline
    - RegressionComparator: Diffs new outputs vs. baseline with configurable tolerance
    - RegressionReport: Structured pass/fail report per record

Enterprise considerations:
    - Snapshots are stored as JSON for portability and diffability
    - Each snapshot carries version metadata for audit trails
    - Tolerance-based comparison allows controlled quality drift
    - Reports integrate with CI pipelines for automated regression gates
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sarathi_agent_inspect.core.sanitizer import InputSanitizer

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sarathi_agent_inspect.core.types import DatasetRecord
    from sarathi_agent_inspect.reporting.base import EvaluationSummary


@dataclass
class RegressionResult:
    """Result of comparing a single record against its baseline.

    Attributes:
        test_id: Unique identifier for the test case.
        passed: Whether the record passed regression checks.
        baseline_score: Score from the baseline run.
        current_score: Score from the current run.
        score_delta: Difference (current - baseline). Negative = regression.
        tolerance: The configured acceptable tolerance.
        details: Additional comparison details.
    """

    test_id: str
    passed: bool
    baseline_score: float
    current_score: float
    score_delta: float
    tolerance: float
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class RegressionReport:
    """Aggregated regression report across all records.

    Attributes:
        total_records: Total number of records compared.
        passed_count: Number of records that passed.
        failed_count: Number of records that regressed.
        results: Per-record regression results.
        snapshot_version: The baseline version used.
        timestamp: When the comparison was run.
        overall_passed: Whether the entire suite passed.
    """

    total_records: int = 0
    passed_count: int = 0
    failed_count: int = 0
    results: list[RegressionResult] = field(default_factory=list)
    snapshot_version: str = ""
    timestamp: float = field(default_factory=time.time)

    @property
    def overall_passed(self) -> bool:
        """Return True if all records passed regression checks."""
        return self.failed_count == 0

    @property
    def pass_rate(self) -> float:
        """Return the pass rate as a percentage."""
        if self.total_records == 0:
            return 100.0
        return (self.passed_count / self.total_records) * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the report to a JSON-compatible dict."""
        return {
            "overall_passed": self.overall_passed,
            "pass_rate": round(self.pass_rate, 2),
            "total_records": self.total_records,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "snapshot_version": self.snapshot_version,
            "timestamp": self.timestamp,
            "results": [
                {
                    "test_id": r.test_id,
                    "passed": r.passed,
                    "baseline_score": r.baseline_score,
                    "current_score": r.current_score,
                    "score_delta": round(r.score_delta, 4),
                    "tolerance": r.tolerance,
                    "details": r.details,
                }
                for r in self.results
            ],
        }


class RegressionSnapshot:
    """Captures and persists model outputs as a versioned baseline.

    Usage:
        snapshot = RegressionSnapshot(version="1.0.0")
        snapshot.add_record("test_001", score=0.95, output="The answer is 42")
        snapshot.save("/baselines/v1.json")

        # Later, load it back
        loaded = RegressionSnapshot.load("/baselines/v1.json")
    """

    def __init__(self, version: str = "1.0.0") -> None:
        """Initialize a regression snapshot.

        Args:
            version: Version string for this baseline.
        """
        self.version = version
        self._records: dict[str, DatasetRecord] = {}
        self._timestamp: float = time.time()

    @property
    def record_count(self) -> int:
        """Return the number of records in the snapshot."""
        return len(self._records)

    def add_record(
        self,
        test_id: str,
        *,
        score: float,
        output: str,
        input_prompt: str = "",
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Add a record to the snapshot.

        Args:
            test_id: Unique test case identifier.
            score: The evaluation score for this record.
            output: The model output for this record.
            input_prompt: The input prompt (for reference).
            extra: Additional data to store.
        """
        self._records[test_id] = {
            "test_id": test_id,
            "score": score,
            "output": output,
            "input": input_prompt,
            "extra": extra or {},
        }

    def get_record(self, test_id: str) -> DatasetRecord | None:
        """Get a specific record by test_id."""
        return self._records.get(test_id)

    def save(self, path: str | Path) -> None:
        """Save the snapshot to a JSON file.

        Args:
            path: File path to save the snapshot.
        """
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "version": self.version,
            "timestamp": self._timestamp,
            "record_count": self.record_count,
            "records": self._records,
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(InputSanitizer.sanitize_for_export(payload), f, indent=2, default=str)

    @classmethod
    def load(cls, path: str | Path) -> RegressionSnapshot:
        """Load a snapshot from a JSON file.

        Args:
            path: File path to load the snapshot from.

        Returns:
            A RegressionSnapshot populated with the saved data.

        Raises:
            FileNotFoundError: If the snapshot file doesn't exist.
            ValueError: If the snapshot format is invalid.
        """
        file_path = Path(path)
        if not file_path.exists():
            raise FileNotFoundError(f"Snapshot not found: {file_path}")

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)

        if "version" not in data or "records" not in data:
            raise ValueError(f"Invalid snapshot format: {file_path}")

        snapshot = cls(version=data["version"])
        snapshot._timestamp = data.get("timestamp", time.time())
        snapshot._records = data["records"]
        return snapshot

    @classmethod
    def from_summary(cls, summary: EvaluationSummary, version: str = "1.0.0") -> RegressionSnapshot:
        """Create a snapshot from an existing EvaluationSummary.

        Note: This only populates metadata and aggregates. Detailed per-record
        outputs must be added manually since EvaluationSummary is an aggregate.
        """
        snapshot = cls(version=version)
        snapshot._timestamp = datetime.fromisoformat(summary.metadata.timestamp).timestamp()
        return snapshot

    def __iter__(self) -> Iterator[DatasetRecord]:
        """Iterate over snapshot records."""
        yield from self._records.values()

    def __len__(self) -> int:
        """Return record count."""
        return self.record_count


class RegressionComparator:
    """Compares current model outputs against a baseline snapshot.

    Usage:
        baseline = RegressionSnapshot.load("/baselines/v1.json")
        comparator = RegressionComparator(baseline, default_tolerance=0.05)
        report = comparator.compare(current_results)
    """

    def __init__(
        self,
        baseline: RegressionSnapshot,
        default_tolerance: float = 0.05,
    ) -> None:
        """Initialize the comparator.

        Args:
            baseline: The baseline snapshot to compare against.
            default_tolerance: Default acceptable score regression (0.05 = 5%).
        """
        self._baseline = baseline
        self._default_tolerance = default_tolerance

    def compare(
        self,
        current_results: dict[str, float],
        tolerances: dict[str, float] | None = None,
    ) -> RegressionReport:
        """Compare current scores against the baseline.

        Args:
            current_results: Dict mapping test_id → current score.
            tolerances: Optional per-test-id tolerance overrides.

        Returns:
            A RegressionReport with per-record pass/fail and aggregates.
        """
        overrides = tolerances or {}
        report = RegressionReport(snapshot_version=self._baseline.version)

        for test_id, current_score in current_results.items():
            baseline_record = self._baseline.get_record(test_id)

            if baseline_record is None:
                # New test — no regression possible, auto-pass
                result = RegressionResult(
                    test_id=test_id,
                    passed=True,
                    baseline_score=0.0,
                    current_score=current_score,
                    score_delta=current_score,
                    tolerance=self._default_tolerance,
                    details={"status": "new_test"},
                )
            else:
                baseline_score = float(baseline_record.get("score", 0.0))
                tolerance = overrides.get(test_id, self._default_tolerance)
                delta = current_score - baseline_score

                # Regression = current score dropped below (baseline - tolerance)
                passed = current_score >= (baseline_score - tolerance)

                result = RegressionResult(
                    test_id=test_id,
                    passed=passed,
                    baseline_score=baseline_score,
                    current_score=current_score,
                    score_delta=delta,
                    tolerance=tolerance,
                    details={
                        "baseline_output": baseline_record.get("output", ""),
                    },
                )

            report.results.append(result)
            report.total_records += 1
            if result.passed:
                report.passed_count += 1
            else:
                report.failed_count += 1

        return report


@dataclass(frozen=True)
class BaselineSnapshotInfo:
    """Metadata describing a stored regression baseline snapshot."""

    snapshot_id: str
    version: str
    branch: str
    path: str
    created_at: float
    label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RegressionBaselineStore:
    """Versioned filesystem-backed baseline storage.

    Stores immutable snapshots per branch and maintains a manifest pointing
    to the latest baseline, giving CI and local workflows a stable place to
    resolve regression baselines.
    """

    def __init__(self, base_dir: str | Path = ".sarathi/baselines", branch: str = "main") -> None:
        self.base_dir = Path(base_dir)
        self.branch = self._sanitize_branch(branch)
        self.branch_dir = self.base_dir / self.branch
        self.branch_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sanitize_branch(branch: str) -> str:
        return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in branch).strip("-") or "main"

    @property
    def manifest_path(self) -> Path:
        return self.branch_dir / "manifest.json"

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            return {"branch": self.branch, "latest": None, "snapshots": []}
        with self.manifest_path.open(encoding="utf-8") as file_obj:
            return json.load(file_obj)

    def _save_manifest(self, manifest: dict[str, Any]) -> None:
        self.branch_dir.mkdir(parents=True, exist_ok=True)
        with self.manifest_path.open("w", encoding="utf-8") as file_obj:
            json.dump(manifest, file_obj, indent=2)

    def save_snapshot(
        self,
        snapshot: RegressionSnapshot,
        *,
        label: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BaselineSnapshotInfo:
        """Persist a snapshot and update the branch manifest."""
        created_at = time.time()
        snapshot_id = f"{int(created_at)}-{uuid4().hex[:8]}"
        file_name = f"{snapshot_id}.json"
        path = self.branch_dir / file_name
        snapshot.save(path)

        info = BaselineSnapshotInfo(
            snapshot_id=snapshot_id,
            version=snapshot.version,
            branch=self.branch,
            path=str(path),
            created_at=created_at,
            label=label,
            metadata=metadata or {},
        )

        manifest = self._load_manifest()
        manifest["latest"] = snapshot_id
        manifest["snapshots"].append(info.__dict__)
        self._save_manifest(InputSanitizer.sanitize_for_export(manifest))
        return info

    def list_snapshots(self) -> list[BaselineSnapshotInfo]:
        """Return stored snapshot metadata for the current branch."""
        manifest = self._load_manifest()
        return [BaselineSnapshotInfo(**entry) for entry in manifest.get("snapshots", [])]

    def load_snapshot(self, snapshot_id: str) -> RegressionSnapshot:
        """Load a specific snapshot by identifier."""
        for info in self.list_snapshots():
            if info.snapshot_id == snapshot_id:
                return RegressionSnapshot.load(info.path)
        raise FileNotFoundError(f"Baseline snapshot not found: {snapshot_id}")

    def load_latest(self) -> RegressionSnapshot | None:
        """Load the most recently saved snapshot for the branch."""
        manifest = self._load_manifest()
        latest_id = manifest.get("latest")
        if not latest_id:
            return None
        return self.load_snapshot(latest_id)
