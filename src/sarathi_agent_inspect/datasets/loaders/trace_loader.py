"""Production trace dataset loader.

Stub for integrating with Langfuse, Datadog, or other trace providers
to build datasets directly from production logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from sarathi_agent_inspect.core.exceptions.base import DatasetFormatError, DatasetLoadError
from sarathi_agent_inspect.datasets.base import BaseDataset, DatasetMetadata, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sarathi_agent_inspect.core.types import DatasetRecord


class TraceLoader(BaseDataset):
    """Dataset loader for production traces (e.g., Langfuse API)."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize trace loader."""
        super().__init__(**kwargs)
        self._source_name: str | None = None
        self._metadata: DatasetMetadata | None = None
        self._records: list[DatasetRecord] = []

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        if not self._metadata:
            raise DatasetLoadError("Dataset not loaded. Call load() first.")
        return self._metadata

    def load(self, path: Path | str) -> None:
        """Load dataset from trace source API or JSON export.

        In a real implementation, 'path' could be a query string, a Langfuse project ID,
        or a file path to an exported trace JSON.
        """
        self._source_name = str(path)

        # Stub: If it's an actual file, we might parse it. Otherwise, we'd query an API.
        p = Path(path)
        if p.exists() and p.suffix.lower() == ".json":
            import json

            try:
                with open(p, encoding="utf-8") as f:
                    self._records = json.load(f)
            except Exception as e:
                raise DatasetFormatError(f"Failed to parse trace export: {e}") from e
        else:
            # Here we would query production trace API (e.g. Langfuse)
            pass

        self._metadata = DatasetMetadata(
            name=f"trace_{p.stem}",
            format="trace",
            source_path=self._source_name,
            record_count=len(self._records),
        )

    def validate(self) -> ValidationResult:
        """Validate all records against the schema class."""
        if not self._metadata:
            raise DatasetLoadError("Dataset not loaded. Call load() first.")

        result = ValidationResult(is_valid=True)

        for i, record in enumerate(self):
            result.record_count += 1
            if self.schema_class:
                try:
                    self.schema_class.model_validate(record)
                except ValidationError as e:
                    result.is_valid = False
                    result.errors.append(f"Record {i} validation failed: {e}")
                    result.invalid_records.append(i)

        return result

    def __iter__(self) -> Iterator[DatasetRecord]:
        """Iterate over trace records."""
        if self._source_name is None:
            raise DatasetLoadError("Dataset not loaded. Call load() first.")

        yield from self._records

    def __len__(self) -> int:
        """Return the number of records."""
        return len(self._records)
