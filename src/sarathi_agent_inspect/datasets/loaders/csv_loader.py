"""CSV dataset loader.

Supports streaming reads for CSV datasets.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from sarathi_agent_inspect.core.exceptions.base import DatasetFormatError, DatasetLoadError
from sarathi_agent_inspect.datasets.base import BaseDataset, DatasetMetadata, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sarathi_agent_inspect.core.types import DatasetRecord


class CSVLoader(BaseDataset):
    """Dataset loader for CSV files."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize CSV loader."""
        super().__init__(**kwargs)
        self._path: Path | None = None
        self._metadata: DatasetMetadata | None = None

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        if not self._metadata:
            raise DatasetLoadError("Dataset not loaded. Call load() first.")
        return self._metadata

    def load(self, path: Path | str) -> None:
        """Load dataset from path."""
        self._path = Path(path)
        if not self._path.exists():
            raise DatasetLoadError(f"File not found: {self._path}")

        if self._path.suffix.lower() != ".csv":
            raise DatasetFormatError(f"Unsupported format, expected .csv: {self._path.suffix}")

        self._metadata = DatasetMetadata(
            name=self._path.stem,
            format="csv",
            source_path=str(self._path),
            record_count=0,
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

        # Update record count
        self._metadata = DatasetMetadata(
            name=self._metadata.name,
            format=self._metadata.format,
            source_path=self._metadata.source_path,
            record_count=result.record_count,
        )

        return result

    def __iter__(self) -> Iterator[DatasetRecord]:
        """Iterate over dataset records."""
        if not self._path:
            raise DatasetLoadError("Dataset not loaded. Call load() first.")

        try:
            with open(self._path, encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # DictReader returns str values. Some schemas may expect structured dicts or lists.
                    # Pydantic handles some coercion, but nested structures in CSV are inherently flat.
                    # We yield standard dictionaries.
                    yield dict(row)
        except Exception as e:
            raise DatasetFormatError(f"Error reading CSV: {e}") from e

    def __len__(self) -> int:
        """Return the number of records (requires full pass)."""
        return sum(1 for _ in self)
