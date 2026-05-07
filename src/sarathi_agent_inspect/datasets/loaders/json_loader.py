"""JSON and JSONL dataset loaders.

Supports streaming reads for JSONL to handle large datasets efficiently.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import ValidationError

from sarathi_agent_inspect.core.exceptions.base import DatasetFormatError, DatasetLoadError
from sarathi_agent_inspect.datasets.base import BaseDataset, DatasetMetadata, ValidationResult

if TYPE_CHECKING:
    from collections.abc import Iterator

    from sarathi_agent_inspect.core.types import DatasetRecord


class JSONLoader(BaseDataset):
    """Dataset loader for JSON and JSONL files."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize JSON loader."""
        super().__init__(**kwargs)
        self._path: Path | None = None
        self._is_jsonl: bool = False
        self._metadata: DatasetMetadata | None = None
        self._records: list[DatasetRecord] | None = None

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

        if self._path.suffix.lower() == ".jsonl":
            self._is_jsonl = True
        elif self._path.suffix.lower() == ".json":
            self._is_jsonl = False
        else:
            raise DatasetFormatError(f"Unsupported JSON format: {self._path.suffix}")

        # Compute basic metadata
        # For JSONL, we don't count records here to avoid reading the whole file.
        record_count = 0
        if not self._is_jsonl:
            try:
                with open(self._path, encoding="utf-8") as f:
                    self._records = json.load(f)
                    if not isinstance(self._records, list):
                        raise DatasetFormatError("JSON dataset must be an array of objects.")
                    record_count = len(self._records)
            except json.JSONDecodeError as e:
                raise DatasetFormatError(f"Invalid JSON: {e}") from e

        self._metadata = DatasetMetadata(
            name=self._path.stem,
            format="jsonl" if self._is_jsonl else "json",
            source_path=str(self._path),
            record_count=record_count,
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

        if self._is_jsonl:
            # Update record count in metadata after full iteration
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

        if self._is_jsonl:
            with open(self._path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError as e:
                            raise DatasetFormatError(f"Invalid JSONL line: {e}") from e
        else:
            if self._records is not None:
                yield from self._records

    def __len__(self) -> int:
        """Return the number of records."""
        if self._is_jsonl:
            # Requires full pass
            return sum(1 for _ in self)
        return len(self._records) if self._records is not None else 0
