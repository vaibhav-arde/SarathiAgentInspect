"""Parquet dataset loader.

Utilizes Polars for high-performance lazy loading of Parquet datasets.
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


class ParquetLoader(BaseDataset):
    """Dataset loader for Parquet files using Polars."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize Parquet loader."""
        super().__init__(**kwargs)
        self._path: Path | None = None
        self._metadata: DatasetMetadata | None = None

        # Deferred import to keep base footprint low if polars isn't installed
        try:
            import polars as pl

            self.pl = pl
        except ImportError as e:
            raise ImportError("polars is required for ParquetLoader. Run `pip install polars pyarrow`") from e

    @property
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        if not self._metadata:
            raise DatasetLoadError("Dataset not loaded. Call load() first.")
        return self._metadata

    def load(self, path: Path | str) -> None:
        """Load dataset metadata from Parquet file."""
        self._path = Path(path)
        if not self._path.exists():
            raise DatasetLoadError(f"File not found: {self._path}")

        if self._path.suffix.lower() != ".parquet":
            raise DatasetFormatError(f"Unsupported format, expected .parquet: {self._path.suffix}")

        try:
            # We use scan_parquet to quickly get metadata without loading data
            self.pl.scan_parquet(self._path)

            # Count records might be expensive on massive files, but collect().height is generally okay
            # For pure metadata, we could skip it, but let's try a fast count
            # Actually, `pl.read_parquet(path).height` is fast because it reads metadata footers.
            record_count = self.pl.scan_parquet(self._path).select(self.pl.len()).collect().item()
        except Exception as e:
            raise DatasetFormatError(f"Failed to read Parquet metadata: {e}") from e

        self._metadata = DatasetMetadata(
            name=self._path.stem,
            format="parquet",
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

        return result

    def __iter__(self) -> Iterator[DatasetRecord]:
        """Iterate over dataset records."""
        if not self._path:
            raise DatasetLoadError("Dataset not loaded. Call load() first.")

        try:
            # For massive datasets, streaming in chunks is preferred.
            # Polars iter_rows() on read_parquet reads the whole DF to memory.
            # We'll stick to a simple read_parquet here, but could implement
            # chunked reading with `scan_parquet().slice(offset, len)` for multi-GB sets.
            df = self.pl.read_parquet(self._path)
            yield from df.iter_rows(named=True)
        except Exception as e:
            raise DatasetFormatError(f"Error reading Parquet rows: {e}") from e

    def __len__(self) -> int:
        """Return the number of records."""
        if self._metadata:
            return self._metadata.record_count
        return 0
