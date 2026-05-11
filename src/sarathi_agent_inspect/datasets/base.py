"""Abstract base dataset interface.

Defines the contract for dataset loaders that handle
JSON, JSONL, CSV, Parquet, and custom dataset formats.

A dataset:
    1. Loads records from a source (file, URL, database)
    2. Validates records against a schema
    3. Provides iteration over records
    4. Carries metadata (name, version, format, size)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from pydantic import BaseModel

    from sarathi_agent_inspect.core.types import DatasetRecord, JsonDict


@dataclass(frozen=True)
class DatasetMetadata:
    """Metadata about a dataset.

    Attributes:
        name: Dataset identifier.
        version: Dataset version string.
        format: File format (json, jsonl, csv, parquet).
        record_count: Number of records in the dataset.
        description: Human-readable description.
        tags: Classification tags (e.g., 'rag', 'chatbot', 'safety').
        source_path: Original source path or URL.
        schema_version: Schema version for the dataset format.
        dvc_hash: Optional DVC commit hash for reproducibility tracking.
        extra: Additional metadata.
    """

    name: str
    version: str = "1.0.0"
    format: str = "json"
    record_count: int = 0
    description: str = ""
    tags: list[str] = field(default_factory=list)
    source_path: str = ""
    schema_version: str = "1.0.0"
    dvc_hash: str | None = None
    extra: JsonDict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of dataset validation.

    Attributes:
        is_valid: Whether the dataset passed all validation checks.
        errors: List of validation error messages.
        warnings: List of validation warning messages.
        record_count: Number of records validated.
        invalid_records: Indices of invalid records.
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    record_count: int = 0
    invalid_records: list[int] = field(default_factory=list)


class BaseDataset(ABC):
    """Abstract base class for dataset loaders.

    All dataset implementations must inherit from this class
    and implement the abstract methods.
    """

    def __init__(self, schema_class: type[BaseModel] | None = None) -> None:
        """Initialize the dataset loader.

        Args:
            schema_class: Optional Pydantic model class for record validation.
        """
        self.schema_class = schema_class

    @property
    @abstractmethod
    def metadata(self) -> DatasetMetadata:
        """Return dataset metadata."""
        ...

    @abstractmethod
    def load(self, path: Path | str) -> None:
        """Load the dataset from a source.

        Args:
            path: Path to the dataset file or directory.

        Raises:
            DatasetLoadError: If the dataset cannot be loaded.
            DatasetFormatError: If the format is unsupported.
        """
        ...

    @abstractmethod
    def validate(self) -> ValidationResult:
        """Validate all records in the dataset.

        Returns:
            ValidationResult with errors and warnings.

        Raises:
            DatasetValidationError: If validation encounters a fatal error.
        """
        ...

    @abstractmethod
    def __iter__(self) -> Iterator[DatasetRecord]:
        """Iterate over dataset records.

        Yields:
            Individual dataset records as dictionaries.
        """
        ...

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of records in the dataset."""
        ...

    def get_record(self, index: int) -> DatasetRecord:
        """Get a specific record by index.

        Args:
            index: Zero-based record index.

        Returns:
            The dataset record at the given index.

        Raises:
            IndexError: If the index is out of range.
        """
        for i, record in enumerate(self):
            if i == index:
                return record
        raise IndexError(f"Record index {index} out of range")

    def filter(self, predicate: Any) -> Iterator[DatasetRecord]:
        """Filter records using a predicate function (Lazy Evaluation).

        Args:
            predicate: A callable that accepts a record and returns bool.

        Yields:
            Records matching the predicate.
        """
        for record in self:
            if predicate(record):
                yield record
