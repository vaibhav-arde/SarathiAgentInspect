"""Dataset transformation and filtering pipeline.

Provides a lazy-evaluation pipeline for mapping, filtering,
deduplicating, batching, and sampling dataset records.

Scalability strategy:
    - All operations are lazy (generator-based) until materialized
    - batch() yields fixed-size chunks for memory-bounded processing
    - sample() provides random subset selection for quick iterations
    - Chaining operations builds a pipeline graph, not a data copy
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from sarathi_agent_inspect.core.types import DatasetRecord
    from sarathi_agent_inspect.datasets.base import BaseDataset


class DatasetPipeline:
    """Lazy evaluation pipeline for dataset transformations."""

    def __init__(self, source: BaseDataset | Iterable[DatasetRecord]) -> None:
        """Initialize the pipeline with a data source."""
        self._source = source
        self._operations: list[tuple[str, Any]] = []

    def map(self, func: Callable[[DatasetRecord], DatasetRecord]) -> DatasetPipeline:
        """Add a mapping operation.

        Args:
            func: Function taking a record and returning a transformed record.
        """
        self._operations.append(("map", func))
        return self

    def filter(self, predicate: Callable[[DatasetRecord], bool]) -> DatasetPipeline:
        """Add a filter operation.

        Args:
            predicate: Function returning True to keep a record, False to drop.
        """
        self._operations.append(("filter", predicate))
        return self

    def tag(self, tags: list[str]) -> DatasetPipeline:
        """Add tags to the metadata of matching records.

        Args:
            tags: List of tags to append.
        """

        def _add_tags(record: DatasetRecord) -> DatasetRecord:
            if "metadata" not in record:
                record["metadata"] = {}
            existing_tags = record["metadata"].get("tags", [])
            # Only add tags that don't already exist
            for tag in tags:
                if tag not in existing_tags:
                    existing_tags.append(tag)
            record["metadata"]["tags"] = existing_tags
            return record

        self._operations.append(("map", _add_tags))
        return self

    def deduplicate(self, key_func: Callable[[DatasetRecord], Any] | None = None) -> DatasetPipeline:
        """Deduplicate records.

        Args:
            key_func: Optional function to generate a uniqueness key.
                      If None, uses the string representation of the record.
        """
        self._operations.append(("dedup", key_func))
        return self

    def __iter__(self) -> Iterator[DatasetRecord]:
        """Execute the pipeline lazily and yield records."""
        seen = set()

        # Build an iterator from the source
        if hasattr(self._source, "__iter__"):
            iterator = iter(self._source)
        else:
            raise TypeError("Source must be iterable")

        for record in iterator:
            drop = False
            current_record = record

            # Apply operations in order
            for op_type, func in self._operations:
                if op_type == "map":
                    current_record = func(current_record)
                elif op_type == "filter":
                    if not func(current_record):
                        drop = True
                        break
                elif op_type == "dedup":
                    # Generate key
                    if func is None:
                        # Fallback: convert dict to a sorted string for hashing
                        try:
                            import json

                            key = json.dumps(current_record, sort_keys=True)
                        except Exception:
                            key = str(current_record)
                    else:
                        key = func(current_record)

                    if key in seen:
                        drop = True
                        break
                    seen.add(key)

            if not drop:
                yield current_record

    def to_list(self) -> list[DatasetRecord]:
        """Execute pipeline and materialize as a list."""
        return list(self)

    def batch(self, batch_size: int = 100) -> Iterator[list[DatasetRecord]]:
        """Yield records in fixed-size batches for memory-bounded processing.

        This is critical for large datasets where materializing the entire
        dataset into memory is impractical. Each batch is a list that can
        be processed independently.

        Args:
            batch_size: Number of records per batch. Must be > 0.

        Yields:
            Lists of records, each containing at most batch_size items.
            The final batch may contain fewer items.

        Raises:
            ValueError: If batch_size is not positive.
        """
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")

        current_batch: list[DatasetRecord] = []
        for record in self:
            current_batch.append(record)
            if len(current_batch) >= batch_size:
                yield current_batch
                current_batch = []

        # Yield the remaining partial batch
        if current_batch:
            yield current_batch

    def sample(self, n: int, *, seed: int | None = None) -> list[DatasetRecord]:
        """Select a random subset of records from the pipeline.

        Useful for quick iteration during development, smoke tests,
        or when evaluating a representative subset of a large dataset.

        Note: This materializes the pipeline first (requires full pass).
        For very large datasets, consider using filter() with a probability
        check instead.

        Args:
            n: Number of records to sample.
            seed: Optional random seed for reproducibility.

        Returns:
            List of n randomly selected records (or all records if n > total).
        """
        all_records = self.to_list()

        rng = random.Random(seed) if seed is not None else random.Random()  # noqa: S311

        if n >= len(all_records):
            return all_records

        return rng.sample(all_records, n)

    def head(self, n: int = 10) -> list[DatasetRecord]:
        """Return the first n records from the pipeline.

        Useful for quick inspection without materializing the full dataset.

        Args:
            n: Number of records to return.

        Returns:
            List of up to n records from the beginning of the pipeline.
        """
        results: list[DatasetRecord] = []
        for record in self:
            results.append(record)
            if len(results) >= n:
                break
        return results
