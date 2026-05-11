"""Dataset caching strategy.

Provides an in-memory LRU cache for loaded datasets to avoid
redundant disk I/O during repeated evaluation runs.

Architecture:
    - Cache key = (file_path, mtime) — invalidates automatically when file changes
    - Bounded by max entry count (not bytes) for simplicity
    - Thread-safe via functools.lru_cache semantics
    - Optional TTL-based expiry for long-running processes

Enterprise considerations:
    - In CI pipelines, caching avoids re-reading fixtures across parametrized tests
    - In production, cache prevents re-parsing 100k+ record datasets per evaluation run
    - mtime-based invalidation ensures stale data is never served
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sarathi_agent_inspect.core.types import DatasetRecord


@dataclass
class CacheEntry:
    """A single cached dataset entry.

    Attributes:
        records: The cached dataset records.
        file_path: Original file path.
        file_mtime: File modification time at cache time.
        file_hash: SHA256 hash of the file for integrity verification.
        cached_at: Timestamp when the entry was cached.
        access_count: Number of times this entry has been accessed.
    """

    records: list[DatasetRecord]
    file_path: str
    file_mtime: float
    file_hash: str
    cached_at: float = field(default_factory=time.time)
    access_count: int = 0


class DatasetCache:
    """In-memory LRU cache for loaded dataset records.

    Usage:
        cache = DatasetCache(max_entries=50, ttl_seconds=3600)

        # Check cache before loading
        records = cache.get("/data/eval.json")
        if records is None:
            loader.load("/data/eval.json")
            records = list(loader)
            cache.put("/data/eval.json", records)
    """

    def __init__(
        self,
        max_entries: int = 50,
        ttl_seconds: float | None = None,
        verify_integrity: bool = False,
    ) -> None:
        """Initialize the dataset cache.

        Args:
            max_entries: Maximum number of datasets to cache.
            ttl_seconds: Time-to-live in seconds. None = no expiry.
            verify_integrity: If True, verify file hash on cache hit (slower but safer).
        """
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._verify_integrity = verify_integrity
        self._cache: dict[str, CacheEntry] = {}
        self._access_order: list[str] = []

    @property
    def size(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)

    @property
    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        return {
            "size": self.size,
            "max_entries": self._max_entries,
            "ttl_seconds": self._ttl_seconds,
            "entries": {
                k: {
                    "records": len(v.records),
                    "cached_at": v.cached_at,
                    "access_count": v.access_count,
                }
                for k, v in self._cache.items()
            },
        }

    @staticmethod
    def _file_hash(path: str) -> str:
        """Compute SHA256 hash of a file for integrity verification."""
        sha = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha.update(chunk)
        return sha.hexdigest()

    @staticmethod
    def _file_mtime(path: str) -> float:
        """Get file modification time."""
        return os.path.getmtime(path)

    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if a cache entry has expired."""
        if self._ttl_seconds is None:
            return False
        return (time.time() - entry.cached_at) > self._ttl_seconds

    def _is_stale(self, entry: CacheEntry) -> bool:
        """Check if the source file has been modified since caching."""
        path = Path(entry.file_path)
        if not path.exists():
            return True

        current_mtime = self._file_mtime(entry.file_path)
        if current_mtime != entry.file_mtime:
            return True

        if self._verify_integrity:
            current_hash = self._file_hash(entry.file_path)
            if current_hash != entry.file_hash:
                return True

        return False

    def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if self._access_order:
            lru_key = self._access_order.pop(0)
            self._cache.pop(lru_key, None)

    def _touch(self, key: str) -> None:
        """Move a key to the end of the access order (most recent)."""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)

    def get(self, path: str) -> list[DatasetRecord] | None:
        """Retrieve cached records for a file path.

        Returns None if not cached, expired, or stale.

        Args:
            path: The dataset file path.

        Returns:
            List of cached records, or None if cache miss.
        """
        resolved = str(Path(path).resolve())
        entry = self._cache.get(resolved)

        if entry is None:
            return None

        # Check expiry
        if self._is_expired(entry):
            self.invalidate(path)
            return None

        # Check staleness
        if self._is_stale(entry):
            self.invalidate(path)
            return None

        # Cache hit
        entry.access_count += 1
        self._touch(resolved)
        return entry.records

    def put(self, path: str, records: list[DatasetRecord]) -> None:
        """Cache dataset records for a file path.

        Args:
            path: The dataset file path.
            records: The dataset records to cache.
        """
        resolved = str(Path(path).resolve())

        # Evict if at capacity
        if len(self._cache) >= self._max_entries and resolved not in self._cache:
            self._evict_lru()

        file_path = Path(resolved)
        mtime = self._file_mtime(resolved) if file_path.exists() else 0.0
        file_hash = self._file_hash(resolved) if self._verify_integrity and file_path.exists() else ""

        self._cache[resolved] = CacheEntry(
            records=records,
            file_path=resolved,
            file_mtime=mtime,
            file_hash=file_hash,
        )
        self._touch(resolved)

    def invalidate(self, path: str) -> bool:
        """Remove a specific entry from the cache.

        Args:
            path: The dataset file path to invalidate.

        Returns:
            True if an entry was removed, False if not found.
        """
        resolved = str(Path(path).resolve())
        if resolved in self._cache:
            del self._cache[resolved]
            if resolved in self._access_order:
                self._access_order.remove(resolved)
            return True
        return False

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()
        self._access_order.clear()
