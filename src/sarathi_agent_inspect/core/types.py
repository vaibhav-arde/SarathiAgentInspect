"""Shared type aliases for the framework.

Centralizes common type definitions to ensure consistency
across all modules without circular imports.
"""

from __future__ import annotations

from typing import Any

# ── Generic Types ─────────────────────────────────────────────────
JsonDict = dict[str, Any]
"""A JSON-serializable dictionary."""

Metadata = dict[str, str | int | float | bool | None]
"""Metadata dictionary with primitive values."""

# ── Provider Types ────────────────────────────────────────────────
ModelName = str
"""An LLM model identifier (e.g., 'gemma4:31b-cloud')."""

ProviderName = str
"""A provider identifier (e.g., 'ollama', 'openai')."""

# ── Evaluation Types ──────────────────────────────────────────────
Score = float
"""A numeric evaluation score, typically 0.0 to 1.0."""

Threshold = float
"""A pass/fail threshold for metric evaluation."""

# ── Dataset Types ─────────────────────────────────────────────────
DatasetRecord = dict[str, Any]
"""A single record from a dataset."""
