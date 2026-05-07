"""Structured exception hierarchy for the framework.

Design principles:
- Every exception carries a machine-readable error_code
- Every exception carries a context dict for structured logging
- Exceptions can be serialized to JSON via to_dict()
- Hierarchy allows catching at any granularity level

Hierarchy:
    SarathiError (base)
    ├── ConfigurationError
    ├── ProviderError
    │   ├── ProviderConnectionError
    │   ├── ProviderTimeoutError
    │   ├── ProviderRateLimitError
    │   └── ProviderAuthenticationError
    ├── EvaluationError
    │   ├── MetricComputationError
    │   └── ThresholdViolationError
    ├── DatasetError
    │   ├── DatasetValidationError
    │   ├── DatasetLoadError
    │   └── DatasetFormatError
    └── RetryExhaustedError
"""

from __future__ import annotations

from typing import Any


class SarathiError(Exception):
    """Base exception for all framework errors.

    All framework exceptions inherit from this class, allowing
    callers to catch all framework errors with a single handler.

    Attributes:
        message: Human-readable error description.
        error_code: Machine-readable error identifier.
        context: Structured context for logging/debugging.
    """

    error_code: str = "SARATHI_ERROR"

    def __init__(
        self,
        message: str = "An error occurred in SarathiAgentInspect",
        context: dict[str, Any] | None = None,
    ) -> None:
        self.message = message
        self.context = context or {}
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to a JSON-compatible dictionary."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "type": type(self).__name__,
        }

    def __str__(self) -> str:
        if self.context:
            ctx_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"[{self.error_code}] {self.message} ({ctx_str})"
        return f"[{self.error_code}] {self.message}"


# ── Configuration Errors ─────────────────────────────────────────


class ConfigurationError(SarathiError):
    """Raised when framework configuration is invalid or missing."""

    error_code: str = "CONFIG_ERROR"


# ── Provider Errors ──────────────────────────────────────────────


class ProviderError(SarathiError):
    """Base error for all LLM provider operations."""

    error_code: str = "PROVIDER_ERROR"


class ProviderConnectionError(ProviderError):
    """Raised when a provider connection cannot be established."""

    error_code: str = "PROVIDER_CONNECTION_ERROR"


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request exceeds the timeout."""

    error_code: str = "PROVIDER_TIMEOUT_ERROR"


class ProviderRateLimitError(ProviderError):
    """Raised when a provider returns a rate limit response."""

    error_code: str = "PROVIDER_RATE_LIMIT_ERROR"


class ProviderAuthenticationError(ProviderError):
    """Raised when provider authentication fails."""

    error_code: str = "PROVIDER_AUTH_ERROR"


# ── Evaluation Errors ────────────────────────────────────────────


class EvaluationError(SarathiError):
    """Base error for evaluation operations."""

    error_code: str = "EVALUATION_ERROR"


class MetricComputationError(EvaluationError):
    """Raised when a metric fails to compute a score."""

    error_code: str = "METRIC_COMPUTATION_ERROR"


class ThresholdViolationError(EvaluationError):
    """Raised when an evaluation score falls below the threshold."""

    error_code: str = "THRESHOLD_VIOLATION_ERROR"


# ── Dataset Errors ───────────────────────────────────────────────


class DatasetError(SarathiError):
    """Base error for dataset operations."""

    error_code: str = "DATASET_ERROR"


class DatasetValidationError(DatasetError):
    """Raised when dataset validation fails."""

    error_code: str = "DATASET_VALIDATION_ERROR"


class DatasetLoadError(DatasetError):
    """Raised when a dataset cannot be loaded."""

    error_code: str = "DATASET_LOAD_ERROR"


class DatasetFormatError(DatasetError):
    """Raised when a dataset format is unsupported or malformed."""

    error_code: str = "DATASET_FORMAT_ERROR"


# ── Retry Errors ─────────────────────────────────────────────────


class RetryExhaustedError(SarathiError):
    """Raised when all retry attempts have been exhausted."""

    error_code: str = "RETRY_EXHAUSTED_ERROR"
