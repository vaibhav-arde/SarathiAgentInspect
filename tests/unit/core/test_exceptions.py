"""Unit tests for the exception hierarchy.

Tests:
- Exception creation with messages and context
- Error codes
- Serialization to dict
- Hierarchy relationships
- String representation
"""

from __future__ import annotations

import pytest

from sarathi_agent_inspect.core.exceptions.base import (
    ConfigurationError,
    DatasetError,
    DatasetFormatError,
    DatasetLoadError,
    DatasetValidationError,
    EvaluationError,
    MetricComputationError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RetryExhaustedError,
    SarathiError,
    ThresholdViolationError,
)


class TestSarathiError:
    """Tests for the base SarathiError."""

    def test_default_message(self) -> None:
        err = SarathiError()
        assert "An error occurred" in str(err)

    def test_custom_message(self) -> None:
        err = SarathiError(message="Something broke")
        assert err.message == "Something broke"

    def test_context(self) -> None:
        ctx = {"provider": "ollama", "model": "gemma4"}
        err = SarathiError(message="Error", context=ctx)
        assert err.context == ctx

    def test_error_code(self) -> None:
        err = SarathiError()
        assert err.error_code == "SARATHI_ERROR"

    def test_to_dict(self) -> None:
        err = SarathiError(message="Test", context={"key": "val"})
        d = err.to_dict()
        assert d["error_code"] == "SARATHI_ERROR"
        assert d["message"] == "Test"
        assert d["context"] == {"key": "val"}
        assert d["type"] == "SarathiError"

    def test_str_with_context(self) -> None:
        err = SarathiError(message="Fail", context={"a": 1})
        s = str(err)
        assert "[SARATHI_ERROR]" in s
        assert "Fail" in s
        assert "a=1" in s

    def test_str_without_context(self) -> None:
        err = SarathiError(message="Fail")
        s = str(err)
        assert s == "[SARATHI_ERROR] Fail"


class TestExceptionHierarchy:
    """Tests for the exception inheritance chain."""

    def test_provider_errors_inherit_from_sarathi_error(self) -> None:
        errors = [
            ProviderError,
            ProviderConnectionError,
            ProviderTimeoutError,
            ProviderRateLimitError,
            ProviderAuthenticationError,
        ]
        for error_cls in errors:
            err = error_cls(message="test")
            assert isinstance(err, SarathiError)

    def test_provider_subtypes_inherit_from_provider_error(self) -> None:
        subtypes = [
            ProviderConnectionError,
            ProviderTimeoutError,
            ProviderRateLimitError,
            ProviderAuthenticationError,
        ]
        for error_cls in subtypes:
            err = error_cls(message="test")
            assert isinstance(err, ProviderError)

    def test_evaluation_errors_inherit_correctly(self) -> None:
        assert issubclass(MetricComputationError, EvaluationError)
        assert issubclass(ThresholdViolationError, EvaluationError)
        assert issubclass(EvaluationError, SarathiError)

    def test_dataset_errors_inherit_correctly(self) -> None:
        assert issubclass(DatasetValidationError, DatasetError)
        assert issubclass(DatasetLoadError, DatasetError)
        assert issubclass(DatasetFormatError, DatasetError)
        assert issubclass(DatasetError, SarathiError)

    def test_retry_exhausted_inherits_correctly(self) -> None:
        assert issubclass(RetryExhaustedError, SarathiError)


class TestErrorCodes:
    """Tests for unique error codes on each exception type."""

    def test_unique_error_codes(self) -> None:
        error_classes = [
            SarathiError,
            ConfigurationError,
            ProviderError,
            ProviderConnectionError,
            ProviderTimeoutError,
            ProviderRateLimitError,
            ProviderAuthenticationError,
            EvaluationError,
            MetricComputationError,
            ThresholdViolationError,
            DatasetError,
            DatasetValidationError,
            DatasetLoadError,
            DatasetFormatError,
            RetryExhaustedError,
        ]
        codes = [cls.error_code for cls in error_classes]
        assert len(codes) == len(set(codes)), "Duplicate error codes found"

    def test_specific_error_codes(self) -> None:
        assert ConfigurationError.error_code == "CONFIG_ERROR"
        assert ProviderRateLimitError.error_code == "PROVIDER_RATE_LIMIT_ERROR"
        assert MetricComputationError.error_code == "METRIC_COMPUTATION_ERROR"

    def test_catching_broad_exception(self) -> None:
        """Verify broad catch works for all framework errors."""
        with pytest.raises(SarathiError):
            raise ProviderRateLimitError(message="Rate limited")

    def test_catching_narrow_exception(self) -> None:
        """Verify narrow catch works for specific errors."""
        with pytest.raises(ProviderRateLimitError):
            raise ProviderRateLimitError(message="Rate limited")
