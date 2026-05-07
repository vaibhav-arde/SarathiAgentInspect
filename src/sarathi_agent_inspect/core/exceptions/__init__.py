"""Exception subsystem — structured exception hierarchy."""

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

__all__ = [
    "ConfigurationError",
    "DatasetError",
    "DatasetFormatError",
    "DatasetLoadError",
    "DatasetValidationError",
    "EvaluationError",
    "MetricComputationError",
    "ProviderAuthenticationError",
    "ProviderConnectionError",
    "ProviderError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "RetryExhaustedError",
    "SarathiError",
    "ThresholdViolationError",
]
