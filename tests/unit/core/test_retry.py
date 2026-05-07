"""Unit tests for the retry strategy system.

Tests:
- RetryStrategy creation
- Creating from settings
- Pre-built strategies (LLM, API, file)
- Sync retry wrapping behavior
"""

from __future__ import annotations

import pytest

from sarathi_agent_inspect.core.config.settings import RetrySettings
from sarathi_agent_inspect.core.exceptions.base import (
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sarathi_agent_inspect.core.retry.strategy import (
    RetryStrategy,
    create_api_retry,
    create_file_retry,
    create_llm_retry,
)


class TestRetryStrategy:
    """Tests for RetryStrategy configuration."""

    def test_default_construction(self) -> None:
        strategy = RetryStrategy()
        assert strategy.max_attempts == 3
        assert strategy.backoff_multiplier == 2.0
        assert strategy.max_delay == 60
        assert strategy.jitter is True

    def test_custom_construction(self) -> None:
        strategy = RetryStrategy(
            max_attempts=5,
            backoff_multiplier=3.0,
            max_delay=120,
            jitter=False,
        )
        assert strategy.max_attempts == 5
        assert strategy.backoff_multiplier == 3.0
        assert strategy.max_delay == 120
        assert strategy.jitter is False

    def test_from_settings(self) -> None:
        settings = RetrySettings(
            max_attempts=4,
            backoff_multiplier=1.5,
            max_delay=30,
            jitter=True,
        )
        strategy = RetryStrategy.from_settings(settings)
        assert strategy.max_attempts == 4
        assert strategy.backoff_multiplier == 1.5
        assert strategy.max_delay == 30
        assert strategy.jitter is True

    def test_default_retry_exceptions(self) -> None:
        strategy = RetryStrategy()
        assert ProviderConnectionError in strategy.retry_exceptions
        assert ProviderTimeoutError in strategy.retry_exceptions
        assert ProviderRateLimitError in strategy.retry_exceptions

    def test_custom_retry_exceptions(self) -> None:
        strategy = RetryStrategy(
            retry_exceptions=(ConnectionError, TimeoutError),
        )
        assert ConnectionError in strategy.retry_exceptions
        assert TimeoutError in strategy.retry_exceptions
        assert ProviderConnectionError not in strategy.retry_exceptions


class TestSyncRetry:
    """Tests for synchronous retry wrapping."""

    def test_wrap_succeeds_without_retry(self) -> None:
        strategy = RetryStrategy(max_attempts=3)
        call_count = 0

        def my_func() -> str:
            nonlocal call_count
            call_count += 1
            return "success"

        wrapped = strategy.wrap(my_func)
        result = wrapped()
        assert result == "success"
        assert call_count == 1

    def test_wrap_retries_on_matching_exception(self) -> None:
        strategy = RetryStrategy(
            max_attempts=3,
            backoff_multiplier=0.01,
            max_delay=1,
            jitter=False,
        )
        call_count = 0

        def my_func() -> str:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ProviderConnectionError(message="Connection failed")
            return "success"

        wrapped = strategy.wrap(my_func)
        result = wrapped()
        assert result == "success"
        assert call_count == 3

    def test_wrap_raises_after_max_attempts(self) -> None:
        strategy = RetryStrategy(
            max_attempts=2,
            backoff_multiplier=0.01,
            max_delay=1,
            jitter=False,
        )

        def my_func() -> str:
            raise ProviderTimeoutError(message="Timeout")

        wrapped = strategy.wrap(my_func)
        with pytest.raises(ProviderTimeoutError):
            wrapped()

    def test_wrap_does_not_retry_on_non_matching_exception(self) -> None:
        strategy = RetryStrategy(max_attempts=3)
        call_count = 0

        def my_func() -> str:
            nonlocal call_count
            call_count += 1
            raise ValueError("Not a provider error")

        wrapped = strategy.wrap(my_func)
        with pytest.raises(ValueError, match="Not a provider error"):
            wrapped()
        assert call_count == 1


class TestPrebuiltStrategies:
    """Tests for pre-built retry strategy factories."""

    def test_create_llm_retry_defaults(self) -> None:
        strategy = create_llm_retry()
        assert strategy.max_attempts == 3
        assert strategy.max_delay == 60
        assert ProviderConnectionError in strategy.retry_exceptions

    def test_create_llm_retry_from_settings(self) -> None:
        settings = RetrySettings(max_attempts=5, max_delay=120)
        strategy = create_llm_retry(settings)
        assert strategy.max_attempts == 5
        assert strategy.max_delay == 120

    def test_create_api_retry_defaults(self) -> None:
        strategy = create_api_retry()
        assert strategy.max_attempts == 3
        assert strategy.max_delay == 30
        assert ConnectionError in strategy.retry_exceptions
        assert TimeoutError in strategy.retry_exceptions

    def test_create_file_retry(self) -> None:
        strategy = create_file_retry()
        assert strategy.max_attempts == 2
        assert strategy.jitter is False
        assert PermissionError in strategy.retry_exceptions
        assert OSError in strategy.retry_exceptions
