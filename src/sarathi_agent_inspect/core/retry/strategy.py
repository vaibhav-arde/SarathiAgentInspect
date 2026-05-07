"""Retry strategies using tenacity.

Provides pre-configured retry strategies for different operation types:
- LLM calls: Retry on rate limits, timeouts, connection errors
- API calls: Retry on transient HTTP errors
- File I/O: Retry on permission/lock errors

Each strategy supports:
- Exponential backoff with optional jitter
- Configurable max attempts and delays
- Logging on each retry attempt
- Custom retry predicates
"""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING, ParamSpec, TypeVar

import structlog
from tenacity import (
    AsyncRetrying,
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from sarathi_agent_inspect.core.exceptions.base import (
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    RetryExhaustedError,
)

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from sarathi_agent_inspect.core.config.settings import RetrySettings

logger = structlog.get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


def _log_retry(retry_state: RetryCallState) -> None:
    """Log each retry attempt with context.

    Args:
        retry_state: Tenacity retry state containing attempt info.
    """
    exception = retry_state.outcome.exception() if retry_state.outcome else None
    logger.warning(
        "retry_attempt",
        attempt=retry_state.attempt_number,
        exception_type=type(exception).__name__ if exception else None,
        exception_msg=str(exception) if exception else None,
        wait_seconds=retry_state.next_action.sleep if retry_state.next_action else None,
    )


class RetryStrategy:
    """Configurable retry strategy wrapping tenacity.

    Attributes:
        max_attempts: Maximum number of retry attempts.
        backoff_multiplier: Base multiplier for exponential backoff.
        max_delay: Maximum delay between retries in seconds.
        jitter: Whether to add random jitter to delays.
        retry_exceptions: Exception types to retry on.
    """

    def __init__(
        self,
        max_attempts: int = 3,
        backoff_multiplier: float = 2.0,
        max_delay: int = 60,
        jitter: bool = True,
        retry_exceptions: tuple[type[Exception], ...] | None = None,
    ) -> None:
        self.max_attempts = max_attempts
        self.backoff_multiplier = backoff_multiplier
        self.max_delay = max_delay
        self.jitter = jitter
        self.retry_exceptions = retry_exceptions or (
            ProviderConnectionError,
            ProviderTimeoutError,
            ProviderRateLimitError,
        )

    @classmethod
    def from_settings(cls, settings: RetrySettings) -> RetryStrategy:
        """Create a RetryStrategy from Pydantic settings.

        Args:
            settings: Retry configuration settings.

        Returns:
            Configured RetryStrategy instance.
        """
        return cls(
            max_attempts=settings.max_attempts,
            backoff_multiplier=settings.backoff_multiplier,
            max_delay=settings.max_delay,
            jitter=settings.jitter,
        )

    def wrap(self, func: Callable[P, T]) -> Callable[P, T]:
        """Wrap a synchronous function with retry logic.

        Args:
            func: The function to wrap.

        Returns:
            Wrapped function with retry behavior.
        """

        @retry(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(
                initial=1,
                max=self.max_delay,
                jitter=self.max_delay // 4 if self.jitter else 0,
            ),
            retry=retry_if_exception_type(self.retry_exceptions),
            before_sleep=_log_retry,
            reraise=True,
        )
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            return func(*args, **kwargs)

        return wrapper

    def async_retrying(self) -> AsyncRetrying:
        """Create an AsyncRetrying instance for async operations.

        Returns:
            Configured AsyncRetrying for use in async for loops.

        Example:
            async for attempt in strategy.async_retrying():
                with attempt:
                    result = await some_async_call()
        """
        return AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait_exponential_jitter(
                initial=1,
                max=self.max_delay,
                jitter=self.max_delay // 4 if self.jitter else 0,
            ),
            retry=retry_if_exception_type(self.retry_exceptions),
            before_sleep=_log_retry,
            reraise=True,
        )


def create_llm_retry(settings: RetrySettings | None = None) -> RetryStrategy:
    """Create a retry strategy optimized for LLM API calls.

    Args:
        settings: Optional retry settings. Uses defaults if None.

    Returns:
        RetryStrategy configured for LLM operations.
    """
    if settings:
        return RetryStrategy.from_settings(settings)
    return RetryStrategy(
        max_attempts=3,
        backoff_multiplier=2.0,
        max_delay=60,
        jitter=True,
        retry_exceptions=(
            ProviderConnectionError,
            ProviderTimeoutError,
            ProviderRateLimitError,
        ),
    )


def create_api_retry(settings: RetrySettings | None = None) -> RetryStrategy:
    """Create a retry strategy for general API calls.

    Args:
        settings: Optional retry settings. Uses defaults if None.

    Returns:
        RetryStrategy configured for API operations.
    """
    if settings:
        strategy = RetryStrategy.from_settings(settings)
        strategy.retry_exceptions = (ConnectionError, TimeoutError)
        return strategy
    return RetryStrategy(
        max_attempts=3,
        backoff_multiplier=1.5,
        max_delay=30,
        jitter=True,
        retry_exceptions=(ConnectionError, TimeoutError),
    )


def create_file_retry() -> RetryStrategy:
    """Create a retry strategy for file I/O operations.

    Returns:
        RetryStrategy configured for file operations.
    """
    return RetryStrategy(
        max_attempts=2,
        backoff_multiplier=1.0,
        max_delay=5,
        jitter=False,
        retry_exceptions=(PermissionError, OSError),
    )


def with_retry(
    max_attempts: int = 3,
    retry_exceptions: tuple[type[Exception], ...] | None = None,
) -> Callable[[Callable[P, Awaitable[T]]], Callable[P, Awaitable[T]]]:
    """Decorator for adding retry logic to async functions.

    Args:
        max_attempts: Maximum number of retry attempts.
        retry_exceptions: Exception types to retry on.

    Returns:
        Decorator function.

    Example:
        @with_retry(max_attempts=3)
        async def call_llm(prompt: str) -> str:
            ...
    """
    exceptions = retry_exceptions or (
        ProviderConnectionError,
        ProviderTimeoutError,
        ProviderRateLimitError,
    )

    def decorator(func: Callable[P, Awaitable[T]]) -> Callable[P, Awaitable[T]]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            strategy = RetryStrategy(
                max_attempts=max_attempts,
                retry_exceptions=exceptions,
            )
            last_exception: Exception | None = None
            async for attempt in strategy.async_retrying():
                with attempt:
                    return await func(*args, **kwargs)
            # This should not be reached due to reraise=True, but as a safety net:
            raise RetryExhaustedError(  # pragma: no cover
                message=f"All {max_attempts} retry attempts exhausted for {func.__name__}",
                context={"function": func.__name__, "last_error": str(last_exception)},
            )

        return wrapper

    return decorator
