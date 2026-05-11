"""Global Concurrency and Rate Limiting.

Provides a centralized mechanism to throttle LLM calls and other
external resource interactions across the entire framework.
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
from typing import Any, ClassVar


@dataclass
class RateLimitConfig:
    """Configuration for the rate limiter."""

    requests_per_minute: int = 60
    burst: int = 10


class GlobalRateLimiter:
    """Scoped rate limiter using a token bucket algorithm.

    Limiters are shared intentionally by scope so call sites can opt into
    global coordination without silently inheriting another subsystem's
    configuration.
    """

    _registry: ClassVar[dict[str, GlobalRateLimiter]] = {}
    _registry_lock: ClassVar[threading.Lock] = threading.Lock()

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        self.config = config or RateLimitConfig()
        self.tokens = float(self.config.burst)
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()

    @classmethod
    def for_scope(cls, scope: str, config: RateLimitConfig | None = None) -> GlobalRateLimiter:
        """Return a shared limiter for the given scope.

        Reusing the same scope with a different configuration is treated as
        a programming error to avoid hidden cross-subsystem coupling.
        """
        resolved_config = config or RateLimitConfig()

        with cls._registry_lock:
            existing = cls._registry.get(scope)
            if existing is not None:
                if existing.config != resolved_config:
                    raise ValueError(
                        f"Rate limiter scope '{scope}' already exists with "
                        f"{existing.config.requests_per_minute} rpm / {existing.config.burst} burst."
                    )
                return existing

            limiter = cls(config=resolved_config)
            cls._registry[scope] = limiter
            return limiter

    @classmethod
    def reset_registry(cls) -> None:
        """Clear shared limiter state. Primarily for tests."""
        with cls._registry_lock:
            cls._registry.clear()

    async def acquire(self) -> None:
        """Acquire a token from the bucket, waiting if necessary."""
        async with self.lock:
            while self.tokens < 1:
                self._refill()
                if self.tokens < 1:
                    # Calculate wait time
                    wait_time = (1 - self.tokens) / (self.config.requests_per_minute / 60.0)
                    await asyncio.sleep(wait_time)

            self._refill()
            self.tokens -= 1

    def _refill(self) -> None:
        """Refill tokens based on elapsed time."""
        now = time.monotonic()
        elapsed = now - self.last_update
        refill_amount = elapsed * (self.config.requests_per_minute / 60.0)
        self.tokens = min(float(self.config.burst), self.tokens + refill_amount)
        self.last_update = now


class ConcurrencyManager:
    """Manages semaphore and optional shared rate limits."""

    def __init__(self, max_concurrent: int = 10, rpm: int = 60, scope: str | None = None) -> None:
        self.semaphore = asyncio.Semaphore(max_concurrent)
        config = RateLimitConfig(requests_per_minute=rpm)
        if scope is None:
            self.rate_limiter = GlobalRateLimiter(config)
        else:
            self.rate_limiter = GlobalRateLimiter.for_scope(scope, config)

    def throttle(self) -> Throttler:
        """Context manager to apply both concurrency and rate limits."""
        return Throttler(self.semaphore, self.rate_limiter)


class Throttler:
    """Combined context manager for semaphore and rate limiter."""

    def __init__(self, semaphore: asyncio.Semaphore, limiter: GlobalRateLimiter) -> None:
        self.semaphore = semaphore
        self.limiter = limiter

    async def __aenter__(self) -> None:
        await self.semaphore.acquire()
        await self.limiter.acquire()

    async def __aexit__(self, *args: Any) -> None:
        self.semaphore.release()
