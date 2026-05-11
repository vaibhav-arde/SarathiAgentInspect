"""Global Concurrency and Rate Limiting.

Provides a centralized mechanism to throttle LLM calls and other 
external resource interactions across the entire framework.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field


@dataclass
class RateLimitConfig:
    """Configuration for the rate limiter."""
    requests_per_minute: int = 60
    burst: int = 10


class GlobalRateLimiter:
    """Centralized rate limiter using a token bucket algorithm.
    
    Ensures that multiple parallel evaluations respect a global throughput limit.
    """

    _instance: GlobalRateLimiter | None = None

    def __new__(cls, *args: Any, **kwargs: Any) -> GlobalRateLimiter:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: RateLimitConfig | None = None) -> None:
        if getattr(self, "_initialized", False):
            return
            
        self.config = config or RateLimitConfig()
        self.tokens = float(self.config.burst)
        self.last_update = time.monotonic()
        self.lock = asyncio.Lock()
        self._initialized = True

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
    """Manages global semaphore and rate limits."""
    
    def __init__(self, max_concurrent: int = 10, rpm: int = 60) -> None:
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limiter = GlobalRateLimiter(RateLimitConfig(requests_per_minute=rpm))

    async def throttle(self) -> Any:
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
