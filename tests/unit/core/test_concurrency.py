"""Unit tests for concurrency and rate limiting utilities."""

from __future__ import annotations

import pytest

from sarathi_agent_inspect.core.concurrency import ConcurrencyManager, GlobalRateLimiter


@pytest.fixture(autouse=True)
def reset_limiter_registry() -> None:
    """Reset shared rate limiter state between tests."""
    GlobalRateLimiter.reset_registry()


def test_rate_limiter_reuses_matching_scope() -> None:
    manager_a = ConcurrencyManager(max_concurrent=2, rpm=30, scope="metrics")
    manager_b = ConcurrencyManager(max_concurrent=4, rpm=30, scope="metrics")

    assert manager_a.rate_limiter is manager_b.rate_limiter


def test_rate_limiter_rejects_conflicting_scope_config() -> None:
    ConcurrencyManager(max_concurrent=2, rpm=30, scope="shared")

    with pytest.raises(ValueError, match="scope 'shared'"):
        ConcurrencyManager(max_concurrent=2, rpm=60, scope="shared")


def test_rate_limiter_allows_independent_scopes() -> None:
    manager_a = ConcurrencyManager(max_concurrent=2, rpm=30, scope="metrics")
    manager_b = ConcurrencyManager(max_concurrent=2, rpm=60, scope="providers")

    assert manager_a.rate_limiter is not manager_b.rate_limiter
