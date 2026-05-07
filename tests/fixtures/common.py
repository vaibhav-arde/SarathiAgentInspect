"""Common reusable test fixtures.

Provides factory fixtures for creating test data objects
used across multiple test modules.
"""

from __future__ import annotations

import pytest

from sarathi_agent_inspect.core.config.settings import (
    EvaluationSettings,
    FrameworkSettings,
    LoggingSettings,
    ProviderSettings,
    RetrySettings,
    SarathiSettings,
)


@pytest.fixture
def default_framework_settings() -> FrameworkSettings:
    """Create default framework settings for testing."""
    return FrameworkSettings(
        name="TestFramework",
        version="0.0.1",
        debug=True,
    )


@pytest.fixture
def default_retry_settings() -> RetrySettings:
    """Create default retry settings for testing."""
    return RetrySettings(
        max_attempts=2,
        backoff_multiplier=1.0,
        max_delay=5,
        jitter=False,
    )


@pytest.fixture
def default_evaluation_settings() -> EvaluationSettings:
    """Create default evaluation settings for testing."""
    return EvaluationSettings(
        default_threshold=0.8,
        max_workers=2,
        batch_size=5,
        fail_fast=True,
    )


@pytest.fixture
def minimal_settings() -> SarathiSettings:
    """Create minimal SarathiSettings with all defaults."""
    return SarathiSettings()


@pytest.fixture
def full_settings() -> SarathiSettings:
    """Create fully populated SarathiSettings for testing."""
    return SarathiSettings(
        framework=FrameworkSettings(
            name="TestSarathi",
            version="0.0.1",
            debug=True,
        ),
        logging=LoggingSettings(
            level="DEBUG",
            format="console",
            include_timestamp=True,
            include_caller=True,
        ),
        provider=ProviderSettings(
            default="ollama",
            timeout=60,
            max_retries=2,
        ),
        retry=RetrySettings(
            max_attempts=2,
            backoff_multiplier=1.5,
            max_delay=10,
            jitter=False,
        ),
        evaluation=EvaluationSettings(
            default_threshold=0.8,
            max_workers=2,
            batch_size=5,
            fail_fast=True,
        ),
    )
