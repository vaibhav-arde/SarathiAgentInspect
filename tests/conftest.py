"""Root conftest — shared fixtures for all tests.

This conftest is automatically loaded by pytest for all test modules.
It provides:
- Environment setup (forces SARATHI_ENV=test/local)
- Config loader fixture
- Logger fixture
- Temporary directory fixture
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest

from sarathi_agent_inspect.core.config.loader import ConfigLoader, reset_config
from sarathi_agent_inspect.core.logging.logger import configure_logging

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from sarathi_agent_inspect.core.config.settings import SarathiSettings


@pytest.fixture(autouse=True)
def _reset_config_singleton() -> Generator[None]:
    """Reset the config singleton before each test to ensure isolation."""
    reset_config()
    yield
    reset_config()


@pytest.fixture
def test_env() -> Generator[None]:
    """Set up test environment variables."""
    original_env = os.environ.get("SARATHI_ENV")
    os.environ["SARATHI_ENV"] = "local"
    yield
    if original_env is not None:
        os.environ["SARATHI_ENV"] = original_env
    else:
        os.environ.pop("SARATHI_ENV", None)


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Create a temporary config directory with test YAML files."""
    config_path = tmp_path / "configs"
    config_path.mkdir()

    # Create default.yaml
    default_yaml = config_path / "default.yaml"
    default_yaml.write_text(
        """
framework:
  name: "SarathiAgentInspect"
  version: "0.1.0"
  debug: false

logging:
  level: "INFO"
  format: "json"
  include_timestamp: true
  include_caller: false

provider:
  default: "ollama"
  timeout: 120
  max_retries: 3
  ollama:
    base_url: "http://localhost:11434"
    model: "gemma4:31b-cloud"

judge:
  provider: "ollama"
  model: "gemma4:31b-cloud"
  timeout: 180

retry:
  max_attempts: 3
  backoff_multiplier: 2.0
  max_delay: 60
  jitter: true
  retry_on:
    - "rate_limit"
    - "timeout"
    - "connection_error"

evaluation:
  default_threshold: 0.7
  max_workers: 4
  batch_size: 10
  fail_fast: false
"""
    )

    # Create local.yaml
    local_yaml = config_path / "local.yaml"
    local_yaml.write_text(
        """
framework:
  debug: true

logging:
  level: "DEBUG"
  format: "console"
"""
    )

    return config_path


@pytest.fixture
def sample_settings(config_dir: Path) -> SarathiSettings:
    """Load settings from test config directory."""
    loader = ConfigLoader(env="local", config_dir=config_dir)
    return loader.load()


@pytest.fixture(autouse=True)
def _configure_test_logging() -> None:
    """Configure logging for tests (console format, DEBUG level)."""
    configure_logging(level="DEBUG", log_format="console")
