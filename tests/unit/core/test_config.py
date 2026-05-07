"""Unit tests for the configuration system.

Tests:
- Settings model validation and defaults
- Config loader with YAML merging
- Environment variable overrides
- Environment switching
- Error handling for invalid configs
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import pytest
from pydantic import ValidationError

from sarathi_agent_inspect.core.config.loader import ConfigLoader, _deep_merge
from sarathi_agent_inspect.core.config.settings import (
    EvaluationSettings,
    FrameworkSettings,
    LoggingSettings,
    ProviderSettings,
    RetrySettings,
    SarathiSettings,
)
from sarathi_agent_inspect.core.exceptions.base import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path

# ── Settings Model Tests ─────────────────────────────────────────


class TestFrameworkSettings:
    """Tests for FrameworkSettings Pydantic model."""

    def test_default_values(self) -> None:
        settings = FrameworkSettings()
        assert settings.name == "SarathiAgentInspect"
        assert settings.version == "0.1.0"
        assert settings.debug is False

    def test_custom_values(self) -> None:
        settings = FrameworkSettings(name="Custom", version="2.0.0", debug=True)
        assert settings.name == "Custom"
        assert settings.version == "2.0.0"
        assert settings.debug is True


class TestLoggingSettings:
    """Tests for LoggingSettings Pydantic model."""

    def test_default_values(self) -> None:
        settings = LoggingSettings()
        assert settings.level == "INFO"
        assert settings.format == "json"
        assert settings.include_timestamp is True
        assert settings.include_caller is False

    def test_valid_log_levels(self) -> None:
        for level in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            settings = LoggingSettings(level=level)
            assert settings.level == level

    def test_valid_formats(self) -> None:
        for fmt in ("json", "console"):
            settings = LoggingSettings(format=fmt)
            assert settings.format == fmt


class TestRetrySettings:
    """Tests for RetrySettings Pydantic model."""

    def test_default_values(self) -> None:
        settings = RetrySettings()
        assert settings.max_attempts == 3
        assert settings.backoff_multiplier == 2.0
        assert settings.max_delay == 60
        assert settings.jitter is True

    def test_validation_min_attempts(self) -> None:
        with pytest.raises(ValidationError):
            RetrySettings(max_attempts=0)

    def test_validation_positive_backoff(self) -> None:
        with pytest.raises(ValidationError):
            RetrySettings(backoff_multiplier=-1.0)


class TestEvaluationSettings:
    """Tests for EvaluationSettings Pydantic model."""

    def test_default_values(self) -> None:
        settings = EvaluationSettings()
        assert settings.default_threshold == 0.7
        assert settings.max_workers == 4
        assert settings.batch_size == 10
        assert settings.fail_fast is False

    def test_threshold_range(self) -> None:
        with pytest.raises(ValidationError):
            EvaluationSettings(default_threshold=1.5)

        with pytest.raises(ValidationError):
            EvaluationSettings(default_threshold=-0.1)


class TestSarathiSettings:
    """Tests for the root SarathiSettings model."""

    def test_default_construction(self) -> None:
        settings = SarathiSettings()
        assert isinstance(settings.framework, FrameworkSettings)
        assert isinstance(settings.logging, LoggingSettings)
        assert isinstance(settings.provider, ProviderSettings)
        assert isinstance(settings.retry, RetrySettings)
        assert isinstance(settings.evaluation, EvaluationSettings)

    def test_nested_defaults(self) -> None:
        settings = SarathiSettings()
        assert settings.provider.default == "ollama"
        assert settings.provider.ollama.model == "gemma4:31b-cloud"
        assert settings.provider.ollama.base_url == "http://localhost:11434"


# ── Config Loader Tests ──────────────────────────────────────────


class TestDeepMerge:
    """Tests for the _deep_merge utility."""

    def test_simple_merge(self) -> None:
        base = {"a": 1, "b": 2}
        override = {"b": 3, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3}

    def test_override_replaces_non_dict(self) -> None:
        base = {"a": {"x": 1}}
        override = {"a": "replaced"}
        result = _deep_merge(base, override)
        assert result == {"a": "replaced"}

    def test_empty_override(self) -> None:
        base = {"a": 1}
        result = _deep_merge(base, {})
        assert result == {"a": 1}

    def test_empty_base(self) -> None:
        override = {"a": 1}
        result = _deep_merge({}, override)
        assert result == {"a": 1}


class TestConfigLoader:
    """Tests for the ConfigLoader."""

    def test_load_default_config(self, config_dir: Path) -> None:
        loader = ConfigLoader(env="local", config_dir=config_dir)
        settings = loader.load()
        assert isinstance(settings, SarathiSettings)
        assert settings.framework.name == "SarathiAgentInspect"

    def test_local_overrides_applied(self, config_dir: Path) -> None:
        loader = ConfigLoader(env="local", config_dir=config_dir)
        settings = loader.load()
        # local.yaml overrides debug to true
        assert settings.framework.debug is True
        # local.yaml overrides level to DEBUG
        assert settings.logging.level == "DEBUG"
        # local.yaml overrides format to console
        assert settings.logging.format == "console"

    def test_environment_property(self, config_dir: Path) -> None:
        loader = ConfigLoader(env="ci", config_dir=config_dir)
        assert loader.environment == "ci"

    def test_missing_env_yaml_falls_back(self, config_dir: Path) -> None:
        loader = ConfigLoader(env="staging", config_dir=config_dir)
        # Should still load default.yaml without error
        settings = loader.load()
        assert settings.framework.debug is False

    def test_env_var_override(self, config_dir: Path) -> None:
        os.environ["SARATHI_DEBUG"] = "true"
        try:
            loader = ConfigLoader(env="local", config_dir=config_dir)
            settings = loader.load()
            assert settings.framework.debug is True
        finally:
            os.environ.pop("SARATHI_DEBUG", None)

    def test_invalid_yaml_raises_error(self, tmp_path: Path) -> None:
        bad_config = tmp_path / "configs"
        bad_config.mkdir()
        bad_yaml = bad_config / "default.yaml"
        bad_yaml.write_text("{ invalid: yaml: content: [")

        loader = ConfigLoader(env="local", config_dir=bad_config)
        with pytest.raises(ConfigurationError):
            loader.load()
