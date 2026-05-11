"""Configuration loader with layered resolution.

Resolution order (highest priority wins):
    1. Environment variables
    2. .env file values
    3. Environment-specific YAML (e.g., local.yaml)
    4. Default YAML (default.yaml)

Thread-safe singleton pattern ensures a single config instance
across the entire framework lifecycle.
"""

from __future__ import annotations

import os
import threading
from typing import TYPE_CHECKING, Any

import structlog
import yaml
from dotenv import load_dotenv

from sarathi_agent_inspect.core.config.constants import (
    CONFIGS_DIR,
    DEFAULT_CONFIG_FILE,
    ENV_LOCAL,
    ENV_VAR_SARATHI_ENV,
    SUPPORTED_ENVIRONMENTS,
)
from sarathi_agent_inspect.core.config.settings import SarathiSettings
from sarathi_agent_inspect.core.exceptions.base import ConfigurationError

if TYPE_CHECKING:
    from pathlib import Path

logger = structlog.get_logger(__name__)

# Module-level singleton
_config_instance: SarathiSettings | None = None
_config_lock = threading.Lock()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override dict into base dict.

    Args:
        base: Base dictionary (lower priority).
        override: Override dictionary (higher priority).

    Returns:
        Merged dictionary with override values taking precedence.
    """
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_file(path: Path) -> dict[str, Any]:
    """Load and parse a YAML file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ConfigurationError: If the file cannot be read or parsed.
    """
    try:
        with path.open("r") as f:
            content = yaml.safe_load(f)
            return content if isinstance(content, dict) else {}
    except FileNotFoundError:
        logger.warning("config_file_not_found", path=str(path))
        return {}
    except yaml.YAMLError as e:
        raise ConfigurationError(
            message=f"Failed to parse YAML config: {path}",
            context={"path": str(path), "error": str(e)},
        ) from e


class ConfigLoader:
    """Loads and merges framework configuration from multiple sources.

    Usage:
        loader = ConfigLoader(env="local")
        settings = loader.load()

        # Or use the module-level convenience function:
        settings = get_config()
    """

    def __init__(
        self,
        env: str | None = None,
        config_dir: Path | None = None,
        dotenv_path: Path | None = None,
    ) -> None:
        """Initialize the config loader.

        Args:
            env: Target environment name. Defaults to SARATHI_ENV or 'local'.
            config_dir: Path to the configs directory. Defaults to project configs/.
            dotenv_path: Path to .env file. Defaults to project root .env.
        """
        env_value = env or os.getenv(ENV_VAR_SARATHI_ENV) or ENV_LOCAL
        self._env: str = env_value
        self._config_dir = config_dir or CONFIGS_DIR
        self._dotenv_path = dotenv_path

        if self._env not in SUPPORTED_ENVIRONMENTS:
            logger.warning(
                "unsupported_environment",
                env=self._env,
                supported=list(SUPPORTED_ENVIRONMENTS),
            )

    @property
    def environment(self) -> str:
        """Return the current environment name."""
        return self._env

    def load(self) -> SarathiSettings:
        """Load configuration with full layered resolution.

        Returns:
            Fully resolved and validated SarathiSettings instance.

        Raises:
            ConfigurationError: If configuration validation fails.
        """
        # Step 1: Load .env file (populates os.environ)
        if self._dotenv_path:
            load_dotenv(self._dotenv_path, override=True)
        else:
            load_dotenv(override=True)

        # Step 2: Load default YAML
        default_path = self._config_dir / DEFAULT_CONFIG_FILE
        config_data = _load_yaml_file(default_path)

        # Step 3: Load environment-specific YAML and merge
        env_path = self._config_dir / f"{self._env}.yaml"
        if env_path.exists():
            env_data = _load_yaml_file(env_path)
            config_data = _deep_merge(config_data, env_data)

        # Step 4: Apply environment variable overrides
        config_data = self._apply_env_overrides(config_data)

        # Step 5: Validate with Pydantic
        try:
            settings = SarathiSettings.model_validate(config_data)
        except Exception as e:
            raise ConfigurationError(
                message="Configuration validation failed",
                context={"env": self._env, "error": str(e)},
            ) from e

        logger.info(
            "config_loaded",
            env=self._env,
            debug=settings.framework.debug,
            provider=settings.provider.default,
        )

        return settings

    def _apply_env_overrides(self, config_data: dict[str, Any]) -> dict[str, Any]:
        """Apply environment variable overrides to config data.

        Maps well-known environment variables to their config paths.

        Args:
            config_data: Current configuration dictionary.

        Returns:
            Configuration with environment variable overrides applied.
        """
        env_mappings: list[tuple[str, list[str], type]] = [
            ("SARATHI_DEBUG", ["framework", "debug"], bool),
            ("SARATHI_LOG_LEVEL", ["logging", "level"], str),
            ("OLLAMA_BASE_URL", ["provider", "ollama", "base_url"], str),
            ("SARATHI_PROVIDER__OLLAMA__BASE_URL", ["provider", "ollama", "base_url"], str),
            ("OLLAMA_MODEL", ["provider", "ollama", "model"], str),
            ("SARATHI_PROVIDER__OLLAMA__MODEL", ["provider", "ollama", "model"], str),
            ("OPENAI_API_KEY", ["provider", "openai", "api_key"], str),
            ("OPENAI_MODEL", ["provider", "openai", "model"], str),
            ("GEMINI_API_KEY", ["provider", "gemini", "api_key"], str),
            ("GEMINI_MODEL", ["provider", "gemini", "model"], str),
            ("ANTHROPIC_API_KEY", ["provider", "anthropic", "api_key"], str),
            ("ANTHROPIC_MODEL", ["provider", "anthropic", "model"], str),
            ("AZURE_OPENAI_API_KEY", ["provider", "azure_openai", "api_key"], str),
            ("AZURE_OPENAI_ENDPOINT", ["provider", "azure_openai", "endpoint"], str),
            ("AZURE_OPENAI_DEPLOYMENT", ["provider", "azure_openai", "deployment"], str),
            ("SARATHI_JUDGE_PROVIDER", ["judge", "provider"], str),
            ("SARATHI_JUDGE_MODEL", ["judge", "model"], str),
            ("SARATHI_JUDGE__MODEL", ["judge", "model"], str),
            ("SARATHI_DEFAULT_THRESHOLD", ["evaluation", "default_threshold"], float),
            ("SARATHI_MAX_WORKERS", ["evaluation", "max_workers"], int),
            ("SARATHI_RETRY_MAX_ATTEMPTS", ["retry", "max_attempts"], int),
        ]

        for env_var, path, cast_type in env_mappings:
            value = os.getenv(env_var)
            if value is not None:
                converted = self._cast_value(value, cast_type)
                self._set_nested(config_data, path, converted)

        return config_data

    @staticmethod
    def _cast_value(value: str, cast_type: type) -> Any:
        """Cast string environment variable to target type.

        Args:
            value: String value from environment.
            cast_type: Target Python type.

        Returns:
            Cast value.
        """
        if cast_type is bool:
            return value.lower() in ("true", "1", "yes")
        return cast_type(value)

    @staticmethod
    def _set_nested(data: dict[str, Any], path: list[str], value: Any) -> None:
        """Set a value in a nested dictionary using a key path.

        Args:
            data: Target dictionary.
            path: List of keys forming the path.
            value: Value to set.
        """
        current = data
        for key in path[:-1]:
            current = current.setdefault(key, {})
        current[path[-1]] = value


def get_config(
    env: str | None = None,
    force_reload: bool = False,
) -> SarathiSettings:
    """Get the framework configuration (singleton).

    Thread-safe convenience function that returns the cached
    configuration instance, creating it on first call.

    Args:
        env: Override the environment name.
        force_reload: Force reloading configuration from sources.

    Returns:
        Validated SarathiSettings instance.
    """
    global _config_instance

    if _config_instance is not None and not force_reload and env is None:
        return _config_instance

    with _config_lock:
        if _config_instance is not None and not force_reload and env is None:
            return _config_instance

        loader = ConfigLoader(env=env)
        _config_instance = loader.load()
        return _config_instance


def reset_config() -> None:
    """Reset the singleton config instance. Primarily for testing."""
    global _config_instance
    with _config_lock:
        _config_instance = None
