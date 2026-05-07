"""Pydantic v2 settings models for the framework.

Implements a typed, validated configuration system with
environment variable support and layered defaults.

Architecture:
    SarathiSettings (root)
    ├── FrameworkSettings
    ├── LoggingSettings
    ├── ProviderSettings
    │   └── OllamaProviderSettings
    ├── JudgeSettings
    ├── RetrySettings
    └── EvaluationSettings
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class FrameworkSettings(BaseModel):
    """Top-level framework configuration."""

    name: str = Field(default="SarathiAgentInspect", description="Framework name")
    version: str = Field(default="0.1.0", description="Framework version")
    debug: bool = Field(default=False, description="Enable debug mode")


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Log level",
    )
    format: Literal["json", "console"] = Field(
        default="json",
        description="Log output format: 'json' for structured, 'console' for human-readable",
    )
    include_timestamp: bool = Field(default=True, description="Include timestamp in log output")
    include_caller: bool = Field(default=False, description="Include caller info in log output")


class OllamaProviderSettings(BaseModel):
    """Ollama-specific provider configuration."""

    base_url: str = Field(default="http://localhost:11434", description="Ollama API base URL")
    model: str = Field(default="gemma4:31b-cloud", description="Default Ollama model")


class OpenAIProviderSettings(BaseModel):
    """OpenAI-specific provider configuration."""

    api_key: str = Field(default="", description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="Default OpenAI model")
    base_url: str = Field(default="", description="Optional base URL for OpenAI-compatible APIs")


class GeminiProviderSettings(BaseModel):
    """Gemini-specific provider configuration."""

    api_key: str = Field(default="", description="Google Gemini API key")
    model: str = Field(default="gemini-2.5-flash", description="Default Gemini model")


class AnthropicProviderSettings(BaseModel):
    """Anthropic-specific provider configuration."""

    api_key: str = Field(default="", description="Anthropic API key")
    model: str = Field(default="claude-sonnet-4-20250514", description="Default Anthropic model")


class AzureOpenAIProviderSettings(BaseModel):
    """Azure OpenAI-specific provider configuration."""

    api_key: str = Field(default="", description="Azure OpenAI API key")
    endpoint: str = Field(default="", description="Azure OpenAI endpoint")
    deployment: str = Field(default="", description="Azure OpenAI deployment name")
    api_version: str = Field(default="2024-02-01", description="Azure OpenAI API version")


class ProviderSettings(BaseModel):
    """LLM provider configuration."""

    default: str = Field(default="ollama", description="Default provider name")
    timeout: int = Field(default=120, ge=1, description="Request timeout in seconds")
    max_retries: int = Field(default=3, ge=0, description="Max retries for provider calls")
    ollama: OllamaProviderSettings = Field(default_factory=OllamaProviderSettings)
    openai: OpenAIProviderSettings = Field(default_factory=OpenAIProviderSettings)
    gemini: GeminiProviderSettings = Field(default_factory=GeminiProviderSettings)
    anthropic: AnthropicProviderSettings = Field(default_factory=AnthropicProviderSettings)
    azure_openai: AzureOpenAIProviderSettings = Field(default_factory=AzureOpenAIProviderSettings)


class JudgeSettings(BaseModel):
    """Judge model configuration for LLM-as-a-Judge evaluations."""

    provider: str = Field(default="ollama", description="Provider for judge model")
    model: str = Field(default="gemma4:31b-cloud", description="Judge model name")
    timeout: int = Field(default=180, ge=1, description="Judge model timeout in seconds")


class RetrySettings(BaseModel):
    """Retry strategy configuration."""

    max_attempts: int = Field(default=3, ge=1, description="Maximum retry attempts")
    backoff_multiplier: float = Field(default=2.0, gt=0, description="Exponential backoff multiplier")
    max_delay: int = Field(default=60, ge=1, description="Maximum delay between retries in seconds")
    jitter: bool = Field(default=True, description="Add random jitter to retry delays")
    retry_on: list[str] = Field(
        default_factory=lambda: ["rate_limit", "timeout", "connection_error"],
        description="Error types to retry on",
    )


class EvaluationSettings(BaseModel):
    """Evaluation execution configuration."""

    default_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="Default pass/fail threshold")
    max_workers: int = Field(default=4, ge=1, description="Maximum parallel evaluation workers")
    batch_size: int = Field(default=10, ge=1, description="Batch size for bulk evaluations")
    fail_fast: bool = Field(default=False, description="Stop on first failure")


class DatasetSettings(BaseModel):
    """Dataset management configuration."""

    base_path: str = Field(default="./datasets", description="Base path for dataset storage")
    cache_enabled: bool = Field(default=True, description="Enable dataset caching")
    cache_path: str = Field(default="./.cache/datasets", description="Cache directory path")


class ReportingSettings(BaseModel):
    """Reporting configuration."""

    output_path: str = Field(default="./reports", description="Report output directory")
    formats: list[str] = Field(default_factory=lambda: ["json"], description="Report output formats")


class SarathiSettings(BaseModel):
    """Root configuration model aggregating all sub-settings.

    This is the single source of truth for all framework configuration.
    It is populated by the ConfigLoader which merges YAML files,
    environment variables, and .env files.
    """

    framework: FrameworkSettings = Field(default_factory=FrameworkSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)
    provider: ProviderSettings = Field(default_factory=ProviderSettings)
    judge: JudgeSettings = Field(default_factory=JudgeSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    evaluation: EvaluationSettings = Field(default_factory=EvaluationSettings)
    datasets: DatasetSettings = Field(default_factory=DatasetSettings)
    reporting: ReportingSettings = Field(default_factory=ReportingSettings)
