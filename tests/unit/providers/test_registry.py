"""Unit tests for the ProviderRegistry and ProviderFactory."""

from collections.abc import AsyncIterator
from typing import Any

import pytest
from pydantic import BaseModel

from sarathi_agent_inspect.core.config.settings import SarathiSettings
from sarathi_agent_inspect.core.exceptions.base import ConfigurationError
from sarathi_agent_inspect.providers.base import BaseProvider, ModelInfo, ProviderResponse
from sarathi_agent_inspect.providers.registry import ProviderFactory, ProviderRegistry, register_provider


class MockProviderSettings(BaseModel):
    model: str = "test-model"


class MockSettings(BaseModel):
    class Provider(BaseModel):
        default: str = "mock"
        timeout: int = 10
        mock_provider: MockProviderSettings = MockProviderSettings()

    provider: Provider = Provider()


@pytest.fixture
def mock_settings() -> Any:
    return MockSettings()


@register_provider("mock")
class MockProvider(BaseProvider):
    """Mock provider for testing."""

    def __init__(self, settings: Any, **kwargs: Any) -> None:
        super().__init__(settings, **kwargs)
        self.is_initialized = False

    @property
    def provider_name(self) -> str:
        return "mock"

    @property
    def model_name(self) -> str:
        return "test-model"

    async def initialize(self) -> None:
        self.is_initialized = True

    async def shutdown(self) -> None:
        self.is_initialized = False

    async def health_check(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            provider=self.provider_name, model=self.model_name, supports_streaming=False, supports_tools=False
        )

    async def generate(self, prompt: str, **kwargs: Any) -> ProviderResponse:
        return ProviderResponse(
            content="mock response",
            model=self.model_name,
            provider=self.provider_name,
            prompt_tokens=10,
            completion_tokens=10,
            total_tokens=20,
            latency_ms=100.0,
            cost_usd=0.0,
            finish_reason="stop",
            raw_response={},
        )

    async def generate_stream(self, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        yield "mock response"

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        if isinstance(text, str):
            return [0.1] * 128
        return [[0.1] * 128 for _ in text]

    def get_token_count(self, text: str) -> int:
        return len(text.split())

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        return 0.0


def test_registry_registration() -> None:
    """Test registering and retrieving a provider."""
    providers = ProviderRegistry.list_providers()
    assert "mock" in providers

    cls = ProviderRegistry.get("mock")
    assert cls is MockProvider


def test_registry_unknown_provider() -> None:
    """Test retrieving an unknown provider."""
    with pytest.raises(ConfigurationError) as exc_info:
        ProviderRegistry.get("unknown")
    assert "Unknown provider" in str(exc_info.value)


def test_factory_create(mock_settings: Any) -> None:
    """Test factory creates a provider."""
    # We cast to SarathiSettings for the factory type hint
    provider = ProviderFactory.create(settings=mock_settings)
    assert isinstance(provider, MockProvider)
    assert provider.provider_name == "mock"


def test_factory_override(mock_settings: Any) -> None:
    """Test factory with provider name override."""

    @register_provider("mock2")
    class MockProvider2(MockProvider):
        @property
        def provider_name(self) -> str:
            return "mock2"

    provider = ProviderFactory.create(settings=mock_settings, provider_name="mock2")
    assert isinstance(provider, MockProvider2)
    assert provider.provider_name == "mock2"


def test_factory_rejects_unready_provider() -> None:
    """Stub providers should fail fast unless explicitly allowed."""
    settings = SarathiSettings()
    settings.provider.default = "anthropic"

    with pytest.raises(ConfigurationError, match="not ready for use"):
        ProviderFactory.create(settings=settings)


def test_factory_can_opt_into_unready_provider() -> None:
    settings = SarathiSettings()
    settings.provider.default = "anthropic"

    provider = ProviderFactory.create(settings=settings, allow_unavailable=True)
    assert provider.provider_name == "anthropic"
