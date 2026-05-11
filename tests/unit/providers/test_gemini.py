"""Unit tests for the Gemini provider."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from google.genai.errors import APIError

from sarathi_agent_inspect.core.config.settings import SarathiSettings
from sarathi_agent_inspect.core.exceptions.base import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
)
from sarathi_agent_inspect.providers.gemini import GeminiProvider


@pytest.fixture
def settings() -> Any:
    s = SarathiSettings()
    s.provider.gemini.api_key = "test-key"
    return s


@pytest.fixture
def provider(settings: Any) -> Any:
    return GeminiProvider(settings=settings)


@pytest.mark.asyncio
async def test_health_check_success(provider: Any) -> None:
    """Test successful health check."""
    mock_client = MagicMock()
    mock_client.models.get = MagicMock()

    with patch("sarathi_agent_inspect.providers.gemini.genai.Client", return_value=mock_client):
        assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_health_check_fail(provider: Any) -> None:
    """Test health check failure."""
    mock_client = MagicMock()
    mock_client.models.get = MagicMock(side_effect=Exception("Failed"))

    with patch("sarathi_agent_inspect.providers.gemini.genai.Client", return_value=mock_client):
        assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_generate_success(provider: Any) -> None:
    """Test successful generation."""
    mock_response = MagicMock()
    mock_response.text = "Hello from Gemini"

    mock_candidate = MagicMock()
    mock_candidate.finish_reason.name = "STOP"
    mock_response.candidates = [mock_candidate]

    mock_response.usage_metadata.prompt_token_count = 10
    mock_response.usage_metadata.candidates_token_count = 5
    mock_response.model_dump.return_value = {}

    mock_client = MagicMock()
    mock_client.aio = AsyncMock()
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

    with patch("sarathi_agent_inspect.providers.gemini.genai.Client", return_value=mock_client):
        response = await provider.generate("Hi", temperature=0.7)

    assert response.content == "Hello from Gemini"
    assert response.provider == "gemini"
    assert response.model == "gemini-2.5-flash"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.total_tokens == 15
    assert response.finish_reason == "STOP"


@pytest.mark.asyncio
async def test_generate_auth_error(provider: Any) -> None:
    """Test auth error mapping."""
    mock_client = MagicMock()
    mock_client.aio = AsyncMock()
    error = APIError(401, {"error": "Auth failed"})
    error.code = 401

    mock_client.aio.models.generate_content = AsyncMock(side_effect=error)

    with (
        patch("sarathi_agent_inspect.providers.gemini.genai.Client", return_value=mock_client),
        pytest.raises(ProviderAuthenticationError),
    ):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_generate_rate_limit_error(provider: Any) -> None:
    """Test rate limit error mapping."""
    mock_client = MagicMock()
    mock_client.aio = AsyncMock()
    error = APIError(429, {"error": "Rate limit"})
    error.code = 429

    mock_client.aio.models.generate_content = AsyncMock(side_effect=error)

    with (
        patch("sarathi_agent_inspect.providers.gemini.genai.Client", return_value=mock_client),
        pytest.raises(ProviderRateLimitError),
    ):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_generate_connection_error(provider: Any) -> None:
    """Test model not found connection error mapping."""
    mock_client = MagicMock()
    mock_client.aio = AsyncMock()
    error = APIError(404, {"error": "Model not found"})
    error.code = 404

    mock_client.aio.models.generate_content = AsyncMock(side_effect=error)

    with (
        patch("sarathi_agent_inspect.providers.gemini.genai.Client", return_value=mock_client),
        pytest.raises(ProviderConnectionError),
    ):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_generate_stream(provider: Any) -> None:
    """Test streaming generation."""

    class AsyncIteratorMock:
        def __init__(self, items: Any) -> None:
            self.items = items
            self.index = 0

        def __aiter__(self) -> "AsyncIteratorMock":
            return self

        async def __anext__(self) -> Any:
            if self.index < len(self.items):
                item = self.items[self.index]
                self.index += 1
                return item
            raise StopAsyncIteration

    mock_chunk1 = MagicMock()
    mock_chunk1.text = "Hello"

    mock_chunk2 = MagicMock()
    mock_chunk2.text = " world"

    mock_client = MagicMock()
    mock_client.aio = AsyncMock()
    mock_client.aio.models.generate_content_stream = AsyncMock(
        return_value=AsyncIteratorMock([mock_chunk1, mock_chunk2])
    )

    with patch("sarathi_agent_inspect.providers.gemini.genai.Client", return_value=mock_client):
        chunks = []
        async for chunk in provider.generate_stream("Hi"):
            chunks.append(chunk)

    assert chunks == ["Hello", " world"]
