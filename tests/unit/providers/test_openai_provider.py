"""Unit tests for the OpenAI provider."""

import sys
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import openai
import pytest

from sarathi_agent_inspect.core.config.settings import SarathiSettings
from sarathi_agent_inspect.core.exceptions.base import (
    ProviderAuthenticationError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sarathi_agent_inspect.providers.openai_provider import OpenAIProvider


@pytest.fixture
def settings() -> Any:
    s = SarathiSettings()
    s.provider.openai.api_key = "test-key"
    return s


@pytest.fixture
def provider(settings: Any) -> Any:
    return OpenAIProvider(settings=settings)


@pytest.mark.asyncio
async def test_health_check_success(provider: Any) -> None:
    """Test successful health check."""
    mock_client = AsyncMock()
    mock_client.models.list = AsyncMock()

    with patch("sarathi_agent_inspect.providers.openai_provider.AsyncOpenAI", return_value=mock_client):
        assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_health_check_fail(provider: Any) -> None:
    """Test health check failure."""
    mock_client = AsyncMock()
    mock_client.models.list = AsyncMock(side_effect=Exception("Failed"))

    with patch("sarathi_agent_inspect.providers.openai_provider.AsyncOpenAI", return_value=mock_client):
        assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_generate_success(provider: Any) -> None:
    """Test successful generation."""
    mock_response = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Hello from OpenAI"
    mock_choice.finish_reason = "stop"
    mock_response.choices = [mock_choice]
    mock_response.usage.prompt_tokens = 10
    mock_response.usage.completion_tokens = 5
    mock_response.model_dump.return_value = {}

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=mock_response)

    with patch("sarathi_agent_inspect.providers.openai_provider.AsyncOpenAI", return_value=mock_client):
        response = await provider.generate("Hi", temperature=0.7)

    assert response.content == "Hello from OpenAI"
    assert response.provider == "openai"
    assert response.model == "gpt-4o"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.total_tokens == 15
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_generate_auth_error(provider: Any) -> None:
    """Test auth error mapping."""
    mock_client = AsyncMock()

    # We need to construct a Request mock
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_response = httpx.Response(401, request=mock_request)

    mock_client.chat.completions.create = AsyncMock(
        side_effect=openai.AuthenticationError("Auth failed", response=mock_response, body=None)
    )

    with (
        patch("sarathi_agent_inspect.providers.openai_provider.AsyncOpenAI", return_value=mock_client),
        pytest.raises(ProviderAuthenticationError),
    ):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_generate_rate_limit_error(provider: Any) -> None:
    """Test rate limit error mapping."""
    mock_client = AsyncMock()
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")
    mock_response = httpx.Response(429, request=mock_request)

    mock_client.chat.completions.create = AsyncMock(
        side_effect=openai.RateLimitError("Rate limit", response=mock_response, body=None)
    )

    with (
        patch("sarathi_agent_inspect.providers.openai_provider.AsyncOpenAI", return_value=mock_client),
        pytest.raises(ProviderRateLimitError),
    ):
        await provider.generate("Hi")


@pytest.mark.asyncio
async def test_generate_timeout_error(provider: Any) -> None:
    """Test timeout error mapping."""
    mock_client = AsyncMock()
    mock_request = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

    mock_client.chat.completions.create = AsyncMock(side_effect=openai.APITimeoutError(request=mock_request))

    with (
        patch("sarathi_agent_inspect.providers.openai_provider.AsyncOpenAI", return_value=mock_client),
        pytest.raises(ProviderTimeoutError),
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
    mock_chunk1.choices = [MagicMock()]
    mock_chunk1.choices[0].delta.content = "Hello"

    mock_chunk2 = MagicMock()
    mock_chunk2.choices = [MagicMock()]
    mock_chunk2.choices[0].delta.content = " world"

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=AsyncIteratorMock([mock_chunk1, mock_chunk2]))

    with patch("sarathi_agent_inspect.providers.openai_provider.AsyncOpenAI", return_value=mock_client):
        chunks = []
        async for chunk in provider.generate_stream("Hi"):
            chunks.append(chunk)

    assert chunks == ["Hello", " world"]


def test_get_token_count_falls_back_without_tiktoken(provider: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test token counting fallback when tiktoken is unavailable."""
    monkeypatch.setitem(sys.modules, "tiktoken", None)

    assert provider.get_token_count("one two three") == 3
