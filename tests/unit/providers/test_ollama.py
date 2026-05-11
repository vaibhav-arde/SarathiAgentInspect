"""Unit tests for the Ollama provider."""

import json
from unittest.mock import MagicMock, patch

import httpx
import pytest

from sarathi_agent_inspect.core.config.settings import SarathiSettings
from sarathi_agent_inspect.core.exceptions.base import ProviderConnectionError, ProviderTimeoutError
from sarathi_agent_inspect.providers.ollama import OllamaProvider


@pytest.fixture
def settings():
    return SarathiSettings()


@pytest.fixture
def provider(settings):
    return OllamaProvider(settings=settings)


@pytest.mark.asyncio
async def test_health_check_success(provider):
    """Test successful health check."""
    mock_response_root = MagicMock()
    mock_response_root.status_code = 200

    mock_response_tags = MagicMock()
    mock_response_tags.status_code = 200
    mock_response_tags.json.return_value = {"models": [{"name": "gemma4:31b-cloud"}]}

    with patch("httpx.AsyncClient.get", side_effect=[mock_response_root, mock_response_tags]):
        assert await provider.health_check() is True


@pytest.mark.asyncio
async def test_health_check_fail_root(provider):
    """Test health check failure on root endpoint."""
    mock_response_root = MagicMock()
    mock_response_root.status_code = 500

    with patch("httpx.AsyncClient.get", return_value=mock_response_root):
        assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_health_check_missing_model(provider):
    """Test health check failure when model is not available."""
    mock_response_root = MagicMock()
    mock_response_root.status_code = 200

    mock_response_tags = MagicMock()
    mock_response_tags.status_code = 200
    mock_response_tags.json.return_value = {"models": [{"name": "other_model"}]}

    with patch("httpx.AsyncClient.get", side_effect=[mock_response_root, mock_response_tags]):
        assert await provider.health_check() is False


@pytest.mark.asyncio
async def test_generate_success(provider):
    """Test successful generation."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "model": "gemma4:31b-cloud",
        "response": "Hello world",
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }

    with patch("httpx.AsyncClient.post", return_value=mock_response):
        response = await provider.generate("Hi", temperature=0.7)

    assert response.content == "Hello world"
    assert response.provider == "ollama"
    assert response.model == "gemma4:31b-cloud"
    assert response.prompt_tokens == 10
    assert response.completion_tokens == 5
    assert response.total_tokens == 15
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_generate_timeout_error(provider):
    """Test generation timeout error mapping."""
    with (
        patch("httpx.AsyncClient.post", side_effect=httpx.TimeoutException("Timeout")),
        pytest.raises(ProviderTimeoutError) as exc_info,
    ):
        await provider.generate("Hi")
    assert "timed out" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_connection_error(provider):
    """Test generation connection error mapping."""
    with (
        patch("httpx.AsyncClient.post", side_effect=httpx.ConnectError("Connection refused")),
        pytest.raises(ProviderConnectionError) as exc_info,
    ):
        await provider.generate("Hi")
    assert "Failed to connect" in str(exc_info.value)


@pytest.mark.asyncio
async def test_generate_stream(provider):
    """Test streaming generation."""

    class MockStreamContext:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

        def raise_for_status(self):
            pass

        async def aiter_lines(self):
            yield json.dumps({"response": "Hello", "done": False})
            yield json.dumps({"response": " world", "done": True})

    with patch("httpx.AsyncClient.stream", return_value=MockStreamContext()):
        chunks = []
        async for chunk in provider.generate_stream("Hi"):
            chunks.append(chunk)

    assert chunks == ["Hello", " world"]
