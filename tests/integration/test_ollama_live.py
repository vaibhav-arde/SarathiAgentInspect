"""Live integration tests for Ollama provider.

These tests run against a real local Ollama instance and are skipped by default.
Run them using: pytest -m integration
"""

from typing import Any

import pytest

from sarathi_agent_inspect.core.config.settings import SarathiSettings
from sarathi_agent_inspect.providers.ollama import OllamaProvider


@pytest.fixture
def live_settings() -> Any:
    """Settings configured for a live local Ollama instance."""
    s = SarathiSettings()
    s.provider.default = "ollama"
    s.provider.ollama.base_url = "http://localhost:11434"
    s.provider.ollama.model = "gemma4:31b-cloud"
    return s


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_live_health_check(live_settings: Any) -> None:
    """Test health check against real Ollama instance."""
    provider = OllamaProvider(settings=live_settings)

    is_healthy = await provider.health_check()
    assert is_healthy is True, "Local Ollama instance is not running or model is missing"


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_live_generate(live_settings: Any) -> None:
    """Test generating a simple completion from live Ollama."""
    provider = OllamaProvider(settings=live_settings)

    response = await provider.generate("Say 'Hello integration test' and nothing else.", temperature=0.0)

    assert response.provider == "ollama"
    assert response.model == live_settings.provider.ollama.model
    assert "Hello integration test" in response.content
    assert response.total_tokens > 0
    assert response.latency_ms > 0


@pytest.mark.asyncio
@pytest.mark.integration
async def test_ollama_live_stream(live_settings: Any) -> None:
    """Test streaming a completion from live Ollama."""
    provider = OllamaProvider(settings=live_settings)

    chunks = []
    async for chunk in provider.generate_stream("Count from 1 to 3.", temperature=0.0):
        chunks.append(chunk)

    full_response = "".join(chunks)
    assert len(chunks) > 1, "Expected multiple chunks for streaming"
    assert "1" in full_response
    assert "2" in full_response
    assert "3" in full_response
