"""Anthropic provider stub.

Skeleton for Anthropic integration, ready for future implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from sarathi_agent_inspect.providers.base import BaseProvider, ModelInfo, ProviderResponse
from sarathi_agent_inspect.providers.registry import register_provider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sarathi_agent_inspect.core.config.settings import SarathiSettings


@register_provider("anthropic")
class AnthropicProvider(BaseProvider):
    """Anthropic LLM provider stub."""

    def __init__(
        self,
        settings: SarathiSettings,
        model_override: str | None = None,
        timeout_override: int | None = None,
    ) -> None:
        """Initialize the Anthropic provider stub."""
        self._settings = settings
        self._model = model_override or settings.provider.anthropic.model
        self._timeout = timeout_override or settings.provider.timeout

    @property
    def provider_name(self) -> str:
        """Return 'anthropic'."""
        return "anthropic"

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model

    async def initialize(self) -> None:
        """Initialize the Anthropic client (Not implemented)."""
        pass

    async def health_check(self) -> bool:
        """Check if Anthropic is reachable (Not implemented)."""
        return False

    def get_model_info(self) -> ModelInfo:
        """Return Anthropic model metadata."""
        return ModelInfo(
            provider=self.provider_name,
            model=self._model,
            supports_streaming=True,
            supports_tools=True,
        )

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response using Anthropic (Not implemented)."""
        raise NotImplementedError("AnthropicProvider generate is not implemented yet.")

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response using Anthropic (Not implemented)."""
        raise NotImplementedError("AnthropicProvider generate_stream is not implemented yet.")
        yield ""  # required for the generator signature

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Generate vector embeddings (Not implemented)."""
        raise NotImplementedError("AnthropicProvider embed is not implemented yet.")

    def get_token_count(self, text: str) -> int:
        """Calculate token count (Not implemented)."""
        raise NotImplementedError("AnthropicProvider get_token_count is not implemented yet.")

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost (Not implemented)."""
        raise NotImplementedError("AnthropicProvider get_cost is not implemented yet.")
