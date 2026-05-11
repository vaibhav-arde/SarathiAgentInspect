"""Gemini provider implementation.

Communicates with Google's generative models using the unified google-genai SDK.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from google import genai
from google.genai.errors import APIError

from sarathi_agent_inspect.core.exceptions.base import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
)
from sarathi_agent_inspect.providers.base import BaseProvider, ModelInfo, ProviderResponse
from sarathi_agent_inspect.providers.cost import estimate_cost
from sarathi_agent_inspect.providers.registry import register_provider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from google.genai.types import GenerateContentConfigDict

    from sarathi_agent_inspect.core.config.settings import SarathiSettings


@register_provider("gemini")
class GeminiProvider(BaseProvider):
    """Google Gemini LLM provider."""

    def __init__(
        self,
        settings: SarathiSettings,
        model_override: str | None = None,
        timeout_override: int | None = None,
    ) -> None:
        """Initialize the Gemini provider."""
        self._settings = settings
        self._model = model_override or settings.provider.gemini.model
        self._timeout = timeout_override or settings.provider.timeout
        self._api_key = settings.provider.gemini.api_key

        self._client: genai.Client | None = None

    @property
    def provider_name(self) -> str:
        """Return 'gemini'."""
        return "gemini"

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model

    async def initialize(self) -> None:
        """Initialize the Gemini client."""
        if not self._api_key:
            raise ProviderAuthenticationError(
                message="Google Gemini API key is missing. Set GEMINI_API_KEY environment variable.",
                context={"provider": self.provider_name},
            )

        if self._client is None:
            # The new unified SDK initializes synchronously.
            # We configure the HTTP client via http_options.
            self._client = genai.Client(
                api_key=self._api_key,
                http_options={"timeout": self._timeout},
            )

    async def health_check(self) -> bool:
        """Check if Gemini API is reachable by fetching the model."""
        if not self._client:
            try:
                await self.initialize()
            except ProviderAuthenticationError:
                return False

        try:
            # We use the sync client for a quick metadata fetch
            self._client.models.get(model=self._model)  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    def get_model_info(self) -> ModelInfo:
        """Return Gemini model metadata."""
        return ModelInfo(
            provider=self.provider_name,
            model=self._model,
            supports_streaming=True,
            supports_tools=True,
        )

    def _map_error(self, e: Exception) -> ProviderError:
        """Map genai exceptions to Sarathi errors."""
        context = {"provider": self.provider_name, "model": self._model}

        if isinstance(e, APIError):
            if e.code == 401 or e.code == 403:
                return ProviderAuthenticationError(f"Gemini auth failed: {e.message}", context=context)
            if e.code == 429:
                return ProviderRateLimitError(f"Gemini rate limit: {e.message}", context=context)
            if e.code == 404:
                return ProviderConnectionError(f"Gemini model not found: {e.message}", context=context)

        return ProviderError(f"Gemini error: {e}", context=context)

    def _build_config(
        self,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        **kwargs: Any,
    ) -> GenerateContentConfigDict:
        """Construct the configuration for generation."""
        # Use a standard dict first, then cast
        config_dict: dict[str, Any] = {}

        if system_prompt:
            config_dict["system_instruction"] = system_prompt
        if temperature is not None:
            config_dict["temperature"] = temperature
        if max_tokens is not None:
            config_dict["max_output_tokens"] = max_tokens

        config_dict.update(kwargs)

        # Cast to appease Mypy since kwargs can be anything
        from typing import cast
        return cast("GenerateContentConfigDict", config_dict)

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response using Gemini."""
        if not self._client:
            await self.initialize()

        config = self._build_config(system_prompt, temperature, max_tokens, **kwargs)

        start_time = time.perf_counter()
        try:
            response = await self._client.aio.models.generate_content(  # type: ignore[union-attr]
                model=self._model,
                contents=prompt,
                config=config,
            )
        except Exception as e:
            raise self._map_error(e) from e

        latency = (time.perf_counter() - start_time) * 1000

        # Extract tokens and cost
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count or 0
            completion_tokens = response.usage_metadata.candidates_token_count or 0

        total_tokens = prompt_tokens + completion_tokens
        cost = estimate_cost(self._model, prompt_tokens, completion_tokens, self.provider_name)

        finish_reason = None
        if response.candidates and response.candidates[0].finish_reason:
            finish_reason = response.candidates[0].finish_reason.name

        return ProviderResponse(
            content=response.text or "",
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency,
            cost_usd=cost,
            finish_reason=finish_reason,
            raw_response=response.model_dump() if hasattr(response, "model_dump") else {},
        )

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Generate vector embeddings using Gemini."""
        if not self._client:
            await self.initialize()

        # The SDK supports list[str] natively
        response = await self._client.aio.models.embed_content(
            model="text-embedding-004",  # Default embedding model for Gemini
            contents=text,
        )
        
        if isinstance(text, str):
            return response.embeddings[0].values
        return [e.values for e in response.embeddings]

    def get_token_count(self, text: str) -> int:
        """Calculate token count using Gemini's native API."""
        if not self._client:
            # Fallback to word count if client not initialized
            return len(text.split())
        
        # This is a sync call in the SDK
        response = self._client.models.count_tokens(
            model=self._model,
            contents=text,
        )
        return response.total_tokens

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Return cost using central estimator."""
        return estimate_cost(self._model, prompt_tokens, completion_tokens, self.provider_name)

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response using Gemini."""
        if not self._client:
            await self.initialize()

        config = self._build_config(system_prompt, temperature, max_tokens, **kwargs)

        try:
            stream = await self._client.aio.models.generate_content_stream(  # type: ignore[union-attr]
                model=self._model,
                contents=prompt,
                config=config,
            )
            async for chunk in stream:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            raise self._map_error(e) from e
