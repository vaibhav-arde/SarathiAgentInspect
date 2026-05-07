"""OpenAI provider implementation.

Communicates with OpenAI or OpenAI-compatible endpoints using the official SDK.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import openai
from openai import AsyncOpenAI

from sarathi_agent_inspect.core.exceptions.base import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from sarathi_agent_inspect.providers.base import BaseProvider, ModelInfo, ProviderResponse
from sarathi_agent_inspect.providers.cost import estimate_cost
from sarathi_agent_inspect.providers.registry import register_provider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sarathi_agent_inspect.core.config.settings import SarathiSettings


@register_provider("openai")
class OpenAIProvider(BaseProvider):
    """OpenAI LLM provider."""

    def __init__(
        self,
        settings: SarathiSettings,
        model_override: str | None = None,
        timeout_override: int | None = None,
    ) -> None:
        """Initialize the OpenAI provider."""
        self._settings = settings

        # Check if we should use the default model or an override (e.g., for judge)
        self._model = model_override or settings.provider.openai.model
        self._timeout = timeout_override or settings.provider.timeout

        self._api_key = settings.provider.openai.api_key
        self._base_url = settings.provider.openai.base_url or None

        self._client: AsyncOpenAI | None = None

    @property
    def provider_name(self) -> str:
        """Return 'openai'."""
        return "openai"

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model

    async def initialize(self) -> None:
        """Initialize the OpenAI client."""
        if not self._api_key:
            raise ProviderAuthenticationError(
                message="OpenAI API key is missing. Set OPENAI_API_KEY environment variable.",
                context={"provider": self.provider_name},
            )

        if self._client is None:
            self._client = AsyncOpenAI(
                api_key=self._api_key,
                base_url=self._base_url,
                timeout=self._timeout,
                max_retries=0,  # We handle retries at the framework level
            )

    async def shutdown(self) -> None:
        """Close the OpenAI client."""
        if self._client:
            await self._client.close()
            self._client = None

    async def health_check(self) -> bool:
        """Check if OpenAI API is reachable by listing models."""
        if not self._client:
            try:
                await self.initialize()
            except ProviderAuthenticationError:
                return False

        try:
            await self._client.models.list()  # type: ignore[union-attr]
            return True
        except Exception:
            return False

    def get_model_info(self) -> ModelInfo:
        """Return OpenAI model metadata."""
        return ModelInfo(
            provider=self.provider_name,
            model=self._model,
            supports_streaming=True,
            supports_tools=True,
        )

    def _map_error(self, e: Exception) -> ProviderError:
        """Map openai exceptions to Sarathi errors."""
        context = {"provider": self.provider_name, "model": self._model}

        if isinstance(e, openai.AuthenticationError):
            return ProviderAuthenticationError(f"OpenAI auth failed: {e}", context=context)
        if isinstance(e, openai.RateLimitError):
            return ProviderRateLimitError(f"OpenAI rate limit: {e}", context=context)
        if isinstance(e, openai.APITimeoutError):
            return ProviderTimeoutError("OpenAI request timed out", context=context)
        if isinstance(e, openai.APIConnectionError):
            return ProviderConnectionError("Failed to connect to OpenAI", context=context)

        return ProviderError(f"OpenAI error: {e}", context=context)

    def _build_messages(self, prompt: str, system_prompt: str | None) -> list[dict[str, str]]:
        """Construct the messages list for Chat Completions API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response using Chat Completions API."""
        if not self._client:
            await self.initialize()

        messages = self._build_messages(prompt, system_prompt)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
        }
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens

        call_kwargs.update(kwargs)

        start_time = time.perf_counter()
        try:
            response = await self._client.chat.completions.create(**call_kwargs)  # type: ignore[union-attr]
        except Exception as e:
            raise self._map_error(e) from e

        latency = (time.perf_counter() - start_time) * 1000

        # Extract tokens and cost
        prompt_tokens = 0
        completion_tokens = 0
        if response.usage:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens

        total_tokens = prompt_tokens + completion_tokens
        cost = estimate_cost(self._model, prompt_tokens, completion_tokens, self.provider_name)

        content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason

        return ProviderResponse(
            content=content,
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency,
            cost_usd=cost,
            finish_reason=finish_reason,
            raw_response=response.model_dump(),
        )

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response using Chat Completions API."""
        if not self._client:
            await self.initialize()

        messages = self._build_messages(prompt, system_prompt)

        call_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "stream": True,
        }
        if temperature is not None:
            call_kwargs["temperature"] = temperature
        if max_tokens is not None:
            call_kwargs["max_tokens"] = max_tokens

        call_kwargs.update(kwargs)

        try:
            stream = await self._client.chat.completions.create(**call_kwargs)  # type: ignore[union-attr]
            async for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            raise self._map_error(e) from e
