"""Ollama provider implementation.

Communicates with a local or remote Ollama instance using httpx.
"""

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING, Any

import httpx

from sarathi_agent_inspect.core.exceptions.base import (
    ProviderConnectionError,
    ProviderError,
    ProviderTimeoutError,
)
from sarathi_agent_inspect.providers.base import BaseProvider, ModelInfo, ProviderResponse
from sarathi_agent_inspect.providers.cost import estimate_cost
from sarathi_agent_inspect.providers.registry import register_provider

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sarathi_agent_inspect.core.config.settings import SarathiSettings


@register_provider("ollama")
class OllamaProvider(BaseProvider):
    """Ollama LLM provider."""

    def __init__(
        self,
        settings: SarathiSettings,
        model_override: str | None = None,
        timeout_override: int | None = None,
    ) -> None:
        """Initialize the Ollama provider.

        Args:
            settings: The framework configuration.
            model_override: Optional model name to override the default.
            timeout_override: Optional timeout override (e.g., for judge).
        """
        self._settings = settings
        self._base_url = settings.provider.ollama.base_url.rstrip("/")
        self._model = model_override or settings.provider.ollama.model
        self._timeout = timeout_override or settings.provider.timeout
        self._client: httpx.AsyncClient | None = None

    @property
    def provider_name(self) -> str:
        """Return 'ollama'."""
        return "ollama"

    @property
    def model_name(self) -> str:
        """Return the configured model name."""
        return self._model

    async def initialize(self) -> None:
        """Establish the HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout,
                headers={"Content-Type": "application/json"},
            )

    async def shutdown(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        """Check if Ollama is reachable and the model exists."""
        if not self._client:
            await self.initialize()

        try:
            # First check if Ollama is up
            response = await self._client.get("/")  # type: ignore[union-attr]
            if response.status_code != 200:
                return False

            # Then check if the model is available locally
            tags_response = await self._client.get("/api/tags")  # type: ignore[union-attr]
            tags_response.raise_for_status()
            models = [m["name"] for m in tags_response.json().get("models", [])]

            # exact match or with tag (if user specified base name)
            if self._model in models:
                return True
            # if user said 'llama3' but local is 'llama3:latest'
            return any(m.startswith(f"{self._model}:") for m in models)
        except httpx.RequestError:
            return False

    def get_model_info(self) -> ModelInfo:
        """Return Ollama model metadata."""
        return ModelInfo(
            provider=self.provider_name,
            model=self._model,
            supports_streaming=True,
            supports_tools=True,
        )

    def _map_error(self, e: Exception) -> ProviderError:
        """Map httpx exceptions to Sarathi errors."""
        context = {"provider": self.provider_name, "model": self._model, "url": self._base_url}
        if isinstance(e, httpx.TimeoutException):
            return ProviderTimeoutError("Ollama request timed out", context=context)
        if isinstance(e, httpx.ConnectError):
            return ProviderConnectionError("Failed to connect to Ollama", context=context)
        return ProviderError(f"Ollama error: {e}", context=context)

    def _build_payload(
        self,
        prompt: str,
        system_prompt: str | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Construct the Ollama generation payload."""
        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": stream,
        }
        if system_prompt:
            payload["system"] = system_prompt

        options = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens is not None:
            options["num_predict"] = max_tokens

        # Merge other kwargs into options
        for k, v in kwargs.items():
            options[k] = v

        if options:
            payload["options"] = options

        return payload

    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response synchronously (awaiting full output)."""
        if not self._client:
            await self.initialize()

        payload = self._build_payload(prompt, system_prompt, temperature, max_tokens, stream=False, **kwargs)

        start_time = time.perf_counter()
        try:
            response = await self._client.post("/api/generate", json=payload)  # type: ignore[union-attr]
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPStatusError as e:
            # e.g., 404 Model not found
            if e.response.status_code == 404:
                raise ProviderConnectionError(
                    message=f"Model '{self._model}' not found in Ollama. Run `ollama pull {self._model}`",
                    context={"provider": self.provider_name, "model": self._model},
                ) from e
            raise self._map_error(e) from e
        except Exception as e:
            raise self._map_error(e) from e

        latency = (time.perf_counter() - start_time) * 1000

        # Extract tokens
        prompt_tokens = data.get("prompt_eval_count", 0)
        completion_tokens = data.get("eval_count", 0)
        total_tokens = prompt_tokens + completion_tokens

        # Ollama is free, but we use the central estimator just in case
        cost = estimate_cost(self._model, prompt_tokens, completion_tokens, self.provider_name)

        return ProviderResponse(
            content=data.get("response", ""),
            model=self._model,
            provider=self.provider_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            latency_ms=latency,
            cost_usd=cost,
            finish_reason=data.get("done_reason", "stop" if data.get("done") else None),
            raw_response=data,
        )

    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Generate vector embeddings via Ollama /api/embeddings."""
        if not self._client:
            await self.initialize()

        assert self._client is not None
        if isinstance(text, str):
            payload = {"model": self._model, "prompt": text}
            response = await self._client.post("/api/embeddings", json=payload)
            response.raise_for_status()
            result: list[float] = response.json().get("embedding", [])
            return result
        else:
            # Batch embedding
            results: list[list[float]] = []
            for t in text:
                payload = {"model": self._model, "prompt": t}
                response = await self._client.post("/api/embeddings", json=payload)
                response.raise_for_status()
                results.append(response.json().get("embedding", []))
            return results

    def get_token_count(self, text: str) -> int:
        """Simple word-based token count as fallback for local models."""
        # In enterprise RAG, you'd use tiktoken or sentencepiece.
        # This is a baseline implementation.
        return len(text.split())

    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Return cost using central estimator."""
        return estimate_cost(self._model, prompt_tokens, completion_tokens, self.provider_name) or 0.0

    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response token by token."""
        if not self._client:
            await self.initialize()

        payload = self._build_payload(prompt, system_prompt, temperature, max_tokens, stream=True, **kwargs)

        try:
            async with self._client.stream("POST", "/api/generate", json=payload) as response:  # type: ignore[union-attr]
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            raise self._map_error(e) from e
