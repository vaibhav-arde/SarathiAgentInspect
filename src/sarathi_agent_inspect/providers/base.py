"""Abstract base provider interface for LLM integrations.

Defines the contract that all LLM providers must implement.
This ensures consistent behavior across OpenAI, Anthropic,
Ollama, Gemini, Azure OpenAI, Bedrock, and future providers.

Lifecycle:
    1. __init__() — Configuration injection
    2. initialize() — Async setup (connections, health checks)
    3. generate() / generate_stream() — LLM operations
    4. shutdown() — Cleanup resources
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sarathi_agent_inspect.core.config.settings import SarathiSettings
    from sarathi_agent_inspect.core.types import JsonDict, ModelName, ProviderName


@dataclass(frozen=True)
class ModelInfo:
    """Metadata about an LLM model.

    Attributes:
        provider: Provider identifier (e.g., 'ollama').
        model: Model identifier (e.g., 'gemma4:31b-cloud').
        max_tokens: Maximum context window size, if known.
        supports_streaming: Whether the model supports streaming.
        supports_tools: Whether the model supports tool/function calling.
        metadata: Additional provider-specific metadata.
    """

    provider: ProviderName
    model: ModelName
    max_tokens: int | None = None
    supports_streaming: bool = True
    supports_tools: bool = False
    metadata: JsonDict = field(default_factory=dict)


@dataclass
class ProviderResponse:
    """Standardized response from any LLM provider.

    Normalizes responses across providers into a consistent format.

    Attributes:
        content: Generated text content.
        model: Model that generated the response.
        provider: Provider that served the request.
        prompt_tokens: Number of tokens in the prompt.
        completion_tokens: Number of tokens in the completion.
        total_tokens: Total tokens consumed.
        latency_ms: Response latency in milliseconds.
        cost_usd: Estimated cost in USD, if available.
        finish_reason: Why generation stopped (e.g., 'stop', 'length').
        raw_response: Original provider response for debugging.
        timestamp: When the response was received.
    """

    content: str
    model: ModelName
    provider: ProviderName
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float | None = None
    finish_reason: str | None = None
    raw_response: JsonDict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class
    and implement the abstract methods.
    """

    def __init__(self, settings: SarathiSettings, **kwargs: Any) -> None:
        """Initialize the provider.

        Args:
            settings: The framework configuration.
            **kwargs: Additional provider-specific overrides.
        """
        self._settings = settings

    @property
    @abstractmethod
    def provider_name(self) -> ProviderName:
        """Return the provider identifier."""
        ...

    @property
    @abstractmethod
    def model_name(self) -> ModelName:
        """Return the current model identifier."""
        ...

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize the provider (establish connections, validate config).

        Called once before the first generate() call.
        Should verify that the provider is reachable and configured correctly.

        Raises:
            ProviderConnectionError: If the provider cannot be reached.
            ProviderAuthenticationError: If credentials are invalid.
        """
        ...

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a response from the LLM.

        Args:
            prompt: The user prompt / input text.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific parameters.

        Returns:
            Standardized ProviderResponse.

        Raises:
            ProviderError: If the generation fails.
            ProviderTimeoutError: If the request times out.
            ProviderRateLimitError: If rate limited.
        """
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        *,
        system_prompt: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a response from the LLM token-by-token.

        Args:
            prompt: The user prompt / input text.
            system_prompt: Optional system-level instruction.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.
            **kwargs: Provider-specific parameters.

        Yields:
            Individual text chunks as they are generated.

        Raises:
            ProviderError: If the generation fails.
        """
        yield ""  # pragma: no cover

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is reachable and operational.

        Returns:
            True if the provider is healthy, False otherwise.
        """
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return metadata about the current model.

        Returns:
            ModelInfo with capabilities and configuration.
        """
        ...

    @abstractmethod
    async def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        """Generate vector embeddings for the given text.

        Args:
            text: Single string or list of strings to embed.

        Returns:
            List of floats (single) or list of list of floats (batch).
        """
        ...

    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Calculate the number of tokens in the given text.

        Args:
            text: The text to tokenize.

        Returns:
            Token count.
        """
        ...

    @abstractmethod
    def get_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate estimated cost in USD based on token counts.

        Args:
            prompt_tokens: Input tokens.
            completion_tokens: Output tokens.

        Returns:
            Estimated cost in USD.
        """
        ...

    async def shutdown(self) -> None:  # noqa: B027
        """Clean up provider resources.

        Override this method if the provider holds connections,
        file handles, or other resources that need cleanup.
        """

    async def __aenter__(self) -> BaseProvider:
        """Async context manager entry."""
        await self.initialize()
        return self

    async def __aexit__(self, *_: Any) -> None:
        """Async context manager exit."""
        await self.shutdown()
