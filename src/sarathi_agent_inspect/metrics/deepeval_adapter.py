"""Adapter bridging Sarathi providers to DeepEval's LLM interface.

DeepEval metrics require an LLM evaluator that implements `DeepEvalBaseLLM`.
This adapter wraps our `BaseProvider` so that we can enforce deterministic
evaluations and use our standard logging/tracing infrastructure while leveraging
DeepEval's internal prompts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

try:
    from deepeval.models.base_model import DeepEvalBaseLLM
except ImportError:
    # Fallback if deepeval is not installed
    class DeepEvalBaseLLM:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def load_model(self, *args: Any, **kwargs: Any) -> Any:
            pass

        def generate(self, *args: Any, **kwargs: Any) -> Any:
            pass

        async def a_generate(self, *args: Any, **kwargs: Any) -> Any:
            pass

        def get_model_name(self, *args: Any, **kwargs: Any) -> Any:
            pass


if TYPE_CHECKING:
    from sarathi_agent_inspect.providers.base import BaseProvider


class ProviderAdapter(DeepEvalBaseLLM):  # type: ignore[no-untyped-call]
    """Wraps a BaseProvider to satisfy DeepEval's LLM interface."""

    def __init__(self, provider: BaseProvider) -> None:
        """Initialize the adapter.

        Args:
            provider: A configured Sarathi BaseProvider instance.
        """
        super().__init__()
        self.provider = provider
        # Force deterministic evaluation where possible
        self._temperature = 0.0

    def load_model(self) -> Any:
        """Load the underlying model. Returning the provider itself."""
        return self.provider

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Synchronous generation. DeepEval uses this occasionally.

        Note: Our BaseProvider is async-first. We use an event loop
        to run it synchronously here.
        """
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        if loop.is_running():
            # If we are already in an event loop, we need a nested execution or similar.
            # DeepEval sometimes mixes async/sync. We try to handle it gracefully.
            import nest_asyncio  # type: ignore[import-untyped]

            nest_asyncio.apply()

        return loop.run_until_complete(self.a_generate(prompt, **kwargs))

    async def a_generate(self, prompt: str, **kwargs: Any) -> str:
        """Asynchronous generation used by DeepEval's async evaluation flow."""
        # DeepEval might pass its own kwargs. We map them loosely.
        temperature = kwargs.get("temperature", self._temperature)

        response = await self.provider.generate(
            prompt=prompt,
            temperature=temperature,
        )
        return response.content

    def get_model_name(self) -> str:
        """Return the underlying model name for DeepEval logging."""
        return str(self.provider.model_name)
