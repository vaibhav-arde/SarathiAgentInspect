"""Judge model abstraction for LLM-as-a-Judge evaluations.

Provides utilities for resolving and configuring the specific
LLM used by metrics (like DeepEval) to score agent outputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sarathi_agent_inspect.providers.registry import ProviderFactory

if TYPE_CHECKING:
    from sarathi_agent_inspect.core.config.settings import SarathiSettings
    from sarathi_agent_inspect.providers.base import BaseProvider


def get_judge_provider(settings: SarathiSettings) -> BaseProvider:
    """Resolve and create the provider used for judge evaluations.

    The judge model is independent of the model under test.
    For example, you might test a local Ollama model, but use
    GPT-4o as the judge to evaluate its outputs.

    Args:
        settings: The framework configuration containing judge settings.

    Returns:
        An instantiated provider configured as the judge.
    """
    judge_name = settings.judge.provider

    # Pass the specific judge timeout to the provider via kwargs
    # Providers should respect this override if provided.
    return ProviderFactory.create(
        settings=settings,
        provider_name=judge_name,
        timeout_override=settings.judge.timeout,
        model_override=settings.judge.model,
    )
