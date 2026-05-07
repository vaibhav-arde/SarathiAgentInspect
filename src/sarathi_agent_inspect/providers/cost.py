"""Token cost estimation utilities.

Provides per-token cost calculations for supported LLM providers.
Pricing tables are approximations and should be updated periodically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class ModelPricing:
    """Pricing configuration for a specific model (costs per 1 million tokens)."""

    input_cost_per_m: float
    output_cost_per_m: float


# Current pricing per 1M tokens (as of mid-2026)
PRICING_TABLE: Final[dict[str, ModelPricing]] = {
    # ── OpenAI ──
    "gpt-4o": ModelPricing(input_cost_per_m=2.50, output_cost_per_m=10.00),
    "gpt-4o-mini": ModelPricing(input_cost_per_m=0.15, output_cost_per_m=0.60),
    # ── Anthropic ──
    "claude-opus-4.6": ModelPricing(input_cost_per_m=5.00, output_cost_per_m=25.00),
    "claude-sonnet-4.6": ModelPricing(input_cost_per_m=3.00, output_cost_per_m=15.00),
    "claude-haiku-4.5": ModelPricing(input_cost_per_m=1.00, output_cost_per_m=5.00),
    # legacy
    "claude-3-5-sonnet-20241022": ModelPricing(input_cost_per_m=3.00, output_cost_per_m=15.00),
    "claude-sonnet-4-20250514": ModelPricing(input_cost_per_m=3.00, output_cost_per_m=15.00),
    # ── Google Gemini ──
    "gemini-2.5-pro": ModelPricing(input_cost_per_m=1.25, output_cost_per_m=10.00),
    "gemini-2.5-flash": ModelPricing(input_cost_per_m=0.30, output_cost_per_m=2.50),
    "gemini-2.5-flash-lite": ModelPricing(input_cost_per_m=0.10, output_cost_per_m=0.40),
    # ── Local / Free ──
    "ollama": ModelPricing(input_cost_per_m=0.0, output_cost_per_m=0.0),
}


def get_pricing(model_name: str, provider_name: str | None = None) -> ModelPricing | None:
    """Retrieve pricing information for a given model.

    Uses fuzzy matching to handle model versions (e.g., 'gpt-4o-2024-11-20' -> 'gpt-4o').

    Args:
        model_name: The name of the model to look up.
        provider_name: Optional provider name. If 'ollama', always returns free pricing.

    Returns:
        ModelPricing if found, None if unknown.
    """
    if provider_name == "ollama":
        return PRICING_TABLE["ollama"]

    model_lower = model_name.lower()

    # Exact match first
    if model_lower in PRICING_TABLE:
        return PRICING_TABLE[model_lower]

    # Fuzzy match base names
    for base_model, pricing in PRICING_TABLE.items():
        if base_model != "ollama" and model_lower.startswith(base_model):
            return pricing

    return None


def estimate_cost(
    model_name: str,
    prompt_tokens: int,
    completion_tokens: int,
    provider_name: str | None = None,
) -> float | None:
    """Estimate the cost of a generation request in USD.

    Args:
        model_name: The name of the model used.
        prompt_tokens: Number of input tokens.
        completion_tokens: Number of generated tokens.
        provider_name: Optional provider name.

    Returns:
        Estimated cost in USD, or None if pricing is unknown.
    """
    pricing = get_pricing(model_name, provider_name)
    if not pricing:
        return None

    input_cost = (prompt_tokens / 1_000_000.0) * pricing.input_cost_per_m
    output_cost = (completion_tokens / 1_000_000.0) * pricing.output_cost_per_m

    return input_cost + output_cost
