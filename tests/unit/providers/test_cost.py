"""Unit tests for the cost estimator."""

from sarathi_agent_inspect.providers.cost import estimate_cost, get_pricing


def test_get_pricing_exact_match():
    """Test exact match pricing."""
    pricing = get_pricing("gpt-4o")
    assert pricing is not None
    assert pricing.input_cost_per_m == 2.50
    assert pricing.output_cost_per_m == 10.00


def test_get_pricing_fuzzy_match():
    """Test fuzzy match pricing."""
    pricing = get_pricing("gpt-4o-2024-11-20")
    assert pricing is not None
    assert pricing.input_cost_per_m == 2.50
    assert pricing.output_cost_per_m == 10.00


def test_get_pricing_ollama():
    """Test Ollama free pricing."""
    pricing = get_pricing("llama3", provider_name="ollama")
    assert pricing is not None
    assert pricing.input_cost_per_m == 0.0
    assert pricing.output_cost_per_m == 0.0


def test_get_pricing_unknown():
    """Test unknown model."""
    pricing = get_pricing("unknown-model-123")
    assert pricing is None


def test_estimate_cost():
    """Test cost estimation calculation."""
    # 1000 input tokens = 0.0025 USD
    # 2000 output tokens = 0.020 USD
    # Total = 0.0225 USD
    cost = estimate_cost("gpt-4o", prompt_tokens=1000, completion_tokens=2000)
    assert cost is not None
    assert round(cost, 4) == 0.0225


def test_estimate_cost_unknown():
    """Test cost estimation calculation for unknown model."""
    cost = estimate_cost("unknown-model-123", prompt_tokens=1000, completion_tokens=2000)
    assert cost is None
