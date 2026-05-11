"""Unit tests for the token tracking utilities."""

from sarathi_agent_inspect.providers.base import ProviderResponse
from sarathi_agent_inspect.providers.tokens import TokenTracker, TokenUsage, format_token_report


def test_token_usage_add() -> None:
    """Test token usage addition."""
    usage1 = TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30, cost_usd=0.01)
    usage2 = TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10, cost_usd=0.005)

    usage1.add(usage2)

    assert usage1.prompt_tokens == 15
    assert usage1.completion_tokens == 25
    assert usage1.total_tokens == 40
    assert usage1.cost_usd == 0.015


def test_token_tracker_track() -> None:
    """Test token tracker accumulation from provider responses."""
    tracker = TokenTracker()

    response1 = ProviderResponse(
        content="test 1",
        model="gpt-4o",
        provider="openai",
        prompt_tokens=10,
        completion_tokens=20,
        total_tokens=30,
        latency_ms=100.0,
        cost_usd=0.01,
        finish_reason="stop",
        raw_response={},
    )

    response2 = ProviderResponse(
        content="test 2",
        model="gpt-4o",
        provider="openai",
        prompt_tokens=5,
        completion_tokens=10,
        total_tokens=15,
        latency_ms=50.0,
        cost_usd=0.005,
        finish_reason="stop",
        raw_response={},
    )

    response3 = ProviderResponse(
        content="test 3",
        model="claude-sonnet-4.6",
        provider="anthropic",
        prompt_tokens=100,
        completion_tokens=50,
        total_tokens=150,
        latency_ms=200.0,
        cost_usd=0.05,
        finish_reason="stop",
        raw_response={},
    )

    tracker.track(response1)
    tracker.track(response2)
    tracker.track(response3)

    assert tracker.calls_tracked == 3
    assert tracker.total_usage.prompt_tokens == 115
    assert tracker.total_usage.completion_tokens == 80
    assert tracker.total_usage.total_tokens == 195
    assert round(tracker.total_usage.cost_usd, 3) == 0.065

    assert "gpt-4o" in tracker.model_usage
    assert tracker.model_usage["gpt-4o"].total_tokens == 45
    assert round(tracker.model_usage["gpt-4o"].cost_usd, 3) == 0.015

    assert "claude-sonnet-4.6" in tracker.model_usage
    assert tracker.model_usage["claude-sonnet-4.6"].total_tokens == 150


def test_format_token_report() -> None:
    """Test token report formatting."""
    tracker = TokenTracker()

    response = ProviderResponse(
        content="test 1",
        model="gpt-4o",
        provider="openai",
        prompt_tokens=1000,
        completion_tokens=2000,
        total_tokens=3000,
        latency_ms=100.0,
        cost_usd=0.0225,
        finish_reason="stop",
        raw_response={},
    )

    tracker.track(response)

    report = format_token_report(tracker)

    assert "=== Token Usage Report ===" in report
    assert "Total Calls: 1" in report
    assert "Total Tokens: 3,000" in report
    assert "Prompt: 1,000" in report
    assert "Completion: 2,000" in report
    assert "Estimated Cost: $0.0225" in report
    assert "Breakdown by Model:" in report
    assert "gpt-4o: 3,000 tokens" in report
