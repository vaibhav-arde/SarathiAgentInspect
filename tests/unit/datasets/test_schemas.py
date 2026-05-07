"""Unit tests for dataset schemas."""

import pytest
from pydantic import ValidationError

from sarathi_agent_inspect.datasets.schemas import (
    AIAgentRecord,
    ChatbotRecord,
    MultiTurnRecord,
    RAGRecord,
    ToolCallingRecord,
)


def test_chatbot_record_valid():
    """Test valid chatbot record."""
    record = ChatbotRecord(input="Hello", expected_output="Hi")
    assert record.input == "Hello"
    assert record.expected_output == "Hi"

def test_chatbot_record_missing_input():
    """Test chatbot record missing required input."""
    with pytest.raises(ValidationError):
        ChatbotRecord(expected_output="Hi")

def test_rag_record_valid():
    """Test valid RAG record."""
    record = RAGRecord(
        query="What is Sarathi?",
        retrieved_contexts=["Sarathi is an AI framework"],
        expected_response="An AI framework"
    )
    assert record.query == "What is Sarathi?"
    assert len(record.retrieved_contexts) == 1

def test_ai_agent_record_valid():
    """Test valid AI agent record."""
    record = AIAgentRecord(
        task="Search for weather",
        expected_actions=[
            {"tool": "weather_api", "tool_input": {"city": "London"}}
        ]
    )
    assert record.task == "Search for weather"
    assert record.expected_actions[0].tool == "weather_api"

def test_multi_turn_record_valid():
    """Test valid multi-turn record."""
    record = MultiTurnRecord(
        dialogue_history=[
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"}
        ],
        expected_next_turn={"role": "user", "content": "How are you?"}
    )
    assert len(record.dialogue_history) == 2
    assert record.expected_next_turn.role == "user"

def test_tool_calling_record_valid():
    """Test valid tool calling record."""
    record = ToolCallingRecord(
        user_prompt="Add 2 and 2",
        available_tools=[
            {
                "name": "add",
                "description": "Add numbers",
                "parameters": {"type": "object", "properties": {"a": {"type": "number"}}}
            }
        ],
        expected_tool_calls=[
            {"name": "add", "arguments": {"a": 2, "b": 2}}
        ]
    )
    assert record.user_prompt == "Add 2 and 2"
    assert record.available_tools[0].name == "add"
