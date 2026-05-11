"""Pydantic schemas for different dataset evaluation types.

Provides strict validation for various enterprise use cases including
Chatbots, RAG systems, Agents, and Tool-Calling scenarios.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DatasetRecordSchema(BaseModel):
    """Base schema for all dataset records."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata associated with the record",
    )


class ChatbotRecord(DatasetRecordSchema):
    """Schema for standard Chatbot/Conversational evaluation."""

    input: str = Field(..., description="User input/prompt")
    expected_output: str | None = Field(default=None, description="Expected model response")
    context: dict[str, Any] = Field(default_factory=dict, description="Context variables")


class RAGRecord(DatasetRecordSchema):
    """Schema for Retrieval-Augmented Generation evaluation."""

    query: str = Field(..., description="User query")
    retrieved_contexts: list[str] = Field(default_factory=list, description="Documents retrieved by the search system")
    expected_response: str | None = Field(default=None, description="Expected generated response")


class AgentAction(BaseModel):
    """Schema for expected actions taken by an AI Agent."""

    tool: str = Field(..., description="Name of the tool used")
    tool_input: dict[str, Any] = Field(..., description="Input provided to the tool")
    log: str = Field(default="", description="Agent reasoning or logs")


class AIAgentRecord(DatasetRecordSchema):
    """Schema for AI Agent task evaluation."""

    task: str = Field(..., description="The objective/task given to the agent")
    expected_actions: list[AgentAction] = Field(
        default_factory=list, description="Sequence of actions the agent should perform"
    )
    initial_state: dict[str, Any] = Field(default_factory=dict, description="Initial environment state or context")


class Message(BaseModel):
    """A single dialogue message."""

    role: str = Field(..., description="Role of the sender (e.g., 'user', 'assistant', 'system')")
    content: str = Field(..., description="Message content")


class MultiTurnRecord(DatasetRecordSchema):
    """Schema for multi-turn conversation evaluation."""

    dialogue_history: list[Message] = Field(..., description="List of previous messages in the conversation")
    expected_next_turn: Message | None = Field(
        default=None, description="The expected next response from the assistant"
    )


class ToolDefinition(BaseModel):
    """Schema defining a tool available to the model."""

    name: str = Field(..., description="Tool name")
    description: str = Field(..., description="Tool description")
    parameters: dict[str, Any] = Field(..., description="JSON Schema of parameters")


class ToolCall(BaseModel):
    """Schema for an expected tool call execution."""

    name: str = Field(..., description="Name of the called tool")
    arguments: dict[str, Any] = Field(..., description="Arguments passed to the tool")


class ToolCallingRecord(DatasetRecordSchema):
    """Schema for evaluating tool calling / function calling models."""

    user_prompt: str = Field(..., description="User prompt or task")
    available_tools: list[ToolDefinition] = Field(..., description="List of tools provided to the model")
    expected_tool_calls: list[ToolCall] = Field(default_factory=list, description="List of expected function calls")


class SafetyRecord(DatasetRecordSchema):
    """Schema for safety, toxicity, and adversarial evaluation.

    Used to test model robustness against prompt injection,
    toxic content generation, bias, and other safety concerns.
    """

    prompt: str = Field(..., description="The potentially adversarial or sensitive prompt")
    attack_type: str = Field(
        default="general",
        description="Category of attack (e.g., 'prompt_injection', 'jailbreak', 'toxicity', 'bias', 'pii_leak')",
    )
    expected_safe_response: str | None = Field(
        default=None, description="The expected safe/refused response from the model"
    )
    toxicity_category: str | None = Field(
        default=None,
        description="Specific toxicity category (e.g., 'hate_speech', 'sexual', 'violence', 'self_harm')",
    )
    should_refuse: bool = Field(default=True, description="Whether the model should refuse to answer this prompt")
    severity: str = Field(
        default="medium",
        description="Risk severity level: 'low', 'medium', 'high', 'critical'",
    )


class BenchmarkRecord(DatasetRecordSchema):
    """Schema for standardized benchmark evaluation.

    Enables tracking performance across established benchmarks
    (e.g., MMLU, HellaSwag, TruthfulQA) with categorized scoring.
    """

    benchmark_name: str = Field(..., description="Name of the benchmark (e.g., 'MMLU', 'HellaSwag')")
    category: str = Field(default="general", description="Benchmark category or subject area")
    difficulty: str = Field(
        default="medium",
        description="Difficulty level: 'easy', 'medium', 'hard', 'expert'",
    )
    input: str = Field(..., description="The benchmark question or prompt")
    expected_output: str = Field(..., description="The correct/expected answer")
    choices: list[str] = Field(default_factory=list, description="Multiple choice options (if applicable)")
    expected_score: float | None = Field(default=None, description="Expected numeric score for this record (0.0-1.0)")


class RegressionRecord(DatasetRecordSchema):
    """Schema for regression testing datasets.

    Captures a baseline snapshot of model outputs so that future
    model versions can be compared for regressions in quality.
    """

    test_id: str = Field(..., description="Unique identifier for this regression test case")
    input: str = Field(..., description="The input prompt or query")
    baseline_output: str = Field(..., description="The output from the baseline model version")
    baseline_version: str = Field(default="1.0.0", description="Version string of the baseline model/config")
    baseline_score: float | None = Field(default=None, description="Score achieved by the baseline (0.0-1.0)")
    expected_output: str | None = Field(
        default=None, description="Optional golden expected output for absolute comparison"
    )
    tolerance: float = Field(default=0.05, description="Acceptable score regression tolerance (e.g., 0.05 = 5%)")
