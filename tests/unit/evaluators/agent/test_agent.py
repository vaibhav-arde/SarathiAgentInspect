"""Unit tests for the AI Agent Evaluation System."""

from sarathi_agent_inspect.evaluators.agent import (
    AgentSpan,
    AgentStep,
    AgentTrace,
    InfiniteLoopProtector,
    LoopDetector,
    MemoryRetentionEvaluator,
    ReasoningEvaluator,
    ReplayEngine,
    StepType,
    ToolEvaluator,
    TraceScorer,
)


def test_agent_trace_efficiency() -> None:
    """Test the efficiency scoring of an agent trace."""
    trace = AgentTrace(trace_id="t1", input_text="Find a flight")
    span = AgentSpan(span_id="s1", name="search")

    # 1 thought followed by 1 action = 1.0 efficiency
    span.add_step(AgentStep(step_id="1", type=StepType.THOUGHT, content="I need to search"))
    span.add_step(AgentStep(step_id="2", type=StepType.ACTION, content="search_flights()"))

    # Another thought without an action = reduces efficiency
    span.add_step(AgentStep(step_id="3", type=StepType.THOUGHT, content="Still thinking..."))

    trace.add_span(span)

    scorer = TraceScorer()
    # 1 action / 2 thoughts = 0.5 efficiency
    assert scorer.calculate_efficiency(trace) == 0.5


def test_tool_evaluator_schema() -> None:
    """Test strict schema validation for tool calls (Dict and Pydantic)."""
    evaluator = ToolEvaluator()

    # Test Dict Schema
    dict_schema = {"location": str, "date": str}
    assert evaluator.validate_schema('{"location": "SF", "date": "today"}', dict_schema) is True
    assert evaluator.validate_schema('{"location": "SF"}', dict_schema) is False

    # Test Pydantic Schema
    from pydantic import BaseModel

    class SearchSchema(BaseModel):
        location: str
        date: str

    assert evaluator.validate_schema('{"location": "SF", "date": "today"}', SearchSchema) is True
    assert evaluator.validate_schema('{"location": "SF"}', SearchSchema) is False


def test_loop_detection() -> None:
    """Test detection of repeating action sequences."""
    detector = LoopDetector()

    actions = ["search", "click", "read", "search", "click", "read"]
    assert detector.detect_action_loops(actions, window_size=3) is True

    actions = ["search", "click", "read", "back"]
    assert detector.detect_action_loops(actions, window_size=2) is False


def test_infinite_loop_protector() -> None:
    """Test active monitoring for infinite loops."""
    protector = InfiniteLoopProtector(max_steps=5, max_repeats=3)

    # Repeat "search" twice (within max_repeats)
    assert protector.should_terminate("search") is False
    assert protector.should_terminate("search") is False

    # Third time should flag for termination
    assert protector.should_terminate("search") is True


def test_memory_retention() -> None:
    """Test if agent remembers facts from earlier steps."""
    trace = AgentTrace(trace_id="t1", input_text="Test memory")
    span = AgentSpan(span_id="s1", name="mem")
    span.add_step(AgentStep(step_id="1", type=StepType.OBSERVATION, content="The user's age is 30"))
    span.add_step(
        AgentStep(
            step_id="2",
            type=StepType.THOUGHT,
            content="I should recommend products for a 30 year old",
        )
    )
    trace.add_span(span)

    evaluator = MemoryRetentionEvaluator(trace)
    assert evaluator.check_fact_usage("30") is True
    assert evaluator.check_fact_usage("unknown") is False


def test_replay_engine_mock_generation() -> None:
    """Test extraction of mock responses from a trace."""
    trace = AgentTrace(trace_id="t1", input_text="Replay test")
    span = AgentSpan(span_id="s1", name="replay")
    span.add_step(AgentStep(step_id="1", type=StepType.ACTION, content="get_weather(city='London')"))
    span.add_step(AgentStep(step_id="2", type=StepType.OBSERVATION, content="Sunny, 20C"))
    trace.add_span(span)

    engine = ReplayEngine(trace)
    mocks = engine.get_mock_responses()

    assert mocks["get_weather(city='London')"] == "Sunny, 20C"


def test_reasoning_redundancy() -> None:
    """Test redundancy detection in reasoning chains."""
    thoughts = ["I need to search", "I will search now", "I need to search"]
    score = ReasoningEvaluator.detect_redundancy(thoughts)
    # 2 unique / 3 total = 0.66
    assert 0.6 < score < 0.7
