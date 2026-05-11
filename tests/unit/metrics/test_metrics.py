"""Unit tests for metrics configuration, registry, and execution."""

from pathlib import Path

import pytest

from sarathi_agent_inspect.metrics.base import BaseMetric, MetricResult
from sarathi_agent_inspect.metrics.composite import CompositeMetric
from sarathi_agent_inspect.metrics.config import MetricConfig
from sarathi_agent_inspect.metrics.execution import MetricExecutor
from sarathi_agent_inspect.metrics.observability import MetricObserver
from sarathi_agent_inspect.metrics.registry import MetricRegistry
from sarathi_agent_inspect.metrics.sdk import SimpleAsyncMetric, metric

# --- Test Configuration ---

def test_metric_config_strict():
    """Test strict pass/fail."""
    config = MetricConfig(threshold=0.8, strict_mode=True, tolerance=0.1)
    assert config.is_passing(0.8) is True
    assert config.is_passing(0.79) is False


def test_metric_config_soft():
    """Test soft pass/fail with tolerance."""
    config = MetricConfig(threshold=0.8, strict_mode=False, tolerance=0.1)
    assert config.is_passing(0.8) is True
    assert config.is_passing(0.75) is True  # Within tolerance
    assert config.is_passing(0.69) is False  # Outside tolerance


# --- Test Registry ---

def test_registry_registration_and_retrieval():
    """Test registering and fetching a metric."""
    MetricRegistry.clear()

    @metric("test_metric_abc")
    class DummyMetric(BaseMetric):
        @property
        def metric_name(self): return "test_metric_abc"
        @property
        def description(self): return "Dummy"
        async def compute(self, **kwargs):
            return MetricResult(self.metric_name, 1.0, True, 0.5)

    assert "test_metric_abc" in MetricRegistry.list_metrics()

    cls = MetricRegistry.get_metric_class("test_metric_abc")
    assert cls is DummyMetric

    instance = MetricRegistry.create_metric("test_metric_abc")
    assert isinstance(instance, DummyMetric)


def test_registry_missing():
    """Test retrieving non-existent metric."""
    MetricRegistry.clear()
    with pytest.raises(KeyError):
        MetricRegistry.get_metric_class("not_found")


# --- Test SDK / Simple Metric ---

@pytest.mark.asyncio
async def test_simple_async_metric():
    """Test SDK SimpleAsyncMetric creation."""
    async def my_eval(input_text, actual_output, **kwargs):
        return 0.9

    sm = SimpleAsyncMetric("my_simple", my_eval)
    result = await sm.compute(input_text="in", actual_output="out")

    assert result.metric_name == "my_simple"
    assert result.score == 0.9


# --- Test Composite Metric ---

@pytest.mark.asyncio
async def test_composite_metric_mean():
    """Test composite metric aggregation (mean)."""
    async def eval1(**kwargs): return 1.0
    async def eval2(**kwargs): return 0.5

    m1 = SimpleAsyncMetric("m1", eval1)
    m2 = SimpleAsyncMetric("m2", eval2)

    comp = CompositeMetric("comp_mean", [m1, m2], aggregation="mean", threshold=0.7)

    result = await comp.compute(input_text="", actual_output="")
    assert result.score == 0.75  # (1.0 + 0.5) / 2
    assert result.passed is True


@pytest.mark.asyncio
async def test_composite_metric_weighted():
    """Test composite metric aggregation (weighted)."""
    async def eval1(**kwargs): return 1.0
    async def eval2(**kwargs): return 0.0

    m1 = SimpleAsyncMetric("m1", eval1)
    m2 = SimpleAsyncMetric("m2", eval2)

    comp = CompositeMetric("comp_weighted", [m1, m2], weights=[0.8, 0.2], aggregation="weighted", threshold=0.7)

    result = await comp.compute(input_text="", actual_output="")
    assert result.score == 0.8  # (1.0*0.8 + 0.0*0.2) / 1.0
    assert result.passed is True


# --- Test Execution ---

@pytest.mark.asyncio
async def test_metric_executor_retry():
    """Test the executor retries failures."""

    class FlakyMetric(BaseMetric):
        def __init__(self):
            self.calls = 0

        @property
        def metric_name(self): return "flaky"
        @property
        def description(self): return ""

        async def compute(self, **kwargs):
            self.calls += 1
            if self.calls < 3:
                raise ValueError("Random failure")
            return MetricResult(self.metric_name, 0.9, True, 0.5)

    metric_obj = FlakyMetric()
    config = MetricConfig(threshold=0.5)
    executor = MetricExecutor(max_retries=3, retry_delay=0.01)

    result = await executor.execute_metric(
        metric_obj, config, input_text="in", actual_output="out"
    )

    assert result.score == 0.9
    assert metric_obj.calls == 3


@pytest.mark.asyncio
async def test_metric_executor_normalization():
    """Test the executor normalizes scores."""

    class RawMetric(BaseMetric):
        @property
        def metric_name(self): return "raw"
        @property
        def description(self): return ""
        @property
        def score_range(self): return (0.0, 10.0)

        async def compute(self, **kwargs):
            return MetricResult(self.metric_name, 8.0, True, 5.0)

    metric_obj = RawMetric()
    config = MetricConfig(normalization_strategy="min_max")
    executor = MetricExecutor()

    result = await executor.execute_metric(
        metric_obj, config, input_text="in", actual_output="out"
    )

    # 8.0 on a 0-10 scale should normalize to 0.8
    assert result.score == 0.8


# --- Test Observability ---

def test_metric_observer(tmp_path):
    """Test metric tracing and persistence."""
    obs = MetricObserver(storage_dir=tmp_path)

    results = [
        MetricResult("m1", 0.9, True, 0.5, "Good"),
        MetricResult("m2", 0.2, False, 0.5, "Bad"),
    ]

    # Trace execution (just logging)
    obs.trace_execution("test_123", results)

    # Persist
    file_path = obs.persist_results("run_123", "test_123", results, {"meta": "data"})

    assert Path(file_path).exists()

    import json
    with open(file_path) as f:
        data = json.load(f)

    assert data["test_id"] == "test_123"
    assert len(data["metrics"]) == 2
    assert data["metrics"][0]["metric_name"] == "m1"
