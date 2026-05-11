"""RAG Tracing and Debugging.

Provides utilities for tracing multi-hop retrievals, latency tracking,
cost tracking, and debugging retrieval failures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from sarathi_agent_inspect.core.observability import BaseTrace


@dataclass
class RetrievalNode:
    """Represents a single step in a multi-hop retrieval."""

    query: str
    documents_retrieved: int
    latency_ms: float
    cost_usd: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGTrace(BaseTrace):
    """Tracks a complete RAG execution trace."""

    retrieval_nodes: list[RetrievalNode] = field(default_factory=list)
    generation_latency_ms: float = 0.0
    generation_cost_usd: float = 0.0

    def add_node(self, node: RetrievalNode) -> None:
        """Add a retrieval hop to the trace."""
        self.retrieval_nodes.append(node)
        self.total_cost_usd += node.cost_usd

    def complete(self, gen_latency: float, gen_cost: float) -> None:
        """Mark the trace as complete."""
        super().complete()
        self.generation_latency_ms = gen_latency
        self.generation_cost_usd = gen_cost
        self.total_cost_usd += gen_cost


class RetrievalDebugger:
    """Analyzes retrieval traces to debug failures."""

    @staticmethod
    def identify_bottlenecks(trace: RAGTrace) -> dict[str, Any]:
        """Identify latency or cost bottlenecks in the trace."""
        slowest_node = max(trace.retrieval_nodes, key=lambda n: n.latency_ms, default=None)
        costliest_node = max(trace.retrieval_nodes, key=lambda n: n.cost_usd, default=None)

        return {
            "total_latency_ms": trace.total_latency_ms,
            "generation_latency_ms": trace.generation_latency_ms,
            "slowest_retrieval_query": slowest_node.query if slowest_node else None,
            "slowest_retrieval_ms": slowest_node.latency_ms if slowest_node else 0.0,
            "total_cost_usd": trace.total_cost_usd,
            "costliest_query": costliest_node.query if costliest_node else None,
        }

    @staticmethod
    def missing_context_analysis(retrieved_context: list[str], golden_context: list[str]) -> dict[str, Any]:
        """Analyze if critical expected documents were entirely missed.

        A simple set difference to find which golden docs were dropped.
        """
        retrieved_set = set(retrieved_context)
        golden_set = set(golden_context)

        missed = list(golden_set - retrieved_set)
        extra = list(retrieved_set - golden_set)

        return {
            "missed_documents": len(missed),
            "extra_documents": len(extra),
            "recall_ratio": len(golden_set.intersection(retrieved_set)) / len(golden_set) if golden_set else 1.0,
        }
