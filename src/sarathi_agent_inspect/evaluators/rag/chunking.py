"""Chunking and context evaluation utilities.

Provides analysis of chunk overlap and optimal context ranking.
"""

from __future__ import annotations

from typing import Protocol


class Tokenizer(Protocol):
    """Abstract interface for pluggable tokenizers."""
    def tokenize(self, text: str) -> list[str]:
        ...


class DefaultTokenizer:
    """Lightweight reference tokenizer using simple whitespace splitting."""
    def tokenize(self, text: str) -> list[str]:
        return text.lower().split()


class ChunkAnalyzer:
    """Analyzes chunk effectiveness in retrieval context."""

    def __init__(self, tokenizer: Tokenizer | None = None) -> None:
        self.tokenizer = tokenizer or DefaultTokenizer()

    def calculate_overlap_ratio(self, chunks: list[str]) -> float:
        """Calculate a rough heuristic of token overlap across chunks.

        Enterprise RAG often fails when chunks have too little overlap
        (missing context at boundaries) or too much overlap (wasted context window).
        """
        if len(chunks) < 2:
            return 0.0

        total_overlap_tokens = 0
        total_tokens = 0

        for i in range(len(chunks) - 1):
            chunk1_tokens = set(self.tokenizer.tokenize(chunks[i]))
            chunk2_tokens = set(self.tokenizer.tokenize(chunks[i+1]))

            overlap = len(chunk1_tokens.intersection(chunk2_tokens))
            total_overlap_tokens += overlap
            total_tokens += len(chunk1_tokens)

        # Add the last chunk's tokens
        total_tokens += len(self.tokenizer.tokenize(chunks[-1]))

        if total_tokens == 0:
            return 0.0

        return total_overlap_tokens / total_tokens

    @staticmethod
    def rank_context_distribution(retrieved_context: list[str], expected_snippets: list[str]) -> list[int]:
        """Evaluate where the golden expected snippets appear in the context.

        Returns the indices of the retrieved chunks that contain the expected snippets.
        Ideally, these should be ranked near 0. If they are at the end,
        position bias in LLMs might cause them to be ignored.
        """
        found_indices = []
        for snippet in expected_snippets:
            snippet_lower = snippet.lower()
            for idx, chunk in enumerate(retrieved_context):
                if snippet_lower in chunk.lower():
                    found_indices.append(idx)
                    break

        return found_indices
