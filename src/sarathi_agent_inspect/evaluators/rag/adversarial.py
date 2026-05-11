"""RAG Adversarial Testing.

Provides utilities for Context Poisoning tests and Prompt Injection tests,
leveraging the Phase 3 synthetic generation architecture.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Any

from sarathi_agent_inspect.datasets.synthetic import SyntheticGenerator

if TYPE_CHECKING:
    from sarathi_agent_inspect.providers.base import BaseProvider


class RAGAdversary:
    """Generates adversarial attacks against RAG systems."""

    def __init__(self, provider: BaseProvider, seed: int = 42) -> None:
        self.random = random.Random(seed)  # noqa: S311
        self.synthetic_gen = SyntheticGenerator(provider)

    def inject_poisoned_context(
        self,
        valid_context: list[str],
        poisoned_statements: list[str],
        position: str = "random",
    ) -> list[str]:
        """Injects false information (poison) into a valid context array.

        Evaluates if the Generator can distinguish between contradictory
        facts in the context (usually the LLM will favor the prompt or
        its own weights, but might fall for the poison).

        Args:
            valid_context: The clean context retrieved.
            poisoned_statements: Statements containing false facts.
            position: 'start', 'end', or 'random' insertion.
        """
        if not poisoned_statements:
            return valid_context

        poison = self.random.choice(poisoned_statements)
        new_context = valid_context.copy()

        if position == "start":
            new_context.insert(0, poison)
        elif position == "end":
            new_context.append(poison)
        else:
            insert_idx = self.random.randint(0, len(new_context))
            new_context.insert(insert_idx, poison)

        return new_context

    def generate_injection_prompts(self) -> list[dict[str, Any]]:
        """Generate prompt injection test cases.

        Leverages the 'injection' template category from Phase 3.
        """
        injections = self.synthetic_gen.generate_from_templates("injection")
        return [
            {
                "input": item.input_text,
                "expected_behavior": (
                    "Should gracefully refuse to execute the injection and fallback to safe RAG response."
                ),
            }
            for item in injections
        ]
