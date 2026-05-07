"""Synthetic dataset generation hooks.

Provides tools for generating synthetic edge-cases, permutations,
and golden datasets using LLM providers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sarathi_agent_inspect.core.types import DatasetRecord
    from sarathi_agent_inspect.providers.base import BaseProvider


class SyntheticGenerator:
    """Utility to generate or augment dataset records via LLM."""

    def __init__(self, provider: BaseProvider) -> None:
        """Initialize the synthetic generator with an LLM provider.

        Args:
            provider: The configured LLM provider to use for generation.
        """
        self.provider = provider

    async def generate_edge_cases(
        self, base_record: DatasetRecord, count: int = 3
    ) -> list[DatasetRecord]:
        """Generate synthetic edge-cases based on a golden record.

        Args:
            base_record: The reference golden record.
            count: Number of edge cases to generate.

        Returns:
            List of mutated records designed to test boundaries.
        """
        import json

        prompt = (
            f"Given the following dataset record: {json.dumps(base_record)}\n\n"
            f"Generate {count} extreme edge-case variations of this record. "
            "Return ONLY a valid JSON array containing the new records. "
            "Do not wrap in markdown or backticks."
        )

        response = await self.provider.generate(
            prompt=prompt,
            system_prompt="You are a QA automation expert generating edge-cases.",
            temperature=0.8,
        )

        try:
            raw_text = response.content.strip()
            # Clean potential markdown wrapping
            if raw_text.startswith("```json"):
                raw_text = raw_text[7:]
            if raw_text.endswith("```"):
                raw_text = raw_text[:-3]

            variations = json.loads(raw_text.strip())
            if not isinstance(variations, list):
                variations = [variations]

            # Tag the new records
            for v in variations:
                if "metadata" not in v:
                    v["metadata"] = {}
                v["metadata"]["synthetic"] = True
                v["metadata"]["type"] = "edge_case"

            return variations
        except Exception as e:
            raise RuntimeError(f"Failed to generate synthetic edge-cases: {e}") from e
