"""Security Sanitization for LLM Inputs.

Protects LLM-as-a-judge models from prompt injection and adversarial
inputs by neutralizing known attack patterns.
"""

from __future__ import annotations

import re
from typing import ClassVar


class InputSanitizer:
    """Detects and neutralizes malicious patterns in inputs."""

    INJECTION_PATTERNS: ClassVar[list[str]] = [
        r"(?i)ignore all previous instructions",
        r"(?i)you are now an? (unfiltered|unrestricted)",
        r"(?i)disregard the system prompt",
        r"(?i)switch to (developer|god) mode",
        r"(?i)forget everything you know",
        r"(?i)system check: override",
    ]

    @classmethod
    def is_clean(cls, text: str) -> bool:
        """Check if the text contains any known injection patterns."""
        return all(not re.search(pattern, text) for pattern in cls.INJECTION_PATTERNS)

    @classmethod
    def sanitize(cls, text: str) -> str:
        """Neutralize known patterns by adding delimiters or warning tokens."""
        sanitized = text
        for pattern in cls.INJECTION_PATTERNS:
            # We don't delete the text (to preserve eval intent),
            # but we wrap it in a 'potentially malicious' tag
            # to warn the judge model.
            sanitized = re.sub(pattern, lambda m: f"[POTENTIAL INJECTION DETECTED: {m.group(0)}]", sanitized)
        return sanitized
