"""Security Sanitization for LLM Inputs.

Protects LLM-as-a-judge models from prompt injection and adversarial
inputs by neutralizing known attack patterns.
"""

from __future__ import annotations

import re
from dataclasses import asdict, is_dataclass
from typing import Any, ClassVar


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
    SENSITIVE_KEY_PATTERNS: ClassVar[list[str]] = [
        r"(?i)api[_-]?key",
        r"(?i)secret",
        r"(?i)(access|refresh|bearer)[_-]?token",
        r"(?i)password",
        r"(?i)authorization",
        r"(?i)cookie",
        r"(?i)session",
    ]
    SECRET_VALUE_PATTERNS: ClassVar[list[str]] = [
        r"(?i)\bbearer\s+[a-z0-9._\-]+\b",
        r"\bsk-[A-Za-z0-9]{12,}\b",
        r"\bAIza[0-9A-Za-z\-_]{20,}\b",
        r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
    ]
    REDACTED: ClassVar[str] = "[REDACTED]"

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
        return cls._redact_secret_values(sanitized)

    @classmethod
    def is_sensitive_key(cls, key: str) -> bool:
        """Return True when a field name likely contains secret material."""
        return any(re.search(pattern, key) for pattern in cls.SENSITIVE_KEY_PATTERNS)

    @classmethod
    def _redact_secret_values(cls, text: str) -> str:
        """Redact common token and key formats from free-form text."""
        redacted = text
        for pattern in cls.SECRET_VALUE_PATTERNS:
            redacted = re.sub(pattern, cls.REDACTED, redacted)
        return redacted

    @classmethod
    def sanitize_for_export(cls, value: Any, field_name: str | None = None) -> Any:
        """Recursively sanitize structured payloads before persistence.

        This applies both prompt-injection neutralization and secret redaction
        across nested dict/list artifacts.
        """
        if isinstance(value, dict):
            return {key: cls.sanitize_for_export(item, field_name=key) for key, item in value.items()}

        if isinstance(value, list):
            return [cls.sanitize_for_export(item, field_name=field_name) for item in value]

        if isinstance(value, tuple):
            return [cls.sanitize_for_export(item, field_name=field_name) for item in value]

        if is_dataclass(value) and not isinstance(value, type):
            return cls.sanitize_for_export(asdict(value), field_name=field_name)

        if hasattr(value, "model_dump"):
            return cls.sanitize_for_export(value.model_dump(), field_name=field_name)

        if hasattr(value, "to_dict"):
            return cls.sanitize_for_export(value.to_dict(), field_name=field_name)

        if isinstance(value, str):
            if field_name and cls.is_sensitive_key(field_name):
                return cls.REDACTED
            return cls.sanitize(value)

        return value
