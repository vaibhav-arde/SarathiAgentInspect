"""Notification registry and plugin system.

Allows for dynamic registration and invocation of different
notification providers (Slack, Teams, etc.).
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from sarathi_agent_inspect.reporting.base import EvaluationSummary

logger = logging.getLogger(__name__)


class BaseNotifier(ABC):
    """Abstract base class for all notifiers."""

    @abstractmethod
    def send_summary(self, summary: EvaluationSummary, trend: dict[str, Any] | None = None) -> None:
        """Send evaluation summary via the notifier channel."""
        ...


class NotifierRegistry:
    """Registry for managing notification providers."""

    _registry: ClassVar[dict[str, type[BaseNotifier]]] = {}

    @classmethod
    def register(cls, name: str) -> Any:
        """Register a notifier class."""

        def decorator(notifier_cls: type[BaseNotifier]) -> type[BaseNotifier]:
            cls._registry[name.lower()] = notifier_cls
            return notifier_cls

        return decorator

    @classmethod
    def get_notifier(cls, name: str, **kwargs: Any) -> BaseNotifier:
        """Factory method to create a notifier instance."""
        notifier_cls = cls._registry.get(name.lower())
        if not notifier_cls:
            raise ValueError(f"Unknown notifier provider: {name}")
        return notifier_cls(**kwargs)
