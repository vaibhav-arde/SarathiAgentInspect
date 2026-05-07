"""Provider registry and factory pattern.

Enables dynamic registration and instantiation of LLM providers
based on configuration settings, keeping the core framework decoupled
from specific implementations.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any, ClassVar

from sarathi_agent_inspect.core.exceptions.base import ConfigurationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from sarathi_agent_inspect.core.config.settings import SarathiSettings
    from sarathi_agent_inspect.providers.base import BaseProvider


class ProviderRegistry:
    """Thread-safe registry for LLM providers.

    Maintains a mapping of provider names to their implementing classes.
    """

    _registry: ClassVar[dict[str, type[BaseProvider]]] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def register(cls, name: str, provider_cls: type[BaseProvider]) -> None:
        """Register a provider class under a given name.

        Args:
            name: The provider identifier (e.g., 'ollama').
            provider_cls: The class implementing BaseProvider.
        """
        with cls._lock:
            cls._registry[name] = provider_cls

    @classmethod
    def get(cls, name: str) -> type[BaseProvider]:
        """Retrieve a provider class by name.

        Args:
            name: The provider identifier.

        Returns:
            The registered provider class.

        Raises:
            ConfigurationError: If the provider is not registered.
        """
        with cls._lock:
            if name not in cls._registry:
                available = ", ".join(cls._registry.keys())
                raise ConfigurationError(
                    message=f"Unknown provider: '{name}'. Available providers: [{available}]",
                    context={"provider": name, "available": list(cls._registry.keys())},
                )
            return cls._registry[name]

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        with cls._lock:
            return list(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (useful for testing)."""
        with cls._lock:
            cls._registry.clear()


def register_provider(name: str) -> Callable[[type[BaseProvider]], type[BaseProvider]]:
    """Decorator to automatically register a provider class.

    Args:
        name: The provider identifier.

    Returns:
        Decorator function.
    """

    def decorator(cls: type[BaseProvider]) -> type[BaseProvider]:
        ProviderRegistry.register(name, cls)
        return cls

    return decorator


class ProviderFactory:
    """Factory for instantiating providers based on settings."""

    @staticmethod
    def create(
        settings: SarathiSettings,
        provider_name: str | None = None,
        **kwargs: Any,
    ) -> BaseProvider:
        """Create a provider instance.

        Args:
            settings: The framework configuration.
            provider_name: Optional provider name to override the default.
            **kwargs: Additional arguments passed to the provider constructor.

        Returns:
            An instance of the requested provider.
        """
        name = provider_name or settings.provider.default
        provider_cls = ProviderRegistry.get(name)
        return provider_cls(settings=settings, **kwargs)
