"""Metric registry and dynamic loading system.

Provides a centralized registry for all available metrics, enabling
dynamic loading by name and plugin-based architecture for custom metrics.
"""

from __future__ import annotations

import importlib
import logging
import typing
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from sarathi_agent_inspect.metrics.base import BaseMetric

logger = logging.getLogger(__name__)


class MetricRegistry:
    """Central registry for evaluation metrics.

    Follows the Singleton pattern internally to maintain a global map
    of metric_name -> MetricClass.
    """

    _registry: typing.ClassVar[dict[str, type[BaseMetric]]] = {}

    @classmethod
    def register(cls, name: str | None = None) -> Callable[[type[BaseMetric]], type[BaseMetric]]:
        """Decorator to register a metric class.

        Args:
            name: Optional explicit name. Defaults to the class's metric_name
                  property or class name if not instantiated.
        """

        def decorator(metric_cls: type[BaseMetric]) -> type[BaseMetric]:
            registry_name = name
            if not registry_name:
                # Try to get metric_name from the class if defined
                if hasattr(metric_cls, "metric_name") and isinstance(metric_cls.metric_name, str):
                    registry_name = metric_cls.metric_name
                elif hasattr(metric_cls, "metric_name") and isinstance(metric_cls.metric_name, property):
                    # It's a property without an instance, fallback to class name
                    registry_name = metric_cls.__name__
                else:
                    registry_name = metric_cls.__name__

            if registry_name in cls._registry:
                logger.warning(f"Overwriting existing metric registration for '{registry_name}'")

            cls._registry[registry_name] = metric_cls
            logger.debug(f"Registered metric: '{registry_name}'")
            return metric_cls

        return decorator

    @classmethod
    def get_metric_class(cls, name: str) -> type[BaseMetric]:
        """Retrieve a metric class by name.

        Args:
            name: The registered name of the metric.

        Returns:
            The metric class.

        Raises:
            KeyError: If the metric is not registered.
        """
        if name not in cls._registry:
            raise KeyError(f"Metric '{name}' not found in registry.")
        return cls._registry[name]

    @classmethod
    def create_metric(cls, name: str, **kwargs: Any) -> BaseMetric:
        """Instantiate a metric by name.

        Args:
            name: The registered name of the metric.
            **kwargs: Arguments passed to the metric constructor.

        Returns:
            An instance of the requested metric.
        """
        metric_cls = cls.get_metric_class(name)
        return metric_cls(**kwargs)

    @classmethod
    def list_metrics(cls) -> list[str]:
        """List all registered metric names."""
        return sorted(cls._registry.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear the registry (useful for testing)."""
        cls._registry.clear()

    @classmethod
    def load_plugin(cls, module_path: str) -> None:
        """Dynamically load an external metric module/plugin.

        Args:
            module_path: Python import path (e.g., 'my_company.metrics.custom_geval').

        Raises:
            ImportError: If the module cannot be loaded.
        """
        try:
            importlib.import_module(module_path)
            logger.info(f"Successfully loaded metric plugin: {module_path}")
        except ImportError as e:
            logger.error(f"Failed to load metric plugin '{module_path}': {e}")
            raise
