"""Structured logging configuration using structlog.

Provides:
- JSON output for CI/production (machine-parseable)
- Colored console output for local development (human-readable)
- Context binding (correlation IDs, provider info, etc.)
- Automatic caller information in debug mode
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(
    level: str = "INFO",
    log_format: str = "json",
    include_timestamp: bool = True,
    include_caller: bool = False,
) -> None:
    """Configure the global structlog logging pipeline.

    Should be called once during framework initialization.

    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        log_format: Output format — 'json' or 'console'.
        include_timestamp: Whether to include timestamps in output.
        include_caller: Whether to include caller file/line info.
    """
    # Shared processors applied to all log entries
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
    ]

    if include_timestamp:
        shared_processors.append(structlog.processors.TimeStamper(fmt="iso"))

    if include_caller:
        shared_processors.append(
            structlog.processors.CallsiteParameterAdder(
                parameters=[
                    structlog.processors.CallsiteParameter.FILENAME,
                    structlog.processors.CallsiteParameter.LINENO,
                    structlog.processors.CallsiteParameter.FUNC_NAME,
                ],
            )
        )

    shared_processors.append(structlog.processors.StackInfoRenderer())
    shared_processors.append(structlog.processors.UnicodeDecoder())

    # Choose renderer based on format
    if log_format == "console":
        renderer: structlog.types.Processor = structlog.dev.ConsoleRenderer(
            colors=True,
        )
    else:
        renderer = structlog.processors.JSONRenderer()

    # Configure structlog
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure stdlib logging to route through structlog
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Suppress noisy third-party loggers
    for noisy_logger in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)


def get_logger(name: str, **initial_context: Any) -> structlog.stdlib.BoundLogger:
    """Get a structured logger with optional initial context.

    Args:
        name: Logger name (typically __name__).
        **initial_context: Key-value pairs to bind to all log entries.

    Returns:
        A bound structlog logger instance.
    """
    log: structlog.stdlib.BoundLogger = structlog.get_logger(name)
    if initial_context:
        log = log.bind(**initial_context)
    return log
