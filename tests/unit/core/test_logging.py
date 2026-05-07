"""Unit tests for the structured logging system.

Tests:
- Logger creation with context
- Logging configuration for different formats
- Context binding
"""

from __future__ import annotations

from sarathi_agent_inspect.core.logging.logger import configure_logging, get_logger


class TestGetLogger:
    """Tests for the get_logger factory function."""

    def test_returns_bound_logger(self) -> None:
        log = get_logger("test.module")
        assert log is not None

    def test_logger_with_context(self) -> None:
        log = get_logger("test.module", provider="ollama", model="gemma4")
        # Verify it's a bound logger (structlog bound loggers are callable)
        assert log is not None

    def test_logger_can_log(self) -> None:
        log = get_logger("test.logging")
        # Should not raise
        log.info("test_message", key="value")
        log.debug("debug_message", count=42)
        log.warning("warning_message")


class TestConfigureLogging:
    """Tests for the configure_logging function."""

    def test_configure_json_format(self) -> None:
        configure_logging(level="INFO", log_format="json")
        log = get_logger("test.json")
        # Should not raise
        log.info("json_test")

    def test_configure_console_format(self) -> None:
        configure_logging(level="DEBUG", log_format="console")
        log = get_logger("test.console")
        # Should not raise
        log.debug("console_test")

    def test_configure_with_caller(self) -> None:
        configure_logging(
            level="DEBUG",
            log_format="console",
            include_caller=True,
        )
        log = get_logger("test.caller")
        log.info("caller_test")

    def test_configure_without_timestamp(self) -> None:
        configure_logging(
            level="INFO",
            log_format="json",
            include_timestamp=False,
        )
        log = get_logger("test.no_timestamp")
        log.info("no_timestamp_test")
