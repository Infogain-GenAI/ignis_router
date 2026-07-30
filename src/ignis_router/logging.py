"""Centralized logging framework for ignis_router.

Provides structured JSON logging, correlation IDs for request tracing,
and a standardized configuration interface for operations engineers.

Usage:
    from ignis_router.logging import setup_logging, get_logger, correlation_context

    # Configure once at application startup
    setup_logging(level="INFO", format="json")

    # Get a logger
    logger = get_logger(__name__)

    # Use correlation context for request tracing
    with correlation_context() as correlation_id:
        logger.info("Processing request", extra={"query": "hello"})

Environment variables:
    IGNIS_LOG_LEVEL: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    IGNIS_LOG_FORMAT: Output format ("json" or "text")
    IGNIS_LOG_FILE: Optional file path for log output
    IGNIS_LOG_CORRELATION_HEADER: HTTP header name for correlation ID propagation
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import sys
import threading
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Generator, Optional

# Context variable for request correlation ID
_correlation_id: ContextVar[Optional[str]] = ContextVar("correlation_id", default=None)

# Module-level flag to prevent double-initialization
_initialized = False


def get_correlation_id() -> Optional[str]:
    """Get the current correlation ID from context."""
    return _correlation_id.get()


def set_correlation_id(cid: str) -> None:
    """Set the correlation ID for the current context."""
    _correlation_id.set(cid)


@contextlib.contextmanager
def correlation_context(correlation_id: Optional[str] = None) -> Generator[str, None, None]:
    """Context manager that sets a correlation ID for the duration of a block.

    Args:
        correlation_id: Optional ID to use. Generates a UUID4 if not provided.

    Yields:
        The active correlation ID.
    """
    cid = correlation_id or uuid.uuid4().hex[:16]
    token = _correlation_id.set(cid)
    try:
        yield cid
    finally:
        _correlation_id.reset(token)


class StructuredJsonFormatter(logging.Formatter):
    """Formats log records as single-line JSON for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Add correlation ID if present
        cid = _correlation_id.get()
        if cid:
            log_entry["correlation_id"] = cid

        # Add extra fields (excluding standard LogRecord attrs)
        _STANDARD_ATTRS = {
            "name", "msg", "args", "created", "relativeCreated", "thread",
            "threadName", "msecs", "filename", "funcName", "levelno", "levelname",
            "lineno", "module", "exc_info", "exc_text", "stack_info",
            "pathname", "processName", "process", "message", "taskName",
        }
        for key, value in record.__dict__.items():
            if key.startswith("_") or key in _STANDARD_ATTRS:
                continue
            log_entry[key] = value

        # Add exception info if present
        if record.exc_info and record.exc_info[1]:
            import traceback
            tb_lines = traceback.format_exception(*record.exc_info)
            log_entry["exception"] = {
                "type": type(record.exc_info[1]).__name__,
                "message": str(record.exc_info[1]),
                "traceback": "".join(tb_lines),
            }
            # Extract crash location from traceback
            tb = record.exc_info[2]
            if tb:
                # Walk to the innermost frame
                while tb.tb_next:
                    tb = tb.tb_next
                log_entry["crash_location"] = {
                    "file": tb.tb_frame.f_code.co_filename,
                    "line": tb.tb_lineno,
                    "function": tb.tb_frame.f_code.co_name,
                }

        # Always include source location for ERROR and CRITICAL
        if record.levelno >= logging.ERROR:
            log_entry["source"] = {
                "file": record.pathname,
                "line": record.lineno,
                "function": record.funcName,
            }

        return json.dumps(log_entry, default=str)


class StructuredTextFormatter(logging.Formatter):
    """Human-readable structured format with correlation ID and extra fields."""

    FORMAT = "%(asctime)s [%(levelname)-8s] %(name)s | %(message)s"

    def __init__(self) -> None:
        super().__init__(fmt=self.FORMAT, datefmt="%Y-%m-%d %H:%M:%S")

    def format(self, record: logging.LogRecord) -> str:
        # Prepend correlation ID if available
        cid = _correlation_id.get()
        if cid:
            record.msg = f"[{cid}] {record.msg}"

        return super().format(record)


def _install_crash_handler(logger_instance: logging.Logger) -> None:
    """Install a global exception hook that logs unhandled crashes."""
    _original_hook = sys.excepthook

    def _crash_hook(exc_type, exc_value, exc_tb):
        if issubclass(exc_type, KeyboardInterrupt):
            _original_hook(exc_type, exc_value, exc_tb)
            return
        logger_instance.critical(
            "UNHANDLED CRASH: %s: %s",
            exc_type.__name__,
            exc_value,
            exc_info=(exc_type, exc_value, exc_tb),
            extra={
                "event": "unhandled_crash",
                "error_type": exc_type.__name__,
                "error_message": str(exc_value),
            },
        )
        _original_hook(exc_type, exc_value, exc_tb)

    sys.excepthook = _crash_hook


def setup_logging(
    *,
    level: Optional[str] = None,
    format: Optional[str] = None,
    log_file: Optional[str] = None,
) -> None:
    """Configure the ignis_router logging framework.

    Should be called once at application startup. Subsequent calls are no-ops
    unless the module-level _initialized flag is reset.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
               Defaults to IGNIS_LOG_LEVEL env var or "INFO".
        format: Output format ("json" or "text").
                Defaults to IGNIS_LOG_FORMAT env var or "json".
        log_file: Optional file path to write logs to.
                  Defaults to IGNIS_LOG_FILE env var or None (stdout only).
    """
    global _initialized
    if _initialized:
        return

    # Load .env so IGNIS_LOG_* variables are available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    # Check ENABLE_LOGGING feature flag
    if os.getenv("ENABLE_LOGGING", "true").lower() in ("false", "0", "no"):
        # Disable all logging — set to WARNING-only with no handlers
        root_logger = logging.getLogger("ignis_router")
        root_logger.setLevel(logging.WARNING)
        root_logger.handlers.clear()
        root_logger.addHandler(logging.NullHandler())
        root_logger.propagate = False
        _initialized = True
        return

    log_level = (level or os.getenv("IGNIS_LOG_LEVEL", "INFO")).upper()
    log_format = (format or os.getenv("IGNIS_LOG_FORMAT", "json")).lower()
    log_file_path = log_file or os.getenv("IGNIS_LOG_FILE")
    log_console = os.getenv("IGNIS_LOG_CONSOLE", "true").lower() in ("true", "1", "yes")

    # Create formatter
    if log_format == "json":
        formatter = StructuredJsonFormatter()
    else:
        formatter = StructuredTextFormatter()

    # Configure root ignis_router logger
    root_logger = logging.getLogger("ignis_router")
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    root_logger.handlers.clear()

    # Console handler (stdout) — can be disabled via IGNIS_LOG_CONSOLE=false
    if log_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler (optional)
    if log_file_path:
        from pathlib import Path
        Path(log_file_path).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate output
    root_logger.propagate = False

    # Install global unhandled exception hook
    _install_crash_handler(root_logger)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the ignis_router namespace.

    Args:
        name: Module name (typically __name__).

    Returns:
        A configured logger instance.
    """
    # Ensure logging is initialized with defaults if not already done
    if not _initialized:
        setup_logging()
    return logging.getLogger(name)


class RequestLogger:
    """High-level logger for routing request lifecycle events.

    Provides structured methods for logging requests, routing decisions,
    and failures with consistent field names.
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._logger = logger or get_logger("ignis_router.requests")

    def log_request(
        self,
        query: str,
        *,
        correlation_id: Optional[str] = None,
        source: str = "unknown",
        **extra: Any,
    ) -> None:
        """Log an incoming routing request."""
        self._logger.info(
            "Routing request received",
            extra={
                "event": "request_received",
                "query_length": len(query),
                "query_preview": query[:100],
                "source": source,
                **extra,
            },
        )

    def log_routing_decision(
        self,
        query: str,
        *,
        selected_model: str,
        provider: str,
        intent: str,
        complexity: str,
        confidence: float,
        latency_ms: float,
        selection_mode: str = "",
        ml_predicted: str = "",
        strategy: str = "",
        **extra: Any,
    ) -> None:
        """Log a completed routing decision."""
        self._logger.info(
            "Routing decision completed",
            extra={
                "event": "routing_decision",
                "query_length": len(query),
                "query_preview": query[:100],
                "selected_model": selected_model,
                "provider": provider,
                "intent": intent,
                "complexity": complexity,
                "confidence": round(confidence, 4),
                "latency_ms": round(latency_ms, 2),
                "selection_mode": selection_mode,
                "ml_predicted": ml_predicted,
                "strategy": strategy,
                **extra,
            },
        )

    def log_failure(
        self,
        query: str,
        *,
        error_type: str,
        error_message: str,
        phase: str = "routing",
        exception: Optional[BaseException] = None,
        **extra: Any,
    ) -> None:
        """Log a routing or LLM failure with full crash details.

        If `exception` is provided, the full traceback, crash file, line number,
        and function name are included in the log entry.
        """
        self._logger.error(
            "Routing failure occurred",
            exc_info=exception if exception else None,
            extra={
                "event": "routing_failure",
                "query_length": len(query),
                "query_preview": query[:100],
                "error_type": error_type,
                "error_message": error_message,
                "phase": phase,
                **extra,
            },
        )

    def log_fallback(
        self,
        query: str,
        *,
        original_model: str,
        fallback_model: str,
        reason: str,
        **extra: Any,
    ) -> None:
        """Log a fallback event when primary model is unavailable."""
        self._logger.warning(
            "Fallback triggered",
            extra={
                "event": "routing_fallback",
                "query_length": len(query),
                "query_preview": query[:100],
                "original_model": original_model,
                "fallback_model": fallback_model,
                "reason": reason,
                **extra,
            },
        )


# Module-level convenience instance
request_logger = RequestLogger()
