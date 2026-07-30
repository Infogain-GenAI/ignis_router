"""Tests for the centralized logging framework."""

import io
import json
import logging
from unittest.mock import patch

import pytest

from ignis_router.logging import (
    StructuredJsonFormatter,
    StructuredTextFormatter,
    RequestLogger,
    correlation_context,
    get_correlation_id,
    get_logger,
    set_correlation_id,
    setup_logging,
    _correlation_id,
)


class TestStructuredJsonFormatter:
    """Tests for JSON log output format."""

    def setup_method(self):
        self.formatter = StructuredJsonFormatter()

    def _make_record(self, msg="test", level=logging.INFO, **extra):
        logger = logging.getLogger("ignis_router.test")
        record = logger.makeRecord(
            "ignis_router.test", level, "test.py", 1, msg, (), None
        )
        for k, v in extra.items():
            setattr(record, k, v)
        return record

    def test_basic_json_output(self):
        record = self._make_record("hello world")
        output = self.formatter.format(record)
        entry = json.loads(output)

        assert entry["level"] == "INFO"
        assert entry["logger"] == "ignis_router.test"
        assert entry["message"] == "hello world"
        assert "timestamp" in entry

    def test_extra_fields_included(self):
        record = self._make_record("routing", event="request_received", model="gpt-4")
        output = self.formatter.format(record)
        entry = json.loads(output)

        assert entry["event"] == "request_received"
        assert entry["model"] == "gpt-4"

    def test_correlation_id_included(self):
        token = _correlation_id.set("test-cid-123")
        try:
            record = self._make_record("with correlation")
            output = self.formatter.format(record)
            entry = json.loads(output)
            assert entry["correlation_id"] == "test-cid-123"
        finally:
            _correlation_id.reset(token)

    def test_no_correlation_id_when_absent(self):
        record = self._make_record("no correlation")
        output = self.formatter.format(record)
        entry = json.loads(output)
        assert "correlation_id" not in entry

    def test_exception_info_included(self):
        try:
            raise ValueError("test error")
        except ValueError:
            import sys
            exc_info = sys.exc_info()

        logger = logging.getLogger("ignis_router.test")
        record = logger.makeRecord(
            "ignis_router.test", logging.ERROR, "test.py", 1,
            "failed", (), exc_info
        )
        output = self.formatter.format(record)
        entry = json.loads(output)

        assert entry["exception"]["type"] == "ValueError"
        assert entry["exception"]["message"] == "test error"
        assert "traceback" in entry["exception"]
        assert "ValueError: test error" in entry["exception"]["traceback"]

    def test_crash_location_extracted(self):
        try:
            raise RuntimeError("crash here")
        except RuntimeError:
            import sys
            exc_info = sys.exc_info()

        logger = logging.getLogger("ignis_router.test")
        record = logger.makeRecord(
            "ignis_router.test", logging.ERROR, "test.py", 1,
            "crashed", (), exc_info
        )
        output = self.formatter.format(record)
        entry = json.loads(output)

        assert "crash_location" in entry
        assert "file" in entry["crash_location"]
        assert "line" in entry["crash_location"]
        assert "function" in entry["crash_location"]

    def test_error_level_includes_source(self):
        record = self._make_record("error msg", level=logging.ERROR)
        output = self.formatter.format(record)
        entry = json.loads(output)

        assert "source" in entry
        assert "file" in entry["source"]
        assert "line" in entry["source"]
        assert "function" in entry["source"]

    def test_info_level_excludes_source(self):
        record = self._make_record("info msg", level=logging.INFO)
        output = self.formatter.format(record)
        entry = json.loads(output)

        assert "source" not in entry


class TestCorrelationContext:
    """Tests for correlation ID context management."""

    def test_generates_id_when_none_provided(self):
        assert get_correlation_id() is None
        with correlation_context() as cid:
            assert cid is not None
            assert len(cid) == 16
            assert get_correlation_id() == cid
        assert get_correlation_id() is None

    def test_uses_provided_id(self):
        with correlation_context("my-custom-id") as cid:
            assert cid == "my-custom-id"
            assert get_correlation_id() == "my-custom-id"

    def test_nested_contexts(self):
        with correlation_context("outer") as outer:
            assert get_correlation_id() == "outer"
            with correlation_context("inner") as inner:
                assert get_correlation_id() == "inner"
            assert get_correlation_id() == "outer"

    def test_set_correlation_id_directly(self):
        set_correlation_id("direct-set")
        assert get_correlation_id() == "direct-set"
        # Clean up
        _correlation_id.set(None)


class TestRequestLogger:
    """Tests for the RequestLogger high-level interface."""

    def setup_method(self):
        self.buf = io.StringIO()
        self.handler = logging.StreamHandler(self.buf)
        self.handler.setFormatter(StructuredJsonFormatter())
        self.logger = logging.getLogger("ignis_router.test_requests")
        self.logger.handlers = [self.handler]
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False
        self.request_logger = RequestLogger(self.logger)

    def _get_entries(self):
        lines = self.buf.getvalue().strip().split("\n")
        return [json.loads(line) for line in lines if line.strip()]

    def test_log_request(self):
        self.request_logger.log_request("Write Python code", source="api")
        entries = self._get_entries()

        assert len(entries) == 1
        assert entries[0]["event"] == "request_received"
        assert entries[0]["query_length"] == 17
        assert entries[0]["source"] == "api"

    def test_log_routing_decision(self):
        self.request_logger.log_routing_decision(
            "Hello world",
            selected_model="gpt-4",
            provider="openai",
            intent="general_conversation",
            complexity="low",
            confidence=0.95,
            latency_ms=12.5,
            selection_mode="intent-based",
            strategy="balanced",
        )
        entries = self._get_entries()

        assert len(entries) == 1
        assert entries[0]["event"] == "routing_decision"
        assert entries[0]["selected_model"] == "gpt-4"
        assert entries[0]["confidence"] == 0.95
        assert entries[0]["latency_ms"] == 12.5

    def test_log_failure(self):
        self.request_logger.log_failure(
            "Bad query",
            error_type="RoutingError",
            error_message="No model available",
            phase="model_selection",
        )
        entries = self._get_entries()

        assert len(entries) == 1
        assert entries[0]["event"] == "routing_failure"
        assert entries[0]["level"] == "ERROR"
        assert entries[0]["error_type"] == "RoutingError"

    def test_log_failure_with_exception_includes_traceback(self):
        try:
            raise ValueError("something broke")
        except ValueError as exc:
            self.request_logger.log_failure(
                "test query",
                error_type="ValueError",
                error_message=str(exc),
                phase="test",
                exception=exc,
            )
        entries = self._get_entries()

        assert len(entries) == 1
        assert entries[0]["event"] == "routing_failure"
        assert "exception" in entries[0]
        assert entries[0]["exception"]["type"] == "ValueError"
        assert "traceback" in entries[0]["exception"]
        assert "crash_location" in entries[0]

    def test_log_fallback(self):
        self.request_logger.log_fallback(
            "Some query",
            original_model="claude-3-5-sonnet",
            fallback_model="gpt-4o-mini",
            reason="API key unavailable",
        )
        entries = self._get_entries()

        assert len(entries) == 1
        assert entries[0]["event"] == "routing_fallback"
        assert entries[0]["level"] == "WARNING"
        assert entries[0]["original_model"] == "claude-3-5-sonnet"
        assert entries[0]["fallback_model"] == "gpt-4o-mini"


class TestSetupLogging:
    """Tests for the setup_logging configuration function."""

    def teardown_method(self):
        import ignis_router.logging as mod
        mod._initialized = False
        logging.getLogger("ignis_router").handlers.clear()

    def test_setup_json_format(self):
        import ignis_router.logging as mod
        mod._initialized = False
        setup_logging(level="DEBUG", format="json")

        root = logging.getLogger("ignis_router")
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, StructuredJsonFormatter)

    def test_setup_text_format(self):
        import ignis_router.logging as mod
        mod._initialized = False
        setup_logging(level="WARNING", format="text")

        root = logging.getLogger("ignis_router")
        assert root.level == logging.WARNING
        assert isinstance(root.handlers[0].formatter, StructuredTextFormatter)

    def test_idempotent_setup(self):
        import ignis_router.logging as mod
        mod._initialized = False
        setup_logging(level="INFO", format="json")
        setup_logging(level="DEBUG", format="text")  # Should be no-op

        root = logging.getLogger("ignis_router")
        assert root.level == logging.INFO  # First call wins
