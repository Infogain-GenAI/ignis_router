"""Tests for PostgreSQL persistence with ROUTER_DATABASE_URL connection string."""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest

from ignis_router.db.persistence import PostgresRouteLogger, PostgresSettings


class TestPostgresSettings:
    """Tests for PostgresSettings dataclass."""

    def test_defaults(self):
        settings = PostgresSettings()
        assert settings.conninfo == "postgresql://postgres:postgres@localhost:5432/llm_router"
        assert settings.table == "routing_responses"

    def test_custom_conninfo(self):
        settings = PostgresSettings(
            conninfo="postgresql://user:pass@myhost:5433/mydb",
            table="custom_table",
        )
        assert settings.conninfo == "postgresql://user:pass@myhost:5433/mydb"
        assert settings.table == "custom_table"

    @patch.dict(os.environ, {
        "ROUTER_DATABASE_URL": "postgresql://admin:secret@prod-host:5432/prod_db",
        "ROUTER_DB_TABLE": "my_routing_log",
    })
    def test_from_env_reads_database_url(self):
        settings = PostgresSettings.from_env()
        assert settings.conninfo == "postgresql://admin:secret@prod-host:5432/prod_db"
        assert settings.table == "my_routing_log"

    @patch.dict(os.environ, {}, clear=True)
    def test_from_env_uses_defaults_when_unset(self):
        # Remove any existing env vars
        os.environ.pop("ROUTER_DATABASE_URL", None)
        os.environ.pop("ROUTER_DB_TABLE", None)

        settings = PostgresSettings.from_env()
        assert settings.conninfo == "postgresql://postgres:postgres@localhost:5432/llm_router"
        assert settings.table == "routing_responses"

    @patch.dict(os.environ, {
        "ROUTER_DATABASE_URL": "postgresql://postgres:Info1234@ignisapps.postgres.database.azure.com:5432/ignis_route",
    })
    def test_from_env_azure_connection_string(self):
        settings = PostgresSettings.from_env()
        assert "ignisapps.postgres.database.azure.com" in settings.conninfo
        assert "ignis_route" in settings.conninfo


class TestPostgresRouteLogger:
    """Tests for PostgresRouteLogger connection behavior."""

    @patch("ignis_router.db.persistence.psycopg.connect")
    def test_connect_uses_conninfo_string(self, mock_connect):
        mock_connect.return_value = MagicMock()
        settings = PostgresSettings(
            conninfo="postgresql://user:pass@host:5432/db"
        )
        logger = PostgresRouteLogger(settings=settings)
        conn = logger._connect()

        mock_connect.assert_called_once_with("postgresql://user:pass@host:5432/db")

    @patch("ignis_router.db.persistence.psycopg.connect")
    def test_ensure_table_connects_with_conninfo(self, mock_connect):
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_conn.__enter__ = MagicMock(return_value=mock_conn)
        mock_conn.__exit__ = MagicMock(return_value=False)
        mock_conn.cursor.return_value.__enter__ = MagicMock(return_value=mock_cursor)
        mock_conn.cursor.return_value.__exit__ = MagicMock(return_value=False)
        mock_connect.return_value = mock_conn

        settings = PostgresSettings(
            conninfo="postgresql://test:test@localhost:5432/test_db"
        )
        logger = PostgresRouteLogger(settings=settings)
        logger.ensure_table()

        mock_connect.assert_called_with("postgresql://test:test@localhost:5432/test_db")

    @patch.dict(os.environ, {
        "ROUTER_DATABASE_URL": "postgresql://env_user:env_pass@env_host:5432/env_db",
    })
    @patch("ignis_router.db.persistence.psycopg.connect")
    def test_default_init_reads_from_env(self, mock_connect):
        mock_connect.return_value = MagicMock()
        logger = PostgresRouteLogger()
        logger._connect()

        mock_connect.assert_called_once_with(
            "postgresql://env_user:env_pass@env_host:5432/env_db"
        )
