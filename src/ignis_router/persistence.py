"""PostgreSQL persistence helpers for routing responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import psycopg

from .models import RoutingResult


@dataclass
class PostgresSettings:
    """Connection settings for PostgreSQL."""

    host: str = "localhost"
    port: int = 5432
    dbname: str = "llm_router"
    user: str = "postgres"
    password: str = "postgres"
    table: str = "routing_responses"

    @classmethod
    def from_env(cls) -> "PostgresSettings":
        import os

        return cls(
            host=os.getenv("ROUTER_DB_HOST", "localhost"),
            port=int(os.getenv("ROUTER_DB_PORT", "5432")),
            dbname=os.getenv("ROUTER_DB_NAME", "llm_router"),
            user=os.getenv("ROUTER_DB_USER", "postgres"),
            password=os.getenv("ROUTER_DB_PASSWORD", "postgres"),
            table=os.getenv("ROUTER_DB_TABLE", "routing_responses"),
        )


class PostgresRouteLogger:
    """Stores routed request/response records in PostgreSQL."""

    def __init__(self, settings: Optional[PostgresSettings] = None):
        self.settings = settings or PostgresSettings.from_env()

    def _connect(self):
        return psycopg.connect(
            host=self.settings.host,
            port=self.settings.port,
            dbname=self.settings.dbname,
            user=self.settings.user,
            password=self.settings.password,
        )

    def ensure_table(self) -> None:
        """Create the routing responses table when missing."""
        sql = f"""
        CREATE TABLE IF NOT EXISTS {self.settings.table} (
            id BIGSERIAL PRIMARY KEY,
            query_text TEXT NOT NULL,
            selected_model TEXT NOT NULL,
            strategy TEXT NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            response_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    def log_response(self, query: str, result: RoutingResult, strategy: str) -> None:
        """Persist one routing response row."""
        payload = json.loads(result.model_dump_json())
        insert_sql = f"""
        INSERT INTO {self.settings.table}
            (query_text, selected_model, strategy, confidence, response_json)
        VALUES
            (%s, %s, %s, %s, %s::jsonb)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    insert_sql,
                    (
                        query,
                        result.selected_model.model_name,
                        strategy,
                        result.confidence,
                        json.dumps(payload),
                    ),
                )
            conn.commit()
