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
            ml_router_predicted TEXT,
            rule_based_would_pick TEXT,
            default_model_used TEXT NOT NULL,
            provider TEXT NOT NULL,
            note TEXT,
            intent TEXT NOT NULL,
            complexity TEXT NOT NULL,
            confidence DOUBLE PRECISION NOT NULL,
            tokens INTEGER DEFAULT 0,
            strategy TEXT NOT NULL,
            response_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    def log_response(self, query: str, result: RoutingResult, strategy: str) -> None:
        """Persist one routing response row (legacy format)."""
        payload = json.loads(result.model_dump_json())
        insert_sql = f"""
        INSERT INTO {self.settings.table}
            (query_text, default_model_used, provider, intent, complexity,
             confidence, strategy, response_json)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    insert_sql,
                    (
                        query,
                        result.selected_model.model_name,
                        result.selected_model.provider,
                        result.detected_intent.value,
                        result.complexity.value,
                        result.confidence,
                        strategy,
                        json.dumps(payload),
                    ),
                )
            conn.commit()

    def log_routing_decision(
        self,
        query: str,
        routing_decision: dict,
        strategy: str,
        response_content: str = "",
    ) -> None:
        """Persist a full routing decision with ML/rule-based/default model info."""
        insert_sql = f"""
        INSERT INTO {self.settings.table}
            (query_text, ml_router_predicted, rule_based_would_pick,
             default_model_used, provider, note, intent, complexity,
             confidence, tokens, strategy, response_json)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """
        # Extract provider from final_model string like "gpt-4.1 (openai)"
        final_model = routing_decision.get("final_model", "")
        provider = ""
        model_name = final_model
        if "(" in final_model and ")" in final_model:
            model_name = final_model.split("(")[0].strip()
            provider = final_model.split("(")[1].rstrip(")")

        payload = {
            "routing_decision": routing_decision,
            "response_preview": response_content[:500] if response_content else "",
        }

        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    insert_sql,
                    (
                        query,
                        routing_decision.get("ml_router_predicted", ""),
                        routing_decision.get("rule_based_would_pick", ""),
                        model_name,
                        provider,
                        routing_decision.get("note", ""),
                        routing_decision.get("intent", ""),
                        routing_decision.get("complexity", ""),
                        routing_decision.get("confidence", 0.0),
                        routing_decision.get("tokens", 0),
                        strategy,
                        json.dumps(payload),
                    ),
                )
            conn.commit()
