"""PostgreSQL persistence helpers for routing responses."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import psycopg

from ..models import RoutingResult


@dataclass
class PostgresSettings:
    """Connection settings for PostgreSQL."""

    conninfo: str = "postgresql://postgres:postgres@localhost:5432/llm_router"
    table: str = "routing_responses"

    @classmethod
    def from_env(cls) -> "PostgresSettings":
        import os

        return cls(
            conninfo=os.getenv(
                "ROUTER_DATABASE_URL",
                "postgresql://postgres:postgres@localhost:5432/llm_router",
            ),
            table=os.getenv("ROUTER_DB_TABLE", "routing_responses"),
        )


class PostgresRouteLogger:
    """Stores routed request/response records in PostgreSQL."""

    def __init__(self, settings: Optional[PostgresSettings] = None):
        self.settings = settings or PostgresSettings.from_env()

    def _connect(self):
        return psycopg.connect(self.settings.conninfo)

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
            routing_latency_ms DOUBLE PRECISION DEFAULT 0,
            cost_estimate DOUBLE PRECISION DEFAULT 0,
            ml_won BOOLEAN DEFAULT FALSE,
            strategy TEXT NOT NULL,
            response_json JSONB NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                # Add columns if table already exists but is missing new fields
                for col, dtype in [
                    ("routing_latency_ms", "DOUBLE PRECISION DEFAULT 0"),
                    ("cost_estimate", "DOUBLE PRECISION DEFAULT 0"),
                    ("ml_won", "BOOLEAN DEFAULT FALSE"),
                ]:
                    cur.execute(f"""
                        DO $$ BEGIN
                            ALTER TABLE {self.settings.table} ADD COLUMN {col} {dtype};
                        EXCEPTION WHEN duplicate_column THEN NULL;
                        END $$;
                    """)
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
             confidence, tokens, routing_latency_ms, cost_estimate, ml_won,
             strategy, response_json)
        VALUES
            (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
        """
        # Extract provider from final_model string like "gpt-4.1 (openai)"
        final_model = routing_decision.get("final_model", "")
        provider = ""
        model_name = final_model
        if "(" in final_model and ")" in final_model:
            model_name = final_model.split("(")[0].strip()
            provider = final_model.split("(")[1].rstrip(")")

        # Calculate cost estimate
        tokens = routing_decision.get("tokens", 0) or 0
        cost_estimate = self._estimate_cost(model_name, tokens)

        # Latency in ms
        latency_ms = (routing_decision.get("time", 0) or 0) * 1000

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
                        tokens,
                        latency_ms,
                        cost_estimate,
                        routing_decision.get("ml_won", False),
                        strategy,
                        json.dumps(payload),
                    ),
                )
            conn.commit()

    @staticmethod
    def _estimate_cost(model_name: str, tokens: int) -> float:
        """Estimate cost in USD based on model and token count."""
        cost_per_1k = {
            "gpt-4.1": 0.01,
            "gpt-4o-mini": 0.00015,
            "claude-3-5-sonnet": 0.003,
            "gemini-1.5-pro": 0.00125,
        }
        for name, rate in cost_per_1k.items():
            if name in model_name.lower():
                return rate * (tokens / 1000)
        return 0.00015 * (tokens / 1000)
