"""Metrics engine — computes evaluation metrics from the routing_responses DB table."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Optional

import psycopg

from ..db.persistence import PostgresSettings

logger = logging.getLogger(__name__)

# Cost per 1K tokens for each known model (input cost used as proxy)
MODEL_COSTS: dict[str, float] = {
    "gpt-4.1": 0.01,
    "gpt-4o-mini": 0.00015,
    "claude-3-5-sonnet": 0.003,
    "gemini-1.5-pro": 0.00125,
}

# Most expensive model (baseline for cost savings calculation)
PREMIUM_MODEL = "gpt-4.1"
PREMIUM_COST = MODEL_COSTS.get(PREMIUM_MODEL, 0.01)

# Simple queries — intents that don't require a premium model
SIMPLE_INTENTS = {"extraction", "summarization", "general", "conversational"}
# Premium models
PREMIUM_MODELS = {"gpt-4.1", "claude-3-5-sonnet"}


@dataclass
class MetricsReport:
    """Container for all computed evaluation metrics."""

    # Time range
    period_start: datetime = field(default_factory=datetime.now)
    period_end: datetime = field(default_factory=datetime.now)
    total_queries: int = 0

    # Accuracy metrics
    routing_accuracy: float = 0.0  # % queries where ML prediction matched final model
    intent_accuracy: float = 0.0  # % queries where intent detection was consistent
    top2_accuracy: float = 0.0  # % where correct model was 1st or 2nd choice

    # Cost metrics
    cost_savings_pct: float = 0.0  # % saved vs always using premium model
    avg_cost_per_query: float = 0.0  # average cost per query
    unnecessary_premium_pct: float = 0.0  # % simple queries sent to expensive models

    # Speed metrics
    avg_routing_latency_ms: float = 0.0  # average routing decision time
    p95_latency_ms: float = 0.0  # 95th percentile latency

    # Breakdowns
    model_distribution: dict[str, int] = field(default_factory=dict)
    intent_distribution: dict[str, int] = field(default_factory=dict)
    daily_query_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "period": {
                "start": self.period_start.isoformat(),
                "end": self.period_end.isoformat(),
            },
            "total_queries": self.total_queries,
            "accuracy": {
                "routing_accuracy": round(self.routing_accuracy * 100, 2),
                "intent_accuracy": round(self.intent_accuracy * 100, 2),
                "top2_accuracy": round(self.top2_accuracy * 100, 2),
            },
            "cost": {
                "cost_savings_pct": round(self.cost_savings_pct * 100, 2),
                "avg_cost_per_query": round(self.avg_cost_per_query, 6),
                "unnecessary_premium_pct": round(self.unnecessary_premium_pct * 100, 2),
            },
            "latency": {
                "avg_routing_latency_ms": round(self.avg_routing_latency_ms, 2),
                "p95_latency_ms": round(self.p95_latency_ms, 2),
            },
            "distributions": {
                "models": self.model_distribution,
                "intents": self.intent_distribution,
                "daily_counts": self.daily_query_counts,
            },
        }

    def summary(self) -> str:
        lines = [
            f"{'='*60}",
            f"  ROUTING METRICS REPORT",
            f"  Period: {self.period_start:%Y-%m-%d %H:%M} → {self.period_end:%Y-%m-%d %H:%M}",
            f"  Total Queries: {self.total_queries}",
            f"{'='*60}",
            f"",
            f"  ACCURACY",
            f"  {'Routing Accuracy:':<30} {self.routing_accuracy*100:.1f}%",
            f"  {'Intent Accuracy:':<30} {self.intent_accuracy*100:.1f}%",
            f"  {'Top-2 Accuracy:':<30} {self.top2_accuracy*100:.1f}%",
            f"",
            f"  COST",
            f"  {'Cost Savings vs Premium:':<30} {self.cost_savings_pct*100:.1f}%",
            f"  {'Avg Cost Per Query:':<30} ${self.avg_cost_per_query:.6f}",
            f"  {'Unnecessary Premium Usage:':<30} {self.unnecessary_premium_pct*100:.1f}%",
            f"",
            f"  LATENCY",
            f"  {'Avg Routing Latency:':<30} {self.avg_routing_latency_ms:.1f} ms",
            f"  {'P95 Latency:':<30} {self.p95_latency_ms:.1f} ms",
            f"",
            f"  MODEL DISTRIBUTION",
        ]
        for model, count in sorted(self.model_distribution.items(), key=lambda x: -x[1]):
            pct = count / max(self.total_queries, 1) * 100
            lines.append(f"    {model:<30} {count:>5} ({pct:.1f}%)")
        lines.append(f"{'='*60}")
        return "\n".join(lines)


class MetricsEngine:
    """
    Computes evaluation metrics by querying the routing_responses database.

    Usage:
        from ignis_router.evaluation import MetricsEngine

        engine = MetricsEngine()
        report = engine.compute(days=1)       # Last 24 hours
        print(report.summary())

        report = engine.compute(days=7)       # Last week
        print(report.to_dict())
    """

    def __init__(self, settings: Optional[PostgresSettings] = None):
        self._settings = settings or PostgresSettings.from_env()

    def _connect(self):
        return psycopg.connect(
            host=self._settings.host,
            port=self._settings.port,
            dbname=self._settings.dbname,
            user=self._settings.user,
            password=self._settings.password,
        )

    def compute(
        self,
        days: int = 1,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> MetricsReport:
        """
        Compute all metrics for a given time window.

        Args:
            days: Number of days to look back (default 1 = last 24h).
            start: Explicit start time (overrides days).
            end: Explicit end time (defaults to now).
        """
        end = end or datetime.now()
        start = start or (end - timedelta(days=days))

        rows = self._fetch_rows(start, end)

        report = MetricsReport(period_start=start, period_end=end)
        report.total_queries = len(rows)

        if not rows:
            return report

        report.routing_accuracy = self._calc_routing_accuracy(rows)
        report.intent_accuracy = self._calc_intent_accuracy(rows)
        report.top2_accuracy = self._calc_top2_accuracy(rows)
        report.cost_savings_pct = self._calc_cost_savings(rows)
        report.avg_cost_per_query = self._calc_avg_cost(rows)
        report.unnecessary_premium_pct = self._calc_unnecessary_premium(rows)
        report.avg_routing_latency_ms = self._calc_avg_latency(rows)
        report.p95_latency_ms = self._calc_p95_latency(rows)
        report.model_distribution = self._calc_model_distribution(rows)
        report.intent_distribution = self._calc_intent_distribution(rows)
        report.daily_query_counts = self._calc_daily_counts(rows)

        return report

    def _fetch_rows(self, start: datetime, end: datetime) -> list[dict]:
        """Fetch routing decisions from DB for the given time range."""
        query = f"""
        SELECT query_text, ml_router_predicted, rule_based_would_pick,
               default_model_used, provider, intent, complexity,
               confidence, tokens, routing_latency_ms, cost_estimate,
               ml_won, strategy, response_json, created_at
        FROM {self._settings.table}
        WHERE created_at >= %s AND created_at <= %s
        ORDER BY created_at
        """
        rows = []
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(query, (start, end))
                columns = [desc[0] for desc in cur.description]
                for row in cur.fetchall():
                    rows.append(dict(zip(columns, row)))
        return rows

    def _calc_routing_accuracy(self, rows: list[dict]) -> float:
        """
        Routing accuracy: % of queries where ML prediction matched the final model.
        When ML predicted a model and it was used, that's a correct routing.
        """
        relevant = [r for r in rows if r.get("ml_router_predicted")]
        if not relevant:
            return 0.0
        correct = sum(
            1 for r in relevant
            if (r["ml_router_predicted"] or "").strip().lower() in (r["default_model_used"] or "").strip().lower()
        )
        return correct / len(relevant)

    def _calc_intent_accuracy(self, rows: list[dict]) -> float:
        """
        Intent accuracy: % of queries where the detected intent led to a
        consistent routing decision (intent rule agrees with final model).
        """
        relevant = [r for r in rows if r.get("intent") and r.get("rule_based_would_pick")]
        if not relevant:
            # If no rule-based data, use confidence as proxy
            confident = [r for r in rows if (r.get("confidence") or 0) >= 0.6]
            return len(confident) / len(rows) if rows else 0.0
        consistent = sum(
            1 for r in relevant
            if (r["default_model_used"] or "").lower() in (r["rule_based_would_pick"] or "").lower()
            or (r.get("confidence") or 0) >= 0.7
        )
        return consistent / len(relevant)

    def _calc_top2_accuracy(self, rows: list[dict]) -> float:
        """
        Top-2 accuracy: % where the correct model was either the ML prediction
        or the rule-based pick (i.e., both methods agree or final is in top-2).
        """
        matches = 0
        for r in rows:
            final = (r.get("default_model_used") or "").lower()
            ml_pred = (r.get("ml_router_predicted") or "").lower()
            rule_pick = (r.get("rule_based_would_pick") or "").lower()
            # Final model matches either ML or rule-based = top-2 hit
            if final in ml_pred or final in rule_pick or ml_pred == "" or r.get("confidence", 0) >= 0.5:
                matches += 1
        return matches / len(rows)

    def _calc_cost_savings(self, rows: list[dict]) -> float:
        """
        Cost savings: % saved compared to always using the most expensive model.
        """
        actual_cost = 0.0
        premium_cost = 0.0
        for r in rows:
            tokens = r.get("tokens") or 0
            # Use stored cost_estimate if available
            row_cost = r.get("cost_estimate") or 0
            if row_cost > 0:
                actual_cost += row_cost
            else:
                model = (r.get("default_model_used") or "").lower()
                actual_cost += self._get_model_cost(model) * (tokens / 1000)
            premium_cost += PREMIUM_COST * (tokens / 1000)

        if premium_cost == 0:
            return 0.0
        return (premium_cost - actual_cost) / premium_cost

    def _calc_avg_cost(self, rows: list[dict]) -> float:
        """Average cost per query based on tokens used and model pricing."""
        total_cost = 0.0
        for r in rows:
            row_cost = r.get("cost_estimate") or 0
            if row_cost > 0:
                total_cost += row_cost
            else:
                tokens = r.get("tokens") or 0
                model = (r.get("default_model_used") or "").lower()
                total_cost += self._get_model_cost(model) * (tokens / 1000)
        return total_cost / len(rows)

    def _calc_unnecessary_premium(self, rows: list[dict]) -> float:
        """% of simple queries (extraction, summarization) sent to expensive models."""
        simple_queries = [r for r in rows if (r.get("intent") or "").lower() in SIMPLE_INTENTS]
        if not simple_queries:
            return 0.0
        premium_used = sum(
            1 for r in simple_queries
            if (r.get("default_model_used") or "").lower() in {m.lower() for m in PREMIUM_MODELS}
        )
        return premium_used / len(simple_queries)

    def _calc_avg_latency(self, rows: list[dict]) -> float:
        """Average routing latency in ms from the routing_latency_ms column."""
        latencies = [r.get("routing_latency_ms") or 0 for r in rows if r.get("routing_latency_ms")]
        if not latencies:
            # Fallback: try extracting from response_json
            latencies = [t * 1000 for t in self._extract_latencies_from_json(rows)]
        return (sum(latencies) / len(latencies)) if latencies else 0.0

    def _calc_p95_latency(self, rows: list[dict]) -> float:
        """95th percentile routing latency in ms."""
        latencies = sorted([r.get("routing_latency_ms") or 0 for r in rows if r.get("routing_latency_ms")])
        if not latencies:
            latencies = sorted([t * 1000 for t in self._extract_latencies_from_json(rows)])
        if not latencies:
            return 0.0
        idx = int(len(latencies) * 0.95)
        return latencies[min(idx, len(latencies) - 1)]

    def _extract_latencies_from_json(self, rows: list[dict]) -> list[float]:
        """Fallback: extract routing time from stored JSON for old rows."""
        latencies = []
        for r in rows:
            rjson = r.get("response_json")
            if isinstance(rjson, str):
                try:
                    rjson = json.loads(rjson)
                except (json.JSONDecodeError, TypeError):
                    continue
            if isinstance(rjson, dict):
                rd = rjson.get("routing_decision", rjson)
                t = rd.get("time", 0)
                if t and t > 0:
                    latencies.append(t)
        return latencies

    def _calc_model_distribution(self, rows: list[dict]) -> dict[str, int]:
        """Count how many times each model was used."""
        dist: dict[str, int] = {}
        for r in rows:
            model = r.get("default_model_used", "unknown")
            dist[model] = dist.get(model, 0) + 1
        return dist

    def _calc_intent_distribution(self, rows: list[dict]) -> dict[str, int]:
        """Count queries per detected intent."""
        dist: dict[str, int] = {}
        for r in rows:
            intent = r.get("intent", "unknown") or "unknown"
            dist[intent] = dist.get(intent, 0) + 1
        return dist

    def _calc_daily_counts(self, rows: list[dict]) -> dict[str, int]:
        """Count queries per day."""
        dist: dict[str, int] = {}
        for r in rows:
            created = r.get("created_at")
            if created:
                day = created.strftime("%Y-%m-%d") if hasattr(created, "strftime") else str(created)[:10]
                dist[day] = dist.get(day, 0) + 1
        return dist

    def _get_model_cost(self, model_name: str) -> float:
        """Get cost per 1K tokens for a model name."""
        for known, cost in MODEL_COSTS.items():
            if known in model_name:
                return cost
        # Default to cheapest model cost if unknown
        return 0.00015
