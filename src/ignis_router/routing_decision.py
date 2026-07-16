"""Shared routing decision logic used by both decorators and API endpoints."""

from __future__ import annotations

from typing import Any, Optional


def build_routing_decision(
    response: dict[str, Any],
    elapsed: float = 0.0,
) -> dict[str, Any]:
    """
    Build a standardized routing decision dict from a router.chat() response.

    This is the single source of truth for routing decision format,
    used by both @chat() decorator and POST /chat API endpoint.
    """
    routing = response.get("routing", {})
    originally_selected = routing.get("originally_selected", "")
    selection_mode = routing.get("selection_mode", "")
    ml_hint = routing.get("ml_model_hint", "")
    fallback_used = response.get("fallback_used", False)
    intent = routing.get("detected_intent", "")

    # ML predicted model
    ml_predicted = ml_hint or (originally_selected if "ml" in selection_mode else "")

    # Rule-based would pick
    rule_based_pick = _get_rule_based_pick(intent)

    # Note
    note = ""
    if fallback_used and originally_selected:
        note = f"API key not available for '{originally_selected}', switched to available provider."

    return {
        "ml_router_predicted": ml_predicted,
        "rule_based_would_pick": f"{rule_based_pick} (intent rule: {intent})" if rule_based_pick else "",
        "final_model": f"{response.get('model', '')} ({response.get('provider', '')})",
        "note": note,
        "intent": intent,
        "complexity": routing.get("complexity", ""),
        "confidence": routing.get("confidence", 0.0),
        "tokens": response.get("usage", {}).get("total_tokens", 0),
        "time": round(elapsed, 3),
    }


def build_routing_decision_from_result(result: Any, elapsed: float = 0.0) -> dict[str, Any]:
    """
    Build a routing decision dict from a RoutingResult object (for @route decorator).
    """
    selection_mode = result.scoring_details.get("selection_mode", "")
    ml_hint = result.scoring_details.get("model_hint", "")
    intent = result.detected_intent.value

    ml_predicted = ml_hint or (result.selected_model.model_name if "ml" in selection_mode else "")
    rule_based_pick = _get_rule_based_pick(intent)

    return {
        "ml_router_predicted": ml_predicted,
        "rule_based_would_pick": f"{rule_based_pick} (intent rule: {intent})" if rule_based_pick else "",
        "final_model": f"{result.selected_model.model_name} ({result.selected_model.provider})",
        "intent": intent,
        "complexity": result.complexity.value,
        "confidence": result.confidence,
        "time": round(elapsed, 3),
    }


def log_routing_decision_to_db(
    query: str,
    routing_decision: dict[str, Any],
    strategy: str,
    response_content: str = "",
) -> None:
    """
    Save routing decision to PostgreSQL. Silently skips if DB is unavailable.
    Used by both decorators and API endpoints.
    """
    try:
        from .persistence import PostgresRouteLogger
        db = PostgresRouteLogger()
        db.ensure_table()
        db.log_routing_decision(
            query=query,
            routing_decision=routing_decision,
            strategy=strategy,
            response_content=response_content,
        )
    except Exception:
        pass  # DB unavailable — don't break the response


def _get_rule_based_pick(intent: str) -> str:
    """Get the model that rule-based routing would pick for an intent."""
    try:
        from .supported_models import get_default_intent_rules
        return next(
            (r.target_model_id for r in get_default_intent_rules()
             if r.intent and r.intent.value == intent),
            "",
        )
    except Exception:
        return ""
