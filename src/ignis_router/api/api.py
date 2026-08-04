"""REST API for ignis_router routing decisions."""

from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from ..config import RouterConfig
from ..exceptions import ConfigurationError, ModelNotAvailableError, RoutingError
from ..db.persistence import PostgresRouteLogger
from ..core.router import Router
from ..db.routing_decision import build_routing_decision, log_routing_decision_to_db


class FeatureKey(str, Enum):
    """Available feature flags for the PUT /features/{key} dropdown."""

    ml_based_routing = "ml_based_routing"
    rule_based_routing = "rule_based_routing"
    hybrid_routing = "hybrid_routing"


# Map clean dropdown names to internal config keys
_FEATURE_KEY_MAP = {
    "ml_based_routing": "enable_ml_model_hint_routing",
    "rule_based_routing": "enable_rule_based_intent_detection",
    "hybrid_routing": "enable_ml_intent_detection",
}
from ..evaluation.metrics import MetricsEngine
from ..evaluation.dashboard import DashboardEngine


class RouteRequest(BaseModel):
    """Input contract for the route endpoint."""

    query: str = Field(..., min_length=1, description="Prompt text to route")

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned


class ChatRequest(BaseModel):
    """Input contract for the chat endpoint (route + execute)."""

    query: str = Field(..., min_length=1, description="Prompt text to route and execute")
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System message for the LLM",
    )
    max_tokens: int = Field(default=1024, ge=1, le=128000, description="Max response tokens")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")

    @field_validator("query")
    @classmethod
    def strip_and_validate_query(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be empty")
        return cleaned


class RouteResponse(BaseModel):
    """Standardized successful route response."""

    selected_model: str
    strategy: str
    confidence: float


class ChatResponse(BaseModel):
    """Standardized successful chat response (route + LLM execution)."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = Field(default_factory=dict)
    finish_reason: str = ""
    routing: dict[str, Any] = Field(default_factory=dict)
    routing_decision: dict[str, Any] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    """Standardized API error response."""

    error: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _abs_from_root(path_value: str) -> str:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(_project_root() / path_obj)


def build_api_router() -> Router:
    """Create a Router configured from .env and YAML strategy."""
    load_dotenv(_project_root() / ".env")

    yaml_config = os.getenv("ROUTER_YAML_CONFIG")
    if yaml_config:
        config = RouterConfig.from_yaml(_abs_from_root(yaml_config))
    else:
        config = RouterConfig.from_yaml(str(_project_root() / "configs" / "quality-first.yaml"))

    config.ml_model_path = _abs_from_root(config.ml_model_path)

    router = Router(config=config)
    router.register_supported_models()
    router.register_default_intent_rules()
    router.enable_llm_clients()
    return router


def _try_init_db_logger() -> PostgresRouteLogger | None:
    """Attempt to connect to PostgreSQL. Returns None if DB is unavailable."""
    try:
        db_logger = PostgresRouteLogger()
        db_logger.ensure_table()
        return db_logger
    except Exception as exc:
        from ..logging import get_logger as _get_logger
        _get_logger(__name__).warning(
            "PostgreSQL unavailable — DB logging disabled. %s", exc
        )
        return None


def create_app(router: Router | None = None, enable_db: bool = True) -> FastAPI:
    """Application factory used by uvicorn and tests."""
    from ..logging import get_logger, request_logger, correlation_context, set_correlation_id
    from ..feature_flags import FeatureFlags

    load_dotenv(_project_root() / ".env")
    app = FastAPI(title="Ignis Router API", version="0.1.0")
    app.state.router = router or build_api_router()
    app.state.db_logger = _try_init_db_logger() if enable_db else None
    app.state.feature_flags = FeatureFlags.from_config(app.state.router.config)

    _api_logger = get_logger(__name__)

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        """Inject correlation ID from header or generate one for each request."""
        header_name = os.getenv("IGNIS_LOG_CORRELATION_HEADER", "X-Correlation-ID")
        cid = request.headers.get(header_name)
        with correlation_context(cid) as active_cid:
            response = await call_next(request)
            response.headers[header_name] = active_cid
            return response

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(
                error="validation_error",
                message="Request validation failed",
                details=jsonable_encoder(exc.errors()),
            ).model_dump(),
        )

    @app.exception_handler(ModelNotAvailableError)
    async def handle_model_unavailable(request: Request, exc: ModelNotAvailableError) -> JSONResponse:
        _api_logger.error("Model not available: %s", exc, extra={"event": "api_error", "error_type": "model_not_available"})
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="model_not_available",
                message=str(exc),
            ).model_dump(),
        )

    async def handle_routing_error(request: Request, exc: Exception) -> JSONResponse:
        _api_logger.warning("Routing error: %s", exc, extra={"event": "api_error", "error_type": "routing_error"})
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="routing_error",
                message=str(exc),
            ).model_dump(),
        )

    app.add_exception_handler(RoutingError, handle_routing_error)
    app.add_exception_handler(ConfigurationError, handle_routing_error)

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        _api_logger.exception("Unexpected error", extra={"event": "api_error", "error_type": "internal_error"})
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                message="Internal server error",
            ).model_dump(),
        )

    @app.get("/")
    async def root() -> dict[str, str]:
        return {"name": "Ignis Router API", "status": "ok", "docs": "/docs"}

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/route",
        response_model=RouteResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def route_query(payload: RouteRequest) -> RouteResponse:
        result = app.state.router.route(payload.query)
        strategy = app.state.router.config.routing_strategy
        if app.state.db_logger:
            try:
                app.state.db_logger.log_response(payload.query, result, strategy)
            except Exception:
                pass  # non-fatal: don't break the API response
        return RouteResponse(
            selected_model=result.selected_model.model_name,
            strategy=strategy,
            confidence=round(result.confidence, 2),
        )

    @app.get(
        "/route",
        response_model=RouteResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def route_query_get(query: str = Query(..., min_length=1, description="Prompt text to route")) -> RouteResponse:
        result = app.state.router.route(query)
        strategy = app.state.router.config.routing_strategy
        if app.state.db_logger:
            try:
                app.state.db_logger.log_response(query, result, strategy)
            except Exception:
                pass
        return RouteResponse(
            selected_model=result.selected_model.model_name,
            strategy=strategy,
            confidence=round(result.confidence, 2),
        )

    @app.get("/providers")
    async def get_providers() -> dict[str, Any]:
        """Return list of available LLM providers from registered models."""
        providers = set()
        for model in app.state.router.registry.get_all_models():
            providers.add(model.provider)
        return {"available_providers": sorted(providers)}

    @app.post(
        "/chat",
        response_model=ChatResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            500: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
    )
    async def chat_query(payload: ChatRequest) -> ChatResponse:
        try:
            result = app.state.router.chat(
                payload.query,
                system_prompt=payload.system_prompt,
                max_tokens=payload.max_tokens,
                temperature=payload.temperature,
            )
        except RuntimeError as exc:
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error="llm_unavailable",
                    message=str(exc),
                ).model_dump(),
            )

        # Build routing decision using shared logic (same as decorators)
        routing_decision = build_routing_decision(result)
        result["routing_decision"] = routing_decision

        # Save to DB using shared logic
        log_routing_decision_to_db(
            query=payload.query,
            routing_decision=routing_decision,
            strategy=app.state.router.config.routing_strategy,
            response_content=result.get("content", ""),
        )

        return ChatResponse(**result)

    @app.get("/metrics")
    async def get_metrics(days: int = Query(default=1, ge=1, le=365, description="Number of days to report on")) -> dict[str, Any]:
        """Get routing evaluation metrics for the specified time period."""
        try:
            engine = MetricsEngine()
            report = engine.compute(days=days)
            return report.to_dict()
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="metrics_error",
                    message=f"Failed to compute metrics: {exc}",
                ).model_dump(),
            )

    @app.get("/metrics/summary")
    async def get_metrics_summary(days: int = Query(default=1, ge=1, le=365, description="Number of days to report on")) -> dict[str, Any]:
        """Get a concise metrics summary."""
        try:
            engine = MetricsEngine()
            report = engine.compute(days=days)
            return {
                "total_queries": report.total_queries,
                "period_days": days,
                "routing_accuracy": round(report.routing_accuracy * 100, 2),
                "cost_savings_pct": round(report.cost_savings_pct * 100, 2),
                "avg_latency_ms": round(report.avg_routing_latency_ms, 2),
                "top_model": max(report.model_distribution, key=report.model_distribution.get) if report.model_distribution else None,
            }
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="metrics_error",
                    message=f"Failed to compute metrics: {exc}",
                ).model_dump(),
            )

    @app.get("/metrics/models")
    async def get_model_distribution(days: int = Query(default=1, ge=1, le=365)) -> dict[str, Any]:
        """Get model usage distribution."""
        try:
            engine = MetricsEngine()
            report = engine.compute(days=days)
            return {
                "period_days": days,
                "total_queries": report.total_queries,
                "models": report.model_distribution,
                "intents": report.intent_distribution,
            }
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="metrics_error",
                    message=f"Failed to compute metrics: {exc}",
                ).model_dump(),
            )

    @app.get("/dashboard")
    async def get_dashboard(
        days: int = Query(default=7, ge=1, le=365, description="Number of days"),
        strategy: str | None = Query(default=None, description="Optional routing strategy filter"),
        intent: str | None = Query(default=None, description="Optional detected intent filter"),
        page: int = Query(default=1, ge=1, description="Routing log page number"),
        page_size: int = Query(default=20, ge=1, le=100, description="Routing log page size"),
    ) -> dict[str, Any]:
        """
        Full dashboard payload with KPIs, charts, routing log, and feature flags.
        Designed for frontend consumption.
        """
        try:
            engine = DashboardEngine()
            dashboard_data = engine.generate(
                days=days,
                strategy=strategy,
                intent=intent,
                page=page,
                page_size=page_size,
            )
            # Include feature flags so the dashboard can show toggles
            dashboard_data["feature_flags"] = app.state.feature_flags.to_dict()
            return dashboard_data
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="dashboard_error",
                    message=f"Failed to generate dashboard: {exc}",
                ).model_dump(),
            )

    @app.get("/features")
    async def get_features() -> dict[str, Any]:
        """Get all feature flags with their current status, grouped by category."""
        return app.state.feature_flags.to_dict()

    @app.put("/features/{key}")
    async def update_feature(key: FeatureKey, enabled: bool = Query(..., description="Set feature to true or false")) -> dict[str, Any]:
        """Toggle a feature flag at runtime without restarting the server."""
        config_key = _FEATURE_KEY_MAP[key.value]
        try:
            app.state.feature_flags.set(config_key, enabled)
            flag = app.state.feature_flags.get(config_key)
            _api_logger.info(
                "Feature flag updated: %s=%s", key.value, enabled,
                extra={"event": "feature_flag_changed", "flag": key.value, "enabled": enabled},
            )
            return {
                "key": key.value,
                "name": flag.name,
                "enabled": flag.enabled,
                "message": f"Feature '{flag.name}' {'enabled' if enabled else 'disabled'}",
            }
        except KeyError as exc:
            return JSONResponse(
                status_code=404,
                content=ErrorResponse(
                    error="unknown_feature",
                    message=str(exc),
                ).model_dump(),
            )

    return app
