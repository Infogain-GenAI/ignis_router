"""REST API for ignis_router routing decisions."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

from .config import RouterConfig
from .exceptions import ConfigurationError, ModelNotAvailableError, RoutingError
from .persistence import PostgresRouteLogger
from .router import Router


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


class ErrorResponse(BaseModel):
    """Standardized API error response."""

    error: str
    message: str
    details: list[dict[str, Any]] = Field(default_factory=list)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


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
        logger = PostgresRouteLogger()
        logger.ensure_table()
        return logger
    except Exception as exc:
        import logging as _log
        _log.getLogger(__name__).warning(
            "PostgreSQL unavailable — DB logging disabled. %s", exc
        )
        return None


def create_app(router: Router | None = None, enable_db: bool = True) -> FastAPI:
    """Application factory used by uvicorn and tests."""
    app = FastAPI(title="Ignis Router API", version="0.1.0")
    app.state.router = router or build_api_router()
    app.state.db_logger = _try_init_db_logger() if enable_db else None

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
        return JSONResponse(
            status_code=503,
            content=ErrorResponse(
                error="model_not_available",
                message=str(exc),
            ).model_dump(),
        )

    async def handle_routing_error(request: Request, exc: Exception) -> JSONResponse:
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
    async def route_query_get(
        query: str = Query(..., min_length=1, description="Prompt text to route"),
    ) -> RouteResponse:
        cleaned_query = query.strip()
        if not cleaned_query:
            raise RequestValidationError(
                [
                    {
                        "type": "value_error",
                        "loc": ("query", "query"),
                        "msg": "Value error, query must not be empty",
                        "input": query,
                    }
                ]
            )

        result = app.state.router.route(cleaned_query)
        strategy = app.state.router.config.routing_strategy
        if app.state.db_logger:
            try:
                app.state.db_logger.log_response(cleaned_query, result, strategy)
            except Exception:
                pass
        return RouteResponse(
            selected_model=result.selected_model.model_name,
            strategy=strategy,
            confidence=round(result.confidence, 2),
        )

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
            # Route first to capture the RoutingResult for DB logging
            routing_result = app.state.router.route(payload.query)
            strategy = app.state.router.config.routing_strategy

            if app.state.db_logger:
                try:
                    app.state.db_logger.log_response(payload.query, routing_result, strategy)
                except Exception:
                    pass

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
        return ChatResponse(**result)

    @app.get("/providers")
    async def list_providers() -> dict[str, Any]:
        router = app.state.router
        llm = router.llm_clients
        available = llm.get_available_providers() if llm else []
        return {"available_providers": available}

    @app.get("/models")
    async def list_models() -> list[dict[str, Any]]:
        """Return all registered models with their metadata."""
        models = app.state.router.get_registered_models()
        return [
            {
                "model_id": m.model_id,
                "provider": m.provider,
                "model_name": m.model_name,
                "capabilities": [c.value for c in m.capabilities],
                "cost_per_1k_input_tokens": m.cost_per_1k_input_tokens,
                "cost_per_1k_output_tokens": m.cost_per_1k_output_tokens,
                "latency": m.latency,
                "quality": m.quality,
                "reliability": m.reliability,
                "priority": m.priority,
                "enabled": m.enabled,
            }
            for m in models
        ]

    @app.get("/rules")
    async def list_rules() -> list[dict[str, Any]]:
        """Return all active routing rules."""
        rules = app.state.router.registry.get_rules()
        return [
            {
                "rule_id": r.rule_id,
                "intent": r.intent.value if r.intent else None,
                "complexity": r.complexity.value if r.complexity else None,
                "required_capabilities": [c.value for c in r.required_capabilities],
                "target_model_id": r.target_model_id,
                "priority": r.priority,
                "enabled": r.enabled,
            }
            for r in rules
        ]

    @app.get("/history")
    async def get_history(
        limit: int = Query(default=20, ge=1, le=100, description="Number of recent records"),
    ) -> list[dict[str, Any]]:
        """Return recent routing responses from PostgreSQL."""
        db = app.state.db_logger
        if db is None:
            return JSONResponse(
                status_code=503,
                content=ErrorResponse(
                    error="db_unavailable",
                    message="PostgreSQL is not configured or unreachable.",
                ).model_dump(),
            )
        try:
            with db._connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        f"SELECT id, query_text, selected_model, strategy, "
                        f"confidence, created_at FROM {db.settings.table} "
                        f"ORDER BY id DESC LIMIT %s",
                        (limit,),
                    )
                    rows = cur.fetchall()
            return [
                {
                    "id": row[0],
                    "query": row[1],
                    "selected_model": row[2],
                    "strategy": row[3],
                    "confidence": row[4],
                    "created_at": row[5].isoformat(),
                }
                for row in rows
            ]
        except Exception as exc:
            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="db_error",
                    message=f"Failed to fetch history: {exc}",
                ).model_dump(),
            )

    return app
