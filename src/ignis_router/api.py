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
from .routing_decision import build_routing_decision, log_routing_decision_to_db


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

    return app
