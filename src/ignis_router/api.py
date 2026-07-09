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


class RouteResponse(BaseModel):
    """Standardized successful route response."""

    selected_model: str
    strategy: str
    confidence: float


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
    return router


def create_app(router: Router | None = None) -> FastAPI:
    """Application factory used by uvicorn and tests."""
    app = FastAPI(title="Ignis Router API", version="0.1.0")
    app.state.router = router or build_api_router()

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
        return RouteResponse(
            selected_model=result.selected_model.model_name,
            strategy=app.state.router.config.routing_strategy,
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
        return RouteResponse(
            selected_model=result.selected_model.model_name,
            strategy=app.state.router.config.routing_strategy,
            confidence=round(result.confidence, 2),
        )

    return app
