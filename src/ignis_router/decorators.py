"""Decorators for using ignis_router as a package without running an API server."""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Optional

from .config import RouterConfig
from .models import RoutingResult
from .core.router import Router
from .db.routing_decision import (
    build_routing_decision,
    build_routing_decision_from_result,
    log_routing_decision_to_db,
)

logger = logging.getLogger(__name__)

# Module-level shared router instance (lazy-initialized)
_shared_router: Optional[Router] = None


def get_shared_router() -> Router:
    """Get or create the module-level shared router instance."""
    global _shared_router
    if _shared_router is None:
        _shared_router = _build_default_router()
    return _shared_router


def set_shared_router(router: Router) -> None:
    """Override the shared router instance with a custom one."""
    global _shared_router
    _shared_router = router


def _build_default_router() -> Router:
    """Build a default router from environment/.env configuration."""
    import os
    from pathlib import Path

    from dotenv import load_dotenv

    # Try loading .env from current working directory
    env_path = Path.cwd() / ".env"
    if env_path.exists():
        load_dotenv(env_path)

    yaml_config = os.getenv("ROUTER_YAML_CONFIG")
    if yaml_config:
        config = RouterConfig.from_yaml(yaml_config)
    else:
        config = RouterConfig()

    router = Router(config=config)
    router.register_supported_models()
    router.register_default_intent_rules()
    router.enable_llm_clients()
    return router


def route(
    *,
    router: Optional[Router] = None,
    log: bool = True,
) -> Callable:
    """
    Decorator that routes a query and passes the RoutingResult to the function.

    The decorated function must accept (query: str, routing_result: RoutingResult, routing_decision: dict, **kwargs).
    routing_decision contains the full decision chain (ML predicted, rule-based pick, final model, etc.)

    Usage:
        @route()
        def handle(query, routing_result, routing_decision):
            print(routing_decision["ml_router_predicted"])
            print(routing_decision["rule_based_would_pick"])
            print(routing_decision["final_model"])

        handle("Write Python code for sorting")
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(query: str, *args: Any, **kwargs: Any) -> Any:
            r = router or get_shared_router()
            start = time.perf_counter()
            result = r.route(query)
            elapsed = time.perf_counter() - start

            routing_decision = build_routing_decision_from_result(result, elapsed)

            if log:
                logger.info(
                    "Routed query to %s (intent=%s, confidence=%.2f, time=%.3fs)",
                    result.selected_model.model_name,
                    result.detected_intent.value,
                    result.confidence,
                    elapsed,
                )

            return func(query, result, routing_decision, *args, **kwargs)

        return wrapper

    return decorator


def chat(
    *,
    router: Optional[Router] = None,
    system_prompt: str = "You are a helpful assistant.",
    max_tokens: int = 1024,
    temperature: float = 0.7,
    log: bool = True,
) -> Callable:
    """
    Decorator that routes a query, calls the LLM, and passes the full response to the function.

    The decorated function must accept (query: str, response: dict, **kwargs).
    Response dict contains: content, model, provider, usage, routing, routing_decision.

    routing_decision has the full chain:
      - ml_router_predicted
      - rule_based_would_pick
      - final_model
      - note
      - intent, complexity, confidence, tokens, time

    Usage:
        @chat(system_prompt="You are a Python expert")
        def ask(query, response):
            print(response["routing_decision"]["ml_router_predicted"])
            print(response["routing_decision"]["final_model"])
            print(response["content"])

        ask("Write a sorting function")
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(query: str, *args: Any, **kwargs: Any) -> Any:
            r = router or get_shared_router()
            start = time.perf_counter()

            try:
                response = r.chat(
                    query,
                    system_prompt=system_prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
            except RuntimeError as exc:
                logger.warning("LLM unavailable: %s. Returning routing-only result.", exc)
                result = r.route(query)
                response = {
                    "content": f"[LLM unavailable] Routed to: {result.selected_model.model_name}",
                    "model": result.selected_model.model_name,
                    "provider": result.selected_model.provider,
                    "usage": {},
                    "finish_reason": "llm_unavailable",
                    "fallback_used": False,
                    "routing": {
                        "detected_intent": result.detected_intent.value,
                        "complexity": result.complexity.value,
                        "confidence": result.confidence,
                        "reasoning": result.reasoning,
                        "originally_selected": result.selected_model.model_name,
                        "selection_mode": result.scoring_details.get("selection_mode", ""),
                        "ml_model_hint": result.scoring_details.get("model_hint", ""),
                    },
                }

            elapsed = time.perf_counter() - start

            # Build routing decision using shared logic (same as API)
            response["routing_decision"] = build_routing_decision(response, elapsed)

            # Auto-save to PostgreSQL
            log_routing_decision_to_db(
                query=query,
                routing_decision=response["routing_decision"],
                strategy=r.config.routing_strategy,
                response_content=response.get("content", ""),
            )

            if log:
                logger.info(
                    "Chat completed: model=%s provider=%s time=%.3fs",
                    response.get("model", "unknown"),
                    response.get("provider", "unknown"),
                    elapsed,
                )

            return func(query, response, *args, **kwargs)

        return wrapper

    return decorator


def with_router(
    *,
    config: Optional[RouterConfig] = None,
    register_defaults: bool = True,
    enable_llm: bool = True,
) -> Callable:
    """
    Decorator that injects a configured Router instance as the first argument.

    Usage:
        @with_router(enable_llm=True)
        def my_app(router):
            result = router.route("Hello")
            print(result.selected_model.model_name)

        my_app()
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            r = Router(config=config or RouterConfig())
            if register_defaults:
                r.register_supported_models()
                r.register_default_intent_rules()
            if enable_llm:
                r.enable_llm_clients()
            return func(r, *args, **kwargs)

        return wrapper

    return decorator


def retry(
    *,
    max_attempts: int = 3,
    fallback_model: Optional[str] = None,
    log: bool = True,
) -> Callable:
    """
    Decorator that retries routing/chat on failure.

    Usage:
        @retry(max_attempts=3, fallback_model="gpt-4o-mini")
        @chat()
        def ask(query, response):
            print(response["content"])
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            last_exc = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    if log:
                        logger.warning(
                            "Attempt %d/%d failed: %s", attempt, max_attempts, exc
                        )

            if log:
                logger.error("All %d attempts failed.", max_attempts)
            raise last_exc  # type: ignore[misc]

        return wrapper

    return decorator


def timed(*, log: bool = True) -> Callable:
    """
    Decorator that measures and logs execution time of any function.

    Usage:
        @timed()
        def process(query):
            return expensive_operation(query)
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            elapsed = time.perf_counter() - start
            if log:
                logger.info("%s executed in %.3fs", func.__name__, elapsed)
            return result

        return wrapper

    return decorator
