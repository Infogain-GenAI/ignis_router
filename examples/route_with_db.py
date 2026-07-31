"""Interactive routing demo that prints result and stores it in PostgreSQL."""

from __future__ import annotations

import json
import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv

from ignis_router import Router, RouterConfig
from ignis_router.db.persistence import PostgresRouteLogger


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_root(path_value: str) -> str:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(_project_root() / path_obj)


def build_router() -> Router:
    # Load environment variables from project root so script works from any cwd.
    load_dotenv(_project_root() / ".env")

    yaml_config = os.getenv("ROUTER_YAML_CONFIG")
    if yaml_config:
        config = RouterConfig.from_yaml(_resolve_from_root(yaml_config))
    else:
        config = RouterConfig.from_yaml(str(_project_root() / "configs" / "quality-first.yaml"))

    # Keep model path stable regardless of current working directory.
    config.ml_model_path = _resolve_from_root(config.ml_model_path)

    router = Router(config=config)
    router.register_supported_models()
    router.register_default_intent_rules()
    return router


def main() -> None:
    router = build_router()
    db_logger = PostgresRouteLogger()
    try:
        db_logger.ensure_table()
    except psycopg.OperationalError as exc:
        settings = db_logger.settings
        print("PostgreSQL connection failed.")
        print(
            "Check ROUTER_DB_HOST/PORT/NAME/USER/PASSWORD in .env. "
            f"Current target: {settings.user}@{settings.host}:{settings.port}/{settings.dbname}"
        )
        raise

    print("Type your query and press Enter. Type 'exit' to quit.")
    print("Every response will be printed and written to PostgreSQL.")

    while True:
        query = input("\nQuery: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        result = router.route(query)
        response = {
            "selected_model": result.selected_model.model_name,
            "strategy": router.config.routing_strategy,
            "confidence": round(result.confidence, 2),
        }

        print("Response:")
        print(json.dumps(response, indent=2))

        db_logger.log_response(
            query=query,
            result=result,
            strategy=router.config.routing_strategy,
        )
        print("Saved to PostgreSQL.")


if __name__ == "__main__":
    main()
