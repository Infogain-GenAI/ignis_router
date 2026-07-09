"""
Example AI Chat App using Ignis Router.

This script demonstrates the full end-to-end flow:
1. Router selects the best LLM for your query
2. The selected LLM is called via its provider API
3. The AI response is returned to you

Prerequisites:
    pip install -e ".[all]"
    Set OPENAI_API_KEY in your .env file

Usage:
    python examples/ai_chat_app.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from ignis_router import Router, RouterConfig


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_from_root(path_value: str) -> str:
    path_obj = Path(path_value)
    if path_obj.is_absolute():
        return str(path_obj)
    return str(_project_root() / path_obj)


def build_router() -> Router:
    """Build a router with LLM execution enabled."""
    load_dotenv(_project_root() / ".env")

    yaml_config = os.getenv("ROUTER_YAML_CONFIG")
    if yaml_config:
        config = RouterConfig.from_yaml(_resolve_from_root(yaml_config))
    else:
        config = RouterConfig.from_yaml(str(_project_root() / "configs" / "quality-first.yaml"))

    config.ml_model_path = _resolve_from_root(config.ml_model_path)

    router = Router(config=config)
    router.register_supported_models()
    router.register_default_intent_rules()
    router.enable_llm_clients()
    return router


def main() -> None:
    router = build_router()

    # Show available providers
    available = router.llm_clients.get_available_providers() if router.llm_clients else []
    print("=" * 60)
    print("Ignis Router - AI Chat App")
    print("=" * 60)
    print(f"Available LLM providers: {available or ['None configured']}")
    if not available:
        print("\nNo API keys found. Set OPENAI_API_KEY, ANTHROPIC_API_KEY,")
        print("or GOOGLE_API_KEY in your .env file.")
        return
    print("\nType your query and press Enter. Type 'exit' to quit.")
    print("The router will pick the best model and call it for you.\n")

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        try:
            result = router.chat(query)
        except Exception as exc:
            print(f"\nError: {exc}\n")
            continue

        print(f"\n--- Routing ---")
        print(f"Model:      {result['model']}")
        print(f"Provider:   {result['provider']}")
        print(f"Intent:     {result['routing']['detected_intent']}")
        print(f"Complexity: {result['routing']['complexity']}")
        print(f"Confidence: {result['routing']['confidence']:.2f}")

        usage = result.get("usage", {})
        if usage:
            print(f"Tokens:     {usage.get('total_tokens', 'N/A')}")

        print(f"\n--- Response ---")
        print(result["content"])
        print()


if __name__ == "__main__":
    main()
