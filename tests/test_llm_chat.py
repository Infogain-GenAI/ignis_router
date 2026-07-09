"""Tests for the LLM client layer, Router.chat(), and /chat API endpoint."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ignis_router import (
    ModelCapability,
    ModelConfig,
    Router,
    RouterConfig,
)
from ignis_router.api import create_app
from ignis_router.llm_client import (
    BaseLLMClient,
    LLMClientRegistry,
    LLMResponse,
    OpenAIClient,
)


# ---------------------------------------------------------------------------
# Mock LLM client for testing without real API keys
# ---------------------------------------------------------------------------
class _MockLLMClient(BaseLLMClient):
    """Fake LLM client that returns canned responses."""

    def __init__(self, provider: str = "openai"):
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return self._provider

    def is_available(self) -> bool:
        return True

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        user_msg = messages[-1]["content"] if messages else ""
        return LLMResponse(
            content=f"Mock response to: {user_msg}",
            model=model,
            provider=self._provider,
            usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
            finish_reason="stop",
        )


def _build_test_router_with_mock_llm() -> Router:
    """Build a Router with mock LLM clients for testing."""
    config = RouterConfig(
        enable_ml_intent_detection=False,
        enable_rule_based_intent_detection=True,
    )
    router = Router(config=config)
    router.register_model(
        ModelConfig(
            model_id="gpt-4o",
            provider="openai",
            model_name="gpt-4o",
            capabilities=[
                ModelCapability.HIGH_QUALITY,
                ModelCapability.REASONING,
                ModelCapability.CODE,
            ],
            priority=10,
        )
    )
    router.register_model(
        ModelConfig(
            model_id="claude-sonnet",
            provider="anthropic",
            model_name="claude-3-5-sonnet",
            capabilities=[
                ModelCapability.HIGH_QUALITY,
                ModelCapability.CREATIVE,
            ],
            priority=8,
        )
    )

    # Register mock LLM clients
    llm_reg = LLMClientRegistry()
    llm_reg.register(_MockLLMClient("openai"))
    llm_reg.register(_MockLLMClient("anthropic"))
    router.enable_llm_clients(llm_reg)
    return router


# ---------------------------------------------------------------------------
# LLM Client Registry Tests
# ---------------------------------------------------------------------------
class TestLLMClientRegistry:
    def test_register_and_retrieve_client(self):
        reg = LLMClientRegistry()
        client = _MockLLMClient("openai")
        reg.register(client)
        assert reg.get("openai") is client

    def test_get_unknown_provider_returns_none(self):
        reg = LLMClientRegistry()
        assert reg.get("unknown") is None

    def test_available_providers(self):
        reg = LLMClientRegistry()
        reg.register(_MockLLMClient("openai"))
        reg.register(_MockLLMClient("anthropic"))
        assert set(reg.get_available_providers()) == {"openai", "anthropic"}

    def test_from_env_creates_all_providers(self):
        reg = LLMClientRegistry.from_env()
        # All three providers are registered (even if keys are missing)
        assert reg.get("openai") is not None
        assert reg.get("anthropic") is not None
        assert reg.get("gemini") is not None


# ---------------------------------------------------------------------------
# Router.chat() Tests
# ---------------------------------------------------------------------------
class TestRouterChat:
    def test_chat_returns_response(self):
        router = _build_test_router_with_mock_llm()
        result = router.chat("Write Python code for sorting")

        assert "content" in result
        assert "Mock response" in result["content"]
        assert result["provider"] in {"openai", "anthropic"}
        assert result["usage"]["total_tokens"] == 30
        assert "routing" in result
        assert "detected_intent" in result["routing"]

    def test_chat_raises_without_enabling_clients(self):
        router = Router(
            config=RouterConfig(
                enable_ml_intent_detection=False,
                enable_rule_based_intent_detection=True,
            )
        )
        router.register_model(
            ModelConfig(model_id="test", provider="openai", model_name="test")
        )
        with pytest.raises(RuntimeError, match="LLM clients not enabled"):
            router.chat("Hello")

    def test_chat_raises_for_unavailable_provider(self):
        config = RouterConfig(
            enable_ml_intent_detection=False,
            enable_rule_based_intent_detection=True,
        )
        router = Router(config=config)
        router.register_model(
            ModelConfig(
                model_id="test",
                provider="unknown_provider",
                model_name="test-model",
                capabilities=[ModelCapability.FAST_RESPONSE],
                priority=100,
            )
        )
        # Empty LLM registry
        router.enable_llm_clients(LLMClientRegistry())
        with pytest.raises(RuntimeError, match="No API client available"):
            router.chat("Hello")

    def test_chat_with_custom_params(self):
        router = _build_test_router_with_mock_llm()
        result = router.chat(
            "Explain quantum physics",
            system_prompt="You are a physics professor.",
            max_tokens=512,
            temperature=0.3,
        )
        assert "content" in result
        assert result["routing"]["detected_intent"] is not None


# ---------------------------------------------------------------------------
# API /chat Endpoint Tests
# ---------------------------------------------------------------------------
class TestChatApi:
    def test_chat_endpoint_success(self):
        router = _build_test_router_with_mock_llm()
        app = create_app(enable_db=False, router=router)
        client = TestClient(app)

        response = client.post(
            "/chat",
            json={"query": "Write a Python function"},
        )

        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "Mock response" in data["content"]
        assert data["provider"] in {"openai", "anthropic"}
        assert "routing" in data

    def test_chat_endpoint_with_params(self):
        router = _build_test_router_with_mock_llm()
        app = create_app(enable_db=False, router=router)
        client = TestClient(app)

        response = client.post(
            "/chat",
            json={
                "query": "Write a poem",
                "system_prompt": "You are a poet.",
                "max_tokens": 256,
                "temperature": 0.9,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "content" in data

    def test_chat_endpoint_validation_error(self):
        router = _build_test_router_with_mock_llm()
        app = create_app(enable_db=False, router=router)
        client = TestClient(app)

        response = client.post("/chat", json={})

        assert response.status_code == 422
        assert response.json()["error"] == "validation_error"

    def test_providers_endpoint(self):
        router = _build_test_router_with_mock_llm()
        app = create_app(enable_db=False, router=router)
        client = TestClient(app)

        response = client.get("/providers")

        assert response.status_code == 200
        data = response.json()
        assert "available_providers" in data
        assert set(data["available_providers"]) == {"openai", "anthropic"}
