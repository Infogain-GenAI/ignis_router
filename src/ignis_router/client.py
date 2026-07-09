"""Python SDK client for the Ignis Router API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import httpx
except ImportError:
    httpx = None  # type: ignore[assignment]


@dataclass
class RouteResult:
    """Result from the /route endpoint."""

    selected_model: str
    strategy: str
    confidence: float


@dataclass
class ChatResult:
    """Result from the /chat endpoint."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    routing: dict[str, Any] = field(default_factory=dict)


class IgnisClient:
    """
    Python SDK client for interacting with the Ignis Router REST API.

    Usage:
        from ignis_router.client import IgnisClient

        client = IgnisClient("http://127.0.0.1:8080")

        # Route only (no LLM call)
        route = client.route("Write Python code for sorting")
        print(route.selected_model)

        # Route + execute LLM
        chat = client.chat("Write Python code for sorting")
        print(chat.content)
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8080", timeout: float = 60.0):
        if httpx is None:
            raise ImportError(
                "httpx is required for the SDK client. Install with: pip install httpx"
            )
        self._base_url = base_url.rstrip("/")
        self._http = httpx.Client(base_url=self._base_url, timeout=timeout)

    def health(self) -> dict[str, str]:
        """Check API health."""
        response = self._http.get("/health")
        response.raise_for_status()
        return response.json()

    def providers(self) -> list[str]:
        """List available LLM providers."""
        response = self._http.get("/providers")
        response.raise_for_status()
        return response.json().get("available_providers", [])

    def route(self, query: str) -> RouteResult:
        """
        Route a query to the best model (no LLM execution).

        Args:
            query: The user prompt to route.

        Returns:
            RouteResult with selected_model, strategy, confidence.
        """
        response = self._http.post("/route", json={"query": query})
        response.raise_for_status()
        data = response.json()
        return RouteResult(
            selected_model=data["selected_model"],
            strategy=data["strategy"],
            confidence=data["confidence"],
        )

    def chat(
        self,
        query: str,
        *,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResult:
        """
        Route a query AND execute it against the selected LLM.

        Args:
            query: The user prompt.
            system_prompt: System message for the LLM.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

        Returns:
            ChatResult with AI response content, model, provider, usage, and routing info.
        """
        payload = {
            "query": query,
            "system_prompt": system_prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        response = self._http.post("/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        return ChatResult(
            content=data["content"],
            model=data["model"],
            provider=data["provider"],
            usage=data.get("usage", {}),
            finish_reason=data.get("finish_reason", ""),
            routing=data.get("routing", {}),
        )

    def close(self) -> None:
        """Close the HTTP client."""
        self._http.close()

    def __enter__(self) -> "IgnisClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
