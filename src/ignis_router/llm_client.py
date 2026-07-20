"""LLM provider clients for executing queries against routed models."""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""

    content: str
    model: str
    provider: str
    usage: dict[str, int] = field(default_factory=dict)
    finish_reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class BaseLLMClient(ABC):
    """Abstract base class for LLM provider clients."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier (e.g. 'openai', 'anthropic')."""

    @abstractmethod
    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        """Send a chat completion request and return a standardized response."""

    def is_available(self) -> bool:
        """Check if the client is configured and ready."""
        return True


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI chat completions."""

    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self._api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "openai"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "openai package is required. Install with: pip install openai"
                )
            kwargs: dict[str, Any] = {"api_key": self._api_key}
            if self._base_url:
                kwargs["base_url"] = self._base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        choice = response.choices[0]
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
        return LLMResponse(
            content=choice.message.content or "",
            model=response.model,
            provider=self.provider_name,
            usage=usage,
            finish_reason=choice.finish_reason or "",
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic message completions."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self._client: Any = None

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from anthropic import Anthropic
            except ImportError:
                raise ImportError(
                    "anthropic package is required. Install with: pip install anthropic"
                )
            self._client = Anthropic(api_key=self._api_key)
        return self._client

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        client = self._get_client()
        response = client.messages.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        content = ""
        if response.content:
            content = "".join(
                block.text for block in response.content if hasattr(block, "text")
            )
        usage = {}
        if response.usage:
            usage = {
                "prompt_tokens": response.usage.input_tokens,
                "completion_tokens": response.usage.output_tokens,
                "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
            }
        return LLMResponse(
            content=content,
            model=response.model,
            provider=self.provider_name,
            usage=usage,
            finish_reason=response.stop_reason or "",
            raw=response.model_dump() if hasattr(response, "model_dump") else {},
        )


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini generative AI."""

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._client_module: Any = None

    @property
    def provider_name(self) -> str:
        return "gemini" 

    def is_available(self) -> bool:
        return bool(self._api_key)

    def _get_module(self) -> Any:
        if self._client_module is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise ImportError(
                    "google-generativeai package is required. "
                    "Install with: pip install google-generativeai"
                )
            genai.configure(api_key=self._api_key)
            self._client_module = genai
        return self._client_module

    def chat(
        self,
        model: str,
        messages: list[dict[str, str]],
        *,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> LLMResponse:
        genai = self._get_module()
        gen_model = genai.GenerativeModel(model)
        prompt = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        )
        response = gen_model.generate_content(prompt, generation_config=generation_config)
        content = response.text if hasattr(response, "text") else str(response)
        return LLMResponse(
            content=content,
            model=model,
            provider=self.provider_name,
            usage={},
            finish_reason="stop",
            raw={},
        )


class LLMClientRegistry:
    """Registry that maps provider names to LLM client instances."""

    def __init__(self) -> None:
        self._clients: dict[str, BaseLLMClient] = {}

    def register(self, client: BaseLLMClient) -> None:
        """Register a provider client."""
        self._clients[client.provider_name] = client

    def get(self, provider: str) -> Optional[BaseLLMClient]:
        """Retrieve a client by provider name."""
        return self._clients.get(provider)

    def get_available_providers(self) -> list[str]:
        """List providers that have valid API keys configured."""
        return [name for name, client in self._clients.items() if client.is_available()]

    @classmethod
    def from_env(cls) -> "LLMClientRegistry":
        """Build a registry with all supported providers, reading keys from env."""
        registry = cls()
        registry.register(OpenAIClient())
        registry.register(AnthropicClient())
        registry.register(GeminiClient())
        return registry
