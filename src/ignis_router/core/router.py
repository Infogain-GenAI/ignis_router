"""Main Router class - public API for the ignis_router library."""

from typing import Any, Optional

from ..config import RouterConfig, RouterRegistry
from ..detection.intent_detector_factory import IntentDetectorFactory
from ..llm.llm_client import LLMClientRegistry, LLMResponse
from ..models import (
    ModelCapability,
    ModelConfig,
    RoutingRequest,
    RoutingResult,
    RoutingRule,
)
from .routing_engine import RoutingEngine
from .supported_models import get_default_intent_rules, get_default_supported_models


class Router:
    """
    Main entry point for the ignis_router library.

    Provides a high-level API for registering models, defining routing rules,
    and routing queries to the most appropriate LLM.

    Usage:
        from ignis_router import Router, ModelConfig, ModelCapability

        router = Router()
        router.register_model(ModelConfig(
            model_id="gpt-4",
            provider="openai",
            model_name="gpt-4",
            capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.REASONING],
        ))
        result = router.route("Explain quantum computing step by step")
        print(result.selected_model.model_name)
    """

    def __init__(self, config: Optional[RouterConfig] = None):
        """
        Initialize the Router.

        Args:
            config: Optional router configuration. Uses defaults if not provided.
        """
        self._config = config or RouterConfig()
        self._registry = RouterRegistry()
        self._intent_detector = IntentDetectorFactory.create(self._config)
        self._engine = RoutingEngine(
            config=self._config,
            registry=self._registry,
            intent_detector=self._intent_detector,
        )
        self._llm_clients: Optional[LLMClientRegistry] = None

    @property
    def config(self) -> RouterConfig:
        """Get the router configuration."""
        return self._config

    @property
    def registry(self) -> RouterRegistry:
        """Get the model registry."""
        return self._registry

    def register_model(self, model: ModelConfig) -> "Router":
        """
        Register a model for routing.

        Args:
            model: Model configuration to register.

        Returns:
            Self for method chaining.
        """
        self._registry.register_model(model)
        return self

    def unregister_model(self, model_id: str) -> "Router":
        """
        Remove a model from the registry.

        Args:
            model_id: ID of the model to remove.

        Returns:
            Self for method chaining.
        """
        self._registry.unregister_model(model_id)
        return self

    def register_supported_models(self, models: Optional[list[ModelConfig]] = None) -> "Router":
        """
        Register a default or custom catalog of supported models.

        Args:
            models: Optional list of model configurations. If omitted, uses
                    the built-in supported model catalog.

        Returns:
            Self for method chaining.
        """
        self._registry.register_models(models or get_default_supported_models())
        return self

    def register_default_intent_rules(self) -> "Router":
        """Register built-in strict intent-to-model rules."""
        for rule in get_default_intent_rules():
            self._registry.add_rule(rule)
        return self

    def add_rule(self, rule: RoutingRule) -> "Router":
        """
        Add a routing rule.

        Args:
            rule: Routing rule to add.

        Returns:
            Self for method chaining.
        """
        self._registry.add_rule(rule)
        return self

    def remove_rule(self, rule_id: str) -> "Router":
        """
        Remove a routing rule.

        Args:
            rule_id: ID of the rule to remove.

        Returns:
            Self for method chaining.
        """
        self._registry.remove_rule(rule_id)
        return self

    def route(self, query: str, **kwargs) -> RoutingResult:
        """
        Route a query to the most appropriate model.

        Args:
            query: The user query/prompt to route.
            **kwargs: Additional routing parameters (preferred_provider, max_cost,
                      required_capabilities, context).

        Returns:
            RoutingResult with the selected model and metadata.

        Raises:
            RoutingError: If no suitable model can be found.
            ModelNotAvailableError: If no models are registered.
        """
        request = RoutingRequest(query=query, **kwargs)
        return self._engine.route(request)

    def route_request(self, request: RoutingRequest) -> RoutingResult:
        """
        Route a pre-built RoutingRequest.

        Args:
            request: The routing request object.

        Returns:
            RoutingResult with the selected model and metadata.
        """
        return self._engine.route(request)

    def get_registered_models(self) -> list[ModelConfig]:
        """Get all registered models."""
        return self._registry.get_all_models()

    def get_enabled_models(self) -> list[ModelConfig]:
        """Get all enabled models."""
        return self._registry.get_enabled_models()

    def enable_llm_clients(self, llm_registry: Optional[LLMClientRegistry] = None) -> "Router":
        """
        Enable LLM execution so chat() can call real model APIs.

        Args:
            llm_registry: Optional pre-built client registry.
                          If omitted, builds from environment variables.

        Returns:
            Self for method chaining.
        """
        self._llm_clients = llm_registry or LLMClientRegistry.from_env()
        return self

    @property
    def llm_clients(self) -> Optional[LLMClientRegistry]:
        """Get the LLM client registry if enabled."""
        return self._llm_clients

    def chat(
        self,
        query: str,
        *,
        system_prompt: str = "You are a helpful assistant.",
        max_tokens: int = 1024,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Route a query to the best model AND execute it, returning the AI response.

        Args:
            query: The user prompt.
            system_prompt: System message for the LLM.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.
            **kwargs: Additional routing parameters (preferred_provider, max_cost, etc.)

        Returns:
            Dict with routing_result, llm_response content, model, provider, and usage.

        Raises:
            RuntimeError: If LLM clients are not enabled or provider is unavailable.
        """
        if self._llm_clients is None:
            raise RuntimeError(
                "LLM clients not enabled. Call router.enable_llm_clients() first."
            )

        # Extract routing-specific kwargs
        routing_kwargs = {}
        for key in ("preferred_provider", "max_cost", "required_capabilities", "context"):
            if key in kwargs:
                routing_kwargs[key] = kwargs.pop(key)

        # Route to best model
        result = self.route(query, **routing_kwargs)
        provider = result.selected_model.provider
        model_name = result.selected_model.model_name

        # Get LLM client for selected provider
        client = self._llm_clients.get(provider)
        fallback_used = False

        if client is None or not client.is_available():
            # Selected model's provider has no API key — try fallback models
            fallback_client = None
            fallback_model = None

            for fb in result.fallback_models:
                fb_client = self._llm_clients.get(fb.provider)
                if fb_client is not None and fb_client.is_available():
                    fallback_client = fb_client
                    fallback_model = fb
                    break

            # If no fallback from routing result, try any available provider
            if fallback_client is None:
                available_providers = self._llm_clients.get_available_providers()
                for ap in available_providers:
                    for m in self._registry.get_enabled_models():
                        if m.provider == ap:
                            fallback_client = self._llm_clients.get(ap)
                            fallback_model = m
                            break
                    if fallback_client:
                        break

            if fallback_client is None or fallback_model is None:
                raise RuntimeError(
                    f"No API client available for provider '{provider}'. "
                    f"Set the API key or install the provider package."
                )

            import logging
            logging.getLogger(__name__).warning(
                "Provider '%s' unavailable. Switching to '%s' (%s).",
                provider,
                fallback_model.model_name,
                fallback_model.provider,
            )
            client = fallback_client
            provider = fallback_model.provider
            model_name = fallback_model.model_name
            fallback_used = True

        # Build messages and call LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        llm_response = client.chat(
            model=model_name,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )

        return {
            "content": llm_response.content,
            "model": llm_response.model,
            "provider": llm_response.provider,
            "usage": llm_response.usage,
            "finish_reason": llm_response.finish_reason,
            "fallback_used": fallback_used,
            "routing": {
                "detected_intent": result.detected_intent.value,
                "complexity": result.complexity.value,
                "confidence": result.confidence,
                "reasoning": result.reasoning,
                "originally_selected": result.selected_model.model_name,
                "selection_mode": result.scoring_details.get("selection_mode", ""),
                "ml_model_hint": result.scoring_details.get("model_hint", ""),
                "ml_won": result.scoring_details.get("ml_won", False),
            },
        }
