"""Main Router class - public API for the ignis_router library."""

from typing import Optional

from .config import RouterConfig, RouterRegistry
from .intent_detector_factory import IntentDetectorFactory
from .models import (
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
