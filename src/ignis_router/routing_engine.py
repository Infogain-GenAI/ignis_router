"""Routing engine that orchestrates intent detection and model selection."""

import logging
from typing import Optional

from .config import RouterConfig, RouterRegistry
from .exceptions import ModelNotAvailableError, RoutingError
from .intent_detector import BaseIntentDetector
from .model_selector import ModelSelector
from .models import (
    Intent,
    ModelCapability,
    ModelConfig,
    RoutingRequest,
    RoutingResult,
    TaskComplexity,
)

logger = logging.getLogger(__name__)


class RoutingEngine:
    """Core routing engine that coordinates intent detection and model selection."""

    def __init__(
        self,
        config: Optional[RouterConfig] = None,
        registry: Optional[RouterRegistry] = None,
        intent_detector: Optional[BaseIntentDetector] = None,
    ):
        self._config = config or RouterConfig()
        self._registry = registry or RouterRegistry()
        self._intent_detector = intent_detector
        self._model_selector = ModelSelector(self._registry, self._config)
        self._ml_router = None

        if self._intent_detector is None:
            # Delayed import to avoid circular dependency between config and factory modules.
            from .intent_detector_factory import IntentDetectorFactory

            self._intent_detector = IntentDetectorFactory.create(self._config)

        # Initialize ML router adapter independently of intent detector
        if self._config.enable_ml_model_hint_routing:
            try:
                from .ml_router_adapter import MLRouterAdapter
                self._ml_router = MLRouterAdapter(
                    router_type=self._config.ml_router_type,
                    project_root=None,
                )
                if not self._ml_router.is_available:
                    logger.warning("ML router adapter not available. Will use fallback path.")
                    self._ml_router = None
            except Exception as exc:
                logger.warning("Failed to initialize ML router adapter: %s", exc)
                self._ml_router = None

    @property
    def config(self) -> RouterConfig:
        return self._config

    @property
    def registry(self) -> RouterRegistry:
        return self._registry

    def route(self, request: RoutingRequest) -> RoutingResult:
        """
        Route a request to the most appropriate model.

        Args:
            request: The routing request containing the query and constraints.

        Returns:
            RoutingResult with the selected model and routing metadata.

        Raises:
            RoutingError: If no suitable model can be found.
            ModelNotAvailableError: If no models are registered.
        """
        if not self._registry.get_enabled_models():
            raise ModelNotAvailableError("No models are registered or enabled.")

        # Detect intent
        intent, confidence = self._detect_intent(request)

        # Assess complexity
        complexity = self._assess_complexity(request)

        # Try intent-based selection first (includes intent rules and scoring).
        model_hint = self._extract_model_hint()

        # Try ML router adapter (llmrouter-lib) for model prediction
        if self._config.enable_ml_model_hint_routing and self._ml_router is not None and not model_hint:
            ml_prediction = self._ml_router.predict_model(request.query)
            if ml_prediction:
                model_hint = ml_prediction
                logger.info("ML router '%s' predicted: %s", self._ml_router.router_type, ml_prediction)
            else:
                logger.warning(
                    "ML router '%s' returned no prediction. Trying legacy model hint.",
                    self._ml_router.router_type,
                )

        # If ML hint routing is enabled and ML produced a model hint,
        # use the ML prediction directly — skip intent-based selection.
        if self._config.enable_ml_model_hint_routing and model_hint:
            hinted_model = self._build_ml_predicted_model(model_hint)
            if hinted_model is not None:
                selected_model = hinted_model
                fallbacks = self._build_fallback_models(
                    hinted_model,
                    request.preferred_provider,
                    request.max_cost,
                )
                scoring_details = {
                    "selection_mode": "ml-model-hint",
                    "strategy": self._config.routing_strategy,
                    "weights": dict(self._config.routing_weights),
                    "scores": {hinted_model.model_id: round(confidence, 4)},
                    "model_hint": model_hint,
                    "ml_won": True,
                }
            else:
                selected_model = None
                fallbacks = []
                scoring_details = {}
        else:
            selected_model = None
            fallbacks = []
            scoring_details = {}

        # If ML hint was not used, fall back to intent-based selection.
        if selected_model is None:
            try:
                selected_model, fallbacks, scoring_details = self._model_selector.select_with_details(
                    intent=intent,
                    complexity=complexity,
                    required_capabilities=request.required_capabilities or None,
                    preferred_provider=request.preferred_provider,
                    max_cost=request.max_cost,
                )
            except Exception as exc:
                logger.warning(
                    "Routing selection failed. Attempting default fallback model. error=%s",
                    exc,
                )
                default_model = self._resolve_default_fallback_model()
                if default_model is None:
                    raise RoutingError("Routing selection failed and no fallback model is available.") from exc

                fallbacks = self._build_fallback_models(
                    default_model,
                    request.preferred_provider,
                    request.max_cost,
                )
                scoring_details = {
                    "selection_mode": "default-fallback-exception",
                    "strategy": self._config.routing_strategy,
                    "weights": dict(self._config.routing_weights),
                    "scores": {},
                }
                selected_model = default_model

        if selected_model is None:
            default_model = self._resolve_default_fallback_model()
            if default_model is None:
                raise RoutingError(
                    f"No suitable model found for intent={intent.value}, "
                    f"complexity={complexity.value}"
                )

            selected_model = default_model
            fallbacks = self._build_fallback_models(
                default_model,
                request.preferred_provider,
                request.max_cost,
            )
            scoring_details = {
                "selection_mode": "default-fallback-no-candidate",
                "strategy": self._config.routing_strategy,
                "weights": dict(self._config.routing_weights),
                "scores": {},
            }

        # Check confidence threshold
        if confidence < self._config.confidence_threshold:
            default_model = self._resolve_default_fallback_model()
            if default_model is not None:
                logger.info(
                    "Low-confidence routing. Using default fallback model. "
                    "confidence=%.3f threshold=%.3f model_id=%s",
                    confidence,
                    self._config.confidence_threshold,
                    default_model.model_id,
                )
                selected_model = default_model
                scoring_details["selection_mode"] = "default-fallback-low-confidence"

        reasoning = self._build_reasoning(intent, complexity, confidence, selected_model)

        return RoutingResult(
            selected_model=selected_model,
            detected_intent=intent,
            complexity=complexity,
            confidence=confidence,
            reasoning=reasoning,
            scoring_details=scoring_details,
            fallback_models=fallbacks,
        )

    def _resolve_default_fallback_model(self) -> Optional[ModelConfig]:
        """Resolve a configured default fallback model when fallback is enabled."""
        if not self._config.fallback_enabled:
            logger.warning("Fallback is disabled in configuration.")
            return None

        if not self._config.default_model_id:
            logger.warning("No default model configured for fallback.")
            return None

        default_model = self._registry.get_model(self._config.default_model_id)
        if default_model is None:
            logger.warning(
                "Configured default fallback model not found. model_id=%s",
                self._config.default_model_id,
            )
            return None

        if not default_model.enabled:
            logger.warning(
                "Configured default fallback model is disabled. model_id=%s",
                default_model.model_id,
            )
            return None

        logger.info("Using default fallback model. model_id=%s", default_model.model_id)
        return default_model

    def _build_fallback_models(
        self,
        primary: ModelConfig,
        preferred_provider: Optional[str],
        max_cost: Optional[float],
    ) -> list[ModelConfig]:
        """Build fallback candidates from enabled models excluding the primary."""
        fallbacks = [
            model
            for model in self._registry.get_enabled_models()
            if model.model_id != primary.model_id
        ]

        if max_cost is not None:
            fallbacks = [m for m in fallbacks if m.cost_per_1k_input_tokens <= max_cost]

        if preferred_provider:
            fallbacks.sort(key=lambda m: (m.provider != preferred_provider, -m.priority))
        else:
            fallbacks.sort(key=lambda m: -m.priority)

        return fallbacks[:3]

    def route_simple(self, query: str) -> RoutingResult:
        """Convenience method to route a simple query string."""
        request = RoutingRequest(query=query)
        return self.route(request)

    def _detect_intent(self, request: RoutingRequest) -> tuple[Intent, float]:
        """Detect intent from the request."""
        return self._intent_detector.detect_intent(request.query)

    def _assess_complexity(self, request: RoutingRequest) -> TaskComplexity:
        """Assess complexity of the request."""
        if not self._config.enable_complexity_assessment:
            return TaskComplexity.MEDIUM

        return self._intent_detector.assess_complexity(request.query)

    def _build_reasoning(
        self,
        intent: Intent,
        complexity: TaskComplexity,
        confidence: float,
        model: ModelConfig,
    ) -> str:
        """Build a human-readable explanation of the routing decision."""
        return (
            f"Detected intent '{intent.value}' with confidence {confidence:.2f}. "
            f"Task complexity assessed as '{complexity.value}'. "
            f"Selected model '{model.model_name}' (provider: {model.provider}) "
            f"based on capability matching and priority scoring."
        )

    def _extract_model_hint(self) -> Optional[str]:
        """Read model hint from detector if detector supports hint exposure."""
        getter = getattr(self._intent_detector, "get_model_hint", None)
        if callable(getter):
            value = getter()
            return value.strip() if isinstance(value, str) and value.strip() else None
        return None

    def _resolve_model_hint(self, model_hint: str) -> Optional[ModelConfig]:
        """Deprecated compatibility wrapper for older call sites."""
        return self._build_ml_predicted_model(model_hint)

    def _build_ml_predicted_model(self, model_hint: str) -> Optional[ModelConfig]:
        """Build a selected model directly from ML prediction output without mapping."""
        hinted_name = model_hint.strip()
        if not hinted_name:
            return None

        logger.info(
            "Using raw ML model hint for direct selection. model_hint=%s",
            hinted_name,
        )

        return ModelConfig(
            model_id=hinted_name,
            provider="ml-prediction",
            model_name=hinted_name,
            metadata={"source": "ml-model-hint"},
        )
