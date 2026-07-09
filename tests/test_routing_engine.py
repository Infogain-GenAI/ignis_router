"""Tests for the RoutingEngine class."""

import pytest

from ignis_router import (
    Intent,
    HybridIntentDetector,
    MLIntentDetector,
    ModelCapability,
    ModelConfig,
    ModelNotAvailableError,
    RoutingError,
    RuleBasedIntentDetector,
    RouterConfig,
    RoutingRequest,
    TaskComplexity,
)
from ignis_router.config import RouterRegistry
from ignis_router.routing_engine import RoutingEngine


@pytest.fixture
def registry_with_models():
    """Create a registry with test models."""
    registry = RouterRegistry()
    registry.register_model(
        ModelConfig(
            model_id="fast-model",
            provider="openai",
            model_name="gpt-4o-mini",
            capabilities=[ModelCapability.FAST_RESPONSE, ModelCapability.COST_EFFECTIVE],
            cost_per_1k_input_tokens=0.00015,
            priority=3,
        )
    )
    registry.register_model(
        ModelConfig(
            model_id="quality-model",
            provider="anthropic",
            model_name="claude-3-5-sonnet",
            capabilities=[
                ModelCapability.HIGH_QUALITY,
                ModelCapability.REASONING,
                ModelCapability.CODE,
            ],
            cost_per_1k_input_tokens=0.003,
            priority=8,
        )
    )
    return registry


@pytest.fixture
def engine(registry_with_models):
    """Create a routing engine."""
    config = RouterConfig()
    return RoutingEngine(config=config, registry=registry_with_models)


class TestRoutingEngine:
    def test_route_basic(self, engine):
        request = RoutingRequest(query="Hello")
        result = engine.route(request)
        assert result.selected_model is not None

    def test_route_returns_scoring_details(self, engine):
        request = RoutingRequest(query="Write a Python function")
        result = engine.route(request)
        assert isinstance(result.scoring_details, dict)
        assert "selection_mode" in result.scoring_details
        assert "strategy" in result.scoring_details
        assert "weights" in result.scoring_details
        assert "scores" in result.scoring_details

    def test_route_simple_convenience(self, engine):
        result = engine.route_simple("What is Python?")
        assert result.selected_model is not None

    def test_route_no_models_raises(self):
        engine = RoutingEngine(registry=RouterRegistry())
        request = RoutingRequest(query="Hello")
        with pytest.raises(ModelNotAvailableError):
            engine.route(request)

    def test_route_detects_intent(self, registry_with_models):
        config = RouterConfig(
            enable_ml_intent_detection=False,
            enable_rule_based_intent_detection=True,
        )
        engine = RoutingEngine(config=config, registry=registry_with_models)
        request = RoutingRequest(query="Write a Python function to reverse a string")
        result = engine.route(request)
        assert result.detected_intent == Intent.CODE_GENERATION

    def test_unknown_request_is_handled(self, engine):
        request = RoutingRequest(query="qwe asd zxc ghj")
        result = engine.route(request)
        assert result.selected_model is not None
        assert result.detected_intent == Intent.GENERAL_CHAT

    def test_route_assesses_complexity(self, engine):
        request = RoutingRequest(query="What is 2+2?")
        result = engine.route(request)
        assert result.complexity == TaskComplexity.LOW

    def test_route_ml_only_mode(self, registry_with_models):
        class _MlOnlyModel:
            def predict(self, samples):
                return ["code_generation"]

            def predict_proba(self, samples):
                return [[0.92, 0.08]]

        config = RouterConfig(
            enable_ml_intent_detection=True,
            enable_rule_based_intent_detection=False,
        )
        detector = MLIntentDetector(model=_MlOnlyModel())
        engine = RoutingEngine(
            config=config,
            registry=registry_with_models,
            intent_detector=detector,
        )

        request = RoutingRequest(query="Write complex code")
        result = engine.route(request)
        assert result.detected_intent == Intent.CODE_GENERATION
        assert result.confidence == 0.92

    def test_route_rule_based_only_mode(self, registry_with_models):
        config = RouterConfig(
            enable_ml_intent_detection=False,
            enable_rule_based_intent_detection=True,
        )
        engine = RoutingEngine(
            config=config,
            registry=registry_with_models,
            intent_detector=RuleBasedIntentDetector(),
        )

        request = RoutingRequest(query="Translate this sentence to French")
        result = engine.route(request)
        assert result.detected_intent == Intent.TRANSLATION

    def test_route_uses_ml_model_hint_for_direct_selection(self, registry_with_models):
        """ML hint is used only when selector finds no candidate (empty registry scenario)."""
        class _HintingDetector:
            def __init__(self):
                self._hint = "claude-3-5-sonnet"

            def detect_intent(self, text):
                return Intent.GENERAL_CHAT, 0.95

            def assess_complexity(self, text):
                return TaskComplexity.MEDIUM

            def get_model_hint(self):
                return self._hint

        config = RouterConfig(
            ml_confidence_threshold=0.6,
            enable_ml_intent_detection=True,
            enable_rule_based_intent_detection=True,
            enable_ml_model_hint_routing=True,
        )
        engine = RoutingEngine(
            config=config,
            registry=registry_with_models,
            intent_detector=_HintingDetector(),
        )

        # ML hint takes priority when enable_ml_model_hint_routing=True
        result = engine.route(RoutingRequest(query="any"))
        assert result.selected_model is not None
        assert result.scoring_details.get("selection_mode") == "ml-model-hint"

    def test_route_uses_raw_ml_model_hint_when_no_selector_candidate(self):
        """ML hint fires only when selector returns no candidate."""
        empty_registry = RouterRegistry()

        class _AliasedHintingDetector:
            def detect_intent(self, text):
                return Intent.GENERAL_CHAT, 0.95

            def assess_complexity(self, text):
                return TaskComplexity.MEDIUM

            def get_model_hint(self):
                return "gemma-2-9b-it"

        # Register one model so route() doesn't raise ModelNotAvailableError,
        # but make it disabled so selector returns None.
        empty_registry.register_model(
            ModelConfig(
                model_id="placeholder",
                provider="test",
                model_name="placeholder",
                enabled=True,
                capabilities=[],
                cost_per_1k_input_tokens=999,
            )
        )

        config = RouterConfig(
            ml_confidence_threshold=0.6,
            enable_ml_intent_detection=True,
            enable_rule_based_intent_detection=True,
            enable_ml_model_hint_routing=True,
        )

        engine = RoutingEngine(
            config=config,
            registry=empty_registry,
            intent_detector=_AliasedHintingDetector(),
        )

        result = engine.route(RoutingRequest(query="any"))
        # Selector returns a candidate (the placeholder), so hint is not used
        assert result.selected_model is not None

    def test_route_hybrid_fallback_on_low_confidence(self, registry_with_models):
        class _LowConfidenceMlModel:
            def predict(self, samples):
                return ["general_chat"]

            def predict_proba(self, samples):
                return [[0.2, 0.8]]

        config = RouterConfig(
            enable_ml_intent_detection=True,
            enable_rule_based_intent_detection=True,
            ml_confidence_threshold=0.6,
        )
        detector = HybridIntentDetector(
            ml_detector=MLIntentDetector(model=_LowConfidenceMlModel()),
            rule_based_detector=RuleBasedIntentDetector(),
            confidence_threshold=config.ml_confidence_threshold,
        )
        engine = RoutingEngine(
            config=config,
            registry=registry_with_models,
            intent_detector=detector,
        )

        request = RoutingRequest(query="Write a Python function")
        result = engine.route(request)
        assert result.detected_intent == Intent.CODE_GENERATION
        assert result.confidence > 0.2

    def test_route_hybrid_fallback_when_ml_unavailable(self, registry_with_models):
        config = RouterConfig(
            enable_ml_intent_detection=True,
            enable_rule_based_intent_detection=True,
            ml_confidence_threshold=0.6,
        )
        detector = HybridIntentDetector(
            ml_detector=MLIntentDetector(model_path="models/missing.pkl"),
            rule_based_detector=RuleBasedIntentDetector(),
            confidence_threshold=config.ml_confidence_threshold,
        )
        engine = RoutingEngine(
            config=config,
            registry=registry_with_models,
            intent_detector=detector,
        )

        request = RoutingRequest(query="Summarize this text")
        result = engine.route(request)
        assert result.detected_intent == Intent.SUMMARIZATION

    def test_route_with_disabled_complexity_assessment(self, registry_with_models):
        config = RouterConfig(enable_complexity_assessment=False)
        engine = RoutingEngine(config=config, registry=registry_with_models)
        request = RoutingRequest(query="Simple hello")
        result = engine.route(request)
        assert result.complexity == TaskComplexity.MEDIUM

    def test_default_model_on_low_confidence(self, registry_with_models):
        config = RouterConfig(
            confidence_threshold=0.99,
            default_model_id="fast-model",
        )
        engine = RoutingEngine(config=config, registry=registry_with_models)
        request = RoutingRequest(query="xyz abc random tokens")
        result = engine.route(request)
        assert result.selected_model.model_id == "fast-model"

    def test_routing_failure_uses_default_fallback(self, registry_with_models, monkeypatch):
        config = RouterConfig(
            default_model_id="fast-model",
            fallback_enabled=True,
        )
        engine = RoutingEngine(config=config, registry=registry_with_models)

        def _raise(*args, **kwargs):
            raise RuntimeError("selector failure")

        monkeypatch.setattr(engine._model_selector, "select_with_details", _raise)

        result = engine.route(RoutingRequest(query="anything"))
        assert result.selected_model.model_id == "fast-model"
        assert result.scoring_details.get("selection_mode") == "default-fallback-exception"

    def test_routing_failure_without_fallback_raises(self, registry_with_models, monkeypatch):
        config = RouterConfig(
            default_model_id="fast-model",
            fallback_enabled=False,
        )
        engine = RoutingEngine(config=config, registry=registry_with_models)

        def _raise(*args, **kwargs):
            raise RuntimeError("selector failure")

        monkeypatch.setattr(engine._model_selector, "select_with_details", _raise)

        with pytest.raises(RoutingError):
            engine.route(RoutingRequest(query="anything"))

    def test_default_fallback_events_are_logged(self, registry_with_models, caplog):
        config = RouterConfig(
            confidence_threshold=0.99,
            default_model_id="fast-model",
            fallback_enabled=True,
        )
        engine = RoutingEngine(config=config, registry=registry_with_models)

        with caplog.at_level("INFO"):
            result = engine.route(RoutingRequest(query="xyz abc random tokens"))

        assert result.selected_model.model_id == "fast-model"
        assert "Using default fallback model" in caplog.text
