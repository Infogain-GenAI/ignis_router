"""Tests for the Router class (public API)."""

import pytest

from ignis_router import (
    Intent,
    ModelCapability,
    ModelConfig,
    ModelNotAvailableError,
    Router,
    RouterConfig,
    RoutingError,
    RoutingRule,
    TaskComplexity,
)


@pytest.fixture
def sample_models():
    """Create sample model configurations."""
    return [
        ModelConfig(
            model_id="gpt-4o",
            provider="openai",
            model_name="gpt-4o",
            capabilities=[
                ModelCapability.HIGH_QUALITY,
                ModelCapability.REASONING,
                ModelCapability.CODE,
            ],
            cost_per_1k_input_tokens=0.005,
            cost_per_1k_output_tokens=0.015,
            priority=10,
        ),
        ModelConfig(
            model_id="gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            capabilities=[
                ModelCapability.FAST_RESPONSE,
                ModelCapability.COST_EFFECTIVE,
                ModelCapability.CODE,
            ],
            cost_per_1k_input_tokens=0.00015,
            cost_per_1k_output_tokens=0.0006,
            priority=5,
        ),
        ModelConfig(
            model_id="claude-sonnet",
            provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            capabilities=[
                ModelCapability.HIGH_QUALITY,
                ModelCapability.REASONING,
                ModelCapability.CREATIVE,
                ModelCapability.CODE,
            ],
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            priority=8,
        ),
    ]


@pytest.fixture
def router(sample_models):
    """Create a router with sample models registered."""
    r = Router(config=RouterConfig(enable_ml_model_hint_routing=False))
    for model in sample_models:
        r.register_model(model)
    return r


class TestRouterBasic:
    """Basic router functionality tests."""

    def test_create_router_default_config(self):
        router = Router()
        assert router.config is not None
        assert router.config.enable_ml_intent_detection is True
        assert router.config.enable_rule_based_intent_detection is True

    def test_create_router_custom_config(self):
        config = RouterConfig(
            enable_ml_intent_detection=False,
            enable_rule_based_intent_detection=True,
            ml_confidence_threshold=0.8,
        )
        router = Router(config=config)
        assert router.config.enable_ml_intent_detection is False
        assert router.config.enable_rule_based_intent_detection is True
        assert router.config.ml_confidence_threshold == 0.8

    def test_register_model(self):
        router = Router()
        model = ModelConfig(
            model_id="test-model",
            provider="test",
            model_name="test-model-v1",
        )
        result = router.register_model(model)
        assert result is router  # Method chaining
        assert len(router.get_registered_models()) == 1

    def test_unregister_model(self):
        router = Router()
        model = ModelConfig(
            model_id="test-model",
            provider="test",
            model_name="test-model-v1",
        )
        router.register_model(model)
        router.unregister_model("test-model")
        assert len(router.get_registered_models()) == 0

    def test_route_no_models_raises(self):
        router = Router()
        with pytest.raises(ModelNotAvailableError):
            router.route("Hello world")


class TestRouterRouting:
    """Tests for routing decisions."""

    def test_route_simple_query(self, router):
        result = router.route("Hello, how are you?")
        assert result.selected_model is not None
        assert result.detected_intent is not None
        assert result.complexity is not None
        assert 0.0 <= result.confidence <= 1.0

    def test_route_code_query(self, router):
        result = router.route("Write a Python function to sort a list")
        assert result.detected_intent == Intent.CODE_GENERATION
        assert ModelCapability.CODE in result.selected_model.capabilities

    def test_route_reasoning_query(self, router):
        result = router.route("Explain step by step how neural networks learn")
        assert result.detected_intent == Intent.REASONING

    def test_route_with_preferred_provider(self, router):
        result = router.route(
            "Write a function",
            preferred_provider="anthropic",
        )
        assert result.selected_model is not None

    def test_route_with_max_cost(self, router):
        result = router.route(
            "Hello world",
            max_cost=0.001,
        )
        assert result.selected_model.cost_per_1k_input_tokens <= 0.001

    def test_route_with_required_capabilities(self, router):
        result = router.route(
            "Translate this to French",
            required_capabilities=[ModelCapability.CREATIVE],
        )
        assert ModelCapability.CREATIVE in result.selected_model.capabilities

    def test_route_returns_fallbacks(self, router):
        result = router.route("Tell me a joke")
        assert isinstance(result.fallback_models, list)

    def test_route_reasoning_field(self, router):
        result = router.route("Summarize this document")
        assert result.reasoning != ""
        assert "intent" in result.reasoning.lower() or "detected" in result.reasoning.lower()


class TestRouterRules:
    """Tests for rule-based routing."""

    def test_add_rule(self, router):
        rule = RoutingRule(
            rule_id="code-to-gpt4",
            intent=Intent.CODE_GENERATION,
            target_model_id="gpt-4o",
            priority=100,
        )
        result = router.add_rule(rule)
        assert result is router

    def test_rule_based_routing(self):
        r = Router(
            config=RouterConfig(
                enable_ml_intent_detection=False,
                enable_rule_based_intent_detection=True,
                enable_ml_model_hint_routing=False,
            )
        )
        for model in [
            ModelConfig(model_id="gpt-4o", provider="openai", model_name="gpt-4o",
                        capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.REASONING, ModelCapability.CODE],
                        cost_per_1k_input_tokens=0.005, priority=10),
            ModelConfig(model_id="claude-sonnet", provider="anthropic", model_name="claude-3-5-sonnet",
                        capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.REASONING, ModelCapability.CREATIVE, ModelCapability.CODE],
                        cost_per_1k_input_tokens=0.003, priority=8),
        ]:
            r.register_model(model)
        rule = RoutingRule(
            rule_id="code-to-claude",
            intent=Intent.CODE_GENERATION,
            target_model_id="claude-sonnet",
            priority=100,
        )
        r.add_rule(rule)
        result = r.route("Write a Python class for a linked list")
        assert result.selected_model.model_id == "claude-sonnet"

    def test_remove_rule(self, router):
        rule = RoutingRule(
            rule_id="test-rule",
            intent=Intent.GENERAL_CHAT,
            target_model_id="gpt-4o-mini",
            priority=50,
        )
        router.add_rule(rule)
        router.remove_rule("test-rule")
        # Should route normally without the rule
        result = router.route("Hello")
        assert result.selected_model is not None

    def test_register_default_intent_rules(self, router):
        default_router = Router(
            config=RouterConfig(
                enable_ml_intent_detection=False,
                enable_rule_based_intent_detection=True,
                enable_ml_model_hint_routing=False,
            )
        )
        default_router.register_supported_models()
        default_router.register_default_intent_rules()

        result = default_router.route("Write a Python class for a linked list")
        assert result.selected_model.model_id == "claude-3-5-sonnet"
