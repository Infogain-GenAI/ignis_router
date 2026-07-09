"""Tests for the ModelSelector class."""

import pytest

from ignis_router import Intent, ModelCapability, ModelConfig, TaskComplexity
from ignis_router.config import RouterRegistry
from ignis_router.model_selector import ModelSelector
from ignis_router.models import RoutingRule


@pytest.fixture
def registry():
    """Create a registry with diverse models."""
    reg = RouterRegistry()
    reg.register_model(
        ModelConfig(
            model_id="fast-cheap",
            provider="openai",
            model_name="gpt-4o-mini",
            capabilities=[ModelCapability.FAST_RESPONSE, ModelCapability.COST_EFFECTIVE],
            cost_per_1k_input_tokens=0.00015,
            priority=3,
        )
    )
    reg.register_model(
        ModelConfig(
            model_id="quality-reasoning",
            provider="openai",
            model_name="gpt-4o",
            capabilities=[
                ModelCapability.HIGH_QUALITY,
                ModelCapability.REASONING,
                ModelCapability.CODE,
            ],
            cost_per_1k_input_tokens=0.005,
            priority=10,
        )
    )
    reg.register_model(
        ModelConfig(
            model_id="creative-model",
            provider="anthropic",
            model_name="claude-3-5-sonnet",
            capabilities=[
                ModelCapability.CREATIVE,
                ModelCapability.HIGH_QUALITY,
                ModelCapability.MULTILINGUAL,
            ],
            cost_per_1k_input_tokens=0.003,
            priority=8,
        )
    )
    reg.register_model(
        ModelConfig(
            model_id="disabled-model",
            provider="test",
            model_name="disabled",
            capabilities=[ModelCapability.FAST_RESPONSE],
            enabled=False,
            priority=100,
        )
    )
    return reg


@pytest.fixture
def selector(registry):
    return ModelSelector(registry)


class TestModelSelection:
    def test_select_for_code(self, selector):
        model, fallbacks = selector.select(
            intent=Intent.CODE_GENERATION,
            complexity=TaskComplexity.HIGH,
        )
        assert model is not None
        assert ModelCapability.CODE in model.capabilities

    def test_select_for_creative(self, selector):
        model, fallbacks = selector.select(
            intent=Intent.CREATIVE_WRITING,
            complexity=TaskComplexity.MEDIUM,
        )
        assert model is not None
        assert ModelCapability.CREATIVE in model.capabilities

    def test_select_with_cost_constraint(self, selector):
        model, fallbacks = selector.select(
            intent=Intent.GENERAL_CHAT,
            complexity=TaskComplexity.LOW,
            max_cost=0.001,
        )
        assert model is not None
        assert model.cost_per_1k_input_tokens <= 0.001

    def test_select_with_required_capability(self, selector):
        model, fallbacks = selector.select(
            intent=Intent.GENERAL_CHAT,
            complexity=TaskComplexity.MEDIUM,
            required_capabilities=[ModelCapability.MULTILINGUAL],
        )
        assert model is not None
        assert ModelCapability.MULTILINGUAL in model.capabilities

    def test_select_with_preferred_provider(self, selector):
        model, fallbacks = selector.select(
            intent=Intent.GENERAL_CHAT,
            complexity=TaskComplexity.MEDIUM,
            preferred_provider="anthropic",
        )
        assert model is not None
        # Provider preference is a soft constraint, so just check we got a result

    def test_disabled_models_excluded(self, selector):
        model, fallbacks = selector.select(
            intent=Intent.GENERAL_CHAT,
            complexity=TaskComplexity.LOW,
        )
        assert model is not None
        assert model.model_id != "disabled-model"
        for fb in fallbacks:
            assert fb.model_id != "disabled-model"

    def test_fallbacks_returned(self, selector):
        model, fallbacks = selector.select(
            intent=Intent.GENERAL_CHAT,
            complexity=TaskComplexity.MEDIUM,
        )
        assert len(fallbacks) > 0

    def test_empty_registry_returns_none(self):
        empty_reg = RouterRegistry()
        selector = ModelSelector(empty_reg)
        model, fallbacks = selector.select(
            intent=Intent.GENERAL_CHAT,
            complexity=TaskComplexity.LOW,
        )
        assert model is None
        assert fallbacks == []

    def test_select_uses_intent_preference_when_available(self, selector):
        model, _, details = selector.select_with_details(
            intent=Intent.CREATIVE_WRITING,
            complexity=TaskComplexity.MEDIUM,
        )

        assert model is not None
        assert model.model_id == "creative-model"
        assert details["selection_mode"] == "intent-preference"

    def test_select_falls_back_to_scored_when_preference_unavailable(self, selector):
        model, _, details = selector.select_with_details(
            intent=Intent.REASONING,
            complexity=TaskComplexity.HIGH,
        )

        assert model is not None
        assert details["selection_mode"] == "scored"


class TestRuleBasedSelection:
    def test_rule_match(self, registry, selector):
        registry.add_rule(
            RoutingRule(
                rule_id="creative-rule",
                intent=Intent.CREATIVE_WRITING,
                target_model_id="creative-model",
                priority=100,
            )
        )
        model, _ = selector.select(
            intent=Intent.CREATIVE_WRITING,
            complexity=TaskComplexity.MEDIUM,
        )
        assert model.model_id == "creative-model"

    def test_rule_priority(self, registry, selector):
        registry.add_rule(
            RoutingRule(
                rule_id="low-priority",
                intent=Intent.CODE_GENERATION,
                target_model_id="fast-cheap",
                priority=1,
            )
        )
        registry.add_rule(
            RoutingRule(
                rule_id="high-priority",
                intent=Intent.CODE_GENERATION,
                target_model_id="quality-reasoning",
                priority=100,
            )
        )
        model, _ = selector.select(
            intent=Intent.CODE_GENERATION,
            complexity=TaskComplexity.HIGH,
        )
        assert model.model_id == "quality-reasoning"
