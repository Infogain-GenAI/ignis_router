"""Default supported model catalog and starter routing rules."""

from ..models import ModelCapability, ModelConfig
from ..models import Intent, RoutingRule


def get_default_supported_models() -> list[ModelConfig]:
    """Return a baseline list of supported models with metadata for routing."""
    return [
        ModelConfig(
            model_id="gpt-4.1",
            provider="openai",
            model_name="gpt-4.1",
            capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.REASONING, ModelCapability.CODE],
            cost_per_1k_input_tokens=0.01,
            cost_per_1k_output_tokens=0.03,
            latency=0.65,
            quality=0.95,
            reliability=0.97,
            priority=9,
        ),
        ModelConfig(
            model_id="gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            capabilities=[ModelCapability.FAST_RESPONSE, ModelCapability.COST_EFFECTIVE, ModelCapability.CODE],
            cost_per_1k_input_tokens=0.00015,
            cost_per_1k_output_tokens=0.0006,
            latency=0.93,
            quality=0.78,
            reliability=0.94,
            priority=6,
        ),
        ModelConfig(
            model_id="claude-3-5-sonnet",
            provider="anthropic",
            model_name="claude-3-5-sonnet",
            capabilities=[ModelCapability.HIGH_QUALITY, ModelCapability.REASONING, ModelCapability.CREATIVE],
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            latency=0.72,
            quality=0.92,
            reliability=0.96,
            priority=8,
        ),
    ]


def get_default_intent_rules() -> list[RoutingRule]:
    """Return strict intent-to-model starter rules for common tasks."""
    return [
        RoutingRule(
            rule_id="intent-code-generation",
            intent=Intent.CODE_GENERATION,
            target_model_id="claude-3-5-sonnet",
            priority=90,
        ),
        RoutingRule(
            rule_id="intent-summarization",
            intent=Intent.SUMMARIZATION,
            target_model_id="gpt-4.1",
            priority=90,
        ),
        RoutingRule(
            rule_id="intent-extraction",
            intent=Intent.EXTRACTION,
            target_model_id="gpt-4o-mini",
            priority=90,
        ),
        RoutingRule(
            rule_id="intent-reasoning",
            intent=Intent.REASONING,
            target_model_id="gpt-4.1",
            priority=90,
        ),
        RoutingRule(
            rule_id="intent-data-analysis",
            intent=Intent.DATA_ANALYSIS,
            target_model_id="gpt-4.1",
            priority=90,
        ),
        RoutingRule(
            rule_id="intent-creative-writing",
            intent=Intent.CREATIVE_WRITING,
            target_model_id="claude-3-5-sonnet",
            priority=90,
        ),
        RoutingRule(
            rule_id="intent-translation",
            intent=Intent.TRANSLATION,
            target_model_id="gpt-4o-mini",
            priority=90,
        ),
        RoutingRule(
            rule_id="intent-classification",
            intent=Intent.CLASSIFICATION,
            target_model_id="gpt-4o-mini",
            priority=90,
        ),
    ]
