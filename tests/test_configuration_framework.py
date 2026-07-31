"""Tests for YAML configuration framework and routing weight behavior."""

import pytest

from ignis_router import ModelCapability, ModelConfig, RouterConfig
from ignis_router.config import RouterRegistry
from ignis_router.exceptions import ConfigurationError
from ignis_router.core.model_selector import ModelSelector
from ignis_router.models import Intent, TaskComplexity


class TestYamlConfigurationLoader:
    def test_router_config_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "routing.yaml"
        yaml_file.write_text(
            "strategy: balanced\n"
            "weights:\n"
            "  quality: 40\n"
            "  latency: 20\n"
            "  cost: 20\n"
            "  reliability: 20\n",
            encoding="utf-8",
        )

        config = RouterConfig.from_yaml(str(yaml_file))

        assert config.routing_strategy == "balanced"
        assert config.routing_weights == {
            "quality": 40.0,
            "latency": 20.0,
            "cost": 20.0,
            "reliability": 20.0,
        }

    def test_invalid_yaml_strategy_raises_meaningful_error(self, tmp_path):
        yaml_file = tmp_path / "invalid_strategy.yaml"
        yaml_file.write_text("strategy: unknown\n", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="Invalid strategy"):
            RouterConfig.from_yaml(str(yaml_file))

    def test_invalid_yaml_weight_total_raises_meaningful_error(self, tmp_path):
        yaml_file = tmp_path / "invalid_weights.yaml"
        yaml_file.write_text(
            "strategy: balanced\n"
            "weights:\n"
            "  quality: 30\n"
            "  latency: 20\n"
            "  cost: 20\n"
            "  reliability: 20\n",
            encoding="utf-8",
        )

        with pytest.raises(ConfigurationError, match="Expected sum to be 100"):
            RouterConfig.from_yaml(str(yaml_file))

    def test_invalid_yaml_syntax_raises_meaningful_error(self, tmp_path):
        yaml_file = tmp_path / "bad_syntax.yaml"
        yaml_file.write_text("strategy: balanced\nweights: [", encoding="utf-8")

        with pytest.raises(ConfigurationError, match="Failed to parse YAML"):
            RouterConfig.from_yaml(str(yaml_file))

    @pytest.mark.parametrize(
        ("strategy", "expected"),
        [
            ("balanced", {"quality": 40.0, "latency": 20.0, "cost": 20.0, "reliability": 20.0}),
            ("cost-first", {"quality": 20.0, "latency": 20.0, "cost": 45.0, "reliability": 15.0}),
            ("quality-first", {"quality": 50.0, "latency": 15.0, "cost": 10.0, "reliability": 25.0}),
            ("latency-first", {"quality": 20.0, "latency": 50.0, "cost": 15.0, "reliability": 15.0}),
        ],
    )
    def test_strategy_supported_with_default_weights(self, strategy, expected):
        config = RouterConfig(routing_strategy=strategy)
        assert config.routing_strategy == strategy
        assert config.routing_weights == expected


class TestRoutingWeightsBehavior:
    def test_quality_first_weights_favor_quality_model(self):
        registry = RouterRegistry()
        registry.register_model(
            ModelConfig(
                model_id="cheap-fast",
                provider="test",
                model_name="cheap-fast",
                capabilities=[ModelCapability.FAST_RESPONSE],
                cost_per_1k_input_tokens=0.0001,
                latency=0.95,
                quality=0.55,
                reliability=0.7,
                priority=2,
            )
        )
        registry.register_model(
            ModelConfig(
                model_id="quality-premium",
                provider="test",
                model_name="quality-premium",
                capabilities=[ModelCapability.HIGH_QUALITY],
                cost_per_1k_input_tokens=0.01,
                latency=0.65,
                quality=0.98,
                reliability=0.95,
                priority=2,
            )
        )

        config = RouterConfig(
            routing_strategy="quality-first",
            routing_weights={"quality": 70.0, "latency": 10.0, "cost": 5.0, "reliability": 15.0},
        )

        selector = ModelSelector(registry, config)
        selected, _ = selector.select(Intent.GENERAL_CHAT, TaskComplexity.MEDIUM)

        assert selected is not None
        assert selected.model_id == "quality-premium"

    def test_cost_first_weights_favor_low_cost_model(self):
        registry = RouterRegistry()
        registry.register_model(
            ModelConfig(
                model_id="cheap",
                provider="test",
                model_name="cheap",
                capabilities=[ModelCapability.FAST_RESPONSE],
                cost_per_1k_input_tokens=0.0001,
                latency=0.6,
                quality=0.7,
                reliability=0.75,
                priority=1,
            )
        )
        registry.register_model(
            ModelConfig(
                model_id="expensive",
                provider="test",
                model_name="expensive",
                capabilities=[ModelCapability.FAST_RESPONSE],
                cost_per_1k_input_tokens=0.05,
                latency=0.6,
                quality=0.7,
                reliability=0.75,
                priority=1,
            )
        )

        selector = ModelSelector(registry, RouterConfig(routing_strategy="cost-first"))
        selected, _ = selector.select(Intent.GENERAL_CHAT, TaskComplexity.MEDIUM)

        assert selected is not None
        assert selected.model_id == "cheap"

    def test_latency_first_weights_favor_fast_model(self):
        registry = RouterRegistry()
        registry.register_model(
            ModelConfig(
                model_id="fast",
                provider="test",
                model_name="fast",
                capabilities=[ModelCapability.FAST_RESPONSE],
                cost_per_1k_input_tokens=0.01,
                latency=0.95,
                quality=0.65,
                reliability=0.75,
                priority=1,
            )
        )
        registry.register_model(
            ModelConfig(
                model_id="slow",
                provider="test",
                model_name="slow",
                capabilities=[ModelCapability.FAST_RESPONSE],
                cost_per_1k_input_tokens=0.01,
                latency=0.35,
                quality=0.65,
                reliability=0.75,
                priority=1,
            )
        )

        selector = ModelSelector(registry, RouterConfig(routing_strategy="latency-first"))
        selected, _ = selector.select(Intent.GENERAL_CHAT, TaskComplexity.MEDIUM)

        assert selected is not None
        assert selected.model_id == "fast"
