"""Tests for intent detector factory and configuration loading."""

import pytest

from ignis_router.config import RouterConfig
from ignis_router.exceptions import ConfigurationError
from ignis_router.detection.intent_detector import (
    HybridIntentDetector,
    MLIntentDetector,
    RuleBasedIntentDetector,
    SemanticIntentDetector,
)
from ignis_router.detection.intent_detector_factory import IntentDetectorFactory


class TestIntentDetectorFactory:
    def test_factory_selects_ml_only_mode(self):
        config = RouterConfig(
            enable_ml_intent_detection=True,
            enable_rule_based_intent_detection=False,
        )
        detector = IntentDetectorFactory.create(config)
        assert isinstance(detector, (MLIntentDetector, SemanticIntentDetector))

    def test_factory_selects_rule_based_only_mode(self):
        config = RouterConfig(
            enable_ml_intent_detection=False,
            enable_rule_based_intent_detection=True,
        )
        detector = IntentDetectorFactory.create(config)
        assert isinstance(detector, RuleBasedIntentDetector)

    def test_factory_selects_hybrid_mode(self):
        config = RouterConfig(
            enable_ml_intent_detection=True,
            enable_rule_based_intent_detection=True,
            ml_confidence_threshold=0.6,
        )
        detector = IntentDetectorFactory.create(config)
        assert isinstance(detector, HybridIntentDetector)

    def test_invalid_configuration_raises(self):
        with pytest.raises(ConfigurationError):
            RouterConfig(
                enable_ml_intent_detection=False,
                enable_rule_based_intent_detection=False,
            )


class TestConfigurationLoading:
    def test_loads_detector_config_from_env_file(self, tmp_path, monkeypatch):
        # Clear any env vars that could override the .env file values
        for var in (
            "ENABLE_ML_INTENT_DETECTION",
            "ENABLE_RULE_BASED_INTENT_DETECTION",
            "ML_CONFIDENCE_THRESHOLD",
            "IGNIS_ROUTER_ENABLE_ML_INTENT_DETECTION",
            "IGNIS_ROUTER_ENABLE_RULE_BASED_INTENT_DETECTION",
            "IGNIS_ROUTER_ML_CONFIDENCE_THRESHOLD",
        ):
            monkeypatch.delenv(var, raising=False)

        env_file = tmp_path / ".env"
        env_file.write_text(
            "ENABLE_ML_INTENT_DETECTION=true\n"
            "ENABLE_RULE_BASED_INTENT_DETECTION=false\n"
            "ML_CONFIDENCE_THRESHOLD=0.75\n",
            encoding="utf-8",
        )

        config = RouterConfig(_env_file=str(env_file))

        assert config.enable_ml_intent_detection is True
        assert config.enable_rule_based_intent_detection is False
        assert config.ml_confidence_threshold == 0.75
