"""Factory for selecting intent detector strategies from configuration."""

import logging

from .config import RouterConfig
from .exceptions import ConfigurationError
from .intent_detector import (
    BaseIntentDetector,
    HybridIntentDetector,
    MLIntentDetector,
    RuleBasedIntentDetector,
)

logger = logging.getLogger(__name__)


class IntentDetectorFactory:
    """Create intent detector implementations based on configured mode."""

    @staticmethod
    def create(config: RouterConfig) -> BaseIntentDetector:
        """Build the proper detector strategy for the current configuration."""
        if not config.enable_ml_intent_detection and not config.enable_rule_based_intent_detection:
            raise ConfigurationError(
                "Invalid configuration: both ML and Rule-Based intent detection are disabled. "
                "Enable at least one detector strategy."
            )

        if config.enable_ml_intent_detection and config.enable_rule_based_intent_detection:
            logger.info("Using Hybrid Intent Detector")
            return HybridIntentDetector(
                ml_detector=MLIntentDetector(
                    model_path=config.ml_model_path,
                    enable_model_hint_routing=config.enable_ml_model_hint_routing,
                ),
                rule_based_detector=RuleBasedIntentDetector(),
                confidence_threshold=config.ml_confidence_threshold,
            )

        if config.enable_ml_intent_detection:
            logger.info("Using ML Intent Detector")
            return MLIntentDetector(
                model_path=config.ml_model_path,
                enable_model_hint_routing=config.enable_ml_model_hint_routing,
            )

        logger.info("Using Rule-Based Intent Detector")
        return RuleBasedIntentDetector()
