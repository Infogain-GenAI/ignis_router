"""Configuration management for ignis_router."""

from pathlib import Path
from typing import Optional

from pydantic import AliasChoices, Field, model_validator
from pydantic_settings import BaseSettings

from .exceptions import ConfigurationError
from .models import ModelConfig, RoutingRule


def get_bundled_configs_dir() -> Path:
    """Return the path to configs bundled inside the installed package."""
    return Path(__file__).parent / "configs"


def resolve_config_path(path: str) -> Path:
    """
    Resolve a config file path.
    Checks (in order):
    1. Absolute path as given
    2. Relative to current working directory
    3. Bundled configs inside the installed package
    """
    p = Path(path)
    if p.is_absolute() and p.exists():
        return p
    if p.exists():
        return p.resolve()
    # Fall back to bundled configs
    bundled = get_bundled_configs_dir() / p.name
    if bundled.exists():
        return bundled
    # Also try matching subdirectory names
    for candidate in get_bundled_configs_dir().rglob(p.name):
        return candidate
    raise ConfigurationError(
        f"Config file not found: '{path}'. "
        f"Checked CWD and bundled package configs at '{get_bundled_configs_dir()}'."
    )


class RouterConfig(BaseSettings):
    """Configuration settings for the router."""

    _ALLOWED_ROUTING_STRATEGIES = {
        "balanced",
        "cost-first",
        "quality-first",
        "latency-first",
    }
    _DEFAULT_ROUTING_WEIGHTS = {
        "balanced": {"quality": 40.0, "latency": 20.0, "cost": 20.0, "reliability": 20.0},
        "cost-first": {"quality": 20.0, "latency": 20.0, "cost": 45.0, "reliability": 15.0},
        "quality-first": {"quality": 50.0, "latency": 15.0, "cost": 10.0, "reliability": 25.0},
        "latency-first": {"quality": 20.0, "latency": 50.0, "cost": 15.0, "reliability": 15.0},
    }

    default_model_id: Optional[str] = Field(
        default=None, description="Default model to use when no rule matches"
    )
    enable_ml_intent_detection: bool = Field(
        default=True,
        description="Enable ML-based intent detection",
        validation_alias=AliasChoices(
            "ENABLE_ML_INTENT_DETECTION", "IGNIS_ROUTER_ENABLE_ML_INTENT_DETECTION"
        ),
    )
    enable_rule_based_intent_detection: bool = Field(
        default=True,
        description="Enable rule-based intent detection",
        validation_alias=AliasChoices(
            "ENABLE_RULE_BASED_INTENT_DETECTION",
            "IGNIS_ROUTER_ENABLE_RULE_BASED_INTENT_DETECTION",
        ),
    )
    ml_confidence_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum ML confidence before hybrid fallback to rule-based",
        validation_alias=AliasChoices(
            "ML_CONFIDENCE_THRESHOLD", "IGNIS_ROUTER_ML_CONFIDENCE_THRESHOLD"
        ),
    )
    ml_model_path: str = Field(
        default="models/knnrouter.pkl",
        description="Path to the serialized ML intent model",
        validation_alias=AliasChoices("ML_MODEL_PATH", "IGNIS_ROUTER_ML_MODEL_PATH"),
    )
    enable_intent_detection: bool = Field(
        default=True,
        description="Deprecated legacy flag retained for backward compatibility",
    )
    enable_complexity_assessment: bool = Field(
        default=True, description="Enable automatic complexity assessment"
    )
    fallback_enabled: bool = Field(
        default=True, description="Enable fallback model selection"
    )
    max_fallback_attempts: int = Field(
        default=2, description="Maximum number of fallback attempts"
    )
    confidence_threshold: float = Field(
        default=0.5, description="Minimum confidence threshold for routing"
    )
    routing_strategy: str = Field(
        default="balanced",
        description="Routing strategy used by weighted model scoring",
        validation_alias=AliasChoices("ROUTING_STRATEGY", "IGNIS_ROUTER_ROUTING_STRATEGY"),
    )
    routing_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "quality": 40.0,
            "latency": 20.0,
            "cost": 20.0,
            "reliability": 20.0,
        },
        description="Routing score weights. Values must sum to 100.",
    )
    enable_ml_model_hint_routing: bool = Field(
        default=False,
        description="Allow ML model-label predictions to directly select registered models",
        validation_alias=AliasChoices(
            "ENABLE_ML_MODEL_HINT_ROUTING",
            "IGNIS_ROUTER_ENABLE_ML_MODEL_HINT_ROUTING",
        ),
    )
    ml_router_type: str = Field(
        default="knn",
        description="ML router type from llmrouter-lib: knn, svm, graph, or mf",
        validation_alias=AliasChoices(
            "ML_ROUTER_TYPE", "IGNIS_ROUTER_ML_ROUTER_TYPE"
        ),
    )
    ml_router_config_dir: str = Field(
        default="configs/ml_routers",
        description="Directory containing ML router YAML configs",
        validation_alias=AliasChoices(
            "ML_ROUTER_CONFIG_DIR", "IGNIS_ROUTER_ML_ROUTER_CONFIG_DIR"
        ),
    )
    model_hint_aliases: dict[str, str] = Field(
        default_factory=lambda: {
            "codegemma-7b": "gpt-4.1",
            "gemma-2-9b-it": "gpt-4o-mini",
            "llama-3.1-8b-instruct": "gpt-4o-mini",
            "llama-3.1-nemotron-51b-instruct": "claude-3-5-sonnet",
            "llama-3.3-nemotron-super-49b-v1": "claude-3-5-sonnet",
        },
        description="Alias map from ML-predicted model labels to registered model IDs.",
    )

    model_config = {
        "env_prefix": "IGNIS_ROUTER_",
        "extra": "ignore",
        "populate_by_name": True,
    }

    @model_validator(mode="after")
    def validate_intent_detector_configuration(self) -> "RouterConfig":
        """Ensure at least one intent detector strategy is enabled."""
        if (
            not self.enable_ml_intent_detection
            and not self.enable_rule_based_intent_detection
        ):
            raise ConfigurationError(
                "Invalid configuration: both ML and Rule-Based intent detection are disabled. "
                "Enable at least one detector strategy."
            )

        if self.routing_strategy not in self._ALLOWED_ROUTING_STRATEGIES:
            raise ConfigurationError(
                f"Invalid routing strategy '{self.routing_strategy}'. Allowed values: "
                f"{sorted(self._ALLOWED_ROUTING_STRATEGIES)}"
            )

        # If strategy is explicitly set and routing_weights are not provided,
        # apply the strategy defaults automatically.
        if "routing_weights" not in self.model_fields_set:
            self.routing_weights = dict(self._DEFAULT_ROUTING_WEIGHTS[self.routing_strategy])

        required_keys = {"quality", "latency", "cost", "reliability"}
        provided_keys = set(self.routing_weights.keys())
        if provided_keys != required_keys:
            raise ConfigurationError(
                "Invalid routing_weights keys. Required keys: "
                "quality, latency, cost, reliability."
            )

        for key, value in self.routing_weights.items():
            if value < 0 or value > 100:
                raise ConfigurationError(
                    f"Invalid routing weight for '{key}': {value}. Expected value between 0 and 100."
                )

        total_weight = sum(self.routing_weights.values())
        if abs(total_weight - 100.0) > 0.0001:
            raise ConfigurationError(
                f"Invalid routing_weights total: {total_weight}. Expected sum to be 100."
            )

        return self

    @classmethod
    def from_yaml(cls, yaml_path: str, **overrides) -> "RouterConfig":
        """Load router configuration with routing strategy and weights from YAML."""
        from .config_framework import load_routing_yaml

        resolved = resolve_config_path(yaml_path)
        parsed = load_routing_yaml(resolved)
        payload = {
            "routing_strategy": parsed.strategy,
            "routing_weights": parsed.weights,
        }
        payload.update(overrides)
        return cls(**payload)


class RouterRegistry:
    """Registry for models and routing rules.""" #RouterRegistry is the router’s in-memory storage and lookup layer.

    def __init__(self):
        self._models: dict[str, ModelConfig] = {}
        self._rules: list[RoutingRule] = []

    def register_model(self, model: ModelConfig) -> None:
        """Register a model configuration."""
        self._models[model.model_id] = model

    def register_models(self, models: list[ModelConfig]) -> None:
        """Register multiple model configurations in one call."""
        for model in models:
            self.register_model(model)

    def unregister_model(self, model_id: str) -> None:
        """Remove a model from the registry."""
        self._models.pop(model_id, None)

    def get_model(self, model_id: str) -> Optional[ModelConfig]:
        """Get a model by ID."""
        return self._models.get(model_id)

    def get_all_models(self) -> list[ModelConfig]:
        """Get all registered models."""
        return list(self._models.values())

    def get_enabled_models(self) -> list[ModelConfig]:
        """Get all enabled models."""
        return [m for m in self._models.values() if m.enabled]

    def add_rule(self, rule: RoutingRule) -> None:
        """Add a routing rule."""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, rule_id: str) -> None:
        """Remove a routing rule by ID."""
        self._rules = [r for r in self._rules if r.rule_id != rule_id]

    def get_rules(self) -> list[RoutingRule]:
        """Get all routing rules sorted by priority."""
        return [r for r in self._rules if r.enabled]

    def clear(self) -> None:
        """Clear all models and rules."""
        self._models.clear()
        self._rules.clear()
