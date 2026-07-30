"""Feature flags for ignis_router — runtime toggles without code changes.

Provides a centralized registry of all toggleable features with their
current state, descriptions, and the environment variable that controls each.

Usage:
    from ignis_router.feature_flags import FeatureFlags

    flags = FeatureFlags.from_config(router.config)
    print(flags.to_dict())        # All flags with status
    flags.toggle("db_logging")    # Flip a flag at runtime
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .config import RouterConfig


@dataclass
class FeatureFlag:
    """A single feature toggle."""

    key: str
    name: str
    description: str
    enabled: bool
    env_var: str
    category: str


class FeatureFlags:
    """Registry of all toggleable features with runtime mutation support."""

    # Master definition of every flag: (config_attr, name, description, env_var, category)
    _DEFINITIONS: list[tuple[str, str, str, str, str]] = [
        (
            "enable_ml_model_hint_routing",
            "ML Based Routing",
            "Use ML router (KNN/SVM/Graph/MF) to predict and select the best LLM model",
            "ENABLE_ML_MODEL_HINT_ROUTING",
            "routing",
        ),
        (
            "enable_rule_based_intent_detection",
            "Rule Based Routing",
            "Use regex keyword patterns and intent rules to select the best LLM model",
            "ENABLE_RULE_BASED_INTENT_DETECTION",
            "routing",
        ),
        (
            "enable_ml_intent_detection",
            "Hybrid Routing",
            "Combine ML and rule-based detection — ML tries first, falls back to rules if confidence is low",
            "ENABLE_ML_INTENT_DETECTION",
            "routing",
        ),
    ]

    def __init__(self, config: RouterConfig) -> None:
        self._config = config
        self._overrides: dict[str, bool] = {}

    @classmethod
    def from_config(cls, config: RouterConfig) -> "FeatureFlags":
        """Create feature flags from a RouterConfig instance."""
        return cls(config)

    def _get_value(self, config_attr: str) -> bool:
        """Get the effective value of a flag (override > config)."""
        if config_attr in self._overrides:
            return self._overrides[config_attr]
        return getattr(self._config, config_attr, True)

    def get_all(self) -> list[FeatureFlag]:
        """Return all feature flags with their current state."""
        return [
            FeatureFlag(
                key=config_attr,
                name=name,
                description=desc,
                enabled=self._get_value(config_attr),
                env_var=env_var,
                category=category,
            )
            for config_attr, name, desc, env_var, category in self._DEFINITIONS
        ]

    def get(self, key: str) -> Optional[FeatureFlag]:
        """Get a single feature flag by its key."""
        for defn in self._DEFINITIONS:
            if defn[0] == key:
                config_attr, name, desc, env_var, category = defn
                return FeatureFlag(
                    key=config_attr,
                    name=name,
                    description=desc,
                    enabled=self._get_value(config_attr),
                    env_var=env_var,
                    category=category,
                )
        return None

    def is_enabled(self, key: str) -> bool:
        """Check if a feature is enabled."""
        return self._get_value(key)

    def toggle(self, key: str) -> bool:
        """Toggle a feature flag and return the new value."""
        current = self._get_value(key)
        new_value = not current
        self._set(key, new_value)
        return new_value

    def _set(self, key: str, value: bool) -> None:
        """Set a feature flag to a specific value. Updates both override and config."""
        valid_keys = {defn[0] for defn in self._DEFINITIONS}
        if key not in valid_keys:
            raise KeyError(f"Unknown feature flag: '{key}'. Valid keys: {sorted(valid_keys)}")
        self._overrides[key] = value
        if hasattr(self._config, key):
            setattr(self._config, key, value)

    def set(self, key: str, value: bool) -> None:
        """Set a feature flag to a specific value (public API)."""
        self._set(key, value)

    def to_dict(self) -> dict[str, Any]:
        """Serialize all flags grouped by category with usage instructions."""
        flags = self.get_all()
        by_category: dict[str, list[dict[str, Any]]] = {}
        for f in flags:
            entry = {
                "key": f.key,
                "name": f.name,
                "description": f.description,
                "enabled": f.enabled,
                "env_var": f.env_var,
                "toggle_url": f"PUT /features/{f.key}?enabled=true|false",
            }
            by_category.setdefault(f.category, []).append(entry)

        # Build dropdown-friendly list for frontend
        available_keys = [
            {
                "key": f.key,
                "name": f.name,
                "category": f.category,
                "enabled": f.enabled,
            }
            for f in flags
        ]

        return {
            "features": by_category,
            "summary": {f.key: f.enabled for f in flags},
            "available_keys": available_keys,
            "usage": {
                "view_all": "GET /features",
                "toggle": "PUT /features/{key}?enabled=true|false",
                "example_enable": "PUT /features/enable_ml_model_hint_routing?enabled=true",
                "example_disable": "PUT /features/db_logging_enabled?enabled=false",
            },
        }
