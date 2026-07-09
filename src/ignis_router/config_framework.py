"""YAML configuration loader and schema definitions for routing behavior."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator

from .exceptions import ConfigurationError

_ALLOWED_STRATEGIES = {
    "balanced",
    "cost-first",
    "quality-first",
    "latency-first",
}

_STRATEGY_DEFAULTS: dict[str, dict[str, float]] = {
    "balanced": {"quality": 40.0, "latency": 20.0, "cost": 20.0, "reliability": 20.0},
    "cost-first": {"quality": 20.0, "latency": 20.0, "cost": 45.0, "reliability": 15.0},
    "quality-first": {"quality": 50.0, "latency": 15.0, "cost": 10.0, "reliability": 25.0},
    "latency-first": {"quality": 20.0, "latency": 50.0, "cost": 15.0, "reliability": 15.0},
}


class RoutingYamlConfig(BaseModel):
    """Schema for YAML-based routing strategy configuration."""

    strategy: str = Field(default="balanced")
    weights: dict[str, float] | None = None

    @model_validator(mode="after")
    def validate_configuration(self) -> "RoutingYamlConfig":
        if self.strategy not in _ALLOWED_STRATEGIES:
            raise ConfigurationError(
                f"Invalid strategy '{self.strategy}'. Allowed values: {sorted(_ALLOWED_STRATEGIES)}"
            )

        if self.weights is None:
            self.weights = dict(_STRATEGY_DEFAULTS[self.strategy])

        required_keys = {"quality", "latency", "cost", "reliability"}
        keys = set(self.weights.keys())
        if keys != required_keys:
            raise ConfigurationError(
                "Invalid YAML weights. Required keys: quality, latency, cost, reliability."
            )

        total = 0.0
        for key, value in self.weights.items():
            if value < 0 or value > 100:
                raise ConfigurationError(
                    f"Invalid YAML weight for '{key}': {value}. Expected value between 0 and 100."
                )
            total += value

        if abs(total - 100.0) > 0.0001:
            raise ConfigurationError(
                f"Invalid YAML weight total: {total}. Expected sum to be 100."
            )

        return self


def load_routing_yaml(path: Path) -> RoutingYamlConfig:
    """Load and validate routing configuration from a YAML file."""
    if not path.exists():
        raise ConfigurationError(f"YAML configuration file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as file_obj:
            payload: Any = yaml.safe_load(file_obj) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Failed to parse YAML configuration {path}: {exc}") from exc
    except OSError as exc:
        raise ConfigurationError(f"Failed to read YAML configuration {path}: {exc}") from exc

    if not isinstance(payload, dict):
        raise ConfigurationError(
            f"Invalid YAML root type in {path}. Expected a mapping/object at top level."
        )

    try:
        return RoutingYamlConfig(**payload)
    except Exception as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"Invalid YAML routing configuration in {path}: {exc}") from exc
