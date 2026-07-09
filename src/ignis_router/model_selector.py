"""Model selector module for choosing the best model based on routing criteria."""

from typing import Optional

from .config import RouterConfig, RouterRegistry
from .models import (
    Intent,
    ModelCapability,
    ModelConfig,
    RoutingRule,
    TaskComplexity,
)


# Default mapping from intents to preferred capabilities
_INTENT_CAPABILITY_MAP: dict[Intent, list[ModelCapability]] = {
    Intent.GENERAL_CHAT: [ModelCapability.FAST_RESPONSE, ModelCapability.COST_EFFECTIVE],
    Intent.CODE_GENERATION: [ModelCapability.CODE, ModelCapability.HIGH_QUALITY],
    Intent.SUMMARIZATION: [ModelCapability.FAST_RESPONSE, ModelCapability.LONG_CONTEXT],
    Intent.REASONING: [ModelCapability.REASONING, ModelCapability.HIGH_QUALITY],
    Intent.CREATIVE_WRITING: [ModelCapability.CREATIVE, ModelCapability.HIGH_QUALITY],
    Intent.DATA_ANALYSIS: [ModelCapability.REASONING, ModelCapability.CODE],
    Intent.TRANSLATION: [ModelCapability.MULTILINGUAL],
    Intent.CLASSIFICATION: [ModelCapability.FAST_RESPONSE, ModelCapability.COST_EFFECTIVE],
    Intent.EXTRACTION: [ModelCapability.FAST_RESPONSE, ModelCapability.CODE],
    Intent.CUSTOM: [],
}

# Complexity to capability preferences
_COMPLEXITY_CAPABILITY_MAP: dict[TaskComplexity, list[ModelCapability]] = {
    TaskComplexity.LOW: [ModelCapability.FAST_RESPONSE, ModelCapability.COST_EFFECTIVE],
    TaskComplexity.MEDIUM: [ModelCapability.HIGH_QUALITY],
    TaskComplexity.HIGH: [ModelCapability.HIGH_QUALITY, ModelCapability.REASONING],
}

_DEFAULT_WEIGHTS = {"quality": 40.0, "latency": 20.0, "cost": 20.0, "reliability": 20.0}

# Intent-aware preferred model families (frontier first, then budget/fast).
# Matching is fuzzy and only applied when a compatible candidate is available.
_INTENT_MODEL_PREFERENCES: dict[Intent, list[str]] = {
    Intent.CODE_GENERATION: [
        "claude sonnet 4.6",
        "claude opus 4.6",
        "claude-3-5-sonnet",
        "gpt-4o-mini",
        "qwen",
    ],
    Intent.SUMMARIZATION: ["gemini 3.1 pro", "gpt-4o-mini", "gpt-4.1"],
    Intent.EXTRACTION: ["gemini 3.1 pro", "gpt-4o-mini", "gpt-4.1"],
    Intent.REASONING: ["deepseek r1", "gpt-5.4", "gpt-4.1", "claude-3-5-sonnet"],
    Intent.DATA_ANALYSIS: ["deepseek r1", "gpt-5.4", "gpt-4.1", "claude-3-5-sonnet"],
    Intent.CREATIVE_WRITING: ["claude fable 5", "claude sonnet 4.6", "claude-3-5-sonnet"],
    Intent.TRANSLATION: ["mistral large 3", "gpt-4o-mini", "gpt-4.1"],
    Intent.CLASSIFICATION: ["mistral large 3", "gpt-4o-mini", "gpt-4.1"],
}

# Hard capability gates to avoid selecting incompatible models through preference matching.
_INTENT_REQUIRED_CAPS: dict[Intent, set[ModelCapability]] = {
    Intent.CODE_GENERATION: {ModelCapability.CODE},
    Intent.REASONING: {ModelCapability.REASONING},
    Intent.DATA_ANALYSIS: {ModelCapability.REASONING},
    Intent.CREATIVE_WRITING: {ModelCapability.CREATIVE},
}


class ModelSelector:
    """Selects the best model based on intent, complexity, rules, and constraints."""

    def __init__(self, registry: RouterRegistry, config: Optional[RouterConfig] = None):
        self._registry = registry
        self._config = config

    def select(
        self,
        intent: Intent,
        complexity: TaskComplexity,
        required_capabilities: Optional[list[ModelCapability]] = None,
        preferred_provider: Optional[str] = None,
        max_cost: Optional[float] = None,
    ) -> tuple[Optional[ModelConfig], list[ModelConfig]]:
        """
        Select the best model and fallbacks.

        Returns:
            Tuple of (best model or None, list of fallback models)
        """
        selected, fallbacks, _ = self.select_with_details(
            intent=intent,
            complexity=complexity,
            required_capabilities=required_capabilities,
            preferred_provider=preferred_provider,
            max_cost=max_cost,
        )
        return selected, fallbacks

    def select_with_details(
        self,
        intent: Intent,
        complexity: TaskComplexity,
        required_capabilities: Optional[list[ModelCapability]] = None,
        preferred_provider: Optional[str] = None,
        max_cost: Optional[float] = None,
    ) -> tuple[Optional[ModelConfig], list[ModelConfig], dict[str, object]]:
        """
        Select the best model and fallbacks with scoring details.

        Returns:
            Tuple of (best model or None, list of fallback models, scoring details)
        """
        # First try to match against explicit rules
        rule_match = self._match_rules(intent, complexity, required_capabilities)
        if rule_match:
            fallbacks = self._get_fallbacks(rule_match, preferred_provider, max_cost)
            return rule_match, fallbacks, {
                "selection_mode": "rule-match",
                "strategy": self._strategy_name(),
                "weights": self._weights(),
                "scores": {rule_match.model_id: 1.0},
            }

        # Fall back to capability-based selection
        candidates = self._get_candidates(
            intent, complexity, required_capabilities, preferred_provider, max_cost
        )

        if not candidates:
            enabled = self._registry.get_enabled_models()
            if enabled:
                return enabled[0], enabled[1:3], {
                    "selection_mode": "fallback-enabled",
                    "strategy": self._strategy_name(),
                    "weights": self._weights(),
                    "scores": {},
                }
            return None, [], {
                "selection_mode": "no-candidates",
                "strategy": self._strategy_name(),
                "weights": self._weights(),
                "scores": {},
            }

        scored: list[tuple[float, ModelConfig]] = []
        score_breakdown: dict[str, float] = {}
        max_input_cost = max((m.cost_per_1k_input_tokens for m in candidates), default=0.0)
        desired_caps = self._desired_capabilities(intent, complexity, required_capabilities)

        for model in candidates:
            score = self._score_model(
                model=model,
                desired_capabilities=desired_caps,
                preferred_provider=preferred_provider,
                complexity=complexity,
                max_input_cost=max_input_cost,
            )
            scored.append((score, model))
            score_breakdown[model.model_id] = round(score, 4)

        scored.sort(key=lambda item: item[0], reverse=True)
        ranked_models = [model for _, model in scored]

        # If intent-specific preference can be satisfied, promote that model.
        preferred = self._select_intent_preferred_candidate(intent, ranked_models)
        selection_mode = "scored"
        if preferred is not None:
            ranked_models = [preferred] + [m for m in ranked_models if m.model_id != preferred.model_id]
            selection_mode = "intent-preference"

        best = ranked_models[0]
        fallbacks = ranked_models[1:4]

        return best, fallbacks, {
            "selection_mode": selection_mode,
            "strategy": self._strategy_name(),
            "weights": self._weights(),
            "scores": score_breakdown,
        }

    def _select_intent_preferred_candidate(
        self,
        intent: Intent,
        candidates: list[ModelConfig],
    ) -> Optional[ModelConfig]:
        """Pick the first available candidate matching the preferred model family for intent."""
        preferences = _INTENT_MODEL_PREFERENCES.get(intent, [])
        if not preferences or not candidates:
            return None

        required_caps = _INTENT_REQUIRED_CAPS.get(intent, set())
        compatible_candidates = [
            model
            for model in candidates
            if not required_caps or required_caps.issubset(set(model.capabilities))
        ]

        if not compatible_candidates:
            return None

        for preference in preferences:
            for model in compatible_candidates:
                if self._matches_model_preference(model, preference):
                    return model

        return None

    def _matches_model_preference(self, model: ModelConfig, preference: str) -> bool:
        """Fuzzy matching between preferred model family and registered model ID/name."""
        pref = self._normalize_text(preference)
        if not pref:
            return False

        model_id = self._normalize_text(model.model_id)
        model_name = self._normalize_text(model.model_name)

        return pref in model_id or pref in model_name or model_id in pref or model_name in pref

    def _normalize_text(self, value: str) -> str:
        """Normalize model labels for robust matching."""
        normalized = "".join(char if char.isalnum() else " " for char in value.lower())
        return " ".join(normalized.split())

    def _match_rules(
        self,
        intent: Intent,
        complexity: TaskComplexity,
        required_capabilities: Optional[list[ModelCapability]] = None,
    ) -> Optional[ModelConfig]:
        """Try to find a matching routing rule."""
        rules = self._registry.get_rules()

        for rule in rules:
            if not self._rule_matches(rule, intent, complexity, required_capabilities):
                continue
            model = self._registry.get_model(rule.target_model_id)
            if model and model.enabled:
                return model

        return None

    def _rule_matches(
        self,
        rule: RoutingRule,
        intent: Intent,
        complexity: TaskComplexity,
        required_capabilities: Optional[list[ModelCapability]] = None,
    ) -> bool:
        """Check if a rule matches the given criteria."""
        if rule.intent is not None and rule.intent != intent:
            return False
        if rule.complexity is not None and rule.complexity != complexity:
            return False
        if rule.required_capabilities:
            if required_capabilities is None:
                return False
            if not all(cap in required_capabilities for cap in rule.required_capabilities):
                return False
        return True

    def _get_candidates(
        self,
        intent: Intent,
        complexity: TaskComplexity,
        required_capabilities: Optional[list[ModelCapability]] = None,
        preferred_provider: Optional[str] = None,
        max_cost: Optional[float] = None,
    ) -> list[ModelConfig]:
        """Get candidate models filtered by hard constraints."""
        enabled_models = self._registry.get_enabled_models()
        if not enabled_models:
            return []

        candidates: list[ModelConfig] = []
        for model in enabled_models:
            if required_capabilities and not all(
                cap in model.capabilities for cap in required_capabilities
            ):
                continue

            if max_cost is not None and model.cost_per_1k_input_tokens > max_cost:
                continue

            candidates.append(model)

        return candidates

    def _desired_capabilities(
        self,
        intent: Intent,
        complexity: TaskComplexity,
        required_capabilities: Optional[list[ModelCapability]] = None,
    ) -> set[ModelCapability]:
        desired_caps: set[ModelCapability] = set()
        desired_caps.update(_INTENT_CAPABILITY_MAP.get(intent, []))
        desired_caps.update(_COMPLEXITY_CAPABILITY_MAP.get(complexity, []))
        if required_capabilities:
            desired_caps.update(required_capabilities)
        return desired_caps

    def _score_model(
        self,
        model: ModelConfig,
        desired_capabilities: set[ModelCapability],
        preferred_provider: Optional[str],
        complexity: TaskComplexity,
        max_input_cost: float,
    ) -> float:
        """Score a model based on how well it matches the criteria."""
        score = 0.0

        # Capability match score
        if desired_capabilities:
            model_caps = set(model.capabilities)
            overlap = model_caps & desired_capabilities
            score += (len(overlap) / len(desired_capabilities)) * 50

        if preferred_provider and model.provider == preferred_provider:
            score += 20

        score += model.priority * 5

        weights = self._weights()
        normalized_cost = (
            1.0 - (model.cost_per_1k_input_tokens / max_input_cost)
            if max_input_cost > 0
            else 1.0
        )

        metadata_score = (
            model.quality * weights["quality"]
            + model.latency * weights["latency"]
            + normalized_cost * weights["cost"]
            + model.reliability * weights["reliability"]
        )
        score += metadata_score

        if complexity == TaskComplexity.LOW and model.cost_per_1k_input_tokens > 0:
            score += max(0, 10 - model.cost_per_1k_input_tokens * 100)

        return score

    def _get_fallbacks(
        self,
        primary: ModelConfig,
        preferred_provider: Optional[str] = None,
        max_cost: Optional[float] = None,
    ) -> list[ModelConfig]:
        """Get fallback models excluding the primary."""
        enabled = self._registry.get_enabled_models()
        fallbacks = [m for m in enabled if m.model_id != primary.model_id]

        if max_cost is not None:
            fallbacks = [m for m in fallbacks if m.cost_per_1k_input_tokens <= max_cost]

        if preferred_provider:
            fallbacks.sort(key=lambda m: (m.provider != preferred_provider, -m.priority))
        else:
            fallbacks.sort(key=lambda m: -m.priority)

        return fallbacks[:3]

    def _weights(self) -> dict[str, float]:
        if self._config is not None:
            return dict(self._config.routing_weights)
        return dict(_DEFAULT_WEIGHTS)

    def _strategy_name(self) -> str:
        if self._config is not None:
            return self._config.routing_strategy
        return "balanced"
