"""Intent detection strategies for classifying user queries."""

from __future__ import annotations

import logging
import pickle
import re
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

from .models import Intent, TaskComplexity

logger = logging.getLogger(__name__)


# Keyword patterns for intent detection
_INTENT_PATTERNS: dict[Intent, list[str]] = {
    Intent.CODE_GENERATION: [
        r"\b(write|create|generate|implement|code|function|class|script|program|debug|fix|refactor)\b",
        r"\b(python|javascript|java|typescript|rust|go|sql|html|css)\b",
        r"\b(api|endpoint|algorithm|data structure|unit test)\b",
    ],
    Intent.SUMMARIZATION: [
        r"\b(summarize|summary|summarise|tldr|brief|overview|condense|shorten)\b",
        r"\b(key points|main ideas|highlights)\b",
    ],
    Intent.REASONING: [
        r"\b(explain|why|how does|analyze|reason|logic|think through|step by step)\b",
        r"\b(compare|contrast|evaluate|assess|pros and cons)\b",
        r"\b(solve|calculate|math|equation|proof)\b",
    ],
    Intent.CREATIVE_WRITING: [
        r"\b(write a story|poem|creative|fiction|narrative|essay|blog post)\b",
        r"\b(imagine|invent|compose|draft|brainstorm)\b",
    ],
    Intent.DATA_ANALYSIS: [
        r"\b(data|dataset|csv|json|analyze data|statistics|chart|graph|visualization)\b",
        r"\b(trend|pattern|correlation|regression|aggregate)\b",
    ],
    Intent.TRANSLATION: [
        r"\b(translate|translation|convert to|in spanish|in french|in german|in chinese)\b",
        r"\b(localize|multilingual|language)\b",
    ], 
    Intent.CLASSIFICATION: [
        r"\b(classify|categorize|label|tag|sentiment|detect|identify type)\b",
    ],
    Intent.EXTRACTION: [
        r"\b(extract|parse|find in|pull out|get the|entities|named entity)\b",
        r"\b(regex|pattern match|scrape)\b",
    ],
}

# Complexity indicators
_HIGH_COMPLEXITY_PATTERNS = [
    r"\b(complex|advanced|detailed|comprehensive|in-depth|multi-step)\b",
    r"\b(architecture|system design|distributed|microservice)\b",
    r"\b(optimize|performance|scalable|production-ready)\b",
]

_LOW_COMPLEXITY_PATTERNS = [
    r"\b(simple|basic|quick|short|brief|easy|hello world)\b",
    r"\b(what is|define|list|name)\b",
]


def _coerce_intent(raw: Any) -> Intent:
    """Convert model output values to a supported Intent enum."""
    if isinstance(raw, Intent):
        return raw

    if isinstance(raw, str):
        normalized = raw.strip().lower()

        for intent in Intent:
            if normalized in {intent.value.lower(), intent.name.lower()}:
                return intent

    return Intent.GENERAL_CHAT


class BaseIntentDetector(ABC):
    """Strategy interface for intent detectors."""

    @abstractmethod
    def detect_intent(self, text: str) -> tuple[Intent, float]:
        """Detect the user intent and return confidence in range [0.0, 1.0]."""

    def assess_complexity(self, text: str) -> TaskComplexity:
        """Assess query complexity using shared heuristic defaults."""
        query_lower = text.lower()

        high_score = sum(
            1 for pattern in _HIGH_COMPLEXITY_PATTERNS if re.search(pattern, query_lower)
        )
        low_score = sum(
            1 for pattern in _LOW_COMPLEXITY_PATTERNS if re.search(pattern, query_lower)
        )

        word_count = len(text.split())
        if word_count > 100:
            high_score += 1
        elif word_count < 15:
            low_score += 1

        if high_score > low_score:
            return TaskComplexity.HIGH
        if low_score > high_score:
            return TaskComplexity.LOW
        return TaskComplexity.MEDIUM


class RuleBasedIntentDetector(BaseIntentDetector):
    """Detect intent using regex and keyword-based heuristics."""

    def __init__(self, custom_patterns: Optional[dict[Intent, list[str]]] = None):
        self._patterns = dict(_INTENT_PATTERNS)
        if custom_patterns:
            for intent, patterns in custom_patterns.items():
                if intent in self._patterns:
                    self._patterns[intent].extend(patterns)
                else:
                    self._patterns[intent] = patterns

    def detect_intent(self, text: str) -> tuple[Intent, float]:
        """
        Detect the intent of a query.

        Returns:
            Tuple of (detected intent, confidence score 0.0-1.0)
        """
        query_lower = text.lower()
        scores: dict[Intent, float] = {}

        for intent, patterns in self._patterns.items():
            match_count = 0
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    match_count += 1
            if match_count > 0:
                scores[intent] = min(match_count / len(patterns), 1.0)

        if not scores:
            return Intent.GENERAL_CHAT, 0.6

        best_intent = max(scores, key=scores.get)  # type: ignore
        confidence = scores[best_intent]
        # Boost confidence if multiple patterns matched
        confidence = min(confidence * 1.2, 1.0)

        return best_intent, round(confidence, 3)

class MLIntentDetector(BaseIntentDetector):
    """Detect intent by running inference against a serialized ML model."""

    def __init__(
        self,
        model_path: str = "models/knnrouter.pkl",
        model: Any = None,
        enable_model_hint_routing: bool = False,
    ):
        self._model_path = model_path
        self._model = model
        self._last_model_hint: Optional[str] = None
        self._enable_model_hint_routing = enable_model_hint_routing

        if self._model is None:
            self._model = self._safe_load_model(model_path)

    @property
    def is_available(self) -> bool:
        """Whether a usable ML model instance is available."""
        return self._model is not None

    def detect_intent(self, text: str) -> tuple[Intent, float]:
        """Run intent inference. Returns low-confidence fallback on failure."""
        self._last_model_hint = None
        if not self.is_available:
            logger.warning(
                "ML model unavailable. Falling back to GENERAL_CHAT output from ML detector."
            )
            return Intent.GENERAL_CHAT, 0.0

        try:
            predictor, transformer = self._resolve_components()
            if predictor is None or not hasattr(predictor, "predict"):
                raise AttributeError("ML model does not expose 'predict'.")

            prediction_input, prediction = self._predict_with_adapters(
                predictor=predictor,
                transformer=transformer,
                text=text,
            )
            if hasattr(prediction, "tolist"):
                prediction = prediction.tolist()

            if isinstance(prediction, (list, tuple)) and prediction:
                raw_intent = prediction[0]
            else:
                raw_intent = prediction
            intent = _coerce_intent(raw_intent)
            raw_label = str(raw_intent).strip()

            if intent == Intent.GENERAL_CHAT and raw_label and raw_label.lower() not in {
                Intent.GENERAL_CHAT.value,
                Intent.GENERAL_CHAT.name.lower(),
            }:
                # ML artifact predicts model labels; capture hint for direct model selection.
                self._last_model_hint = raw_label

            confidence = 0.0
            if hasattr(predictor, "predict_proba"):
                probabilities = predictor.predict_proba(prediction_input)
                if hasattr(probabilities, "tolist"):
                    probabilities = probabilities.tolist()

                if isinstance(probabilities, (list, tuple)) and probabilities:
                    first_row = probabilities[0]
                    if hasattr(first_row, "tolist"):
                        first_row = first_row.tolist()

                    if isinstance(first_row, (list, tuple)) and first_row:
                        if hasattr(predictor, "classes_"):
                            try:
                                classes = [str(item) for item in list(getattr(predictor, "classes_"))]
                                label = str(
                                    raw_intent.value if isinstance(raw_intent, Intent) else raw_intent
                                )
                                label_idx = classes.index(label)
                                confidence = float(first_row[label_idx])
                            except Exception:
                                confidence = float(first_row[0])
                        else:
                            confidence = float(first_row[0])

            return intent, round(max(0.0, min(confidence, 1.0)), 3)
        except Exception as exc:
            logger.warning(
                "ML inference failed. Falling back to GENERAL_CHAT output from ML detector. %s",
                exc,
            )
            return Intent.GENERAL_CHAT, 0.0

    def get_model_hint(self) -> Optional[str]:
        """Return the last ML-predicted model hint label if available."""
        return self._last_model_hint

    def clear_model_hint(self) -> None:
        """Clear any retained ML model hint from prior predictions."""
        self._last_model_hint = None

    def _resolve_components(self) -> tuple[Optional[Any], Optional[Any]]:
        """Resolve predictor and optional transformer from loaded artifact."""
        artifact = self._model
        if artifact is None:
            return None, None

        if isinstance(artifact, dict):
            predictor = (
                artifact.get("pipeline")
                or artifact.get("model")
                or artifact.get("classifier")
                or artifact.get("estimator")
                or artifact.get("predictor")
                or artifact.get("clf")
            )
            transformer = (
                artifact.get("vectorizer")
                or artifact.get("transformer")
                or artifact.get("preprocessor")
            )
            return predictor, transformer

        if isinstance(artifact, tuple) and len(artifact) == 2:
            first, second = artifact
            if hasattr(first, "predict") and hasattr(second, "transform"):
                return first, second
            if hasattr(second, "predict") and hasattr(first, "transform"):
                return second, first

        return artifact, None

    def _predict_with_adapters(
        self,
        predictor: Any,
        transformer: Optional[Any],
        text: str,
    ) -> tuple[Any, Any]:
        """Try common input encodings until model inference succeeds."""
        attempted_errors: list[str] = []

        prepared_variants = self._prepare_input_variants(text, transformer)
        numeric_variant = self._numeric_feature_fallback(text, predictor)
        if numeric_variant is not None:
            prepared_variants.append(numeric_variant)

        for prepared_input in prepared_variants:
            try:
                prediction = predictor.predict(prepared_input)
                return prepared_input, prediction
            except Exception as exc:
                attempted_errors.append(str(exc))

        raise RuntimeError(
            "No compatible input adapter for ML model. "
            f"Tried {len(attempted_errors)} input variants. Last error: "
            f"{attempted_errors[-1] if attempted_errors else 'unknown'}"
        )

    def _prepare_input_variants(self, text: str, transformer: Optional[Any]) -> list[Any]:
        """Build candidate input formats for predictor compatibility."""
        variants: list[Any] = []

        if transformer is not None and hasattr(transformer, "transform"):
            try:
                variants.append(transformer.transform([text]))
            except Exception:
                pass

        # Raw text single sample format used by sklearn text pipelines.
        variants.append([text])

        # Nested single-sample format for estimators expecting 2D arrays.
        variants.append([[text]])

        return variants

    def _numeric_feature_fallback(self, text: str, predictor: Any) -> Optional[list[list[float]]]:
        """Create deterministic numeric features when model expects numeric vectors."""
        n_features = getattr(predictor, "n_features_in_", None)
        if not isinstance(n_features, int) or n_features <= 0:
            return None

        vector = [0.0] * n_features
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return [vector]

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
            token_index = int(digest[:8], 16) % n_features
            vector[token_index] += 1.0

        max_value = max(vector) if vector else 0.0
        if max_value > 0:
            vector = [value / max_value for value in vector]

        return [vector]

    def _safe_load_model(self, model_path: str) -> Any:
        """Load model from disk without raising fatal startup errors."""
        try:
            path = Path(model_path)
            if not path.exists():
                logger.warning("ML model file missing at %s", path)
                return None

            with path.open("rb") as file_obj:
                return pickle.load(file_obj)
        except Exception as exc:
            logger.warning("Failed to load ML model from %s: %s", model_path, exc)
            return None


class HybridIntentDetector(BaseIntentDetector):
    """Prefer ML intent detection with automatic rule-based fallback."""

    def __init__(
        self,
        ml_detector: MLIntentDetector,
        rule_based_detector: RuleBasedIntentDetector,
        confidence_threshold: float = 0.6,
    ):
        self._ml_detector = ml_detector
        self._rule_based_detector = rule_based_detector
        self._confidence_threshold = confidence_threshold

    def detect_intent(self, text: str) -> tuple[Intent, float]:
        """Try ML first, then fall back when unavailable, failed, or low confidence."""
        self._ml_won = False

        if not self._ml_detector.is_available:
            self._ml_detector.clear_model_hint()
            logger.warning("ML model unavailable. Falling back to Rule-Based Detector.")
            return self._rule_based_detector.detect_intent(text)

        try:
            intent, confidence = self._ml_detector.detect_intent(text)
        except Exception as exc:
            self._ml_detector.clear_model_hint()
            logger.warning(
                "ML inference raised an exception. Falling back to Rule-Based Detector. %s",
                exc,
            )
            return self._rule_based_detector.detect_intent(text)

        if confidence < self._confidence_threshold:
            self._ml_detector.clear_model_hint()
            logger.warning(
                "ML confidence below threshold. Falling back to Rule-Based Detector. "
                "confidence=%.3f threshold=%.3f",
                confidence,
                self._confidence_threshold,
            )
            return self._rule_based_detector.detect_intent(text)

        # ML confidence is high enough — ML wins.
        # If ML predicted a model label (not intent), keep the hint for ML router.
        # Still use rule-based for intent detection internally, but mark ML as winner.
        if self._ml_detector.get_model_hint():
            self._ml_won = True
            # Use rule-based for internal intent (needed for scoring fallback),
            # but ML model hint takes priority for model selection.
            rb_intent, rb_conf = self._rule_based_detector.detect_intent(text)
            return rb_intent, confidence  # keep ML confidence

        return intent, confidence

    @property
    def ml_won(self) -> bool:
        """Whether ML prediction won (confidence >= threshold) on last query."""
        return getattr(self, "_ml_won", False)

    def assess_complexity(self, text: str) -> TaskComplexity:
        """Complexity remains heuristic-driven regardless of detector strategy."""
        return self._rule_based_detector.assess_complexity(text)

    def get_model_hint(self) -> Optional[str]:
        """Expose ML model hint for routing engines that support model-hint selection."""
        return self._ml_detector.get_model_hint()


class IntentDetector(RuleBasedIntentDetector):
    """Backward-compatible alias for the historical rule-based detector."""

    pass
