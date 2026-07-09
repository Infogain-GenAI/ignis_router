"""Tests for intent detector strategies."""

from typing import Any

from ignis_router import Intent, TaskComplexity
from ignis_router.intent_detector import (
    HybridIntentDetector,
    MLIntentDetector,
    RuleBasedIntentDetector,
)


class _MockMlModel:
    def __init__(self, intent: str, confidence: float, fail: bool = False):
        self._intent = intent
        self._confidence = confidence
        self._fail = fail

    def predict(self, samples: list[str]) -> list[str]:
        if self._fail:
            raise RuntimeError("inference failed")
        return [self._intent for _ in samples]

    def predict_proba(self, samples: list[str]) -> list[list[float]]:
        if self._fail:
            raise RuntimeError("proba failed")
        return [[self._confidence, 1.0 - self._confidence] for _ in samples]


class _MockVectorizer:
    def transform(self, samples: list[str]) -> list[list[float]]:
        return [[float(len(sample))] for sample in samples]


class _MockFeatureModel:
    classes_ = ["code_generation", "general_chat"]

    def predict(self, samples):
        first = samples[0][0]
        if isinstance(first, str):
            raise ValueError("Expected 2D numeric features")
        return ["code_generation"]

    def predict_proba(self, samples):
        first = samples[0][0]
        if isinstance(first, str):
            raise ValueError("Expected 2D numeric features")
        return [[0.88, 0.12]]


class _MockNumericOnlyModel:
    classes_ = ["general_chat", "code_generation"]
    n_features_in_ = 8

    def predict(self, samples):
        first = samples[0]
        if not isinstance(first, list):
            raise ValueError("Expected list-like numeric vector")
        if not all(isinstance(value, (int, float)) for value in first):
            raise ValueError("Expected numeric values")
        return ["code_generation"]

    def predict_proba(self, samples):
        first = samples[0]
        if not all(isinstance(value, (int, float)) for value in first):
            raise ValueError("Expected numeric values")
        return [[0.2, 0.8]]


class _MockNumpyProbabilityModel:
    classes_ = ["general_chat", "code_generation"]

    def predict(self, samples):
        return ["code_generation"]

    def predict_proba(self, samples):
        import numpy as np

        return np.array([[0.1, 0.9]])


class TestRuleBasedIntentDetection:
    def setup_method(self):
        self.detector = RuleBasedIntentDetector()

    def test_detect_code_generation(self):
        intent, confidence = self.detector.detect_intent(
            "Write a Python function to calculate fibonacci"
        )
        assert intent == Intent.CODE_GENERATION
        assert confidence > 0.3

    def test_detect_translation(self):
        intent, confidence = self.detector.detect_intent("Translate this sentence to French")
        assert intent == Intent.TRANSLATION
        assert confidence > 0.3

    def test_detect_general_chat(self):
        intent, confidence = self.detector.detect_intent("Hello, how are you today?")
        assert intent == Intent.GENERAL_CHAT
        assert 0.0 <= confidence <= 1.0

    def test_custom_patterns(self):
        custom = {Intent.CUSTOM: [r"\b(deploy|kubernetes|docker)\b"]}
        detector = RuleBasedIntentDetector(custom_patterns=custom)
        intent, confidence = detector.detect_intent("Deploy this to kubernetes")
        assert intent == Intent.CUSTOM
        assert confidence > 0.3


class TestComplexityAssessment:
    def setup_method(self):
        self.detector = RuleBasedIntentDetector()

    def test_low_complexity(self):
        complexity = self.detector.assess_complexity("What is Python?")
        assert complexity == TaskComplexity.LOW

    def test_high_complexity(self):
        complexity = self.detector.assess_complexity(
            "Design a comprehensive distributed system architecture with microservices"
        )
        assert complexity == TaskComplexity.HIGH


class TestMlIntentDetection:
    def test_ml_detector_success(self):
        detector = MLIntentDetector(model=_MockMlModel("code_generation", 0.91))
        intent, confidence = detector.detect_intent("Any prompt")
        assert intent == Intent.CODE_GENERATION
        assert confidence == 0.91

    def test_ml_detector_unavailable_model(self):
        detector = MLIntentDetector(model_path="models/does_not_exist.pkl")
        intent, confidence = detector.detect_intent("Any prompt")
        assert intent == Intent.GENERAL_CHAT
        assert confidence == 0.0

    def test_ml_detector_inference_exception(self):
        detector = MLIntentDetector(model=_MockMlModel("reasoning", 0.9, fail=True))
        intent, confidence = detector.detect_intent("Any prompt")
        assert intent == Intent.GENERAL_CHAT
        assert confidence == 0.0

    def test_ml_detector_supports_model_vectorizer_artifact(self):
        artifact = {
            "model": _MockFeatureModel(),
            "vectorizer": _MockVectorizer(),
        }
        detector = MLIntentDetector(model=artifact)

        intent, confidence = detector.detect_intent("Write code")

        assert intent == Intent.CODE_GENERATION
        assert confidence == 0.88

    def test_ml_detector_supports_numeric_only_model(self):
        detector = MLIntentDetector(model=_MockNumericOnlyModel())

        intent, confidence = detector.detect_intent("What is capital of india?")

        assert intent == Intent.CODE_GENERATION
        assert confidence == 0.8

    def test_ml_detector_parses_numpy_probability_output(self):
        detector = MLIntentDetector(model=_MockNumpyProbabilityModel())

        intent, confidence = detector.detect_intent("Write code")

        assert intent == Intent.CODE_GENERATION
        assert confidence == 0.9


class TestHybridIntentDetection:
    def test_hybrid_uses_ml_when_confident(self):
        hybrid = HybridIntentDetector(
            ml_detector=MLIntentDetector(model=_MockMlModel("reasoning", 0.93)),
            rule_based_detector=RuleBasedIntentDetector(),
            confidence_threshold=0.6,
        )

        intent, confidence = hybrid.detect_intent("Explain why compilers optimize loops")
        assert intent == Intent.REASONING
        assert confidence == 0.93

    def test_hybrid_falls_back_when_confidence_low(self):
        hybrid = HybridIntentDetector(
            ml_detector=MLIntentDetector(model=_MockMlModel("general_chat", 0.2)),
            rule_based_detector=RuleBasedIntentDetector(),
            confidence_threshold=0.6,
        )

        intent, confidence = hybrid.detect_intent("Write a Python script for sorting")
        assert intent == Intent.CODE_GENERATION
        assert confidence > 0.2

    def test_hybrid_falls_back_when_ml_unavailable(self):
        hybrid = HybridIntentDetector(
            ml_detector=MLIntentDetector(model_path="models/missing.pkl"),
            rule_based_detector=RuleBasedIntentDetector(),
            confidence_threshold=0.6,
        )

        intent, confidence = hybrid.detect_intent("Summarize this meeting transcript")
        assert intent == Intent.SUMMARIZATION
        assert confidence > 0.0
