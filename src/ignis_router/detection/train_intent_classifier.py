"""
Train the Semantic Intent Classifier.

Usage:
    python -m ignis_router.detection.train_intent_classifier
    python -m ignis_router.detection.train_intent_classifier --data path/to/custom_data.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Semantic Intent Classifier")
    parser.add_argument("--data", type=str, default=None, help="Path to training data JSON")
    parser.add_argument("--output", type=str, default=None, help="Path to save model")
    args = parser.parse_args()

    from .semantic_intent_classifier import SemanticIntentClassifier

    print("=" * 60)
    print("  Training Semantic Intent Classifier")
    print("  Model: all-MiniLM-L6-v2 + Logistic Regression")
    print("=" * 60)
    print()

    classifier = SemanticIntentClassifier()
    metrics = classifier.train(training_data_path=args.data)

    print(f"\n  Results:")
    print(f"  {'Accuracy:':<25} {metrics['accuracy']*100:.1f}%")
    print(f"  {'Training samples:':<25} {metrics['num_samples']}")
    print(f"  {'Intent classes:':<25} {metrics['num_classes']}")
    print()
    print(f"  Per-class F1 scores:")
    for intent, scores in metrics["per_class"].items():
        print(f"    {intent:<25} F1={scores['f1']:.3f}  P={scores['precision']:.3f}  R={scores['recall']:.3f}")

    save_path = classifier.save(path=args.output)
    print(f"\n  Model saved to: {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
