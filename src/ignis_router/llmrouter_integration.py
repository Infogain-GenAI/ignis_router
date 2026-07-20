"""
LLMRouter Integration Layer
============================

This module provides the integration between:
  - **LLMRouter (open-source package)** → ML routing decisions, embeddings, training
  - **LLM Router Accelerator** → orchestration, rules, scoring, API, logging

Architecture:
┌─────────────────────────────────────────────────────────┐
│ LLM Router Accelerator (ignis_router)                   │
│                                                         │
│  Intent Detection ─► Rule Engine ─► Scoring Engine      │
│         │                                               │
│         ▼                                               │
│  ┌─────────────────────────────────────────────┐        │
│  │ LLMRouter Package (inference layer)         │        │
│  │  • KNN / SVM / Graph / MF routing           │        │
│  │  • Longformer embedding generation          │        │
│  │  • Training pipeline (optional retrain)     │        │
│  └─────────────────────────────────────────────┘        │
│         │                                               │
│         ▼                                               │
│  Provider Integration (OpenAI, Anthropic, Gemini)       │
│  REST API / SDK / Decorators / PostgreSQL Logging       │
└─────────────────────────────────────────────────────────┘

Usage:
    from ignis_router.llmrouter_integration import (
        MLInferenceEngine,    # Predict best model for a query
        EmbeddingEngine,      # Generate query embeddings
        TrainingPipeline,     # Retrain routers with new data
    )
"""

from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .exceptions import IntentDetectionError

logger = logging.getLogger(__name__)

# Supported ML router types from LLMRouter package
SUPPORTED_ML_ROUTERS = {"knn", "svm", "graph", "mf"}

DEFAULT_FALLBACK_LLM = "gpt-4o-mini"


def _get_llmrouter_root() -> str:
    """Get the root directory of the installed llmrouter-lib package."""
    import llmrouter
    return str(Path(llmrouter.__file__).resolve().parent.parent)


def _get_project_root() -> str:
    """Get ignis_router project root (2 levels up from this file)."""
    return str(Path(__file__).resolve().parents[2])


# ═══════════════════════════════════════════════════════════════════════════════
# 1. INFERENCE ENGINE — Predict best LLM model for a query
# ═══════════════════════════════════════════════════════════════════════════════


class MLInferenceEngine:
    """
    Wraps LLMRouter's ML routers for inference.

    Reuses the open-source inference engine (KNN, SVM, Graph, MF)
    to predict which LLM model is best for a given query.

    Usage:
        engine = MLInferenceEngine(router_type="knn")
        prediction = engine.predict("Write Python code for sorting")
        # prediction = "qwen2.5-7b-instruct"
    """

    def __init__(
        self,
        router_type: str = "knn",
        config_dir: str = "configs/ml_routers",
        project_root: Optional[str] = None,
    ):
        if router_type not in SUPPORTED_ML_ROUTERS:
            raise IntentDetectionError(
                f"Unsupported ML router type: '{router_type}'. "
                f"Supported: {sorted(SUPPORTED_ML_ROUTERS)}"
            )

        self._router_type = router_type
        self._project_root = project_root or _get_project_root()
        self._config_dir = os.path.join(self._project_root, config_dir)
        self._router: Any = None
        self._available = False

        self._load_router()

    @property
    def router_type(self) -> str:
        return self._router_type

    @property
    def is_available(self) -> bool:
        return self._available

    def predict(self, query: str) -> Optional[str]:
        """
        Predict the best LLM model for a query using the ML router.

        Args:
            query: User query text.

        Returns:
            Predicted model name string, or None if prediction fails.
        """
        if not self._available or self._router is None:
            logger.warning("ML router '%s' is not available.", self._router_type)
            return None

        try:
            llmrouter_root = _get_llmrouter_root()
            original_cwd = os.getcwd()
            os.chdir(llmrouter_root)
            try:
                result = self._router.route_single({"query": query})
            finally:
                os.chdir(original_cwd)

            model_name = result.get("model_name")
            if model_name:
                logger.info(
                    "ML router '%s' predicted model: %s", self._router_type, model_name
                )
                return str(model_name).strip()
            return None
        except Exception as exc:
            logger.warning(
                "ML router '%s' prediction failed: %s", self._router_type, exc
            )
            return None

    def _load_router(self) -> None:
        """Load the ML router from llmrouter-lib package."""
        yaml_path = self._get_yaml_path()
        if not os.path.exists(yaml_path):
            logger.warning("ML router config not found: %s. Disabled.", yaml_path)
            return

        try:
            llmrouter_root = _get_llmrouter_root()
            self._ensure_model_file(llmrouter_root)

            original_cwd = os.getcwd()
            os.chdir(llmrouter_root)
            try:
                if self._router_type == "knn":
                    from llmrouter.models.knnrouter.router import KNNRouter
                    self._router = KNNRouter(yaml_path=yaml_path)
                elif self._router_type == "svm":
                    from llmrouter.models.svmrouter.router import SVMRouter
                    self._router = SVMRouter(yaml_path=yaml_path)
                elif self._router_type == "graph":
                    from llmrouter.models.graphrouter.router import GraphRouter
                    self._router = GraphRouter(yaml_path=yaml_path)
                elif self._router_type == "mf":
                    from llmrouter.models.mfrouter.router import MFRouter
                    self._router = MFRouter(yaml_path=yaml_path)

                self._available = True
                logger.info("ML router '%s' loaded successfully.", self._router_type)
            finally:
                os.chdir(original_cwd)

        except Exception as exc:
            logger.warning("Failed to load ML router '%s': %s", self._router_type, exc)
            self._available = False

    def _ensure_model_file(self, llmrouter_root: str) -> None:
        """Copy local trained model to llmrouter-lib's expected location."""
        model_dir_map = {"knn": "knnrouter", "svm": "svmrouter", "graph": "graphrouter", "mf": "mfrouter"}
        model_file_map = {
            "knn": "knnrouter.pkl", "svm": "svmrouter.pkl",
            "graph": "graphrouter.pt", "mf": "mfrouter.pt",
        }
        subdir = model_dir_map[self._router_type]
        filename = model_file_map[self._router_type]

        local_model = os.path.join(self._project_root, "models", subdir, filename)
        if not os.path.exists(local_model):
            logger.warning("Local model file not found: %s", local_model)
            return

        # Copy to both possible locations llmrouter-lib may check
        targets = [
            os.path.join(llmrouter_root, "models", subdir, filename),
            os.path.join(llmrouter_root, "llmrouter", "models", subdir, filename),
        ]
        for target in targets:
            if not os.path.exists(target):
                os.makedirs(os.path.dirname(target), exist_ok=True)
                shutil.copy2(local_model, target)
                logger.info("Copied model: %s -> %s", local_model, target)

    def _get_yaml_path(self) -> str:
        filename_map = {
            "knn": "knnrouter.yaml", "svm": "svmrouter.yaml",
            "graph": "graphrouter.yaml", "mf": "mfrouter.yaml",
        }
        return os.path.join(self._config_dir, filename_map[self._router_type])


# ═══════════════════════════════════════════════════════════════════════════════
# 2. EMBEDDING ENGINE — Reuse LLMRouter's Longformer embeddings
# ═══════════════════════════════════════════════════════════════════════════════


class EmbeddingEngine:
    """
    Reuses LLMRouter's Longformer embedding generation.

    Provides query embeddings that can be used for:
    - Similarity search
    - Custom model selection logic
    - Caching/clustering queries

    Usage:
        embedder = EmbeddingEngine()
        embedding = embedder.embed("Write Python code for sorting")
        # embedding.shape = (768,)
    """

    def __init__(self):
        self._available = False
        self._check_availability()

    @property
    def is_available(self) -> bool:
        return self._available

    def _check_availability(self) -> None:
        try:
            from llmrouter.utils import get_longformer_embedding  # noqa: F401
            self._available = True
        except ImportError:
            logger.warning("LLMRouter embedding engine not available.")
            self._available = False

    def embed(self, text: str) -> Optional[np.ndarray]:
        """
        Generate a Longformer embedding for a text query.

        Args:
            text: Query text to embed.

        Returns:
            Numpy array of shape (768,), or None if unavailable.
        """
        if not self._available:
            return None

        try:
            from llmrouter.utils import get_longformer_embedding
            embedding = get_longformer_embedding(text)
            if hasattr(embedding, "numpy"):
                return embedding.numpy()
            return np.array(embedding)
        except Exception as exc:
            logger.warning("Embedding generation failed: %s", exc)
            return None

    def embed_batch(self, texts: list[str]) -> Optional[np.ndarray]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of query texts.

        Returns:
            Numpy array of shape (n, 768), or None if unavailable.
        """
        if not self._available:
            return None

        results = []
        for text in texts:
            emb = self.embed(text)
            if emb is not None:
                results.append(emb)
        return np.array(results) if results else None


# ═══════════════════════════════════════════════════════════════════════════════
# 3. TRAINING PIPELINE — Retrain routers when new models are introduced
# ═══════════════════════════════════════════════════════════════════════════════


class TrainingPipeline:
    """
    Exposes LLMRouter's training pipeline as a programmatic capability.

    Enables retraining when:
    - New LLM models are introduced
    - You have new query/response data
    - You want to optimize routing for your specific use case

    Usage:
        pipeline = TrainingPipeline()
        pipeline.train("knn")   # Train KNN router
        pipeline.train("svm")   # Train SVM router
        pipeline.train_all()    # Train all routers
    """

    def __init__(self, project_root: Optional[str] = None):
        self._project_root = project_root or _get_project_root()

    def train(self, router_type: str) -> bool:
        """
        Train a specific ML router.

        Args:
            router_type: One of 'knn', 'svm', 'graph', 'mf'.

        Returns:
            True if training succeeded, False otherwise.
        """
        if router_type not in SUPPORTED_ML_ROUTERS:
            raise ValueError(f"Unsupported router: '{router_type}'. Use: {sorted(SUPPORTED_ML_ROUTERS)}")

        llmrouter_root = _get_llmrouter_root()
        yaml_path = os.path.join(
            self._project_root, "configs", "ml_routers",
            {"knn": "knnrouter.yaml", "svm": "svmrouter.yaml",
             "graph": "graphrouter.yaml", "mf": "mfrouter.yaml"}[router_type]
        )

        original_cwd = os.getcwd()
        os.chdir(llmrouter_root)

        try:
            if router_type == "knn":
                from llmrouter.models.knnrouter.router import KNNRouter
                from llmrouter.models.knnrouter.trainer import KNNRouterTrainer
                router = KNNRouter(yaml_path=yaml_path)
                trainer = KNNRouterTrainer(router=router)

            elif router_type == "svm":
                from llmrouter.models.svmrouter.router import SVMRouter
                from llmrouter.models.svmrouter.trainer import SVMRouterTrainer
                router = SVMRouter(yaml_path=yaml_path)
                trainer = SVMRouterTrainer(router=router)

            elif router_type == "graph":
                from llmrouter.models.graphrouter.router import GraphRouter
                from llmrouter.models.graphrouter.trainer import GraphTrainer
                router = GraphRouter(yaml_path=yaml_path)
                trainer = GraphTrainer(router=router)

            elif router_type == "mf":
                from llmrouter.models.mfrouter.router import MFRouter
                from llmrouter.models.mfrouter.trainer import MFRouterTrainer
                router = MFRouter(yaml_path=yaml_path)
                trainer = MFRouterTrainer(router=router)

            trainer.train()
            self._copy_trained_model(router_type, llmrouter_root)
            logger.info("Successfully trained '%s' router.", router_type)
            return True

        except Exception as exc:
            logger.error("Training failed for '%s': %s", router_type, exc)
            return False
        finally:
            os.chdir(original_cwd)

    def train_all(self) -> dict[str, bool]:
        """
        Train all ML routers.

        Returns:
            Dict mapping router_type to success boolean.
        """
        results = {}
        for rt in sorted(SUPPORTED_ML_ROUTERS):
            results[rt] = self.train(rt)
        return results

    def _copy_trained_model(self, router_type: str, llmrouter_root: str) -> None:
        """Copy freshly trained model from llmrouter-lib location to our models/ folder."""
        model_dir_map = {"knn": "knnrouter", "svm": "svmrouter", "graph": "graphrouter", "mf": "mfrouter"}
        model_file_map = {
            "knn": "knnrouter.pkl", "svm": "svmrouter.pkl",
            "graph": "graphrouter.pt", "mf": "mfrouter.pt",
        }
        subdir = model_dir_map[router_type]
        filename = model_file_map[router_type]

        # Check both possible source locations
        sources = [
            os.path.join(llmrouter_root, "models", subdir, filename),
            os.path.join(llmrouter_root, "llmrouter", "models", subdir, filename),
        ]
        dst_dir = os.path.join(self._project_root, "models", subdir)
        os.makedirs(dst_dir, exist_ok=True)

        for src in sources:
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(dst_dir, filename))
                logger.info("Trained model saved: %s", os.path.join(dst_dir, filename))
                return

        logger.warning("Trained model file not found for '%s'.", router_type)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HELPER — Check API key availability for predicted model
# ═══════════════════════════════════════════════════════════════════════════════


def check_llm_key_available(model_name: str) -> bool:
    """Check if an API key is available for the predicted LLM model."""
    model_lower = model_name.lower()

    if any(k in model_lower for k in ("gpt", "openai", "o1", "o3")):
        return bool(os.getenv("OPENAI_API_KEY"))
    if any(k in model_lower for k in ("claude", "anthropic", "sonnet", "opus", "haiku")):
        return bool(os.getenv("ANTHROPIC_API_KEY"))
    if any(k in model_lower for k in ("gemini", "gemma", "google", "palm")):
        return bool(os.getenv("GOOGLE_API_KEY"))
    if any(k in model_lower for k in ("mistral",)):
        return bool(os.getenv("MISTRAL_API_KEY"))
    if any(k in model_lower for k in ("llama", "meta", "qwen")):
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("TOGETHER_API_KEY"))
    return bool(os.getenv("OPENAI_API_KEY"))


def resolve_llm_for_prediction(
    predicted_model: str,
    default_fallback: str = DEFAULT_FALLBACK_LLM,
) -> tuple[str, str]:
    """
    Given an ML-predicted model name, resolve which LLM to actually call.

    If the predicted model's API key is available, use it directly.
    Otherwise fall back to default model.

    Returns:
        Tuple of (model_to_use, message). Message is empty if no fallback needed.
    """
    if check_llm_key_available(predicted_model):
        return predicted_model, ""

    msg = (
        f"API key not available for '{predicted_model}'. "
        f"Switching to default model '{default_fallback}' for LLM response."
    )
    logger.warning(msg)
    return default_fallback, msg
