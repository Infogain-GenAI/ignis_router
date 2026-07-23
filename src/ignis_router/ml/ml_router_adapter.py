"""Adapter module to integrate llmrouter-lib ML routers (KNN, SVM, Graph, MF) into ignis_router."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional

from ..exceptions import IntentDetectionError

logger = logging.getLogger(__name__)

# Supported ML router types
SUPPORTED_ML_ROUTERS = {"knn", "svm", "graph", "mf"}

# Default LLM to fall back to when selected model's API key is unavailable
DEFAULT_FALLBACK_LLM = "gpt-4o-mini"


class MLRouterAdapter:
    """
    Adapter that wraps llmrouter-lib's ML routers (KNN, SVM, GraphRouter, MFRouter)
    into a unified interface for ignis_router's routing pipeline.

    Each router takes a query and returns a predicted LLM model name.
    """

    def __init__(
        self,
        router_type: str = "knn",
        config_dir: str = "configs/ml_routers",
        project_root: Optional[str] = None,
    ):
        """
        Initialize the ML router adapter.

        Args:
            router_type: One of 'knn', 'svm', 'graph', 'mf'.
            config_dir: Directory containing router YAML configs (relative to project root).
            project_root: Project root path. Auto-detected if not provided.
        """
        if router_type not in SUPPORTED_ML_ROUTERS:
            raise IntentDetectionError(
                f"Unsupported ML router type: '{router_type}'. "
                f"Supported: {sorted(SUPPORTED_ML_ROUTERS)}"
            )

        self._router_type = router_type
        self._project_root = project_root or str(Path(__file__).resolve().parents[3])
        # Prefer bundled configs inside the installed package; fall back to project root
        bundled_config_dir = str(Path(__file__).resolve().parents[1] / "configs" / "ml_routers")
        project_config_dir = os.path.join(self._project_root, config_dir)
        self._config_dir = bundled_config_dir if os.path.isdir(bundled_config_dir) else project_config_dir
        self._router: Any = None
        self._available = False

        self._load_router()

    @property
    def router_type(self) -> str:
        """Return the active ML router type."""
        return self._router_type

    @property
    def is_available(self) -> bool:
        """Whether the ML router loaded successfully."""
        return self._available

    def predict_model(self, query: str) -> Optional[str]:
        """
        Run ML routing on a query and return the predicted LLM model name.

        Args:
            query: User query text.

        Returns:
            Predicted model name string, or None if prediction fails.
        """
        if not self._available or self._router is None:
            logger.warning("ML router '%s' is not available.", self._router_type)
            return None

        try:
            # llmrouter-lib uses relative paths internally during prediction too
            import llmrouter
            import io
            import sys
            llmrouter_root = str(Path(llmrouter.__file__).resolve().parent.parent)
            original_cwd = os.getcwd()
            os.chdir(llmrouter_root)
            try:
                # Suppress llmrouter-lib's internal print/tqdm output
                old_stdout = sys.stdout
                sys.stdout = io.StringIO()
                try:
                    result = self._router.route_single({"query": query})
                finally:
                    sys.stdout = old_stdout
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
        """Load the appropriate llmrouter-lib router class."""
        yaml_path = self._get_yaml_path()
        if not os.path.exists(yaml_path):
            logger.warning(
                "ML router config not found: %s. ML routing disabled.", yaml_path
            )
            return

        try:
            # llmrouter-lib resolves relative paths from cwd;
            # temporarily switch to its package root so data/model paths resolve.
            import llmrouter
            llmrouter_root = str(Path(llmrouter.__file__).resolve().parent.parent)
            original_cwd = os.getcwd()

            # Copy our local model file into llmrouter-lib location if missing
            self._ensure_model_file(llmrouter_root)
            # Copy data folder if missing in llmrouter-lib location
            self._ensure_data_folder(llmrouter_root)

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
            logger.warning(
                "Failed to load ML router '%s': %s. ML routing disabled.",
                self._router_type,
                exc,
            )
            self._available = False

    def _ensure_model_file(self, llmrouter_root: str) -> None:
        """Copy local model file to llmrouter-lib location if it doesn't exist there."""
        model_dir_map = {
            "knn": "knnrouter",
            "svm": "svmrouter",
            "graph": "graphrouter",
            "mf": "mfrouter",
        }
        model_file_map = {
            "knn": "knnrouter.pkl",
            "svm": "svmrouter.pkl",
            "graph": "graphrouter.pt",
            "mf": "mfrouter.pt",
        }
        subdir = model_dir_map[self._router_type]
        filename = model_file_map[self._router_type]

        # Source: our project's models/ folder (bundled inside package)
        _pkg_root = str(Path(__file__).resolve().parents[1])
        local_model = os.path.join(_pkg_root, "models", subdir, filename)
        # Also check flat structure (e.g. knnrouter.pkl directly in models/)
        local_model_flat = os.path.join(_pkg_root, "models", filename)

        # Target locations: llmrouter-lib resolves from both root and package subfolder
        target_locations = [
            os.path.join(llmrouter_root, "models", subdir, filename),
            os.path.join(llmrouter_root, "llmrouter", "models", subdir, filename),
        ]

        if not os.path.exists(local_model):
            # Try flat structure fallback
            if os.path.exists(local_model_flat):
                local_model = local_model_flat
            else:
                logger.warning("ML model file missing at %s", local_model)
                return

        import shutil
        for target_model in target_locations:
            if not os.path.exists(target_model):
                os.makedirs(os.path.dirname(target_model), exist_ok=True)
                shutil.copy2(local_model, target_model)
                logger.info(
                    "Copied model file: %s -> %s", local_model, target_model,
                )

    def _ensure_data_folder(self, llmrouter_root: str) -> None:
        """Copy project data/ folder to llmrouter-lib location if missing."""
        import shutil
        _pkg_root = str(Path(__file__).resolve().parents[1])
        local_data = os.path.join(_pkg_root, "data")
        target_data = os.path.join(llmrouter_root, "data")

        if not os.path.exists(local_data):
            return

        if not os.path.exists(target_data):
            shutil.copytree(local_data, target_data)
            logger.info("Copied data folder: %s -> %s", local_data, target_data)

    def _get_yaml_path(self) -> str:
        """Get the YAML config path for the current router type."""
        filename_map = {
            "knn": "knnrouter.yaml",
            "svm": "svmrouter.yaml",
            "graph": "graphrouter.yaml",
            "mf": "mfrouter.yaml",
        }
        return os.path.join(self._config_dir, filename_map[self._router_type])


def check_llm_key_available(provider: str, model_name: str) -> bool:
    """
    Check if an API key is available for the given LLM provider/model.

    Args:
        provider: Provider name or model name from ML prediction.
        model_name: The predicted model name.

    Returns:
        True if an API key is configured for this model's provider.
    """
    # Map common model name patterns to required env vars
    model_lower = model_name.lower()

    if any(k in model_lower for k in ("gpt", "openai", "o1", "o3")):
        return bool(os.getenv("OPENAI_API_KEY"))

    if any(k in model_lower for k in ("claude", "anthropic", "sonnet", "opus", "haiku")):
        return bool(os.getenv("ANTHROPIC_API_KEY"))

    if any(k in model_lower for k in ("gemini", "gemma", "google", "palm")):
        return bool(os.getenv("GOOGLE_API_KEY"))

    if any(k in model_lower for k in ("mistral",)):
        return bool(os.getenv("MISTRAL_API_KEY"))

    if any(k in model_lower for k in ("llama", "meta")):
        # Llama models often served via OpenAI-compatible endpoints
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("TOGETHER_API_KEY"))

    if any(k in model_lower for k in ("qwen",)):
        return bool(os.getenv("OPENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY"))

    # Default: check OpenAI as most common fallback
    return bool(os.getenv("OPENAI_API_KEY"))


def resolve_llm_for_prediction(
    predicted_model: str,
    default_fallback: str = DEFAULT_FALLBACK_LLM,
) -> tuple[str, str]:
    """
    Given an ML-predicted model name, check if its API key is available.
    If not, fall back to default model.

    Args:
        predicted_model: Model name predicted by ML router.
        default_fallback: Default model to use when key is unavailable.

    Returns:
        Tuple of (model_to_use, message).
        message is empty if original model is used, or contains fallback explanation.
    """
    if check_llm_key_available("", predicted_model):
        return predicted_model, ""

    fallback_msg = (
        f"API key not available for '{predicted_model}'. "
        f"Switching to default model '{default_fallback}' for LLM response."
    )
    logger.warning(fallback_msg)
    return default_fallback, fallback_msg
