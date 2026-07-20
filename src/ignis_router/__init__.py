"""
ignis_router - Reusable LLM Routing Library

A centralized routing engine for intelligent model selection across applications.
Supports intent detection, complexity assessment, and rule-based routing.
"""

from .config import RouterConfig, RouterRegistry
from .config_framework import RoutingYamlConfig, load_routing_yaml
from .api import ChatRequest, ChatResponse, ErrorResponse, RouteRequest, RouteResponse, create_app
from .client import ChatResult, IgnisClient, RouteResult
from .llm_client import (
    AnthropicClient,
    BaseLLMClient,
    GeminiClient,
    LLMClientRegistry,
    LLMResponse,
    OpenAIClient,
)
from .exceptions import (
    ConfigurationError,
    IgnisRouterError,
    IntentDetectionError,
    ModelNotAvailableError,
    RoutingError,
)
from .intent_detector import (
    BaseIntentDetector,
    HybridIntentDetector,
    IntentDetector,
    MLIntentDetector,
    RuleBasedIntentDetector,
)
from .intent_detector_factory import IntentDetectorFactory
from .ml_router_adapter import MLRouterAdapter, resolve_llm_for_prediction
from .llmrouter_integration import (
    EmbeddingEngine,
    MLInferenceEngine,
    TrainingPipeline,
    check_llm_key_available,
)
from .decorators import chat, get_shared_router, retry, route, set_shared_router, timed, with_router
from .model_selector import ModelSelector
from .models import (
    Intent,
    ModelCapability,
    ModelConfig,
    RoutingRequest,
    RoutingResult,
    RoutingRule,
    TaskComplexity,
)
from .router import Router
from .routing_engine import RoutingEngine
from .supported_models import get_default_intent_rules, get_default_supported_models

__version__ = "0.1.0"

__all__ = [
    # Main API
    "Router",
    "RoutingEngine",
    "create_app",
    # SDK Client
    "IgnisClient",
    "RouteResult",
    "ChatResult",
    # LLM Clients
    "BaseLLMClient",
    "OpenAIClient",
    "AnthropicClient",
    "GeminiClient",
    "LLMClientRegistry",
    "LLMResponse",
    # Configuration
    "RouterConfig",
    "RouterRegistry",
    "RoutingYamlConfig",
    "load_routing_yaml",
    # Models
    "Intent",
    "ModelCapability",
    "ModelConfig",
    "RoutingRequest",
    "RoutingResult",
    "RoutingRule",
    "TaskComplexity",
    "RouteRequest",
    "RouteResponse",
    "ChatRequest",
    "ChatResponse",
    "ErrorResponse",
    # Components
    "BaseIntentDetector",
    "RuleBasedIntentDetector",
    "MLIntentDetector",
    "HybridIntentDetector",
    "IntentDetector",
    "IntentDetectorFactory",
    "ModelSelector",
    "get_default_supported_models",
    "get_default_intent_rules",
    # Exceptions
    "IgnisRouterError",
    "RoutingError",
    "IntentDetectionError",
    "ModelNotAvailableError",
    "ConfigurationError",
    # Decorators
    "route",
    "chat",
    "with_router",
    "retry",
    "timed",
    "get_shared_router",
    "set_shared_router",
    # LLMRouter Integration (ML inference, embeddings, training)
    "MLInferenceEngine",
    "EmbeddingEngine",
    "TrainingPipeline",
    "check_llm_key_available",
]
