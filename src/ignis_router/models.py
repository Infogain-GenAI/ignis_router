"""Data models for the ignis_router library."""

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class TaskComplexity(str, Enum):
    """Complexity levels for incoming tasks."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Intent(str, Enum):
    """Supported intent categories for routing decisions."""

    GENERAL_CHAT = "general_chat"
    CODE_GENERATION = "code_generation"
    SUMMARIZATION = "summarization"
    REASONING = "reasoning"
    CREATIVE_WRITING = "creative_writing"
    DATA_ANALYSIS = "data_analysis"
    TRANSLATION = "translation"
    CLASSIFICATION = "classification"
    EXTRACTION = "extraction"
    CUSTOM = "custom"


class ModelCapability(str, Enum):
    """Capabilities that a model can have."""

    FAST_RESPONSE = "fast_response"
    HIGH_QUALITY = "high_quality"
    CODE = "code"
    REASONING = "reasoning"
    CREATIVE = "creative"
    MULTILINGUAL = "multilingual"
    LONG_CONTEXT = "long_context"
    COST_EFFECTIVE = "cost_effective"


class ModelConfig(BaseModel):
    """Configuration for a registered model."""

    model_id: str = Field(..., description="Unique identifier for the model")
    provider: str = Field(..., description="Provider name (e.g., openai, anthropic, gemini)")
    model_name: str = Field(..., description="Model name as recognized by the provider")
    capabilities: list[ModelCapability] = Field(
        default_factory=list, description="List of model capabilities"
    )
    max_tokens: int = Field(default=4096, description="Maximum token limit")
    cost_per_1k_input_tokens: float = Field(
        default=0.0, description="Cost per 1K input tokens in USD"
    )
    cost_per_1k_output_tokens: float = Field(
        default=0.0, description="Cost per 1K output tokens in USD"
    )
    latency: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized latency score (higher means faster response)",
    )
    quality: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized output quality score",
    )
    reliability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Normalized reliability score",
    )
    priority: int = Field(
        default=0, description="Priority for selection (higher = preferred)"
    )
    enabled: bool = Field(default=True, description="Whether this model is currently enabled")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RoutingRequest(BaseModel):
    """A request to be routed to an appropriate model."""

    query: str = Field(..., description="The user query/prompt to route")
    context: Optional[dict[str, Any]] = Field(
        default=None, description="Additional context for routing"
    )
    preferred_provider: Optional[str] = Field(
        default=None, description="Preferred provider if any"
    )
    max_cost: Optional[float] = Field(
        default=None, description="Maximum acceptable cost per request"
    )
    required_capabilities: list[ModelCapability] = Field(
        default_factory=list, description="Required model capabilities"
    )


class RoutingResult(BaseModel):
    """Result of a routing decision."""

    selected_model: ModelConfig = Field(..., description="The selected model configuration")
    detected_intent: Intent = Field(..., description="The detected intent")
    complexity: TaskComplexity = Field(..., description="Assessed task complexity")
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence score of the routing decision"
    )
    reasoning: str = Field(default="", description="Explanation for the routing decision")
    scoring_details: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured scoring details used for model selection",
    )
    fallback_models: list[ModelConfig] = Field(
        default_factory=list, description="Alternative models if primary fails"
    )


class RoutingRule(BaseModel):
    """A rule that maps intents/conditions to model selections."""

    rule_id: str = Field(..., description="Unique rule identifier")
    intent: Optional[Intent] = Field(default=None, description="Intent to match")
    complexity: Optional[TaskComplexity] = Field(default=None, description="Complexity to match")
    required_capabilities: list[ModelCapability] = Field(
        default_factory=list, description="Required capabilities"
    )
    target_model_id: str = Field(..., description="Model ID to route to")
    priority: int = Field(default=0, description="Rule priority (higher = checked first)")
    enabled: bool = Field(default=True, description="Whether this rule is active")
