"""
Basic example demonstrating ignis_router usage.

This example shows how to:
1. Create a Router instance
2. Register models with capabilities
3. Add routing rules
4. Route queries to the appropriate model
"""

from ignis_router import (
    Intent,
    ModelCapability,
    ModelConfig,
    Router,
    RouterConfig,
    RoutingRule,
)


def main():
    # 1. Create a router with custom config
    config = RouterConfig(
        enable_intent_detection=True,
        enable_complexity_assessment=True,
        fallback_enabled=True,
        confidence_threshold=0.5,
    )
    router = Router(config=config)

    # 2. Register models
    router.register_model(
        ModelConfig(
            model_id="gpt-4o",
            provider="openai",
            model_name="gpt-4o",
            capabilities=[
                ModelCapability.HIGH_QUALITY,
                ModelCapability.REASONING,
                ModelCapability.CODE,
                ModelCapability.LONG_CONTEXT,
            ],
            max_tokens=128000,
            cost_per_1k_input_tokens=0.005,
            cost_per_1k_output_tokens=0.015,
            priority=10,
        )
    )

    router.register_model(
        ModelConfig(
            model_id="gpt-4o-mini",
            provider="openai",
            model_name="gpt-4o-mini",
            capabilities=[
                ModelCapability.FAST_RESPONSE,
                ModelCapability.COST_EFFECTIVE,
                ModelCapability.CODE,
            ],
            max_tokens=128000,
            cost_per_1k_input_tokens=0.00015,
            cost_per_1k_output_tokens=0.0006,
            priority=5,
        )
    )

    router.register_model(
        ModelConfig(
            model_id="claude-sonnet",
            provider="anthropic",
            model_name="claude-3-5-sonnet-20241022",
            capabilities=[
                ModelCapability.HIGH_QUALITY,
                ModelCapability.REASONING,
                ModelCapability.CREATIVE,
                ModelCapability.CODE,
            ],
            max_tokens=200000,
            cost_per_1k_input_tokens=0.003,
            cost_per_1k_output_tokens=0.015,
            priority=8,
        )
    )

    router.register_model(
        ModelConfig(
            model_id="gemini-pro",
            provider="gemini",
            model_name="gemini-1.5-pro",
            capabilities=[
                ModelCapability.HIGH_QUALITY,
                ModelCapability.MULTILINGUAL,
                ModelCapability.LONG_CONTEXT,
                ModelCapability.REASONING,
            ],
            max_tokens=1000000,
            cost_per_1k_input_tokens=0.00125,
            cost_per_1k_output_tokens=0.005,
            priority=7,
        )
    )

    # 3. Add optional routing rules (override automatic selection)
    router.add_rule(
        RoutingRule(
            rule_id="creative-to-claude",
            intent=Intent.CREATIVE_WRITING,
            target_model_id="claude-sonnet",
            priority=100,
        )
    )

    # 4. Route various queries
    queries = [
        "Hello, how are you today?",
        "Write a Python function to implement binary search",
        "Explain step by step how transformers work in neural networks",
        "Write a short story about AI becoming sentient",
        "Summarize the key points of this research paper",
        "Translate this paragraph to French",
    ]

    print("=" * 70)
    print("ignis_router - LLM Routing Demo")
    print("=" * 70)

    for query in queries:
        result = router.route(query)
        print(f"\nQuery: {query[:60]}...")
        print(f"  Intent:     {result.detected_intent.value}")
        print(f"  Complexity: {result.complexity.value}")
        print(f"  Model:      {result.selected_model.model_name} ({result.selected_model.provider})")
        print(f"  Confidence: {result.confidence:.2f}")
        print(f"  Fallbacks:  {[m.model_name for m in result.fallback_models]}")
        print(f"  Reasoning:  {result.reasoning}")

    # 5. Route with constraints
    print("\n" + "=" * 70)
    print("Routing with constraints")
    print("=" * 70)

    result = router.route(
        "Analyze this data",
        max_cost=0.002,
        preferred_provider="openai",
    )
    print(f"\nQuery: 'Analyze this data' (max_cost=0.002, preferred=openai)")
    print(f"  Selected: {result.selected_model.model_name}")
    print(f"  Cost: ${result.selected_model.cost_per_1k_input_tokens}/1K tokens")


if __name__ == "__main__":
    main()
