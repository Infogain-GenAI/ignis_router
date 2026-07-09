# Ignis Router Manager File-by-File Guide

## 1) Quick Architecture Summary

Ignis Router is a Python library plus FastAPI service that performs model routing in this order:
1. Build request data model.
2. Detect intent and complexity.
3. Apply strict routing rules first.
4. If no strict rule, score candidate models using weights and metadata.
5. Return selected model, confidence, reasoning, and fallbacks.
6. Optionally expose API responses and persist routing responses to PostgreSQL.

Main orchestration chain:
Router -> RoutingEngine -> IntentDetector + ModelSelector -> RoutingResult

---

## 2) Root-Level Files

### LICENSE
- Purpose: Project legal license (MIT).
- Classes: None.
- Objects/Functions: None.
- Talking point: Defines permissions and warranty disclaimer for using/distributing the code.

### .gitignore
- Purpose: Excludes venv, caches, build artifacts, env files, logs, IDE folders.
- Classes: None.
- Objects/Functions: None.
- Talking point: Keeps repository clean and prevents sensitive/local files from being committed.

### pyproject.toml
- Purpose: Packaging/build metadata, dependencies, tool config.
- Classes: None.
- Objects/Functions: None.
- Key objects:
  - build-system: setuptools backend.
  - project: name/version/dependencies.
  - optional dependencies: dev/openai/anthropic/gemini/all.
  - tool configs: pytest, black, ruff, mypy.
- Talking point: Single source for install/build/test/lint configuration.

### MANIFEST.in
- Purpose: Controls additional files included in source distribution.
- Classes: None.
- Objects/Functions: None.
- Talking point: Ensures README, LICENSE, env example, source, examples, tests are packaged.

### .env.example
- Purpose: Sample runtime environment variables.
- Classes: None.
- Objects/Functions: None.
- Key objects:
  - detector toggles, ML threshold, API port/reload, DB settings.
- Talking point: Copy to .env and edit values for local runtime.

### README.md
- Purpose: End-user guide and setup instructions.
- Classes: None.
- Objects/Functions: None.
- Talking point: Onboarding doc for install, API run, DB logging, and troubleshooting.

### models/knnrouter.pkl
- Purpose: Serialized ML artifact used by MLIntentDetector.
- Classes: Not source code.
- Objects/Functions: Not source code.
- Talking point: Optional ML model; if missing/unusable, system falls back safely.

---

## 3) Config Folder

### configs/balanced.yaml
- Purpose: Balanced routing strategy weights.
- Classes: None.
- Objects:
  - strategy = balanced
  - weights: quality 40, latency 20, cost 20, reliability 20

### configs/cost-first.yaml
- Purpose: Cost-prioritized strategy.
- Objects:
  - strategy = cost-first
  - weights: quality 20, latency 20, cost 45, reliability 15

### configs/quality-first.yaml
- Purpose: Quality-prioritized strategy.
- Objects:
  - strategy = quality-first
  - weights: quality 50, latency 15, cost 10, reliability 25

### configs/latency-first.yaml
- Purpose: Latency-prioritized strategy.
- Objects:
  - strategy = latency-first
  - weights: quality 20, latency 50, cost 15, reliability 15

### configs/postgres_schema.sql
- Purpose: SQL schema for routing response persistence.
- Classes: None.
- Objects:
  - table routing_responses
  - index idx_routing_responses_created_at
- Talking point: Matches structure used by PostgresRouteLogger in persistence.py.

---

## 4) Examples Folder

### examples/basic_routing.py
- Purpose: Standalone demo of direct Router usage.
- Classes: None.
- Functions:
  - main(): builds router, registers models/rules, routes sample queries, prints results.
- Key objects:
  - Multiple ModelConfig objects with capabilities/metadata.
  - One high-priority RoutingRule for creative writing.
- Talking point: Best file to explain manual registration and constrained routing.

### examples/route_with_db.py
- Purpose: Interactive terminal routing + PostgreSQL logging demo.
- Classes: None.
- Functions:
  - _project_root(): resolves repo root.
  - _resolve_from_root(path_value): makes relative paths absolute.
  - build_router(): loads env, yaml config, model path, registers defaults.
  - main(): REPL loop; routes query; prints response; stores row in Postgres.
- Key objects:
  - Router, RouterConfig, PostgresRouteLogger.
- Talking point: End-to-end operational script (routing + persistence).

---

## 5) Package: src/ignis_router

### src/ignis_router/__init__.py
- Purpose: Public package exports and API surface.
- Classes defined here: None.
- Objects:
  - __version__ = 0.1.0
  - __all__ list controls exported symbols.
- Talking point: Re-exports core classes/functions from internal modules for clean imports.

### src/ignis_router/exceptions.py
- Purpose: Custom exception hierarchy.
- Classes:
  - IgnisRouterError: base domain exception.
  - RoutingError: selection/routing failures.
  - IntentDetectionError: intent detection failures.
  - ModelNotAvailableError: no enabled/registered model available.
  - ConfigurationError: invalid configuration.
- Talking point: Centralized domain errors used across engine and API handlers.

### src/ignis_router/models.py
- Purpose: Core pydantic data contracts and enums.
- Enums:
  - TaskComplexity: low, medium, high.
  - Intent: general_chat, code_generation, summarization, reasoning, creative_writing, data_analysis, translation, classification, extraction, custom.
  - ModelCapability: fast_response, high_quality, code, reasoning, creative, multilingual, long_context, cost_effective.
- Classes:
  - ModelConfig: model metadata and routing attributes.
  - RoutingRequest: input query and constraints.
  - RoutingResult: selected model plus diagnostics.
  - RoutingRule: strict matching rule for intent/complexity/capability.
- Key object behavior:
  - priority in ModelConfig is used as a score bonus and fallback sorter.
- Talking point: This file defines the domain language used by all modules.

### src/ignis_router/config_framework.py
- Purpose: YAML strategy schema + loader.
- Module objects:
  - _ALLOWED_STRATEGIES
  - _STRATEGY_DEFAULTS
- Classes:
  - RoutingYamlConfig with validate_configuration() validator.
- Functions:
  - load_routing_yaml(path): parse/validate yaml and return typed config.
- Talking point: Converts yaml strategy files into validated runtime weight config.

### src/ignis_router/config.py
- Purpose: Runtime settings and in-memory registry.
- Classes:
  - RouterConfig (BaseSettings): env-backed settings, strategy/weights, detector toggles, model hint aliases.
  - RouterRegistry: model/rule registration and lookup store.
- RouterConfig key methods:
  - validate_intent_detector_configuration(): validates detector toggles and routing weights.
  - from_yaml(yaml_path, **overrides): creates config from yaml loader.
- RouterRegistry key methods:
  - register_model(s), unregister_model, get_model(s), get_enabled_models.
  - add_rule/remove_rule/get_rules (sorted by rule priority descending).
- Talking point: RouterConfig is policy; RouterRegistry is runtime state.

### src/ignis_router/intent_detector.py
- Purpose: intent and complexity detection strategies.
- Module objects:
  - _INTENT_PATTERNS, _HIGH_COMPLEXITY_PATTERNS, _LOW_COMPLEXITY_PATTERNS.
- Functions:
  - _coerce_intent(raw): converts model output into Intent enum safely.
- Classes:
  - BaseIntentDetector (abstract): detect_intent + shared assess_complexity heuristic.
  - RuleBasedIntentDetector: regex-based detection with optional custom patterns.
  - MLIntentDetector: supports multiple artifact shapes and adapters.
  - HybridIntentDetector: ML first, fallback to rules when unavailable/low confidence.
  - IntentDetector: backward-compatible alias of RuleBasedIntentDetector.
- MLIntentDetector key methods:
  - detect_intent(): inference + confidence extraction + optional model hint capture.
  - _resolve_components(): detect predictor/vectorizer in artifact.
  - _predict_with_adapters(): tries transformed text, raw text, numeric fallback.
  - _safe_load_model(): safe load with non-fatal fallback.
  - get_model_hint()/clear_model_hint().
- HybridIntentDetector key methods:
  - detect_intent(): threshold-based fallback orchestration.
  - get_model_hint(): exposes ML hint to engine.
- Talking point: Most robust compatibility logic lives in this file.

### src/ignis_router/intent_detector_factory.py
- Purpose: Chooses detector implementation from RouterConfig.
- Classes:
  - IntentDetectorFactory.
- Methods:
  - create(config): returns HybridIntentDetector, MLIntentDetector, or RuleBasedIntentDetector.
- Talking point: Keeps detector strategy decision centralized and testable.

### src/ignis_router/model_selector.py
- Purpose: model selection logic (rules first, scoring second).
- Module objects:
  - _INTENT_CAPABILITY_MAP
  - _COMPLEXITY_CAPABILITY_MAP
  - _DEFAULT_WEIGHTS
  - _INTENT_MODEL_PREFERENCES
  - _INTENT_REQUIRED_CAPS
- Classes:
  - ModelSelector.
- Key methods:
  - select(): basic API.
  - select_with_details(): returns model, fallbacks, scoring diagnostics.
  - _match_rules(): strict RoutingRule evaluation first.
  - _get_candidates(): constraint filtering.
  - _desired_capabilities(): expected caps from intent + complexity + request.
  - _score_model(): weighted score including priority bonus.
  - _get_fallbacks(): fallback ordering by provider preference + priority.
- Important formula:
  - score adds model.priority times 5.
- Talking point: This file decides final winner when strict rules do not force a model.

### src/ignis_router/routing_engine.py
- Purpose: main orchestrator from request to routing result.
- Classes:
  - RoutingEngine.
- Key methods:
  - route(request): end-to-end flow with fallback protections.
  - route_simple(query): convenience wrapper.
  - _detect_intent(), _assess_complexity().
  - _extract_model_hint(), _build_ml_predicted_model().
  - _resolve_default_fallback_model(), _build_fallback_models().
  - _build_reasoning().
- Flow behavior:
  - checks enabled models.
  - gets intent/confidence and complexity.
  - optionally uses ML model hint route when enabled.
  - otherwise uses ModelSelector.
  - uses default fallback on selection exceptions/no candidates/low confidence.
  - returns RoutingResult.
- Talking point: Single most important runtime file for explaining execution path.

### src/ignis_router/supported_models.py
- Purpose: builtin model catalog and builtin strict intent rules.
- Functions:
  - get_default_supported_models(): returns list of three default ModelConfig objects.
  - get_default_intent_rules(): returns strict intent-to-model RoutingRule objects.
- Talking point:
  - Model priorities here are defaults, not mandatory framework constants.
  - Rule priorities here are high to enforce intent mappings when registered.

### src/ignis_router/router.py
- Purpose: high-level public facade for users.
- Classes:
  - Router.
- Key methods:
  - register_model, register_supported_models, unregister_model.
  - add_rule, remove_rule, register_default_intent_rules.
  - route(query, **kwargs), route_request(request).
  - get_registered_models, get_enabled_models.
- Talking point: Most apps should use Router directly instead of engine internals.

### src/ignis_router/persistence.py
- Purpose: PostgreSQL persistence utilities.
- Classes:
  - PostgresSettings dataclass: env-driven DB configuration.
  - PostgresRouteLogger: ensure_table and log_response methods.
- Key methods:
  - PostgresSettings.from_env().
  - PostgresRouteLogger.ensure_table().
  - PostgresRouteLogger.log_response().
- Talking point: Isolated persistence component; routing works without DB.

### src/ignis_router/api.py
- Purpose: FastAPI app factory and HTTP contracts.
- Classes:
  - RouteRequest: request payload schema + query cleanup validator.
  - RouteResponse: success response schema.
  - ErrorResponse: standardized error schema.
- Functions:
  - _project_root(), _abs_from_root().
  - build_api_router(): loads env/yaml, configures Router.
  - create_app(router=None): builds FastAPI app, handlers, endpoints.
- Endpoints:
  - GET /
  - GET /health
  - POST /route
  - GET /route
- Talking point: API layer is thin; delegates routing logic to Router.

### src/ignis_router/run_api.py
- Purpose: stable entrypoint for running API server.
- Functions:
  - _resolve_port(), _resolve_reload().
  - _is_port_available(), _is_api_healthy().
  - main(): validates port and runs uvicorn factory app.
- Talking point: Includes conflict checks so startup errors are explicit.

---

## 6) Tests Folder

### tests/__init__.py
- Purpose: package marker for tests.
- Classes: None.
- Objects/Functions: None.

### tests/test_intent_detector.py
- Purpose: unit tests for rule-based, ML, hybrid detectors and complexity.
- Test helper classes:
  - _MockMlModel, _MockVectorizer, _MockFeatureModel, _MockNumericOnlyModel, _MockNumpyProbabilityModel.
- Test classes:
  - TestRuleBasedIntentDetection
  - TestComplexityAssessment
  - TestMlIntentDetection
  - TestHybridIntentDetection
- Talking point: Verifies compatibility paths and fallback behavior thoroughly.

### tests/test_intent_detector_factory.py
- Purpose: tests factory mode selection and env loading into config.
- Test classes:
  - TestIntentDetectorFactory
  - TestConfigurationLoading
- Talking point: Ensures config toggles produce the intended detector strategy.

### tests/test_model_selector.py
- Purpose: tests scoring, constraints, intent preference, and rule precedence.
- Fixtures:
  - registry, selector.
- Test classes:
  - TestModelSelection
  - TestRuleBasedSelection
- Talking point: Validates selection hierarchy (rules over scores) and candidate filtering.

### tests/test_registry.py
- Purpose: tests metadata fields and supported-model registration paths.
- Test classes:
  - TestRegistryMetadata
  - TestSupportedModelRegistration
- Talking point: Confirms registry behavior and Story 2 metadata support.

### tests/test_router.py
- Purpose: tests public Router facade behaviors.
- Fixtures:
  - sample_models, router.
- Test classes:
  - TestRouterBasic
  - TestRouterRouting
  - TestRouterRules
- Talking point: Validates user-facing API, constraints, and default intent rules registration.

### tests/test_routing_engine.py
- Purpose: tests orchestration logic including fallbacks and logging signals.
- Fixtures:
  - registry_with_models, engine.
- Test class:
  - TestRoutingEngine
- Talking point: Covers low-confidence fallback, selector exceptions, and ML hint path behavior.

### tests/test_configuration_framework.py
- Purpose: tests yaml loader validation and weight-driven selection outcomes.
- Test classes:
  - TestYamlConfigurationLoader
  - TestRoutingWeightsBehavior
- Talking point: Confirms strategy profiles materially influence chosen model.

### tests/test_api.py
- Purpose: tests API endpoints, schemas, and exception mapping.
- Helper classes:
  - _SuccessfulRouter
  - _RoutingErrorRouter
  - _ModelUnavailableRouter
- Test class:
  - TestRouterApi
- Talking point: Ensures consistent API contracts for success and error cases.

---

## 7) Generated Packaging Metadata (egg-info)

These files are generated during packaging/install and reflect current distribution metadata.

### src/ignis_router.egg-info/PKG-INFO
- Purpose: package metadata snapshot (name, version, dependencies, extras, URLs).
- Classes: None.
- Objects/Functions: None.

### src/ignis_router.egg-info/requires.txt
- Purpose: dependency and extra groups used in packaged distribution.
- Classes: None.
- Objects/Functions: None.

### src/ignis_router.egg-info/SOURCES.txt
- Purpose: source file list included in distribution.
- Classes: None.
- Objects/Functions: None.

### src/ignis_router.egg-info/dependency_links.txt
- Purpose: legacy dependency link metadata (currently empty/whitespace).
- Classes: None.
- Objects/Functions: None.

### src/ignis_router.egg-info/top_level.txt
- Purpose: top-level import package name list.
- Classes: None.
- Objects/Functions: None.

---

## 8) Manager Q and A Cheat Sheet

### Q: Where is the main routing logic?
A: In src/ignis_router/routing_engine.py method route, which orchestrates detector + selector + fallback.

### Q: Where do we decide model score?
A: In src/ignis_router/model_selector.py method _score_model.

### Q: Where are default models and priorities?
A: In src/ignis_router/supported_models.py inside get_default_supported_models.

### Q: Where are strict intent to model mappings?
A: In src/ignis_router/supported_models.py inside get_default_intent_rules.

### Q: Where is API contract defined?
A: In src/ignis_router/api.py classes RouteRequest, RouteResponse, ErrorResponse.

### Q: Where is DB logging code?
A: In src/ignis_router/persistence.py and usage example in examples/route_with_db.py.

### Q: How are detector modes selected?
A: In src/ignis_router/intent_detector_factory.py based on RouterConfig toggles.

---

## 9) Suggested Walkthrough Order in Meeting

1. models.py (domain types)
2. config.py and config_framework.py (policy + registry)
3. intent_detector.py and intent_detector_factory.py (classification strategy)
4. model_selector.py (rules and scoring)
5. routing_engine.py (end-to-end orchestration)
6. router.py (public API)
7. supported_models.py (defaults)
8. api.py and run_api.py (service runtime)
9. persistence.py and examples/route_with_db.py (DB flow)
10. tests folder (evidence and behavior coverage)

This order helps explain the system from building blocks to runtime behavior.
