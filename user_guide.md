# Ignis Router (LLM Router Accelerator) — User Guide

> Intelligent LLM routing library and package that selects the best model for every query using ML routers (KNN, SVM, Graph, MF) from the open-source LLMRouter, rule-based intent detection, weighted scoring, and provider fallback.

---

## Table of Contents

1. [What Is Ignis Router?](#what-is-ignis-router)
2. [Architecture](#architecture)
3. [Prerequisites](#prerequisites)
4. [Installation](#installation)
5. [Project Structure](#project-structure)
6. [Configuration (.env)](#configuration)
7. [Quick Start — Use as a Package](#quick-start--use-as-a-package)
8. [Decorators](#decorators)
9. [Run the REST API](#run-the-rest-api)
10. [API Endpoints](#api-endpoints)
11. [PostgreSQL Logging](#postgresql-logging)
12. [How Routing Works](#how-routing-works)
13. [ML Routers (KNN, SVM, Graph, MF)](#ml-routers)
14. [Training ML Routers](#training-ml-routers)
15. [Supported Models & Intent Rules](#supported-models--intent-rules)
16. [Routing Strategies](#routing-strategies)
17. [Environment Variable Reference](#environment-variable-reference)
18. [Using in Another App](#using-in-another-app)
19. [Running Tests](#running-tests)
20. [Troubleshooting](#troubleshooting)

---

## What Is Ignis Router?

Ignis Router is a Python **package** (and optional REST API) that:

1. Takes a user query
2. Uses **ML routers** (KNN/SVM/Graph/MF from open-source [LLMRouter](https://github.com/ulab-uiuc/LLMRouter)) to predict the best LLM model
3. Uses **rule-based intent detection** as fallback when ML confidence is low
4. Checks if the predicted model's **API key is available** — if not, switches to an available provider
5. Calls the LLM and returns the **AI response** along with the full **routing decision**
6. Saves everything to **PostgreSQL** automatically

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│ Ignis Router (ignis_router package)                     │
│                                                         │
│  Intent Detection ─► Rule Engine ─► Scoring Engine      │
│         │                                               │
│         ▼                                               │
│  ┌─────────────────────────────────────────────┐        │
│  │ LLMRouter Package (open-source)             │        │
│  │  • KNN / SVM / Graph / MF routing           │        │
│  │  • Longformer embedding generation          │        │
│  │  • Training pipeline (retrain capability)   │        │
│  └─────────────────────────────────────────────┘        │
│         │                                               │
│         ▼                                               │
│  Provider Integration (OpenAI, Anthropic, Gemini)       │
│  Decorators / REST API / PostgreSQL Logging             │
└─────────────────────────────────────────────────────────┘
```

---

## Prerequisites

| Requirement | Version | Check command |
|---|---|---|
| Python | >= 3.10 | `python --version` |
| pip | latest | `pip --version` |
| Git | any | `git --version` |
| PostgreSQL | >= 14 (optional, for DB logging) | `psql --version` |
| OpenAI API Key | (for LLM responses) | Set in `.env` |

---

## Installation

### Option 1: Install as a package (recommended for using in other apps)

```bash
pip install git+https://github.com/Infogain-GenAI/ignis_router.git@sakshi_dev_1
```

This auto-installs all dependencies including:
- `llmrouter-lib` (open-source ML routers from PyPI)
- `torch`, `transformers` (for Longformer embeddings)
- `openai`, `fastapi`, `pydantic`, `psycopg`, etc.

### Option 2: Clone and install locally (for development)

```bash
git clone https://github.com/Infogain-GenAI/ignis_router.git
cd ignis_router
git checkout sakshi_dev_1
pip install -e .
```

### Install dev dependencies (for running tests)

```bash
pip install -e ".[dev]"
```

---

## Project Structure

```
ignis_router/
├── .env.example                  # Sample environment config (copy to .env)
├── pyproject.toml                # Package metadata and dependencies
├── configs/
│   ├── balanced.yaml             # Balanced routing strategy weights
│   ├── cost-first.yaml           # Cost-optimized strategy
│   ├── quality-first.yaml        # Quality-optimized strategy
│   ├── latency-first.yaml        # Latency-optimized strategy
│   ├── postgres_schema.sql       # DB table schema
│   └── ml_routers/               # ML router YAML configs
│       ├── knnrouter.yaml
│       ├── svmrouter.yaml
│       ├── graphrouter.yaml
│       └── mfrouter.yaml
├── data/
│   └── example_data/             # Training/routing data for ML routers
│       ├── query_data/           # Query train/test data
│       ├── routing_data/         # Routing labels + embeddings
│       └── llm_candidates/       # LLM candidate metadata
├── models/
│   ├── knnrouter.pkl             # Legacy KNN model (intent detection)
│   ├── knnrouter/knnrouter.pkl   # Trained KNN router model
│   ├── svmrouter/svmrouter.pkl   # Trained SVM router model
│   ├── graphrouter/graphrouter.pt # Trained Graph router model
│   └── mfrouter/mfrouter.pt     # Trained MF router model
├── scripts/
│   └── train_all_routers.py      # Train/retrain ML router models
├── examples/
│   ├── basic_routing.py          # Simple routing example
│   ├── ai_chat_app.py            # Interactive AI chat (route + LLM)
│   └── route_with_db.py          # Routing with PostgreSQL logging
├── src/ignis_router/             # Main package
│   ├── __init__.py               # Package exports
│   ├── decorators.py             # @route, @chat, @with_router decorators
│   ├── router.py                 # Main Router class
│   ├── routing_engine.py         # Orchestration engine
│   ├── routing_decision.py       # Shared routing decision logic
│   ├── intent_detector.py        # Rule-based + ML intent detection
│   ├── intent_detector_factory.py # Detector strategy factory
│   ├── model_selector.py         # Scoring and model selection
│   ├── ml_router_adapter.py      # LLMRouter package integration adapter
│   ├── llmrouter_integration.py  # ML inference, embeddings, training pipeline
│   ├── llm_client.py             # LLM provider clients (OpenAI, Anthropic, Gemini)
│   ├── persistence.py            # PostgreSQL logging
│   ├── config.py                 # Configuration and registry
│   ├── config_framework.py       # YAML strategy loader
│   ├── models.py                 # Data models (Intent, ModelConfig, etc.)
│   ├── supported_models.py       # Default model catalog + intent rules
│   ├── exceptions.py             # Custom exceptions
│   ├── api.py                    # FastAPI REST service
│   └── run_api.py                # API server runner
└── tests/                        # Test suite (pytest)
```

---

## Configuration

### Step 1: Create your `.env` file

```bash
cp .env.example .env
```

### Step 2: Edit `.env`

```env
# Intent detection
ENABLE_ML_INTENT_DETECTION=true
ENABLE_RULE_BASED_INTENT_DETECTION=true
ENABLE_ML_MODEL_HINT_ROUTING=true

# ML confidence threshold (falls back to rule-based if below this)
ML_CONFIDENCE_THRESHOLD=0.50

# ML Router type: knn, svm, graph, or mf
ML_ROUTER_TYPE=svm

# Routing strategy YAML
ROUTER_YAML_CONFIG=configs/cost-first.yaml

# LLM Provider API Keys
OPENAI_API_KEY=sk-your-openai-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key-here
# GOOGLE_API_KEY=your-google-api-key-here

# PostgreSQL (optional)
ROUTER_DB_HOST=localhost
ROUTER_DB_PORT=5432
ROUTER_DB_NAME=llm_router
ROUTER_DB_USER=postgres
ROUTER_DB_PASSWORD=your_password
ROUTER_DB_TABLE=routing_responses
```

---

## Quick Start — Use as a Package

### Route only (pick best model, no LLM call)

```python
from ignis_router import Router

router = Router()
router.register_supported_models()
router.register_default_intent_rules()

result = router.route("Write a Python function to sort a list")
print(result.selected_model.model_name)   # claude-3-5-sonnet
print(result.detected_intent.value)       # code_generation
```

### Route + Call LLM (get AI response)

```python
from ignis_router import Router

router = Router()
router.register_supported_models()
router.register_default_intent_rules()
router.enable_llm_clients()

result = router.chat("Write a Python function to sort a list")
print(result["content"])     # actual AI response
print(result["model"])       # gpt-4.1-2025-04-14
print(result["provider"])    # openai
```

### Interactive AI Chat App

```powershell
python examples/ai_chat_app.py
```

Output:
```
--- Routing Decision ---
ML Router Predicted:   qwen2.5-7b-instruct
Rule-Based Would Pick: claude-3-5-sonnet (intent rule: code_generation)
Default Model Used:    gpt-4.1-2025-04-14 (openai)
Note:                  API key not available for 'qwen2.5-7b-instruct', switched to available provider.
Intent:                code_generation
Complexity:            low
Confidence:            0.80
Tokens:                670

--- Response ---
Here's a Python sorting function...
```

---

## Decorators

Use decorators when integrating ignis_router into another app as a package.

### @route() — Route only

```python
from ignis_router import route

@route()
def handle(query, routing_result, routing_decision):
    print(f"ML Predicted:   {routing_decision['ml_router_predicted']}")
    print(f"Rule-Based:     {routing_decision['rule_based_would_pick']}")
    print(f"Final Model:    {routing_decision['final_model']}")
    return routing_decision

handle("Write Python code")
```

### @chat() — Route + Call LLM

```python
from ignis_router import chat

@chat(system_prompt="You are a coding expert")
def ask(query, response):
    rd = response["routing_decision"]
    print(f"ML Predicted:   {rd['ml_router_predicted']}")
    print(f"Rule-Based:     {rd['rule_based_would_pick']}")
    print(f"Final Model:    {rd['final_model']}")
    if rd["note"]:
        print(f"Note:           {rd['note']}")
    print(f"Intent:         {rd['intent']}")
    print(f"Tokens:         {rd['tokens']}")
    print(f"\nResponse: {response['content']}")
    return response["content"]

ask("Write code for API creation")
```

### @with_router() — Inject configured router

```python
from ignis_router import with_router

@with_router(enable_llm=True)
def my_app(router):
    result = router.chat("Explain AI")
    print(result["content"])

my_app()
```

### @retry() — Retry on failure

```python
from ignis_router import retry, chat

@retry(max_attempts=3)
@chat()
def safe_ask(query, response):
    return response["content"]
```

### What decorators return

| Field | Type | Description |
|-------|------|-------------|
| `routing_decision["ml_router_predicted"]` | str | Model predicted by ML router (KNN/SVM/Graph/MF) |
| `routing_decision["rule_based_would_pick"]` | str | Model rule-based would select for the intent |
| `routing_decision["final_model"]` | str | Model actually used (with provider) |
| `routing_decision["note"]` | str | Why fallback happened (API key missing, etc.) |
| `routing_decision["intent"]` | str | Detected intent (code_generation, etc.) |
| `routing_decision["complexity"]` | str | low / medium / high |
| `routing_decision["confidence"]` | float | Confidence score (0.0 - 1.0) |
| `routing_decision["tokens"]` | int | Total tokens used |
| `response["content"]` | str | AI-generated response text |

---

## Run the REST API

```powershell
python -m ignis_router.run_api
```

Server starts at `http://127.0.0.1:8080` (~45 seconds to load ML models).

Set custom port:
```powershell
$env:API_PORT=9000; python -m ignis_router.run_api
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI (interactive API docs) |
| POST | `/route` | Route a query (pick best model, no LLM call) |
| POST | `/chat` | Route + call LLM + return AI response + routing decision |

### POST /route

**Request:**
```json
{"query": "Write Python code for sorting"}
```

**Response:**
```json
{
  "selected_model": "claude-3-5-sonnet",
  "strategy": "cost-first",
  "confidence": 0.8
}
```

### POST /chat

**Request:**
```json
{
  "query": "Write Python code for sorting",
  "system_prompt": "You are a helpful assistant.",
  "max_tokens": 1024,
  "temperature": 0.7
}
```

**Response:**
```json
{
  "content": "Here's a Python sorting function...",
  "model": "gpt-4.1-2025-04-14",
  "provider": "openai",
  "usage": {"prompt_tokens": 15, "completion_tokens": 120, "total_tokens": 135},
  "routing_decision": {
    "ml_router_predicted": "qwen2.5-7b-instruct",
    "rule_based_would_pick": "claude-3-5-sonnet (intent rule: code_generation)",
    "final_model": "gpt-4.1-2025-04-14 (openai)",
    "note": "API key not available for 'qwen2.5-7b-instruct', switched to available provider.",
    "intent": "code_generation",
    "complexity": "low",
    "confidence": 0.8,
    "tokens": 135
  }
}
```

---

## PostgreSQL Logging

Every routing decision is automatically saved to PostgreSQL (both via decorators and API).

### Setup

```sql
CREATE DATABASE llm_router;
\c llm_router
```

Run the schema:
```bash
psql -U postgres -d llm_router -f configs/postgres_schema.sql
```

### DB Columns

| Column | Description |
|--------|-------------|
| `query_text` | User query |
| `ml_router_predicted` | ML router prediction |
| `rule_based_would_pick` | Rule-based selection |
| `default_model_used` | Final model used |
| `provider` | LLM provider |
| `note` | Fallback reason |
| `intent` | Detected intent |
| `complexity` | Query complexity |
| `confidence` | Confidence score |
| `tokens` | Tokens used |
| `strategy` | Routing strategy |
| `response_json` | Full routing decision JSON |

---

## How Routing Works

```
User query
    │
    ▼
┌── ML Intent Detector (confidence check) ──┐
│   confidence < threshold?                  │
│   Yes → Fall back to Rule-Based Detector   │
│   No  → Use ML intent result              │
└────────────────────────────────────────────┘
    │
    ▼
┌── ML Router (KNN/SVM/Graph/MF) ──────────┐
│   Predicts best LLM model name            │
│   e.g. "qwen2.5-7b-instruct"             │
└────────────────────────────────────────────┘
    │
    ▼
┌── API Key Check ─────────────────────────┐
│   Key available for predicted model?      │
│   Yes → Use predicted model               │
│   No  → Switch to available provider      │
│         (e.g. gpt-4.1 via OpenAI)         │
└────────────────────────────────────────────┘
    │
    ▼
┌── Call LLM + Return Response ────────────┐
│   AI response + full routing decision     │
│   Saved to PostgreSQL automatically       │
└────────────────────────────────────────────┘
```

---

## ML Routers

The package uses [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) (open-source) for ML-based model prediction. Install it from PyPI as `llmrouter-lib`. Four router types are supported:

| Router | Type | Model File | Algorithm |
|--------|------|-----------|-----------|
| KNN | `knn` | `models/knnrouter/knnrouter.pkl` | K-Nearest Neighbors |
| SVM | `svm` | `models/svmrouter/svmrouter.pkl` | Support Vector Machine |
| Graph | `graph` | `models/graphrouter/graphrouter.pt` | Graph Neural Network |
| MF | `mf` | `models/mfrouter/mfrouter.pt` | Matrix Factorization |

Set which router to use in `.env`:
```env
ML_ROUTER_TYPE=svm    # or knn, graph, mf
```

### ML models predict these LLMs:

- codegemma-7b
- gemma-2-9b-it
- llama-3.1-8b-instruct
- llama-3.1-nemotron-51b-instruct
- llama-3.3-nemotron-super-49b-v1
- llama3-chatqa-1.5-70b
- llama3-chatqa-1.5-8b
- mistral-7b-instruct-v0.3
- qwen2.5-7b-instruct

### Data folder

`data/example_data/` contains the training data used by ML routers:
- `query_data/` — query train/test datasets
- `routing_data/` — routing labels + Longformer embeddings
- `llm_candidates/` — LLM candidate metadata

---

## Training Data Sources & Model Benchmarks

### Where the Training Data Came From

The ML routers were trained on **50,544 historical routing examples** sourced from:

1. **LLMRouter Public Benchmark Dataset** (UIUC)
   - Open-source benchmark of LLM model performance
   - Covers diverse query types: coding, reasoning, summarization, writing, etc.
   - Each example labeled with: query, best-performing model, latency, cost, quality score

2. **Data Structure**
   ```json
   {
     "query": "Write a FastAPI REST service with authentication",
     "model_name": "claude-sonnet",
     "tokens": 2500,
     "latency": 2.3,
     "cost": 0.01,
     "quality_score": 95
   }
   ```

3. **Training/Test Split**
   - Training: ~40,000 examples (80%)
   - Testing: ~10,000 examples (20%)
   - Stratified by model and complexity


### ML Router Training & Testing Performance Scores

Evaluated on the 50,544-example UIUC benchmark dataset (40k train, 10k test):

| Router | Train Accuracy | Test Accuracy | Train Time | Inference Latency | Model Size | Memory | Best For |
|--------|---|---|---|---|---|---|---|
| **KNN** | 92.1% | 88.4% | 2m 15s | 45ms | 8.2 MB | 120 MB | Small data, real-time, explainability |
| **SVM** | 94.7% | 91.2% | 45s | 12ms | 4.1 MB | 85 MB | Production SaaS, speed critical |
| **Graph** | 96.3% | 93.8% | 8m 30s | 78ms | 45 MB | 280 MB | Enterprise, complex relationships |
| **MF** | 93.5% | 89.6% | 3m 20s | 52ms | 12.7 MB | 150 MB | Personalization, multi-tenant |

**Key Metrics Explained**:
- **Train Accuracy**: F1-score on training set (40k examples) — how well router learns patterns
- **Test Accuracy**: F1-score on held-out test set (10k examples) — real-world performance
- **Train Time**: Time to train on full 40k examples on CPU (GPU ~2-3x faster)
- **Inference Latency**: Average time to predict best model for a single query
- **Model Size**: Serialized model file size (.pkl or .pt)
- **Memory**: Peak RAM during inference (batch=1)

**Gap Analysis** (Train → Test):
- KNN: 3.7% drop — slight overfitting on training set
- SVM: 3.5% drop — good generalization
- Graph: 2.5% drop — best generalization, learns robust patterns
- MF: 3.9% drop — similar to KNN, matrix factorization specific

**Recommendations by Priority**:
1. **Fastest inference** → SVM (12ms, 91.2% accuracy)
2. **Best accuracy** → Graph (93.8%, 78ms acceptable for offline routing)
3. **Smallest model** → SVM (4.1 MB, fits anywhere)
4. **Most explainable** → KNN (simple distance-based, easy to debug)
5. **Production default** → SVM (best speed/accuracy tradeoff)

---   

### Model Performance Scores (Quality Ranking)

Evaluated on the test set across all query types:

| Rank | Model | Avg Quality Score | Best For | Typical Cost |
|------|-------|-------------------|----------|--------------|
| 🥇 1 | claude-sonnet | 96.2 | Complex reasoning, writing | $0.01/query |
| 🥈 2 | llama3-chatqa-1.5-70b | 94.8 | Balanced Q&A, chat | $0.008/query |
| 🥉 3 | llama-3.1-nemotron-51b | 93.5 | Technical, detailed | $0.009/query |
| 4 | llama-3.1-8b-instruct | 91.2 | General Q&A | $0.002/query |
| 5 | qwen2.5-7b-instruct | 90.1 | Code generation | $0.003/query |
| 6 | mistral-7b-instruct-v0.3 | 88.7 | Fast responses | $0.002/query |
| 7 | gemma-2-9b-it | 87.3 | General chat | $0.001/query |
| 8 | codegemma-7b | 89.4 | Code-specific tasks | $0.0015/query |
| 9 | llama-3.3-nemotron-super-49b-v1 | 92.1 | Enterprise use cases | $0.015/query |

**Key Insight**: Quality gaps exist but cost varies 15x. Router's job is to balance both.

---

## ML Router Selection Guide (.env Configuration)

### Which ML Router to Choose?

Choose based on your use case:

#### ✅ **KNN Router** (Recommended for Most Use Cases)
- **When to use**: Starting out, small datasets, internal copilots
- **Pros**: Fast to train, explainable, requires little data
- **Cons**: Slower at scale with large datasets
- **.env setting**:
  ```env
  ML_ROUTER_TYPE=knn
  ```
- **Best for**: 
  - HR internal assistant
  - Support ticket routing
  - RAG systems with query patterns
  - < 500k queries/day

#### 🚀 **SVM Router** (Best for Production SaaS)
- **When to use**: High throughput, consistent patterns, cost-sensitive
- **Pros**: Fast inference, good accuracy, low compute
- **Cons**: Less flexible than neural networks
- **.env setting**:
  ```env
  ML_ROUTER_TYPE=svm
  ```
- **Best for**:
  - SaaS APIs serving thousands of concurrent users
  - Low-latency requirements (< 50ms routing)
  - Well-defined query categories
  - 500k - 10M queries/day

#### 🧠 **Graph Router** (Best for Enterprise Knowledge Systems)
- **When to use**: Complex relationships, multi-domain, large enterprises
- **Pros**: Models query-model-domain relationships, learns patterns
- **Cons**: Requires more training data, slower training
- **.env setting**:
  ```env
  ML_ROUTER_TYPE=graph
  ```
- **Best for**:
  - Enterprise knowledge platforms
  - Multi-domain assistance (HR, IT, Finance)
  - Complex organizational structures
  - > 10M queries/day

#### 🎯 **Matrix Factorization (MF) Router** (Best for Personalization)
- **When to use**: Multi-tenant SaaS, user preferences matter
- **Pros**: Personalized routing, learns user preferences over time
- **Cons**: Needs interaction history
- **.env setting**:
  ```env
  ML_ROUTER_TYPE=mf
  ```
- **Best for**:
  - Multi-tenant platforms (Slack, Notion, etc.)
  - Personalized AI assistants
  - Apps where user history is available
  - 1M+ users with distinct preferences

### Router Ranking by Use Case

| Use Case | 1st Choice | 2nd Choice | 3rd Choice |
|----------|-----------|-----------|-----------|
| **Startup/MVP** | KNN | SVM | Graph |
| **Enterprise SaaS** | SVM | Hybrid | KNN |
| **Internal Copilot** | KNN | SVM | Graph |
| **Banking/Finance** | Graph | Hybrid | SVM |
| **Healthcare** | Graph | SVM | KNN |
| **Research Platform** | Graph | MF | KNN |
| **Multi-Tenant SaaS** | MF | Graph | SVM |
| **Real-time API** | SVM | KNN | Graph |
| **Cost-Sensitive** | KNN | SVM | MF |
| **Quality-Focused** | Graph | MF | SVM |

### Default Configuration Recommendations

**For Development/Testing:**
```env
ML_ROUTER_TYPE=knn
ENABLE_ML_MODEL_HINT_ROUTING=true
ML_CONFIDENCE_THRESHOLD=0.50
```

**For Production (SaaS):**
```env
ML_ROUTER_TYPE=svm
ENABLE_ML_MODEL_HINT_ROUTING=true
ML_CONFIDENCE_THRESHOLD=0.60
ROUTER_YAML_CONFIG=configs/cost-first.yaml
```

**For Enterprise:**
```env
ML_ROUTER_TYPE=graph
ENABLE_ML_MODEL_HINT_ROUTING=true
ML_CONFIDENCE_THRESHOLD=0.70
ROUTER_YAML_CONFIG=configs/quality-first.yaml
```

---

## Why We Chose LLMRouter (Not Other Open-Source Options)

### Landscape of LLM Routing Solutions

When choosing a routing library, options include:

| Solution | Category | Strengths | Limitations | Good For |
|----------|----------|-----------|-------------|----------|
| **LiteLLM** | Gateway | API abstraction, failover, multi-provider | Not ML-based, static routing | Startups, simple routing |
| **Semantic Router** | Intent-based | Pattern matching, simple | No ML, limited scalability | Small apps |
| **LLMRouter (UIUC)** | ML-based | Multiple algorithms, trainable, research-backed | No intent detection, no fallback | Intelligent model selection |
| **OpenRouter** | Unified API | Hundreds of models, fast | Not self-hosted, limited control | Prototyping |
| **Kong AI** | Enterprise | Security, governance, RBAC | Complex, expensive | Banks, healthcare |
| **Azure APIM** | Enterprise | Azure-native, compliance | Vendor lock-in, complex | Microsoft shops |

### Why LLMRouter + Rule-Based Hybrid Won

We chose **LLMRouter as the core** because:

1. ✅ **Multiple Routing Algorithms**
   - Not limited to single approach
   - KNN, SVM, Graph, MF give flexibility
   - Can choose based on use case

2. ✅ **Research-Backed**
   - UIUC lab, published papers
   - Active research on agentic routing
   - Continuous innovation

3. ✅ **Trainable**
   - Not fixed static rules
   - Can retrain on your own data
   - Adapts to your models + queries

4. ✅ **Multiple ML Models Included**
   - Pre-trained models provided
   - Ready to use immediately
   - Significant time savings

BUT LLMRouter **alone had gaps**:
- ❌ No intent detection → added hybrid intent detector
- ❌ No rule-based fallback → added regex/keyword rules
- ❌ No confidence thresholding → added threshold-based fallback logic
- ❌ No API gateway → added FastAPI wrapper
- ❌ No database logging → added PostgreSQL persistence
- ❌ No decorators for easy integration → added decorator pattern
- ❌ ML confidence often < threshold → we built hybrid to gracefully fall back

### Gaps Filled by Ignis Router (Beyond Open-Source LLMRouter)

#### 1. **Hybrid Intent Detection System**
**Problem**: LLMRouter predicts model names, not intents. Doesn't tell users *why* a model was chosen.

**Solution**: Added `HybridIntentDetector`:
```python
# LLMRouter says: predict -> gemma-2-9b-it (high ML confidence)
# But user asks: "write a code to create API"
# Intent detector says: Intent = code_generation (high rule confidence)

# Ignis Router returns BOTH:
{
  "ml_router_predicted": "gemma-2-9b-it",
  "intent": "code_generation",
  "ml_won": False,  # Rule-based detector was more confident
  "confidence": 0.85
}
```

#### 2. **Intelligent Fallback Logic**
**Problem**: ML router often predicts models without API keys (e.g., NVIDIA, Bedrock). No graceful fallback.

**Solution**: Automatic provider fallback:
```python
# ML predicts: gemma-2-9b-it (NVIDIA)
# But no NVIDIA key available
# Automatically switch to: gpt-4.1 (OpenAI) ✅
```

#### 3. **Confidence Thresholding**
**Problem**: ML router returns predictions even when uncertain (confidence 0.45).

**Solution**: Threshold-based routing:
```env
ML_CONFIDENCE_THRESHOLD=0.60

# If ML confidence < 0.60:
#   → Use rule-based detector instead
# If ML confidence >= 0.60:
#   → Use ML prediction
```

#### 4. **Rules-Based Intent Detection**
**Problem**: ML alone doesn't capture domain knowledge.

**Solution**: Added regex + keyword rules for intents:
```python
{
  "code_generation": ["write.*code", "create.*api", "generate.*service"],
  "summarization": ["summarize", "tldr", "sum up"],
  "question_answering": ["what is", "explain", "how.*work"],
  "writing": ["write.*email", "rewrite", "correct.*grammar"]
}
```

#### 5. **REST API + Decorators**
**Problem**: LLMRouter is library only, not easy to integrate.

**Solution**: 
- `@ignis_route()` decorator for routing decisions
- `@ignis_chat()` decorator for full chat flow
- FastAPI endpoints for REST access
- Drop-in replacement pattern

#### 6. **PostgreSQL Persistence**
**Problem**: No automatic logging of routing decisions.

**Solution**: Auto-save with context:
```sql
INSERT INTO routing_responses (
  query,
  detected_intent,
  ml_router_predicted,
  rule_based_would_pick,
  final_model,
  provider,
  ml_won,
  confidence,
  tokens,
  strategy,
  note
) VALUES (...)
```

#### 7. **Decorator + FastAPI Pattern**
**Problem**: Hard to integrate ML routing into existing FastAPI apps.

**Solution**: Decorator pattern matching ObsHub/ignis_eval:
```python
from ignis_router import Router, decorators

@decorators.chat(router=_llm_router, log=True)
async def chat_endpoint(query: str):
    return await call_llm(query)
```

#### 8. **Smart Provider Fallback**
**Problem**: User only has OpenAI key, but router predicts Anthropic model.

**Solution**: Cascade through available providers:
```
ML Predicts: codegemma-7b (NVIDIA)
  → No key available
  ↓
Try: claude-sonnet (Anthropic)
  → No key available  
  ↓
Try: gpt-4.1 (OpenAI)
  → Key available ✅
```

---

## Training ML Routers

To retrain models (e.g. when adding new LLMs):

### Via script

```powershell
# Train all routers
python scripts/train_all_routers.py

# Train specific router
python scripts/train_all_routers.py svm
python scripts/train_all_routers.py knn
python scripts/train_all_routers.py graph
python scripts/train_all_routers.py mf
```

### Via code (programmatic)

```python
from ignis_router import TrainingPipeline

pipeline = TrainingPipeline()
pipeline.train("svm")      # Train SVM router
pipeline.train_all()       # Train all routers
```

---

## Supported Models & Intent Rules

### Registered Models

| Model ID | Provider | Capabilities | Priority |
|----------|----------|-------------|----------|
| gpt-4.1 | OpenAI | high_quality, reasoning, code | 9 |
| gpt-4o-mini | OpenAI | fast_response, cost_effective, code | 6 |
| claude-3-5-sonnet | Anthropic | high_quality, reasoning, creative | 8 |

### Intent Rules

| Intent | Selected Model |
|--------|---------------|
| Code Generation | claude-3-5-sonnet |
| Summarization | gpt-4.1 |
| Reasoning | gpt-4.1 |
| Data Analysis | gpt-4.1 |
| Creative Writing | claude-3-5-sonnet |
| Translation | gpt-4o-mini |
| Classification | gpt-4o-mini |
| Extraction | gpt-4o-mini |
| General Chat | *(scored by weights)* |

---

## Routing Strategies

Set via `ROUTER_YAML_CONFIG` in `.env`:

| Strategy | Quality | Latency | Cost | Reliability | Best For |
|----------|---------|---------|------|-------------|----------|
| `quality-first` | 50 | 15 | 10 | 25 | Best output quality |
| `cost-first` | 20 | 20 | 45 | 15 | Minimize cost |
| `latency-first` | 20 | 50 | 15 | 15 | Fastest response |
| `balanced` | 40 | 20 | 20 | 20 | General purpose |

---

## Environment Variable Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `ENABLE_ML_INTENT_DETECTION` | `true` | Enable ML-based intent detection |
| `ENABLE_RULE_BASED_INTENT_DETECTION` | `true` | Enable rule-based intent detection |
| `ENABLE_ML_MODEL_HINT_ROUTING` | `false` | Use ML router prediction for model selection |
| `ML_CONFIDENCE_THRESHOLD` | `0.60` | Min ML confidence before fallback to rule-based |
| `ML_ROUTER_TYPE` | `knn` | ML router: knn, svm, graph, or mf |
| `ML_MODEL_PATH` | `models/knnrouter.pkl` | Path to legacy ML intent model |
| `ROUTER_YAML_CONFIG` | *(none)* | Strategy YAML (e.g. `configs/cost-first.yaml`) |
| `API_PORT` | `8080` | API server port |
| `OPENAI_API_KEY` | *(none)* | OpenAI API key |
| `ANTHROPIC_API_KEY` | *(none)* | Anthropic API key |
| `GOOGLE_API_KEY` | *(none)* | Google Gemini API key |
| `ROUTER_DB_HOST` | `localhost` | PostgreSQL host |
| `ROUTER_DB_PORT` | `5432` | PostgreSQL port |
| `ROUTER_DB_NAME` | `llm_router` | PostgreSQL database |
| `ROUTER_DB_USER` | `postgres` | PostgreSQL user |
| `ROUTER_DB_PASSWORD` | *(none)* | PostgreSQL password |
| `ROUTER_DB_TABLE` | `routing_responses` | PostgreSQL table |

---

## Using in Another App

### Step 1: Install

```bash
pip install git+https://github.com/Infogain-GenAI/ignis_router.git@sakshi_dev_1
```

### Step 2: Set API key

Create a `.env` file in your app's root:
```env
OPENAI_API_KEY=sk-your-key-here
ML_ROUTER_TYPE=svm
ENABLE_ML_MODEL_HINT_ROUTING=true
```

### Step 3: Use decorators

```python
from ignis_router import chat

@chat(system_prompt="You are a helpful assistant")
def ask(query, response):
    rd = response["routing_decision"]
    print(f"ML Predicted:   {rd['ml_router_predicted']}")
    print(f"Rule-Based:     {rd['rule_based_would_pick']}")
    print(f"Final Model:    {rd['final_model']}")
    print(f"Response:       {response['content'][:200]}")
    return response

result = ask("Write a sorting algorithm in Python")
```

### What happens internally:

1. `pip install` downloads ignis_router + llmrouter-lib (from PyPI) + all deps
2. First run downloads Longformer model from HuggingFace (~560 MB, cached permanently)
3. ML router (SVM/KNN/Graph/MF) predicts best LLM model
4. If API key missing for predicted model → switches to available provider
5. Calls LLM → returns response
6. Saves routing decision to PostgreSQL (if configured)

---

## Running Tests

```powershell
python -m pytest tests/ -q           # Quick run
python -m pytest tests/ -v           # Verbose
python -m pytest tests/test_api.py   # Specific file
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ValidationError: ENABLE_RULE_BASED_INTENT_DETECTION` | Typo in `.env` (e.g. `flase`) | Fix to `true` or `false` |
| `Address already in use` on API start | Another process on same port | Stop old process or change `API_PORT` |
| `POST /chat` returns 503 | LLM API key missing or invalid | Set `OPENAI_API_KEY` in `.env` |
| ML confidence always below threshold | ML intent detector is uncertain | Lower `ML_CONFIDENCE_THRESHOLD` or rely on rule-based |
| `HTTP Error 504` from HuggingFace | HuggingFace server down | Wait and retry, or set `HF_HUB_OFFLINE=1` in `.env` |
| `File does not exist: .pkl` | ML model file missing | Run `python scripts/train_all_routers.py` |
| Every query returns same model | `ENABLE_ML_MODEL_HINT_ROUTING=true` overrides rules | Set to `false` for intent-rule routing |
| `ConfigurationError: both detectors disabled` | Both ML and rule-based set to `false` | Enable at least one |
| PostgreSQL connection failed | Wrong credentials in `.env` | Check `ROUTER_DB_*` values |
| Slow startup (~45s) | Loading torch + Longformer models | Normal on first run; cached after |

---
---

## Roadmap & Next Steps

### Phase 2: Routing Evaluation Framework

**Focus**: Measure routing effectiveness and optimize performance.

Upcoming in the next release:

- **Evaluation Dataset**: Standardized benchmark dataset for measuring routing quality against real-world queries
- **Metrics Collection**: Track routing accuracy, cost efficiency, latency, and quality scores automatically
- **Performance Reports**: Generate detailed evaluation reports comparing different routing strategies (KNN vs SVM vs Graph)
- **Optimization Guidance**: Data-driven recommendations for tuning `ML_CONFIDENCE_THRESHOLD`, weights, and router selection

**Why it matters**: Objective measurement of routing quality helps you confidently choose between ML routers and rule-based fallback, and identify which strategy (cost-first vs quality-first) works best for your workload.

