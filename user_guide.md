# Ignis Router — User Guide

> **Intelligent LLM routing for Python.**
> Selects the best language model for every query using ML routers, rule-based intent detection, weighted scoring, and automatic provider fallback.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

---

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Quick Start](#quick-start)
4. [Decorators](#decorators)
5. [REST API](#rest-api)
6. [SDK Client](#sdk-client)
7. [Streamlit Dashboard](#streamlit-dashboard)
8. [ML Routers](#ml-routers)
9. [Logging](#logging)
10. [Environment Variables](#environment-variables)
11. [Troubleshooting](#troubleshooting)

---

## Installation

```bash
pip install git+https://github.com/Infogain-GenAI/ignis_router.git@sakshi_dev_1
```

Optional extras:

```bash
pip install "ignis_router[all]"         # All LLM providers (OpenAI + Anthropic + Gemini)
pip install "ignis_router[dashboard]"   # Streamlit dashboard
pip install "ignis_router[dev]"         # Dev tools (pytest, black, ruff)
```

For development:

```bash
git clone https://github.com/Infogain-GenAI/ignis_router.git
cd ignis_router
pip install -e ".[dev,all,dashboard]"
```

**Requirements:** Python ≥ 3.10, at least one LLM API key. PostgreSQL ≥ 14 is optional (for DB logging).

---

## Configuration

Create a `.env` file in your project root:

```env
# LLM Provider API Keys (set at least one)
OPENAI_API_KEY=sk-your-openai-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key-here
# GOOGLE_API_KEY=your-google-api-key-here

# Routing strategy: quality-first, cost-first, latency-first, or balanced
ROUTER_YAML_CONFIG=configs/cost-first.yaml

# ML Router type: knn, svm, graph, or mf
ML_ROUTER_TYPE=svm
ENABLE_ML_MODEL_HINT_ROUTING=true
ML_CONFIDENCE_THRESHOLD=0.50

# Intent detection
ENABLE_ML_INTENT_DETECTION=true
ENABLE_RULE_BASED_INTENT_DETECTION=true

# PostgreSQL (optional — for persisting routing decisions)
ROUTER_DB_HOST=localhost
ROUTER_DB_PORT=5432
ROUTER_DB_NAME=llm_router
ROUTER_DB_USER=postgres
ROUTER_DB_PASSWORD=your_password

# Logging
IGNIS_LOG_FILE=logs/ignis_router.log
IGNIS_LOG_CONSOLE=false
```

### Routing strategies

| Strategy | Optimizes for |
|----------|---------------|
| `configs/quality-first.yaml` | Best output quality |
| `configs/cost-first.yaml` | Lowest cost |
| `configs/latency-first.yaml` | Fastest response |
| `configs/balanced.yaml` | General purpose |

### PostgreSQL setup

```sql
CREATE DATABASE llm_router;
```

The table is created automatically on first use. No manual schema setup required.

---

## Quick Start

### Route a query (pick the best model)

```python
from ignis_router import Router

router = Router()
router.register_supported_models()
router.register_default_intent_rules()

result = router.route("Write a Python function to sort a list")
print(result.selected_model.model_name)   # claude-3-5-sonnet
print(result.detected_intent.value)       # code_generation
print(result.confidence)                  # 0.85
```

### Route + call the LLM

```python
from ignis_router import Router

router = Router()
router.register_supported_models()
router.register_default_intent_rules()
router.enable_llm_clients()

response = router.chat("Write a Python function to sort a list")
print(response["content"])     # AI response text
print(response["model"])       # gpt-4.1-2025-04-14
print(response["provider"])    # openai
```

---

## Decorators

#### `@route()` — Route only

```python
from ignis_router import route

@route()
def handle(query, routing_result, routing_decision):
    print(f"Model: {routing_decision['final_model']}")
    print(f"Intent: {routing_decision['intent']}")
    return routing_decision

handle("Write Python code")
```

#### `@chat()` — Route + call LLM

```python
from ignis_router import chat

@chat(system_prompt="You are a coding expert")
def ask(query, response):
    print(f"Model: {response['routing_decision']['final_model']}")
    print(f"Response: {response['content'][:200]}")
    return response["content"]

ask("Write code for API creation")
```

#### `@with_router()` — Inject configured router

```python
from ignis_router import with_router

@with_router(enable_llm=True)
def my_app(router):
    result = router.chat("Explain quantum computing")
    print(result["content"])

my_app()
```

#### `@retry()` — Automatic retry

```python
from ignis_router import retry, chat

@retry(max_attempts=3)
@chat()
def safe_ask(query, response):
    return response["content"]
```

#### Routing decision fields

| Field | Description |
|-------|-------------|
| `ml_router_predicted` | Model predicted by ML router |
| `rule_based_would_pick` | Model rule-based detection would select |
| `final_model` | Model actually used (with provider) |
| `note` | Fallback reason (e.g. "API key missing") |
| `intent` | Detected intent (code_generation, summarization, etc.) |
| `confidence` | Confidence score (0.0–1.0) |
| `tokens` | Total tokens used |

---

## REST API

### Start the server

```bash
python -m ignis_router.api.run_api
```

Starts at `http://127.0.0.1:8080`. Swagger UI at `/docs`.

```powershell
# Custom port
$env:API_PORT=9000; python -m ignis_router.api.run_api

# With DB password
$env:ROUTER_DB_PASSWORD = 'your_password'; python -m ignis_router.api.run_api
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/route` | Route a query → selected model, strategy, confidence |
| `POST` | `/chat` | Route + call LLM → AI response with routing decision |
| `GET` | `/metrics?days=N` | Routing metrics for last N days |
| `GET` | `/metrics/summary?days=N` | Text summary |
| `GET` | `/metrics/models?days=N` | Model distribution |
| `GET` | `/dashboard?days=N` | Full dashboard data |
| `GET` | `/features` | Feature flag states |
| `PUT` | `/features/{key}?enabled=true` | Toggle a feature at runtime |

### `POST /route`

```json
// Request
{"query": "Write Python code for sorting"}

// Response
{"selected_model": "claude-3-5-sonnet", "strategy": "cost-first", "confidence": 0.8}
```

### `POST /chat`

```json
// Request
{"query": "Write Python code for sorting", "max_tokens": 1024, "temperature": 0.7}

// Response
{
  "content": "Here's a Python sorting function...",
  "model": "gpt-4.1-2025-04-14",
  "provider": "openai",
  "usage": {"prompt_tokens": 15, "completion_tokens": 120, "total_tokens": 135},
  "routing_decision": {
    "ml_router_predicted": "qwen2.5-7b-instruct",
    "rule_based_would_pick": "claude-3-5-sonnet",
    "final_model": "gpt-4.1-2025-04-14 (openai)",
    "intent": "code_generation",
    "confidence": 0.8,
    "tokens": 135
  }
}
```

### Feature flags

Toggle routing behavior without restarting:

```bash
curl -X PUT "http://localhost:8080/features/ml_based_routing?enabled=false"
curl -X PUT "http://localhost:8080/features/rule_based_routing?enabled=true"
```

---

## SDK Client

```python
from ignis_router import IgnisClient

with IgnisClient("http://127.0.0.1:8080") as client:
    # Route only
    result = client.route("Write Python code")
    print(result.selected_model)

    # Route + execute LLM
    chat = client.chat("Write Python code", max_tokens=512)
    print(chat.content)
    print(chat.model)
```

---

## Streamlit Dashboard

```bash
# Install
pip install "ignis_router[dashboard]"

# Run (API must be running first)
python -m streamlit run examples/streamlit_dashboard.py
```

Opens at `http://localhost:8501`. Shows KPIs, model distribution, confidence charts, per-model/per-intent performance, ML vs rule-based comparison, and routing log.

> Requires both the API and PostgreSQL to be running.

---

## ML Routers

Four ML routers from [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) (open-source, UIUC):

| Router | `.env` value | Inference | Accuracy | Best for |
|--------|-------------|-----------|----------|----------|
| **KNN** | `knn` | 45 ms | 88.4% | Startups, explainability |
| **SVM** | `svm` | 12 ms | 91.2% | Production SaaS, low latency |
| **Graph** | `graph` | 78 ms | 93.8% | Enterprise, complex domains |
| **MF** | `mf` | 52 ms | 89.6% | Multi-tenant, personalization |

```env
ML_ROUTER_TYPE=svm
```

### Which router for which use case?

| Use Case | Router | `.env` Config | Why |
|----------|--------|---------------|-----|
| **Startup / MVP** | KNN | `ML_ROUTER_TYPE=knn` | Fast to train, explainable, requires little data |
| **Internal copilot** | KNN | `ML_ROUTER_TYPE=knn` | Simple, easy to debug, < 500k queries/day |
| **Production SaaS** | SVM | `ML_ROUTER_TYPE=svm` | Fastest inference (12 ms), best speed/accuracy tradeoff |
| **Real-time API** | SVM | `ML_ROUTER_TYPE=svm` | Lowest latency, handles 500k–10M queries/day |
| **Cost-sensitive app** | SVM | `ML_ROUTER_TYPE=svm` | Smallest model (4.1 MB), lowest compute |
| **Enterprise platform** | Graph | `ML_ROUTER_TYPE=graph` | Highest accuracy (93.8%), handles complex multi-domain |
| **Banking / Healthcare** | Graph | `ML_ROUTER_TYPE=graph` | Learns robust patterns, best generalization |
| **Research platform** | Graph | `ML_ROUTER_TYPE=graph` | Models query-model-domain relationships |
| **Multi-tenant SaaS** | MF | `ML_ROUTER_TYPE=mf` | Learns user preferences over time |
| **Personalized assistant** | MF | `ML_ROUTER_TYPE=mf` | Adapts to user interaction history |

### Recommended `.env` by environment

**Development / Testing:**
```env
ML_ROUTER_TYPE=knn
ENABLE_ML_MODEL_HINT_ROUTING=true
ML_CONFIDENCE_THRESHOLD=0.50
ROUTER_YAML_CONFIG=configs/balanced.yaml
```

**Production (SaaS):**
```env
ML_ROUTER_TYPE=svm
ENABLE_ML_MODEL_HINT_ROUTING=true
ML_CONFIDENCE_THRESHOLD=0.60
ROUTER_YAML_CONFIG=configs/cost-first.yaml
```

**Enterprise:**
```env
ML_ROUTER_TYPE=graph
ENABLE_ML_MODEL_HINT_ROUTING=true
ML_CONFIDENCE_THRESHOLD=0.70
ROUTER_YAML_CONFIG=configs/quality-first.yaml
```

When the predicted model's API key is unavailable, Ignis Router automatically falls back to an available provider.

### Retraining

```bash
python -m ignis_router.scripts.train_all_routers        # All routers
python -m ignis_router.scripts.train_all_routers svm    # Specific router
```

---

## Logging

Structured JSON logs with correlation IDs.

```env
IGNIS_LOG_FILE=logs/ignis_router.log
IGNIS_LOG_CONSOLE=false
IGNIS_LOG_LEVEL=INFO
IGNIS_LOG_FORMAT=json
```

Example log entry:

```json
{
  "timestamp": "2026-07-24T05:40:48+00:00",
  "level": "INFO",
  "correlation_id": "cfd5595db1b749e5",
  "event": "routing_decision",
  "selected_model": "gpt-4.1-2025-04-14",
  "intent": "code_generation",
  "confidence": 0.85,
  "latency_ms": 15.3
}
```

Errors include full tracebacks with file, line, and function. Compatible with ELK, Datadog, CloudWatch, and Splunk.

```python
from ignis_router import correlation_context

with correlation_context("my-trace-id") as cid:
    result = router.route("Write code")
    # All logs share the same correlation_id
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GOOGLE_API_KEY` | — | Google Gemini API key |
| `ROUTER_YAML_CONFIG` | — | Strategy YAML (e.g. `configs/cost-first.yaml`) |
| `ML_ROUTER_TYPE` | `knn` | ML router: `knn`, `svm`, `graph`, `mf` |
| `ENABLE_ML_MODEL_HINT_ROUTING` | `false` | Use ML prediction for model selection |
| `ML_CONFIDENCE_THRESHOLD` | `0.60` | Min ML confidence before rule-based fallback |
| `ENABLE_ML_INTENT_DETECTION` | `true` | Enable ML intent detection |
| `ENABLE_RULE_BASED_INTENT_DETECTION` | `true` | Enable rule-based intent detection |
| `API_PORT` | `8080` | API server port |
| `ROUTER_DB_HOST` | `localhost` | PostgreSQL host |
| `ROUTER_DB_PORT` | `5432` | PostgreSQL port |
| `ROUTER_DB_NAME` | `llm_router` | Database name |
| `ROUTER_DB_USER` | `postgres` | Database user |
| `ROUTER_DB_PASSWORD` | `postgres` | Database password |
| `ROUTER_DB_TABLE` | `routing_responses` | Table name |
| `IGNIS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `IGNIS_LOG_FORMAT` | `json` | `json` or `text` |
| `IGNIS_LOG_FILE` | — | Log file path |
| `IGNIS_LOG_CONSOLE` | `true` | Print logs to terminal |
| `HF_HUB_OFFLINE` | — | Set `1` to block HuggingFace downloads |

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `POST /chat` returns 503 | Set `OPENAI_API_KEY` in `.env` |
| `Password authentication failed` | Set `$env:ROUTER_DB_PASSWORD` before starting API |
| Dashboard shows 500 error | Ensure API and PostgreSQL are both running |
| `Address already in use` | Stop old process or change `API_PORT` |
| ML confidence always low | Lower `ML_CONFIDENCE_THRESHOLD` |
| Missing `.pkl` model file | Run `python -m ignis_router.scripts.train_all_routers` |
| Same model every time | Set `ENABLE_ML_MODEL_HINT_ROUTING=false` |
| Slow startup (~45 s) | Normal — PyTorch + Longformer loading (cached after first run) |
| No log file | Set `IGNIS_LOG_FILE=logs/ignis_router.log` |
| JSON logs in terminal | Set `IGNIS_LOG_CONSOLE=false` |

---

## Running Tests

```bash
python -m pytest tests/ -v
python -m pytest tests/test_api.py      # Single file
```

---

## License

MIT — see [LICENSE](LICENSE).
