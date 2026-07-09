# Ignis Router — User Guide

> Intelligent LLM routing library that selects the best model for every query.

---

## Table of Contents

1. [What Is Ignis Router?](#what-is-ignis-router)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Project Structure](#project-structure)
5. [Configuration](#configuration)
6. [Quick Start](#quick-start)
7. [Run the REST API](#run-the-rest-api)
8. [API Endpoints](#api-endpoints)
9. [Run with PostgreSQL Logging](#run-with-postgresql-logging)
10. [How Routing Works](#how-routing-works)
11. [Supported Models](#supported-models)
12. [Intent Rules](#intent-rules)
13. [Routing Strategies](#routing-strategies)
14. [Environment Variable Reference](#environment-variable-reference)
15. [Running Tests](#running-tests)
16. [Troubleshooting](#troubleshooting)

---

## What Is Ignis Router?

Ignis Router is a Python library and REST API that takes a user prompt and decides which LLM (Large Language Model) should handle it. It uses:

- **Rule-based intent detection** — keyword/regex matching to classify what the user wants (code, summary, translation, etc.)
- **ML-based detection** — optional ML model (`.pkl` file) for model prediction
- **Intent-to-model rules** — strict mapping from detected intent to a specific LLM
- **Weighted scoring** — fallback scoring based on quality, cost, latency, and reliability
- **LLM execution** — after routing, optionally call the selected model's API (OpenAI, Anthropic, Gemini) and return the AI response
- **PostgreSQL logging** — every routing decision is automatically saved to the database
- **Python SDK client** — other apps can call the router API with a simple Python client

---

## Prerequisites

| Requirement | Version | Check command |
|---|---|---|
| Python | >= 3.10 | `python --version` |
| pip | latest | `pip --version` |
| PostgreSQL | >= 14 (optional, for DB logging) | `psql --version` |
| Git | any | `git --version` |

---

## Installation

### Option 1: Clone from GitHub

```bash
git clone https://github.com/Infogain-GenAI/ignis_router.git
cd ignis_router
git checkout sakshi_dev
```

### Option 2: Install directly via pip

```bash
pip install git+https://github.com/Infogain-GenAI/ignis_router.git@sakshi_dev
```

### Install dependencies (if cloned)

```bash
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
├── .env.example              # Sample environment config (copy to .env)
├── configs/                  # Routing strategy YAML files
│   ├── balanced.yaml
│   ├── cost-first.yaml
│   ├── latency-first.yaml
│   ├── quality-first.yaml
│   └── postgres_schema.sql   # DB table schema
├── examples/
│   ├── basic_routing.py      # Simple Python usage example
│   ├── ai_chat_app.py        # Interactive AI chat (route + call LLM)
│   └── route_with_db.py      # Interactive CLI with PostgreSQL logging
├── models/
│   └── knnrouter.pkl         # ML model file (optional)
├── src/ignis_router/         # Main library package
│   ├── api.py                # FastAPI REST service
│   ├── client.py             # Python SDK client for remote API calls
│   ├── config.py             # Configuration and registry
│   ├── intent_detector.py    # Rule-based + ML intent detection
│   ├── llm_client.py         # LLM provider clients (OpenAI, Anthropic, Gemini)
│   ├── model_selector.py     # Scoring and model selection
│   ├── models.py             # Data models (Intent, ModelConfig, etc.)
│   ├── router.py             # Main Router public API
│   ├── routing_engine.py     # Orchestration engine
│   ├── run_api.py            # API server runner
│   ├── persistence.py        # PostgreSQL logging
│   └── supported_models.py   # Default model catalog + intent rules
├── tests/                    # Test suite (pytest)
└── pyproject.toml            # Package metadata and dependencies
```

---

## Configuration

### Step 1: Create your `.env` file

```bash
cp .env.example .env
```

### Step 2: Edit `.env`

```env
# Intent detection mode
ENABLE_ML_INTENT_DETECTION=true
ENABLE_RULE_BASED_INTENT_DETECTION=true

# ML model hint routing (true = return ML prediction directly)
ENABLE_ML_MODEL_HINT_ROUTING=false

# ML confidence threshold for hybrid fallback
ML_CONFIDENCE_THRESHOLD=0.50

# Routing strategy (choose one YAML file)
ROUTER_YAML_CONFIG=configs/cost-first.yaml

# LLM Provider API Keys (set the ones you have)
OPENAI_API_KEY=sk-your-openai-key-here
# ANTHROPIC_API_KEY=sk-ant-your-key-here
# GOOGLE_API_KEY=your-google-api-key-here

# PostgreSQL (optional, for DB logging)
ROUTER_DB_HOST=localhost
ROUTER_DB_PORT=5432
ROUTER_DB_NAME=llm_router
ROUTER_DB_USER=postgres
ROUTER_DB_PASSWORD=your_password
ROUTER_DB_TABLE=routing_responses
```

---

## Quick Start

### Use as a Python library

```python
from ignis_router import Router, RouterConfig

# Create router with default config
router = Router()
router.register_supported_models()
router.register_default_intent_rules()

# Route a query
result = router.route("Write a Python function to sort a list")

print(result.selected_model.model_name)   # claude-3-5-sonnet
print(result.detected_intent.value)       # code_generation
print(result.confidence)                  # 0.8
```

### Use with custom config

```python
config = RouterConfig.from_yaml("configs/quality-first.yaml")
router = Router(config=config)
router.register_supported_models()
router.register_default_intent_rules()

result = router.route("Summarize this article")
print(result.selected_model.model_name)   # gpt-4.1
```

### Use as an AI Chat (route + call LLM)

```python
from ignis_router import Router

router = Router()
router.register_supported_models()
router.register_default_intent_rules()
router.enable_llm_clients()  # reads OPENAI_API_KEY from env

# Route AND get AI response
result = router.chat("Write a Python function to sort a list")
print(result["content"])     # actual AI-generated response
print(result["model"])       # gpt-4.1 or claude-3-5-sonnet
print(result["provider"])    # openai or anthropic
```

### Interactive AI Chat App

```powershell
python examples/ai_chat_app.py
```

This starts an interactive terminal where you type queries, the router picks the best model, calls it, and prints the AI response.

---

## Run the REST API

### Start the server

```powershell
python -m ignis_router.run_api
```

Default: `http://127.0.0.1:8080`

If port 8080 is busy, the runner tells you and exits. Set a different port:

```powershell
$env:API_PORT=9000; python -m ignis_router.run_api
```

### Test the API

**PowerShell:**
```powershell
Invoke-RestMethod -Uri "http://127.0.0.1:8080/route" -Method Post -ContentType "application/json" -Body '{"query":"Write Python code for sorting"}'
```

**curl:**
```bash
curl -X POST http://127.0.0.1:8080/route \
  -H "Content-Type: application/json" \
  -d '{"query":"Write Python code for sorting"}'
```

**Browser:**
```
http://127.0.0.1:8080/route?query=Write%20Python%20code%20for%20sorting
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | API info and status |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI (interactive docs) |
| POST | `/route` | Route a query (JSON body: `{"query": "..."}`) |
| GET | `/route?query=...` | Route a query (browser-friendly) |
| POST | `/chat` | Route + call LLM and return AI response |
| GET | `/models` | List all registered models with metadata |
| GET | `/rules` | List all active routing rules |
| GET | `/history?limit=20` | Fetch recent routing responses from DB |
| GET | `/providers` | List available LLM providers |

### POST /route — Request

```json
{
  "query": "Generate Python code for data analysis"
}
```

### POST /route — Response

```json
{
  "selected_model": "claude-3-5-sonnet",
  "strategy": "cost-first",
  "confidence": 0.8
}
```

### POST /chat — Request

```json
{
  "query": "Write Python code for sorting",
  "system_prompt": "You are a helpful assistant.",
  "max_tokens": 1024,
  "temperature": 0.7
}
```

### POST /chat — Response

```json
{
  "content": "Here's a Python sorting function...\n",
  "model": "gpt-4.1",
  "provider": "openai",
  "usage": {"prompt_tokens": 15, "completion_tokens": 120, "total_tokens": 135},
  "routing": {
    "detected_intent": "code_generation",
    "complexity": "medium",
    "confidence": 0.8
  }
}
```

### GET /models — Response

```json
[
  {
    "model_id": "gpt-4.1",
    "provider": "openai",
    "capabilities": ["high_quality", "reasoning", "code"],
    "priority": 9,
    "enabled": true
  }
]
```

### GET /rules — Response

```json
[
  {
    "rule_id": "intent-code-generation",
    "intent": "code_generation",
    "target_model_id": "claude-3-5-sonnet",
    "priority": 90
  }
]
```

### GET /history?limit=5 — Response

```json
[
  {
    "id": 34,
    "query": "Generate Python code for data analysis",
    "selected_model": "claude-3-5-sonnet",
    "strategy": "cost-first",
    "confidence": 0.8,
    "created_at": "2026-07-09T16:01:15+05:30"
  }
]
```

### Error Response (all error endpoints)

```json
{
  "error": "validation_error",
  "message": "Request validation failed",
  "details": [...]
}
```

---

## Run with PostgreSQL Logging

### Step 1: Create the database and table

```sql
CREATE DATABASE llm_router;
\c llm_router

CREATE TABLE IF NOT EXISTS routing_responses (
    id BIGSERIAL PRIMARY KEY,
    query_text TEXT NOT NULL,
    selected_model TEXT NOT NULL,
    strategy TEXT NOT NULL,
    confidence DOUBLE PRECISION NOT NULL,
    response_json JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

Or use the provided SQL file:

```bash
psql -U postgres -d llm_router -f configs/postgres_schema.sql
```

### Step 2: Set DB credentials in `.env`

```env
ROUTER_DB_HOST=localhost
ROUTER_DB_PORT=5432
ROUTER_DB_NAME=llm_router
ROUTER_DB_USER=postgres
ROUTER_DB_PASSWORD=your_password
```

### Step 3: Run the interactive CLI

```powershell
python examples/route_with_db.py
```

Every query and response is printed to terminal and saved to PostgreSQL.

---

## How Routing Works

```
User query
    │
    ▼
┌─────────────────────────────────────────────┐
│ ML Detector (knnrouter.pkl)                 │
│ Predicts a model label (e.g. gemma-2-9b-it) │
└─────────────┬───────────────────────────────┘
              │
    ┌─────────▼──────────┐
    │ ENABLE_ML_MODEL_   │
    │ HINT_ROUTING=true? │
    └──┬─────────────┬───┘
     Yes            No
      │              │
      ▼              ▼
  Return ML     Check ML confidence
  prediction    vs threshold
  directly          │
              ┌─────▼─────┐
              │ Above     │ Below
              │ threshold │ threshold
              └──┬────────┴──┐
                 │           │
                 ▼           ▼
            Use ML      Fall back to
            result      Rule-Based
            (GENERAL_   Detector
             CHAT +        │
             scoring)      ▼
                      Detect intent
                      (code, summary,
                       translation...)
                           │
                           ▼
                      Intent rule
                      matched?
                      ┌────┴────┐
                    Yes        No
                      │         │
                      ▼         ▼
                  Mapped      Score all
                  model      models by
                  (claude,   weights
                   gpt-4.1)
```

---

## Supported Models

| Model ID | Provider | Best For |
|----------|----------|----------|
| `gpt-4.1` | OpenAI | Reasoning, summarization, data analysis |
| `gpt-4o-mini` | OpenAI | Fast/cheap tasks, translation, classification |
| `claude-3-5-sonnet` | Anthropic | Code generation, creative writing |

---

## Intent Rules

When rule-based detection identifies the user's intent, these rules select the model:

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

Set via `ROUTER_YAML_CONFIG` in `.env`. Each strategy changes how models are scored:

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
| `ENABLE_ML_INTENT_DETECTION` | `true` | Enable ML-based detection |
| `ENABLE_RULE_BASED_INTENT_DETECTION` | `true` | Enable rule-based detection |
| `ENABLE_ML_MODEL_HINT_ROUTING` | `false` | Return raw ML model prediction directly |
| `ML_CONFIDENCE_THRESHOLD` | `0.60` | Min ML confidence before hybrid fallback |
| `ML_MODEL_PATH` | `models/knnrouter.pkl` | Path to ML model file |
| `ROUTER_YAML_CONFIG` | *(none)* | Path to strategy YAML (e.g. `configs/cost-first.yaml`) |
| `API_PORT` | `8080` | API server port |
| `IGNIS_ROUTER_API_RELOAD` | `false` | Enable auto-reload on code changes |
| `ROUTER_DB_HOST` | `localhost` | PostgreSQL host |
| `ROUTER_DB_PORT` | `5432` | PostgreSQL port |
| `ROUTER_DB_NAME` | `llm_router` | PostgreSQL database name |
| `ROUTER_DB_USER` | `postgres` | PostgreSQL user |
| `ROUTER_DB_PASSWORD` | *(none)* | PostgreSQL password |
| `ROUTER_DB_TABLE` | `routing_responses` | PostgreSQL table name |
| `OPENAI_API_KEY` | *(none)* | OpenAI API key (for `/chat` endpoint) |
| `ANTHROPIC_API_KEY` | *(none)* | Anthropic API key (for `/chat` endpoint) |
| `GOOGLE_API_KEY` | *(none)* | Google Gemini API key (for `/chat` endpoint) |

---

## Running Tests

```powershell
# Run all tests
python -m pytest tests/ -q

# Run specific test file
python -m pytest tests/test_api.py -q

# Run with verbose output
python -m pytest tests/ -v
```

---

## Using the Python SDK Client

Other apps can call the Ignis Router API using the built-in SDK client:

```python
from ignis_router.client import IgnisClient

# Connect to the running API
client = IgnisClient("http://127.0.0.1:8080")

# Route only (no LLM call)
route = client.route("Write Python code for sorting")
print(route.selected_model)   # claude-3-5-sonnet
print(route.strategy)         # cost-first

# Route + call LLM
chat = client.chat("Explain quantum computing")
print(chat.content)           # actual AI response
print(chat.model)             # gpt-4.1
print(chat.provider)          # openai

# Check health
print(client.health())        # {"status": "ok"}

# List available providers
print(client.providers())     # ["openai"]

client.close()
```

Install requirement for SDK client:
```bash
pip install httpx
```

---

## Troubleshooting

| Problem | Cause | Fix |
|---------|-------|-----|
| `ValidationError: ENABLE_RULE_BASED_INTENT_DETECTION` | Typo in `.env` (e.g. `flase`) | Fix to `true` or `false` |
| `EOFError: Ran out of input` | `models/knnrouter.pkl` is empty (0 bytes) | Replace with valid trained model file |
| `Address already in use` on API start | Another process on same port | Stop old process or change `API_PORT` |
| `Method Not Allowed` in browser | `/route` requires POST, browser sends GET | Use `/route?query=...` or `/docs` |
| Every query returns `gpt-4o-mini` | Rule-based disabled + ML returns GENERAL_CHAT | Set `ENABLE_RULE_BASED_INTENT_DETECTION=true` |
| Every query returns ML model name (gemma) | `ENABLE_ML_MODEL_HINT_ROUTING=true` | Set to `false` for intent-based routing |
| `/chat` returns 503 | LLM API key missing or invalid | Set `OPENAI_API_KEY` in `.env` |
| PostgreSQL connection failed | Wrong credentials in `.env` | Check `ROUTER_DB_*` values |
| `ConfigurationError: both detectors disabled` | Both ML and rule-based set to `false` | Enable at least one |

---

## License

MIT — see [LICENSE](LICENSE) for details.
