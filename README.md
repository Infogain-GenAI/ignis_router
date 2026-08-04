<p align="center">
  <h1 align="center">Ignis Router</h1>
  <p align="center">
    <strong>Intelligent LLM Routing Library for Python</strong>
  </p>
  <p align="center">
    Automatically selects the best language model for every query using ML routers, rule-based intent detection, weighted scoring, and provider fallback.
  </p>
</p>

---

## Installation

```bash
pip install ignis_router
```

With optional provider integrations:

```bash
# All LLM providers (OpenAI + Anthropic + Gemini)
pip install "ignis_router[all]"

# Streamlit dashboard
pip install "ignis_router[dashboard]"

# Development tools (pytest, black, ruff)
pip install "ignis_router[dev]"

# All optional integrations
pip install "ignis_router[all,dashboard,dev]"
```

## Key Features

- **Zero-Latency Routing**: ML-powered model selection adds < 15 ms overhead
- **Multi-Model Support**: OpenAI GPT, Anthropic Claude, and Google Gemini out of the box
- **4 ML Routers**: KNN, SVM, Graph, MF — trained on 50k+ examples
- **Hybrid Intent Detection**: Semantic ML classifier + rule-based fallback
- **Automatic Fallback**: Switches to available provider when API key is missing
- **4 Routing Strategies**: Quality-first, cost-first, latency-first, balanced — configurable via YAML
- **PostgreSQL Persistence**: Every routing decision logged automatically
- **Streamlit Dashboard**: Visual analytics for routing metrics
- **Structured Logging**: JSON logs with correlation IDs
- **Feature Flags**: Toggle routing behavior at runtime without restart

## 📋 Table of Contents

- [Architecture](#-architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [Decorators](#decorators)
- [REST API](#rest-api)
- [SDK Client](#sdk-client)
- [ML Routers](#ml-routers)
- [Intent Detection](#intent-detection)
- [Feature Flags](#feature-flags)
- [PostgreSQL Logging](#postgresql-logging)
- [Logging & Observability](#logging--observability)
- [Environment Variables](#environment-variables)
- [Troubleshooting](#-troubleshooting)

## 🏛️ Architecture

### Routing Pipeline

```
┌─────────────┐         ┌──────────────────┐         ┌─────────────────┐
│             │         │                  │         │                 │
│  Your App   │───────▶│  Intent Detection │───────▶│  ML Router      │
│  (Query)    │         │  (Semantic + Rule)│         │  (KNN/SVM/      │
│             │         │                  │         │   Graph/MF)     │
└─────────────┘         └──────────────────┘         └────────┬────────┘
                                                              │
                              ┌────────────────────────────────┘
                              ▼
                    ┌──────────────────┐
                    │  Provider Check  │
                    │  (API key avail?)│
                    └────────┬─────────┘
                             │
                ┌────────────┼────────────┐
                ▼            ▼            ▼
        ┌──────────┐  ┌──────────┐  ┌──────────┐
        │  OpenAI  │  │ Anthropic│  │  Gemini  │
        │  (GPT)   │  │ (Claude) │  │          │
        └────┬─────┘  └────┬─────┘  └────┬─────┘
             └──────────────┼──────────────┘
                            ▼
                  ┌──────────────────┐
                  │   Persistence    │
                  │ PostgreSQL + Logs│
                  └──────────────────┘
```

### Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.10+ | Modern async/await support |
| ML Routers | LLMRouter (UIUC) | Pre-trained model selection |
| Intent Detection | Sentence Transformers | Semantic classification |
| API | FastAPI | REST service with Swagger UI |
| Database | PostgreSQL | Routing decision persistence |
| Dashboard | Streamlit | Visual analytics |
| Configuration | pydantic-settings | Type-safe environment config |
| Logging | JSON structured | Correlation IDs, crash tracebacks |

---

## Quick Start

### 1. Installation

```bash
pip install ignis_router
```

```bash
# For development (includes testing tools)
pip install "ignis_router[all,dashboard,dev]"
```

### 2. Get Your API Keys

#### OpenAI API Key

1. Go to [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the key (starts with `sk-...`)

#### Other Providers (Optional)

- **Anthropic**: [https://console.anthropic.com/settings/keys](https://console.anthropic.com/settings/keys)
- **Google Gemini**: [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 3. Configure Environment

Create a `.env` file in your project root:

```bash
touch .env   # Linux/Mac
New-Item .env # Windows PowerShell
```

Add the following values:

```env
# Required: At least one LLM API key
OPENAI_API_KEY=sk-your-openai-key-here

# Required: Routing strategy
ROUTER_YAML_CONFIG=configs/cost-first.yaml

# Required: ML Router type
ML_ROUTER_TYPE=svm
ENABLE_ML_MODEL_HINT_ROUTING=true
ML_CONFIDENCE_THRESHOLD=0.50

# Optional: Intent detection (both enabled = hybrid mode)
ENABLE_ML_INTENT_DETECTION=true
ENABLE_RULE_BASED_INTENT_DETECTION=true
```

```env
# Optional: Other settings (defaults usually fine)
ROUTER_DATABASE_URL=postgresql://postgres:your_password@localhost:5432/llm_router
IGNIS_LOG_FILE=logs/ignis_router.log
```

### 4. Run Your First Query

Create a `main.py` with the quick-start code from the [Usage Examples](#usage-examples) section below and run it:

```bash
python main.py
```

You should see:

```
============================================================
Ignis Router - AI Chat App
============================================================
Available LLM providers: ['openai']

Type your query and press Enter. Type 'exit' to quit.

You: Write a Python function to sort a list

--- Routing Decision ---
ML Router Predicted:   qwen2.5-7b-instruct
Final Model Used:      gpt-4.1-2025-04-14 (openai)
Confidence:            0.80

--- Response ---
Here's a Python sorting function...
```

### 5. View Your Data

#### Streamlit Dashboard

```bash
# Start the API server first
python -m ignis_router.api.run_api

# Then launch the dashboard
python -m streamlit run examples/streamlit_dashboard.py
```

Opens at `http://localhost:8501` — see KPIs, model distribution, confidence charts, and routing log.

#### PostgreSQL

```sql
-- Query your local data
SELECT query_text, default_model_used, intent, confidence
FROM routing_responses
ORDER BY created_at DESC
LIMIT 5;
```

---

## Configuration

### Routing Strategies

| Strategy | File | Optimizes for |
|----------|------|---------------|
| Quality-first | `configs/quality-first.yaml` | Best output quality |
| Cost-first | `configs/cost-first.yaml` | Lowest cost |
| Latency-first | `configs/latency-first.yaml` | Fastest response |
| Balanced | `configs/balanced.yaml` | General purpose |

### PostgreSQL Setup

```sql
CREATE DATABASE llm_router;
```

> The table is created automatically on first use. No manual schema setup required.

---

## Usage Examples

### AI Chat App (Interactive Terminal)

```python
"""
AI Chat App with intelligent LLM routing.
Usage: python examples/ai_chat_app.py
"""
from dotenv import load_dotenv
from ignis_router import Router, RouterConfig
from ignis_router.db.routing_decision import build_routing_decision, log_routing_decision_to_db
from ignis_router.evaluation import LatencyCollector

load_dotenv()

# Build router with LLM execution enabled
router = Router()
router.register_supported_models()
router.register_default_intent_rules()
router.enable_llm_clients()

# Show available providers
available = router.llm_clients.get_available_providers() if router.llm_clients else []
print("=" * 60)
print("Ignis Router - AI Chat App")
print("=" * 60)
print(f"Available LLM providers: {available or ['None configured']}")
print("\nType your query and press Enter. Type 'exit' to quit.\n")

while True:
    query = input("You: ").strip()
    if not query or query.lower() in {"exit", "quit"}:
        break

    with LatencyCollector() as lc:
        result = router.chat(query)

    # Build and save routing decision
    rd = build_routing_decision(result, elapsed=lc.elapsed)
    log_routing_decision_to_db(
        query=query,
        routing_decision=rd,
        strategy=router.config.routing_strategy,
        response_content=result.get("content", ""),
    )

    # Display routing decision
    print(f"\n--- Routing Decision ---")
    if rd.get("ml_router_predicted"):
        print(f"ML Router Predicted:   {rd['ml_router_predicted']}")
    print(f"Final Model Used:      {rd['final_model']}")
    print(f"Intent:                {rd.get('intent', '')}")
    print(f"Confidence:            {rd.get('confidence', 0):.2f}")
    if rd.get("tokens"):
        print(f"Tokens:                {rd['tokens']}")

    print(f"\n--- Response ---")
    print(result["content"])
    print()
```

**Output:**

```
============================================================
Ignis Router - AI Chat App
============================================================
Available LLM providers: ['openai']

Type your query and press Enter. Type 'exit' to quit.

You: Write a REST API with authentication in FastAPI

--- Routing Decision ---
ML Router Predicted:   qwen2.5-7b-instruct
Final Model Used:      gpt-4.1-2025-04-14 (openai)
Intent:                code_generation
Confidence:            0.80
Tokens:                670

--- Response ---
Here's a FastAPI REST API with JWT authentication...
```

### Streamlit Dashboard

Real-time visual analytics for all routing decisions.

```bash
# 1. Install dashboard dependencies
pip install "ignis_router[dashboard]"

# 2. Start the API server (needs PostgreSQL running)
python -m ignis_router.api.run_api

# 3. Launch the dashboard
python -m streamlit run examples/streamlit_dashboard.py
```

Opens at `http://localhost:8501`. The dashboard provides:

| Panel | What it shows |
|-------|--------------|
| **KPI Cards** | Query count, routing accuracy, cost savings, avg latency, ML win rate |
| **Model Distribution** | Which models are being selected and how often |
| **Confidence Distribution** | Histogram of ML confidence scores |
| **Performance by Intent** | Avg confidence, top model, and ML vs rule-based wins per intent |
| **Performance by Model** | Queries, avg confidence, avg cost, avg latency per model |
| **ML vs Rule-Based** | Side-by-side comparison of routing outcomes |
| **Routing Log** | Paginated table of recent routing decisions with all details |

Filters: Window (24h / 7d / 30d / 90d), Strategy, Intent. Auto-refreshes every 30 seconds.

> Requires both the API and PostgreSQL to be running.

### Basic Decorator Usage

```python
from ignis_router import chat

# Decorate your function — routing happens automatically
@chat(system_prompt="You are a helpful assistant")
def ask(query, response):
    rd = response["routing_decision"]
    print(f"Model: {rd['final_model']}")
    print(f"Intent: {rd['intent']}")
    print(response["content"])
    return response

ask("Write a Python function to sort a list")
```

### Route Only (No LLM Call)

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

### FastAPI Integration

```python
from fastapi import FastAPI
from dotenv import load_dotenv
from ignis_router import Router

load_dotenv()
app = FastAPI()

router = Router()
router.register_supported_models()
router.register_default_intent_rules()
router.enable_llm_clients()

@app.post("/ask")
async def ask(query: str):
    response = router.chat(query)
    return {
        "answer": response["content"],
        "model_used": response["model"],
        "provider": response["provider"],
        "intent": response["routing"]["intent"],
    }
```

### Flask Integration

```python
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from ignis_router import Router

load_dotenv()
app = Flask(__name__)

router = Router()
router.register_supported_models()
router.register_default_intent_rules()
router.enable_llm_clients()

@app.route("/chat", methods=["POST"])
def chat():
    query = request.json["query"]
    response = router.chat(query)
    return jsonify({
        "answer": response["content"],
        "model": response["model"],
        "routing": response.get("routing", {}),
    })

if __name__ == "__main__":
    app.run(port=5000)
```

---

## Decorators

### `@route()` — Route only (no LLM call)

```python
from ignis_router import route

@route()
def handle(query, routing_result, routing_decision):
    print(f"Model: {routing_decision['final_model']}")
    print(f"Intent: {routing_decision['intent']}")
    return routing_decision

handle("Write Python code")
```

### `@chat()` — Route + call LLM

```python
from ignis_router import chat

@chat(system_prompt="You are a coding expert")
def ask(query, response):
    print(f"Model: {response['routing_decision']['final_model']}")
    print(f"Response: {response['content'][:200]}")
    return response["content"]

ask("Write code for API creation")
```

### `@with_router()` — Inject configured router

```python
from ignis_router import with_router

@with_router(enable_llm=True)
def my_app(router):
    result = router.chat("Explain quantum computing")
    print(result["content"])

my_app()
```

### `@retry()` — Automatic retry

```python
from ignis_router import retry, chat

@retry(max_attempts=3)
@chat()
def safe_ask(query, response):
    return response["content"]
```

### Routing Decision Fields

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

# With DB connection string
$env:ROUTER_DATABASE_URL = 'postgresql://postgres:your_password@localhost:5432/llm_router'; python -m ignis_router.api.run_api
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info (name, status, docs) |
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `GET` | `/route?query=...` | Route a query via GET parameter |
| `POST` | `/route` | Route a query → selected model, strategy, confidence |
| `POST` | `/chat` | Route + call LLM → AI response with routing decision |
| `GET` | `/metrics?days=N` | Routing metrics for last N days |
| `GET` | `/metrics/summary?days=N` | Text summary |
| `GET` | `/metrics/models?days=N` | Model distribution |
| `GET` | `/providers` | List available LLM providers |
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

### Feature Flags via API

```bash
# Toggle routing behavior without restarting
curl -X PUT "http://localhost:8080/features/ml_based_routing?enabled=false"
curl -X PUT "http://localhost:8080/features/rule_based_routing?enabled=true"
curl http://localhost:8080/features
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

## ML Routers

Four pre-trained routers from [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) (open-source, UIUC):

| Router | `.env` value | Inference | Accuracy | Best for |
|--------|-------------|-----------|----------|----------|
| **KNN** | `knn` | 45 ms | 88.4% | Startups, explainability |
| **SVM** | `svm` | 12 ms | 91.2% | Production SaaS, low latency |
| **Graph** | `graph` | 78 ms | 93.8% | Enterprise, complex domains |
| **MF** | `mf` | 52 ms | 89.6% | Multi-tenant, personalization |

### Which router for which use case?

| Use Case | Router | Config | Why |
|----------|--------|--------|-----|
| **Startup / MVP** | KNN | `ML_ROUTER_TYPE=knn` | Fast to train, explainable, requires little data |
| **Production SaaS** | SVM | `ML_ROUTER_TYPE=svm` | Fastest inference (12 ms), best speed/accuracy tradeoff |
| **Enterprise platform** | Graph | `ML_ROUTER_TYPE=graph` | Highest accuracy (93.8%), handles complex multi-domain |
| **Multi-tenant SaaS** | MF | `ML_ROUTER_TYPE=mf` | Learns user preferences over time |

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

### Retraining

```bash
python -m ignis_router.scripts.train_all_routers        # All routers
python -m ignis_router.scripts.train_all_routers svm    # Specific router
```

---

## Intent Detection

Ignis Router uses a **hybrid** intent detection system with two layers:

### Layer 1: Semantic ML Classifier (primary)

- Uses Sentence Transformer embeddings + Logistic Regression
- Trained on `data/intent_training_data.json`
- If confidence ≥ `ML_CONFIDENCE_THRESHOLD` → uses ML result
- If confidence < threshold → falls back to Layer 2

### Layer 2: Rule-Based Detector (fallback)

- Regex keyword matching (instant, < 1 ms)
- Always available, no model loading required

### Supported Intents

| Intent | Triggers on | Default Model |
|--------|-------------|---------------|
| `code_generation` | "write code", "create API", "implement" | claude-3-5-sonnet |
| `summarization` | "summarize", "TLDR", "sum up" | gpt-4.1 |
| `reasoning` | "explain why", "compare", "analyze" | gpt-4.1 |
| `creative_writing` | "write a poem", "compose", "story" | claude-3-5-sonnet |
| `data_analysis` | "analyze data", "trends", "statistics" | gpt-4.1 |
| `translation` | "translate", "in Spanish" | gpt-4o-mini |
| `classification` | "classify", "categorize", "sentiment" | gpt-4o-mini |
| `extraction` | "extract", "parse", "pull out" | gpt-4o-mini |
| `general_chat` | anything else | *(scored by strategy weights)* |

---

## Feature Flags

Toggle routing behavior at runtime without restarting the server or changing code.

| Flag | What it controls | Toggle via API |
|------|------------------|----------------|
| `ml_based_routing` | ML router model prediction | `PUT /features/ml_based_routing?enabled=false` |
| `rule_based_routing` | Regex keyword rules | `PUT /features/rule_based_routing?enabled=true` |
| `hybrid_routing` | ML first + rule-based fallback | `PUT /features/hybrid_routing?enabled=true` |

```python
from ignis_router import Router, FeatureFlags

router = Router()
flags = FeatureFlags.from_config(router.config)
flags.set("enable_ml_model_hint_routing", False)
print(flags.to_dict())
```

---

## PostgreSQL Logging

Every routing decision is **automatically** persisted to PostgreSQL.

### Setup

1. Create the database:
```sql
CREATE DATABASE llm_router;
```

2. Configure in `.env`:
```env
ROUTER_DATABASE_URL=postgresql://postgres:your_password@localhost:5432/llm_router
```

> The table is created automatically on first use. No manual schema needed.

### `routing_responses` Table

| Column | Example Value |
|--------|---------------|
| `query_text` | "Write a Python sorting function" |
| `ml_router_predicted` | "qwen2.5-7b-instruct" |
| `rule_based_would_pick` | "claude-3-5-sonnet" |
| `default_model_used` | "gpt-4.1-2025-04-14" |
| `provider` | "openai" |
| `note` | "API key not available, switched provider" |
| `intent` | "code_generation" |
| `complexity` | "low" |
| `confidence` | 0.80 |
| `tokens` | 135 |
| `strategy` | "cost-first" |
| `routing_latency_ms` | 15.3 |
| `ml_won` | true |

### Querying

```sql
-- Model usage distribution
SELECT default_model_used, COUNT(*) as queries
FROM routing_responses
GROUP BY default_model_used
ORDER BY queries DESC;

-- Average confidence by intent
SELECT intent, AVG(confidence) as avg_confidence, COUNT(*) as count
FROM routing_responses
GROUP BY intent;
```

---

## Logging & Observability

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

Compatible with ELK, Datadog, CloudWatch, and Splunk.

```python
from ignis_router import correlation_context

with correlation_context("my-trace-id") as cid:
    result = router.route("Write code")
    # All logs share the same correlation_id
```

---

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key | `sk-proj-abc123...` |
| `ROUTER_YAML_CONFIG` | Strategy YAML path | `configs/cost-first.yaml` |
| `ML_ROUTER_TYPE` | ML router type | `svm` |

### Optional Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GOOGLE_API_KEY` | — | Google Gemini API key |
| `ENABLE_ML_MODEL_HINT_ROUTING` | `false` | Use ML prediction for model selection |
| `ML_CONFIDENCE_THRESHOLD` | `0.60` | Min ML confidence before rule-based fallback |
| `ENABLE_ML_INTENT_DETECTION` | `true` | Enable ML intent detection |
| `ENABLE_RULE_BASED_INTENT_DETECTION` | `true` | Enable rule-based intent detection |
| `API_PORT` | `8080` | API server port |
| `ROUTER_DATABASE_URL` | `postgresql://postgres:postgres@localhost:5432/llm_router` | PostgreSQL connection string |
| `IGNIS_LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL` |
| `IGNIS_LOG_FORMAT` | `json` | `json` or `text` |
| `IGNIS_LOG_FILE` | — | Log file path |
| `IGNIS_LOG_CONSOLE` | `true` | Print logs to terminal |
| `HF_HUB_OFFLINE` | — | Set `1` to block HuggingFace downloads |

---

## 🐛 Troubleshooting

### Import Errors

**Problem:** `ModuleNotFoundError: No module named 'ignis_router'`

**Solution:** Make sure the package is installed:

```bash
pip install ignis_router
```

### Authentication Errors

**Problem:** `POST /chat` returns 503

**Solutions:**

1. Check `.env` file has correct API keys
2. Verify keys are active (not revoked)
3. Ensure OpenAI account has credits
4. Test keys independently

### No Data in Dashboard

**Problem:** Dashboard shows 500 error or empty charts

**Solutions:**

1. Ensure API is running: `python -m ignis_router.api.run_api`
2. Ensure PostgreSQL is running and accessible
3. Check `ROUTER_DATABASE_URL` in `.env`
4. Run some queries first to generate data

### ML Router Issues

| Problem | Fix |
|---------|-----|
| ML confidence always low | Lower `ML_CONFIDENCE_THRESHOLD` |
| Missing `.pkl` model file | Run `python -m ignis_router.scripts.train_all_routers` |
| Same model every time | Set `ENABLE_ML_MODEL_HINT_ROUTING=false` |
| Slow startup (~45 s) | Normal — PyTorch + Longformer loading (cached after first run) |

### Logging Issues

| Problem | Fix |
|---------|-----|
| No log file | Set `IGNIS_LOG_FILE=logs/ignis_router.log` |
| JSON logs in terminal | Set `IGNIS_LOG_CONSOLE=false` |
| `Address already in use` | Stop old process or change `API_PORT` |

---

## Contributing

```bash
git clone https://github.com/Infogain-GenAI/ignis_router.git
cd ignis_router
pip install -e ".[dev,all,dashboard]"
python -m pytest tests/ -v
```

### Project Structure

```
src/ignis_router/
├── api/           # FastAPI REST service + SDK client
├── configs/       # Routing strategy YAMLs + ML router configs
├── core/          # Router, routing engine, model selector
├── data/          # Intent training data
├── db/            # PostgreSQL persistence
├── detection/     # Intent detection (semantic + rule-based)
├── evaluation/    # Metrics, dashboard, reports
├── llm/           # LLM provider clients (OpenAI, Anthropic, Gemini)
├── ml/            # LLMRouter integration + ML inference
├── models/        # Pre-trained ML router models (.pkl, .pt)
└── scripts/       # Training scripts
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## References

Built with:

- [OpenAI](https://openai.com/) — LLM provider
- [Anthropic](https://www.anthropic.com/) — Claude LLM provider
- [Google Gemini](https://deepmind.google/technologies/gemini/) — Gemini LLM provider
- [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) — ML router models (UIUC)
- [FastAPI](https://fastapi.tiangolo.com/) — REST API framework
- [Streamlit](https://streamlit.io/) — Dashboard framework
- [Sentence Transformers](https://www.sbert.net/) — Semantic intent classification
- [pydantic](https://docs.pydantic.dev/) — Data validation and settings

Questions? Open an issue or check the [examples](examples/) directory for more details.

---

<p align="center">
  Built by <a href="https://github.com/Infogain-GenAI">Infogain GenAI</a>
</p>
