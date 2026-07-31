<p align="center">
  <h1 align="center">Ignis Router</h1>
  <p align="center">
    <strong>Intelligent LLM Routing Library for Python</strong>
  </p>
  <p align="center">
    Automatically selects the best language model for every query using ML routers, rule-based intent detection, weighted scoring, and provider fallback.
  </p>
  <p align="center">
    <a href="https://www.python.org/"><img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License: MIT"></a>
    <a href="https://github.com/Infogain-GenAI/ignis_router"><img src="https://img.shields.io/badge/maintained-yes-brightgreen.svg" alt="Maintained"></a>
  </p>
</p>

---

## What is Ignis Router?

Ignis Router is a production-ready Python package that sits between your application and LLM providers. It uses **machine learning** to predict the optimal model for each query, **rule-based intent detection** as an intelligent fallback, and **automatic provider switching** when API keys are unavailable.

```
Your App → Ignis Router → Best LLM (OpenAI / Anthropic / Gemini) → Response
```

### Why use it?

- **Cost savings** — Routes simple queries to cheaper models, complex ones to premium models
- **Quality optimization** — ML routers trained on 50k+ examples learn which model performs best for which query type
- **Zero downtime** — Automatic fallback when a provider is unavailable
- **Full observability** — Every routing decision is logged with correlation IDs

---

## Key Features

| | Feature | Description |
|---|---------|-------------|
| 🧠 | **ML-Based Routing** | 4 router types (KNN, SVM, Graph, MF) predict the best LLM model |
| 🎯 | **Intent Detection** | Hybrid semantic + rule-based classification (code, summarization, reasoning, etc.) |
| 🔄 | **Provider Fallback** | Auto-switches to available provider when API key is missing |
| ⚡ | **4 Strategies** | Quality-first, cost-first, latency-first, balanced — configurable via YAML |
| 🛠️ | **Decorators** | `@route()`, `@chat()`, `@with_router()`, `@retry()` |
| 🌐 | **REST API** | FastAPI with Swagger UI, feature toggles, metrics |
| 📊 | **Dashboard** | Streamlit dashboard for routing analytics |
| 🗄️ | **PostgreSQL** | Automatic persistence of every routing decision |
| 📝 | **Structured Logging** | JSON logs with correlation IDs and crash tracebacks |
| 🔀 | **Feature Flags** | Toggle routing behavior at runtime without restart |

---

## Installation

```bash
pip install git+https://github.com/Infogain-GenAI/ignis_router.git@main
```

<details>
<summary><strong>Optional extras</strong></summary>

```bash
pip install "ignis_router[all]"         # All LLM providers
pip install "ignis_router[dashboard]"   # Streamlit dashboard
pip install "ignis_router[dev]"         # Development tools
```
</details>

---

## Quick Start

### 1. Create `.env`

```env
OPENAI_API_KEY=sk-your-key-here
ML_ROUTER_TYPE=svm
ENABLE_ML_MODEL_HINT_ROUTING=true
```

### 2. Route + Call LLM

```python
from ignis_router import chat

@chat(system_prompt="You are a helpful assistant")
def ask(query, response):
    rd = response["routing_decision"]
    print(f"ML Predicted:  {rd['ml_router_predicted']}")
    print(f"Final Model:   {rd['final_model']}")
    print(f"Intent:        {rd['intent']}")
    print(f"Response:      {response['content'][:100]}")
    return response

ask("Write a Python function to sort a list")
```

### 3. Output

```
ML Predicted:  qwen2.5-7b-instruct
Final Model:   gpt-4.1-2025-04-14 (openai)
Intent:        code_generation
Response:      Here's a Python sorting function...
```

---

## Usage Options

### Decorators (simplest)

```python
from ignis_router import route, chat

@route()
def handle(query, routing_result, routing_decision):
    return routing_decision["final_model"]

@chat()
def ask(query, response):
    return response["content"]
```

### Direct Python API

```python
from ignis_router import Router

router = Router()
router.register_supported_models()
router.register_default_intent_rules()
router.enable_llm_clients()

response = router.chat("Explain quantum computing")
print(response["content"])
```

### REST API

```bash
python -m ignis_router.api.run_api
# Server: http://127.0.0.1:8080
# Swagger: http://127.0.0.1:8080/docs
```

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Write Python code for sorting"}'
```

### SDK Client

```python
from ignis_router import IgnisClient

with IgnisClient("http://127.0.0.1:8080") as client:
    result = client.chat("Write Python code")
    print(result.content)
```

---

## How It Works

```
User Query: "Write a Python API with authentication"
     │
     ▼
┌─ Intent Detection ────────────────────────────┐
│  Semantic ML → confidence 0.92 → code_gen     │
└───────────────────────────────────────────────┘
     │
     ▼
┌─ ML Router (SVM) ────────────────────────────┐
│  Predicts: qwen2.5-7b-instruct               │
└───────────────────────────────────────────────┘
     │
     ▼
┌─ Provider Check ─────────────────────────────┐
│  qwen2.5 → No API key → Fallback to OpenAI  │
└───────────────────────────────────────────────┘
     │
     ▼
┌─ LLM Call ───────────────────────────────────┐
│  gpt-4.1 (OpenAI) → AI Response             │
└───────────────────────────────────────────────┘
     │
     ▼
┌─ Persistence ────────────────────────────────┐
│  PostgreSQL + JSON Logs + Correlation IDs     │
└───────────────────────────────────────────────┘
```

---

## ML Routers

Four pre-trained routers from [LLMRouter](https://github.com/ulab-uiuc/LLMRouter) (open-source, UIUC):

| Router | Inference | Accuracy | Best For |
|--------|-----------|----------|----------|
| **SVM** | 12 ms | 91.2% | Production SaaS, low latency |
| **KNN** | 45 ms | 88.4% | Startups, explainability |
| **Graph** | 78 ms | 93.8% | Enterprise, complex domains |
| **MF** | 52 ms | 89.6% | Multi-tenant, personalization |

```env
ML_ROUTER_TYPE=svm  # or knn, graph, mf
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/docs` | Swagger UI |
| `POST` | `/route` | Route query → model, strategy, confidence |
| `POST` | `/chat` | Route + LLM → AI response + routing decision |
| `GET` | `/metrics?days=N` | Routing metrics |
| `GET` | `/dashboard?days=N` | Full dashboard data |
| `GET` | `/features` | Feature flag states |
| `PUT` | `/features/{key}` | Toggle features at runtime |

---

## Dashboard

```bash
pip install "ignis_router[dashboard]"
python -m streamlit run examples/streamlit_dashboard.py
```

Visual analytics: KPIs, model distribution, confidence histograms, per-model performance, routing log.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | OpenAI API key |
| `ANTHROPIC_API_KEY` | — | Anthropic API key |
| `GOOGLE_API_KEY` | — | Google Gemini API key |
| `ML_ROUTER_TYPE` | `knn` | Router: `knn`, `svm`, `graph`, `mf` |
| `ROUTER_YAML_CONFIG` | — | Strategy YAML path |
| `ENABLE_ML_MODEL_HINT_ROUTING` | `false` | Enable ML model prediction |
| `ML_CONFIDENCE_THRESHOLD` | `0.60` | Fallback threshold |
| `ROUTER_DB_PASSWORD` | `postgres` | PostgreSQL password |

See [user_guide.md](user_guide.md) for the full environment variable reference.

---

## Documentation

| Resource | Description |
|----------|-------------|
| [User Guide](user_guide.md) | Complete setup, configuration, and usage documentation |
| [Swagger UI](http://127.0.0.1:8080/docs) | Interactive API explorer (when API is running) |
| [Examples](examples/) | Sample scripts (AI chat, routing with DB, Streamlit dashboard) |

---

## Project Structure

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

## Development

```bash
git clone https://github.com/Infogain-GenAI/ignis_router.git
cd ignis_router
pip install -e ".[dev,all,dashboard]"
python -m pytest tests/ -v
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  Built by <a href="https://github.com/Infogain-GenAI">Infogain GenAI</a>
</p>
