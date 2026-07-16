# Ignis Router

> Intelligent LLM routing library that selects the best model for every query using ML routers and rule-based intent detection.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Features

- **ML-based routing** — KNN, SVM, Graph, MF routers predict the best LLM (via [LLMRouter](https://github.com/ulab-uiuc/LLMRouter))
- **Rule-based fallback** — intent detection with automatic fallback when ML confidence is low
- **Provider fallback** — auto-switches to available LLM when API key is missing
- **Decorators** — `@route()`, `@chat()` for easy integration into any app
- **REST API** — FastAPI endpoints for routing and chat
- **PostgreSQL logging** — automatic persistence of routing decisions
- **Retrainable** — retrain ML models when new LLMs are introduced

## Quick Install

```bash
pip install git+https://github.com/Infogain-GenAI/ignis_router.git@sakshi_dev_1
```

## Quick Usage

```python
from ignis_router import chat

@chat(system_prompt="You are a helpful assistant")
def ask(query, response):
    print(response["routing_decision"]["ml_router_predicted"])
    print(response["routing_decision"]["final_model"])
    print(response["content"])
    return response

ask("Write a Python sorting function")
```

## Configuration

Create a `.env` file:

```env
OPENAI_API_KEY=sk-your-key-here
ML_ROUTER_TYPE=svm
ENABLE_ML_MODEL_HINT_ROUTING=true
ML_CONFIDENCE_THRESHOLD=0.50
ROUTER_YAML_CONFIG=configs/cost-first.yaml
```

## Run the API

```bash
python -m ignis_router.run_api
# Server at http://127.0.0.1:8080
# Swagger docs at http://127.0.0.1:8080/docs
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/route` | Route a query (pick best model) |
| POST | `/chat` | Route + call LLM + return AI response |

## Architecture

```
User Query
    → ML Router (KNN/SVM/Graph/MF) predicts best model
    → API key check (fallback if unavailable)
    → Call LLM (OpenAI/Anthropic/Gemini)
    → Return response + routing decision
    → Save to PostgreSQL
```

## Documentation

See [user_guide.md](user_guide.md) for complete documentation including:
- Installation steps
- Configuration reference
- Decorator usage
- ML router training
- Database setup
- Troubleshooting

## License

MIT
