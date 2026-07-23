"""Tests for the FastAPI routing service."""

from types import SimpleNamespace

from fastapi.testclient import TestClient

from ignis_router.api.api import create_app
from ignis_router.exceptions import ModelNotAvailableError, RoutingError


class _SuccessfulRouter:
    def __init__(self):
        self.config = SimpleNamespace(routing_strategy="quality-first")

    def route(self, query: str):
        return SimpleNamespace(
            selected_model=SimpleNamespace(model_name="gpt-4.1"),
            confidence=0.923,
        )


class _RoutingErrorRouter:
    def __init__(self):
        self.config = SimpleNamespace(routing_strategy="quality-first")

    def route(self, query: str):
        raise RoutingError("No suitable model found")


class _ModelUnavailableRouter:
    def __init__(self):
        self.config = SimpleNamespace(routing_strategy="quality-first")

    def route(self, query: str):
        raise ModelNotAvailableError("No models are registered or enabled")


class TestRouterApi:
    def test_root_endpoint(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.get("/")

        assert response.status_code == 200
        assert response.json() == {
            "name": "Ignis Router API",
            "status": "ok",
            "docs": "/docs",
        }

    def test_health_endpoint(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_route_endpoint_success(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.post("/route", json={"query": "Generate Python code for data analysis"})

        assert response.status_code == 200
        assert response.json() == {
            "selected_model": "gpt-4.1",
            "strategy": "quality-first",
            "confidence": 0.92,
        }

    def test_route_endpoint_get_success(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.get("/route", params={"query": "Generate Python code for data analysis"})

        assert response.status_code == 200
        assert response.json() == {
            "selected_model": "gpt-4.1",
            "strategy": "quality-first",
            "confidence": 0.92,
        }

    def test_route_endpoint_request_validation(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.post("/route", json={})

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "validation_error"
        assert body["message"] == "Request validation failed"
        assert isinstance(body["details"], list)

    def test_route_endpoint_rejects_blank_query(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.post("/route", json={"query": "   "})

        assert response.status_code == 422
        body = response.json()
        assert body["error"] == "validation_error"
        assert body["message"] == "Request validation failed"

    def test_route_endpoint_routing_error(self):
        app = create_app(enable_db=False, router=_RoutingErrorRouter())
        client = TestClient(app)

        response = client.post("/route", json={"query": "Anything"})

        assert response.status_code == 400
        assert response.json() == {
            "error": "routing_error",
            "message": "No suitable model found",
            "details": [],
        }

    def test_route_endpoint_model_unavailable_error(self):
        app = create_app(enable_db=False, router=_ModelUnavailableRouter())
        client = TestClient(app)

        response = client.post("/route", json={"query": "Anything"})

        assert response.status_code == 503
        assert response.json() == {
            "error": "model_not_available",
            "message": "No models are registered or enabled",
            "details": [],
        }
