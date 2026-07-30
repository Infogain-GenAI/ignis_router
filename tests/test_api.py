"""Tests for the FastAPI routing service."""

from types import SimpleNamespace

from fastapi.testclient import TestClient
import ignis_router.api.api as api_module

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


class _CapturingDashboardEngine:
    last_call = None

    def generate(
        self,
        days: int = 7,
        strategy: str | None = None,
        intent: str | None = None,
        page: int = 1,
        page_size: int = 20,
    ):
        _CapturingDashboardEngine.last_call = {
            "days": days,
            "strategy": strategy,
            "intent": intent,
            "page": page,
            "page_size": page_size,
        }
        return {
            "generated_at": "2026-07-23T00:00:00+00:00",
            "window_hours": days * 24,
            "start_date": "2026-07-22T00:00:00",
            "end_date": "2026-07-23T00:00:00",
            "filters": {
                "strategy": strategy,
                "intent": intent,
            },
            "kpis": {},
            "charts": {},
            "routing_log": [],
            "routing_log_pagination": {
                "page": page,
                "page_size": page_size,
                "total_count": 0,
                "total_pages": 1,
                "has_prev": False,
                "has_next": False,
            },
        }


class TestRouterApi:
    def test_dashboard_endpoint_accepts_days_strategy_and_intent_filters(self, monkeypatch):
        monkeypatch.setattr(api_module, "DashboardEngine", _CapturingDashboardEngine)
        _CapturingDashboardEngine.last_call = None

        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.get(
            "/dashboard",
            params={
                "days": 30,
                "strategy": "quality-first",
                "intent": "general_chat",
                "page": 2,
                "page_size": 15,
            },
        )

        assert response.status_code == 200
        assert _CapturingDashboardEngine.last_call == {
            "days": 30,
            "strategy": "quality-first",
            "intent": "general_chat",
            "page": 2,
            "page_size": 15,
        }
        assert response.json()["filters"] == {
            "strategy": "quality-first",
            "intent": "general_chat",
        }

    def test_dashboard_endpoint_all_filters_behave_like_unfiltered(self, monkeypatch):
        monkeypatch.setattr(api_module, "DashboardEngine", _CapturingDashboardEngine)
        _CapturingDashboardEngine.last_call = None

        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.get(
            "/dashboard",
            params={"days": 7, "strategy": "All", "intent": "all"},
        )

        assert response.status_code == 200
        assert _CapturingDashboardEngine.last_call == {
            "days": 7,
            "strategy": "All",
            "intent": "all",
            "page": 1,
            "page_size": 20,
        }
        assert response.json()["filters"] == {
            "strategy": "All",
            "intent": "all",
        }

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


class TestFeatureFlagsApi:
    """Tests for GET /features and PUT /features/{key} endpoints."""

    def test_get_features_returns_all_flags(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.get("/features")

        assert response.status_code == 200
        body = response.json()
        assert "features" in body
        assert "summary" in body
        assert "available_keys" in body
        assert "routing" in body["features"]

    def test_get_features_summary_has_all_keys(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.get("/features")
        summary = response.json()["summary"]

        assert "enable_ml_model_hint_routing" in summary
        assert "enable_rule_based_intent_detection" in summary
        assert "enable_ml_intent_detection" in summary

    def test_put_feature_toggle_off(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.put("/features/ml_based_routing", params={"enabled": False})

        assert response.status_code == 200
        body = response.json()
        assert body["key"] == "ml_based_routing"
        assert body["enabled"] is False
        assert "disabled" in body["message"]

    def test_put_feature_toggle_on(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.put("/features/hybrid_routing", params={"enabled": True})

        assert response.status_code == 200
        assert response.json()["enabled"] is True

    def test_put_feature_persists_in_get(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        client.put("/features/rule_based_routing", params={"enabled": False})
        response = client.get("/features")
        summary = response.json()["summary"]

        assert summary["enable_rule_based_intent_detection"] is False

    def test_put_feature_invalid_key_returns_422(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        response = client.put("/features/nonexistent_flag", params={"enabled": True})

        assert response.status_code == 422

    def test_dashboard_includes_feature_flags(self):
        app = create_app(enable_db=False, router=_SuccessfulRouter())
        client = TestClient(app)

        import ignis_router.api.api as mod
        original = mod.DashboardEngine
        mod.DashboardEngine = _CapturingDashboardEngine
        try:
            response = client.get("/dashboard")
            assert response.status_code == 200
            body = response.json()
            assert "feature_flags" in body
            assert "summary" in body["feature_flags"]
        finally:
            mod.DashboardEngine = original
