"""Tests for model registry behavior and Story 2 metadata support."""

from ignis_router import ModelConfig, Router
from ignis_router.config import RouterRegistry
from ignis_router.supported_models import get_default_supported_models


class TestRegistryMetadata:
    def test_model_metadata_fields_supported(self):
        model = ModelConfig(
            model_id="gpt-4.1",
            provider="openai",
            model_name="gpt-4.1",
            cost_per_1k_input_tokens=0.8,
            latency=0.7,
            quality=0.95,
            reliability=0.98,
        )

        assert model.cost_per_1k_input_tokens == 0.8
        assert model.latency == 0.7
        assert model.quality == 0.95
        assert model.reliability == 0.98

    def test_register_and_retrieve_model_metadata(self):
        registry = RouterRegistry()
        model = ModelConfig(
            model_id="test-model",
            provider="test",
            model_name="test-model-v1",
            cost_per_1k_input_tokens=0.2,
            latency=0.6,
            quality=0.85,
            reliability=0.9,
        )

        registry.register_model(model)
        loaded = registry.get_model("test-model")

        assert loaded is not None
        assert loaded.cost_per_1k_input_tokens == 0.2
        assert loaded.latency == 0.6
        assert loaded.quality == 0.85
        assert loaded.reliability == 0.9


class TestSupportedModelRegistration:
    def test_registry_bulk_registration(self):
        registry = RouterRegistry()
        models = get_default_supported_models()

        registry.register_models(models)

        assert len(registry.get_all_models()) == len(models)

    def test_router_register_supported_models(self):
        router = Router()

        router.register_supported_models()

        registered_ids = {m.model_id for m in router.get_registered_models()}
        expected_ids = {m.model_id for m in get_default_supported_models()}
        assert expected_ids.issubset(registered_ids)
