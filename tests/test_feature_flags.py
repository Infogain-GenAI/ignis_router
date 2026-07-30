"""Tests for the feature flags system."""

import pytest

from ignis_router.config import RouterConfig
from ignis_router.feature_flags import FeatureFlags


class TestFeatureFlags:
    """Tests for FeatureFlags toggle, get, set, and serialization."""

    def setup_method(self):
        self.config = RouterConfig()
        self.flags = FeatureFlags.from_config(self.config)

    def test_get_all_returns_all_flags(self):
        all_flags = self.flags.get_all()
        assert len(all_flags) == 3
        keys = {f.key for f in all_flags}
        assert "enable_ml_model_hint_routing" in keys
        assert "enable_rule_based_intent_detection" in keys
        assert "enable_ml_intent_detection" in keys

    def test_get_single_flag(self):
        flag = self.flags.get("enable_ml_model_hint_routing")
        assert flag is not None
        assert flag.name == "ML Based Routing"
        assert flag.category == "routing"

    def test_get_unknown_flag_returns_none(self):
        assert self.flags.get("nonexistent_flag") is None

    def test_is_enabled_reads_from_config(self):
        assert self.flags.is_enabled("enable_ml_intent_detection") == self.config.enable_ml_intent_detection

    def test_set_updates_config(self):
        original = self.config.enable_ml_model_hint_routing
        self.flags.set("enable_ml_model_hint_routing", not original)
        assert self.config.enable_ml_model_hint_routing == (not original)
        assert self.flags.is_enabled("enable_ml_model_hint_routing") == (not original)

    def test_toggle_flips_value(self):
        original = self.flags.is_enabled("enable_ml_intent_detection")
        new_value = self.flags.toggle("enable_ml_intent_detection")
        assert new_value == (not original)
        assert self.flags.is_enabled("enable_ml_intent_detection") == (not original)

    def test_toggle_twice_restores_original(self):
        original = self.flags.is_enabled("enable_rule_based_intent_detection")
        self.flags.toggle("enable_rule_based_intent_detection")
        self.flags.toggle("enable_rule_based_intent_detection")
        assert self.flags.is_enabled("enable_rule_based_intent_detection") == original

    def test_set_unknown_key_raises(self):
        with pytest.raises(KeyError, match="Unknown feature flag"):
            self.flags.set("totally_invalid_key", True)

    def test_override_takes_priority_over_config(self):
        self.config.enable_ml_model_hint_routing = True
        self.flags.set("enable_ml_model_hint_routing", False)
        assert self.flags.is_enabled("enable_ml_model_hint_routing") is False

    def test_to_dict_structure(self):
        result = self.flags.to_dict()
        assert "features" in result
        assert "summary" in result
        assert "available_keys" in result
        assert "usage" in result

        # All flags in routing category
        assert "routing" in result["features"]
        routing_flags = result["features"]["routing"]
        assert len(routing_flags) == 3

        # Each flag has required fields
        for flag in routing_flags:
            assert "key" in flag
            assert "name" in flag
            assert "description" in flag
            assert "enabled" in flag
            assert "env_var" in flag
            assert "toggle_url" in flag

    def test_to_dict_summary_matches_flags(self):
        result = self.flags.to_dict()
        for item in result["available_keys"]:
            key = item["key"]
            assert key in result["summary"]
            assert result["summary"][key] == item["enabled"]

    def test_to_dict_available_keys_has_clean_names(self):
        result = self.flags.to_dict()
        names = {item["name"] for item in result["available_keys"]}
        assert "ML Based Routing" in names
        assert "Rule Based Routing" in names
        assert "Hybrid Routing" in names

    def test_set_reflects_in_to_dict(self):
        self.flags.set("enable_ml_model_hint_routing", False)
        result = self.flags.to_dict()
        assert result["summary"]["enable_ml_model_hint_routing"] is False

    def test_usage_contains_examples(self):
        result = self.flags.to_dict()
        assert "toggle" in result["usage"]
        assert "example_enable" in result["usage"]
        assert "example_disable" in result["usage"]
