"""Tests for sax_llm.registry — model parsing and output type registration."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from sax_llm.registry import (
    get_output_type_registry,
    parse_model_id,
    register_output_type,
    reset_output_type_registry,
    reset_provider_cache,
)

# ---------------------------------------------------------------------------
# parse_model_id
# ---------------------------------------------------------------------------


class TestParseModelId:
    def test_explicit_anthropic(self):
        assert parse_model_id("anthropic:claude-sonnet-4-5-20250929") == (
            "anthropic", "claude-sonnet-4-5-20250929",
        )

    def test_explicit_mistral(self):
        assert parse_model_id("mistral:mistral-large-latest") == (
            "mistral", "mistral-large-latest",
        )

    def test_bare_name_defaults_anthropic(self):
        assert parse_model_id("claude-sonnet-4-5-20250929") == (
            "anthropic", "claude-sonnet-4-5-20250929",
        )

    def test_unknown_provider(self):
        provider, model = parse_model_id("openai:gpt-4")
        assert provider == "openai"
        assert model == "gpt-4"


# ---------------------------------------------------------------------------
# Output type registry
# ---------------------------------------------------------------------------


class TestOutputTypeRegistry:
    def setup_method(self):
        reset_output_type_registry()

    def teardown_method(self):
        reset_output_type_registry()

    def test_register_and_retrieve(self):
        class MyModel(BaseModel):
            value: str

        register_output_type("MyModel", MyModel)
        registry = get_output_type_registry()
        assert "MyModel" in registry
        assert registry["MyModel"] is MyModel

    def test_empty_registry(self):
        assert get_output_type_registry() == {}

    def test_multiple_registrations(self):
        class A(BaseModel):
            x: int

        class B(BaseModel):
            y: str

        register_output_type("A", A)
        register_output_type("B", B)

        registry = get_output_type_registry()
        assert len(registry) == 2
        assert registry["A"] is A
        assert registry["B"] is B

    def test_overwrite(self):
        class V1(BaseModel):
            old: str

        class V2(BaseModel):
            new: str

        register_output_type("Model", V1)
        register_output_type("Model", V2)

        assert get_output_type_registry()["Model"] is V2

    def test_reset(self):
        class M(BaseModel):
            v: int

        register_output_type("M", M)
        assert len(get_output_type_registry()) == 1

        reset_output_type_registry()
        assert len(get_output_type_registry()) == 0


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


class TestGetProvider:
    def setup_method(self):
        reset_provider_cache()

    def teardown_method(self):
        reset_provider_cache()

    def test_unknown_provider_raises(self):
        from sax_llm.registry import get_provider_by_name

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_provider_by_name("openai")
