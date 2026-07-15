"""Tests for sax_llm.client — shared LLM client utilities."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest
from pydantic import BaseModel, Field

from sax_llm import get_output_type_registry, register_output_type
from sax_llm.client import (
    build_batch_request,
    build_messages_params,
    build_system_param,
    build_thinking_param,
    build_tool_definition,
    extract_tool_result,
    extract_usage,
    get_anthropic_client,
    parse_batch_response_json,
    reset_client,
)
from sax_llm.registry import reset_output_type_registry

# ---------------------------------------------------------------------------
# Test models
# ---------------------------------------------------------------------------


class SampleOutput(BaseModel):
    """A sample structured output for testing."""

    name: str = Field(description="A name.")
    value: int = Field(description="A value.")


class CamelCaseModel(BaseModel):
    """Model with multi-word CamelCase name."""

    data: str


# ---------------------------------------------------------------------------
# build_tool_definition
# ---------------------------------------------------------------------------


class TestBuildToolDefinition:
    def test_name_is_snake_case(self) -> None:
        tool = build_tool_definition(SampleOutput)
        assert tool["name"] == "sample_output"

    def test_camel_case_conversion(self) -> None:
        tool = build_tool_definition(CamelCaseModel)
        assert tool["name"] == "camel_case_model"

    def test_description_from_docstring(self) -> None:
        tool = build_tool_definition(SampleOutput)
        assert tool["description"] == "A sample structured output for testing."

    def test_input_schema_has_properties(self) -> None:
        tool = build_tool_definition(SampleOutput)
        schema = tool["input_schema"]
        assert "name" in schema["properties"]
        assert "value" in schema["properties"]

    def test_cache_control_added_by_default(self) -> None:
        tool = build_tool_definition(SampleOutput)
        assert tool["cache_control"] == {"type": "ephemeral"}

    def test_cache_control_omitted_when_disabled(self) -> None:
        tool = build_tool_definition(SampleOutput, cache_control=False)
        assert "cache_control" not in tool


# ---------------------------------------------------------------------------
# build_system_param
# ---------------------------------------------------------------------------


class TestBuildSystemParam:
    def test_with_cache_control(self) -> None:
        result = build_system_param("You are helpful.", cache_control=True)
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "You are helpful."
        assert result[0]["cache_control"] == {"type": "ephemeral"}

    def test_without_cache_control(self) -> None:
        result = build_system_param("You are helpful.", cache_control=False)
        assert result == "You are helpful."


# ---------------------------------------------------------------------------
# build_thinking_param
# ---------------------------------------------------------------------------


class TestBuildThinkingParam:
    def test_opus_gets_enabled(self) -> None:
        result = build_thinking_param("claude-opus-4-6", 10_000)
        assert result == {"type": "enabled", "budget_tokens": 10_000}

    def test_sonnet_gets_budget(self) -> None:
        result = build_thinking_param("claude-sonnet-4-5-20250929", 10_000)
        assert result == {"type": "enabled", "budget_tokens": 10_000}

    def test_haiku_returns_none(self) -> None:
        result = build_thinking_param("claude-haiku-4-5-20251001", 10_000)
        assert result is None

    def test_zero_budget_returns_none(self) -> None:
        result = build_thinking_param("claude-opus-4-6", 0)
        assert result is None


# ---------------------------------------------------------------------------
# build_messages_params
# ---------------------------------------------------------------------------


class TestBuildMessagesParams:
    def test_basic_structure(self) -> None:
        params = build_messages_params(
            system_prompt="Be helpful.",
            user_prompt="Do it.",
            output_type=SampleOutput,
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
        )
        assert params["model"] == "claude-sonnet-4-5-20250929"
        assert params["max_tokens"] == 4096
        assert len(params["messages"]) == 1
        assert params["messages"][0]["role"] == "user"
        assert params["messages"][0]["content"] == "Do it."
        assert len(params["tools"]) == 1
        assert params["tool_choice"] == {"type": "tool", "name": "sample_output"}

    def test_no_thinking_by_default(self) -> None:
        params = build_messages_params(
            system_prompt="sys",
            user_prompt="usr",
            output_type=SampleOutput,
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
        )
        assert "thinking" not in params

    def test_thinking_adds_param_and_changes_tool_choice(self) -> None:
        params = build_messages_params(
            system_prompt="sys",
            user_prompt="usr",
            output_type=SampleOutput,
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            thinking_budget_tokens=10_000,
        )
        assert params["thinking"] == {"type": "enabled", "budget_tokens": 10_000}
        assert params["tool_choice"] == {"type": "auto"}
        assert params["max_tokens"] >= 14096

    def test_no_cache_disables_cache_control(self) -> None:
        params = build_messages_params(
            system_prompt="sys",
            user_prompt="usr",
            output_type=SampleOutput,
            model="claude-sonnet-4-5-20250929",
            max_tokens=4096,
            cache_instructions=False,
            cache_tool_definitions=False,
        )
        assert params["system"] == "sys"
        assert "cache_control" not in params["tools"][0]


# ---------------------------------------------------------------------------
# extract_tool_result
# ---------------------------------------------------------------------------


class TestExtractToolResult:
    def test_extracts_tool_use_block(self) -> None:
        mock_block = MagicMock()
        mock_block.type = "tool_use"
        mock_block.input = {"name": "test", "value": 42}

        mock_message = MagicMock()
        mock_message.content = [mock_block]

        result = extract_tool_result(mock_message, SampleOutput)
        assert isinstance(result, SampleOutput)
        assert result.name == "test"
        assert result.value == 42

    def test_skips_text_blocks(self) -> None:
        text_block = MagicMock()
        text_block.type = "text"

        tool_block = MagicMock()
        tool_block.type = "tool_use"
        tool_block.input = {"name": "found", "value": 1}

        mock_message = MagicMock()
        mock_message.content = [text_block, tool_block]

        result = extract_tool_result(mock_message, SampleOutput)
        assert result.name == "found"

    def test_raises_when_no_tool_use(self) -> None:
        text_block = MagicMock()
        text_block.type = "text"

        mock_message = MagicMock()
        mock_message.content = [text_block]

        with pytest.raises(ValueError, match="No tool_use block found"):
            extract_tool_result(mock_message, SampleOutput)


# ---------------------------------------------------------------------------
# extract_usage
# ---------------------------------------------------------------------------


class TestExtractUsage:
    def test_extracts_all_fields(self) -> None:
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 200
        mock_usage.cache_creation_input_tokens = 50
        mock_usage.cache_read_input_tokens = 75

        mock_message = MagicMock()
        mock_message.usage = mock_usage

        in_tok, out_tok, cache_create, cache_read = extract_usage(mock_message)
        assert in_tok == 100
        assert out_tok == 200
        assert cache_create == 50
        assert cache_read == 75

    def test_handles_missing_cache_fields(self) -> None:
        mock_usage = MagicMock(spec=["input_tokens", "output_tokens"])
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 200

        mock_message = MagicMock()
        mock_message.usage = mock_usage

        in_tok, out_tok, cache_create, cache_read = extract_usage(mock_message)
        assert in_tok == 100
        assert out_tok == 200
        assert cache_create == 0
        assert cache_read == 0

    def test_handles_none_cache_fields(self) -> None:
        mock_usage = MagicMock()
        mock_usage.input_tokens = 100
        mock_usage.output_tokens = 200
        mock_usage.cache_creation_input_tokens = None
        mock_usage.cache_read_input_tokens = None

        mock_message = MagicMock()
        mock_message.usage = mock_usage

        _in_tok, _out_tok, cache_create, cache_read = extract_usage(mock_message)
        assert cache_create == 0
        assert cache_read == 0


# ---------------------------------------------------------------------------
# build_batch_request
# ---------------------------------------------------------------------------


class TestBuildBatchRequest:
    def test_wraps_params_with_custom_id(self) -> None:
        params = {"model": "claude-sonnet-4-5-20250929", "max_tokens": 4096}
        result = build_batch_request("req-123", params)

        assert result["custom_id"] == "req-123"
        assert result["params"] is params

    def test_preserves_params_unchanged(self) -> None:
        params = {"model": "test", "system": "hello", "messages": []}
        result = build_batch_request("id-456", params)

        assert result["params"] == params


# ---------------------------------------------------------------------------
# get_output_type_registry
# ---------------------------------------------------------------------------


class TestGetOutputTypeRegistry:
    """The registry is a plugin surface: consumers register their own types."""

    def setup_method(self) -> None:
        reset_output_type_registry()

    def teardown_method(self) -> None:
        reset_output_type_registry()

    def test_returns_registered_types(self) -> None:
        register_output_type("SampleOutput", SampleOutput)
        register_output_type("CamelCaseModel", CamelCaseModel)
        registry = get_output_type_registry()
        assert set(registry.keys()) == {"SampleOutput", "CamelCaseModel"}

    def test_values_are_base_model_subclasses(self) -> None:
        register_output_type("SampleOutput", SampleOutput)
        registry = get_output_type_registry()
        for name, cls in registry.items():
            assert issubclass(cls, BaseModel), f"{name} is not a BaseModel subclass"


# ---------------------------------------------------------------------------
# parse_batch_response_json
# ---------------------------------------------------------------------------


def _build_message_json(
    tool_name: str,
    tool_input: dict,
    *,
    model: str = "claude-sonnet-4-5-20250929",
    input_tokens: int = 100,
    output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> str:
    """Build a minimal valid Anthropic Message JSON string for testing."""
    message = {
        "id": "msg_test123",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": [
            {
                "type": "tool_use",
                "id": "toolu_test123",
                "name": tool_name,
                "input": tool_input,
            }
        ],
        "stop_reason": "tool_use",
        "stop_sequence": None,
        "usage": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_creation_input_tokens": cache_creation_input_tokens,
            "cache_read_input_tokens": cache_read_input_tokens,
        },
    }
    return json.dumps(message)


class TestParseBatchResponseJson:
    """Parsing looks up the type by name in the plugin registry and extracts
    the first tool_use block — so a locally registered model exercises it fully."""

    def setup_method(self) -> None:
        reset_output_type_registry()
        register_output_type("SampleOutput", SampleOutput)

    def teardown_method(self) -> None:
        reset_output_type_registry()

    def test_parses_registered_type(self) -> None:
        raw = _build_message_json("sample_output", {"name": "widget", "value": 7})
        parsed, model_name, in_tok, out_tok, _, _ = parse_batch_response_json(raw, "SampleOutput")

        assert isinstance(parsed, SampleOutput)
        assert parsed.name == "widget"
        assert parsed.value == 7
        assert model_name == "claude-sonnet-4-5-20250929"
        assert in_tok == 100
        assert out_tok == 200

    def test_raises_key_error_for_unknown_type(self) -> None:
        with pytest.raises(KeyError, match="Unknown output type"):
            parse_batch_response_json("{}", "NonExistentType")

    def test_raises_value_error_for_no_tool_use(self) -> None:
        raw_dict = json.loads(_build_message_json("sample_output", {"name": "x", "value": 1}))
        raw_dict["content"] = [{"type": "text", "text": "hello"}]
        raw = json.dumps(raw_dict)

        with pytest.raises(ValueError, match="No tool_use block found"):
            parse_batch_response_json(raw, "SampleOutput")

    def test_extracts_usage(self) -> None:
        raw = _build_message_json(
            "sample_output",
            {"name": "x", "value": 1},
            input_tokens=500,
            output_tokens=300,
            cache_creation_input_tokens=50,
            cache_read_input_tokens=75,
        )
        _, _, in_tok, out_tok, cache_create, cache_read = parse_batch_response_json(
            raw, "SampleOutput"
        )
        assert in_tok == 500
        assert out_tok == 300
        assert cache_create == 50
        assert cache_read == 75


# ---------------------------------------------------------------------------
# get_anthropic_client (imperative shell)
# ---------------------------------------------------------------------------


class TestGetAnthropicClient:
    """SDK-level retries must be disabled: they belong to the consumer's durable
    retry layer, not the Anthropic SDK (stacking hides 429/529 backoff)."""

    def test_constructs_client_with_retries_disabled(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        reset_client()
        captured: dict[str, object] = {}

        class _FakeAsyncAnthropic:
            def __init__(self, **kwargs: object) -> None:
                captured.update(kwargs)

        monkeypatch.setattr("anthropic.AsyncAnthropic", _FakeAsyncAnthropic)
        try:
            get_anthropic_client()
        finally:
            reset_client()

        assert captured["max_retries"] == 0
