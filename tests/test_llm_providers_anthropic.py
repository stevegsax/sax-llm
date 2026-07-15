"""Tests for AnthropicProvider."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from sax_llm.anthropic import AnthropicProvider
from sax_llm.models import (
    DocumentContent,
    ImageContent,
    Message,
    TextContent,
    text_messages,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_message(
    tool_input: dict,
    *,
    model: str = "claude-sonnet-4-5-20250929",
    input_tokens: int = 100,
    output_tokens: int = 200,
    cache_creation_input_tokens: int = 10,
    cache_read_input_tokens: int = 50,
) -> MagicMock:
    """Build a mock Anthropic Message."""
    tool_block = MagicMock()
    tool_block.type = "tool_use"
    tool_block.input = tool_input

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = cache_creation_input_tokens
    usage.cache_read_input_tokens = cache_read_input_tokens

    message = MagicMock()
    message.content = [tool_block]
    message.usage = usage
    message.model = model
    message.model_dump_json = MagicMock(return_value=json.dumps({"model": model}))

    return message


def _make_mock_text_message(
    text: str,
    *,
    model: str = "claude-sonnet-4-5-20250929",
    input_tokens: int = 100,
    output_tokens: int = 50,
) -> MagicMock:
    """Build a mock Anthropic Message with text-only content."""
    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text

    usage = MagicMock()
    usage.input_tokens = input_tokens
    usage.output_tokens = output_tokens
    usage.cache_creation_input_tokens = 0
    usage.cache_read_input_tokens = 0

    message = MagicMock()
    message.content = [text_block]
    message.usage = usage
    message.model = model
    message.model_dump_json = MagicMock(return_value=json.dumps({"model": model}))

    return message


class SampleOutput:
    """Test output type."""

    __name__ = "SampleOutput"
    __doc__ = "A sample output."

    @staticmethod
    def model_json_schema() -> dict:
        return {"type": "object", "properties": {"result": {"type": "string"}}}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBuildRequestParams:
    """Tests for AnthropicProvider.build_request_params."""

    def test_builds_with_messages_and_output_type(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = AnthropicProvider()
        params = provider.build_request_params(
            messages=text_messages("You are helpful.", "Do something."),
            output_type=TestOutput,
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
        )
        assert params["model"] == "claude-sonnet-4-5-20250929"
        assert params["max_tokens"] == 1024
        assert len(params["tools"]) == 1
        assert params["tools"][0]["name"] == "test_output"

    def test_passes_thinking_budget(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = AnthropicProvider()
        params = provider.build_request_params(
            messages=text_messages("sys", "user"),
            output_type=TestOutput,
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            thinking_budget_tokens=5000,
        )
        assert "thinking" in params
        assert params["thinking"]["budget_tokens"] == 5000

    def test_cache_control_disabled(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = AnthropicProvider()
        params = provider.build_request_params(
            messages=text_messages("sys", "user", cache_system=False),
            output_type=TestOutput,
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
            cache_instructions=False,
            cache_tool_definitions=False,
        )
        # System should be plain string without cache control
        assert isinstance(params["system"], str)
        # Tool should not have cache_control
        assert "cache_control" not in params["tools"][0]

    def test_output_type_none_omits_tools(self) -> None:
        provider = AnthropicProvider()
        params = provider.build_request_params(
            messages=text_messages("sys", "user"),
            output_type=None,
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
        )
        assert "tools" not in params
        assert "tool_choice" not in params

    def test_multimodal_image_content(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        messages = [
            Message(role="system", content="Describe the image."),
            Message(
                role="user",
                content=[
                    ImageContent(media_type="image/png", data="base64data"),
                    TextContent(text="What is this?"),
                ],
            ),
        ]
        provider = AnthropicProvider()
        params = provider.build_request_params(
            messages=messages,
            output_type=TestOutput,
            model="claude-sonnet-4-5-20250929",
            max_tokens=1024,
        )
        # User message should have multimodal content
        user_msg = params["messages"][0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "image"
        assert user_msg["content"][0]["source"]["media_type"] == "image/png"
        assert user_msg["content"][1]["type"] == "text"

    def test_multimodal_document_content(self) -> None:
        messages = [
            Message(role="system", content="Extract text from this PDF."),
            Message(
                role="user",
                content=[
                    DocumentContent(
                        media_type="application/pdf",
                        data="pdfbase64",
                    ),
                    TextContent(text="OCR this document."),
                ],
            ),
        ]
        provider = AnthropicProvider()
        params = provider.build_request_params(
            messages=messages,
            output_type=None,
            model="claude-sonnet-4-5-20250929",
            max_tokens=8192,
        )
        user_msg = params["messages"][0]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "document"
        assert user_msg["content"][0]["source"]["media_type"] == "application/pdf"


class TestCall:
    """Tests for AnthropicProvider.call."""

    @pytest.mark.asyncio
    async def test_returns_provider_response(self) -> None:
        tool_input = {"files": [], "edits": [], "explanation": "done"}
        message = _make_mock_message(tool_input)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=message)

        provider = AnthropicProvider()
        provider._get_client = lambda: mock_client

        params = {
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 1024,
            "tools": [{"name": "test"}],
        }
        result = await provider.call(params)

        assert result.tool_input == tool_input
        assert result.model_name == "claude-sonnet-4-5-20250929"
        assert result.input_tokens == 100
        assert result.output_tokens == 200
        assert result.cache_creation_input_tokens == 10
        assert result.cache_read_input_tokens == 50

    @pytest.mark.asyncio
    async def test_raw_response_json_populated(self) -> None:
        tool_input = {"result": "ok"}
        message = _make_mock_message(tool_input)

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=message)

        provider = AnthropicProvider()
        provider._get_client = lambda: mock_client

        result = await provider.call({"model": "test", "tools": [{"name": "t"}]})
        assert result.raw_response_json  # non-empty

    @pytest.mark.asyncio
    async def test_no_tools_extracts_text_content(self) -> None:
        message = _make_mock_text_message("Hello world")

        mock_client = AsyncMock()
        mock_client.messages.create = AsyncMock(return_value=message)

        provider = AnthropicProvider()
        provider._get_client = lambda: mock_client

        result = await provider.call({"model": "test"})
        assert result.text_content == "Hello world"
        assert result.tool_input == {}


class TestSupportsBatch:
    """Tests for batch support flag."""

    def test_supports_batch_is_true(self) -> None:
        provider = AnthropicProvider()
        assert provider.supports_batch is True


class TestBuildBatchRequest:
    """Tests for AnthropicProvider.build_batch_request."""

    def test_wraps_params(self) -> None:
        provider = AnthropicProvider()
        result = provider.build_batch_request("req-123", {"model": "test"})
        assert result["custom_id"] == "req-123"
        assert result["params"]["model"] == "test"
