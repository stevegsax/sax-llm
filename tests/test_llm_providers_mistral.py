"""Tests for MistralProvider."""

from __future__ import annotations

import json
import logging
from io import BytesIO
from unittest.mock import AsyncMock, MagicMock

import pytest

from sax_llm.mistral import (
    MistralProvider,
    _download_file_content,
    _extract_images_from_response,
    _format_batch_errors,
    _is_set,
    _parse_error_file_entries,
)
from sax_llm.models import (
    BatchPollStatus,
    DocumentContent,
    ImageContent,
    Message,
    TextContent,
    text_messages,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(
    tool_input: dict,
    *,
    model: str = "mistral-large-latest",
    prompt_tokens: int = 80,
    completion_tokens: int = 150,
) -> MagicMock:
    """Build a mock Mistral ChatCompletionResponse."""
    func = MagicMock()
    func.arguments = json.dumps(tool_input)

    tool_call = MagicMock()
    tool_call.function = func

    message = MagicMock()
    message.tool_calls = [tool_call]

    choice = MagicMock()
    choice.message = message

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens

    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    response.model = model
    response.model_dump = MagicMock(return_value={"model": model, "choices": []})

    return response


def _make_mock_batch_job(
    status: str = "SUCCESS",
    *,
    output_file: str | None = "file-output-default",
    error_file: object | None = None,
    errors: list | None = None,
    failed_requests: int = 0,
    total_requests: int = 10,
) -> MagicMock:
    """Build a mock Mistral BatchJobOut with explicit error-related fields.

    Using explicit values prevents MagicMock from auto-creating truthy stubs
    for ``errors``, ``error_file``, and ``failed_requests``.
    """
    job = MagicMock()
    job.status = status
    job.output_file = output_file
    job.error_file = error_file
    job.errors = errors or []
    job.failed_requests = failed_requests
    job.total_requests = total_requests
    return job


def _make_mock_file(content: str) -> MagicMock:
    """Build a mock Mistral file download response (async httpx-style)."""
    mock_file = MagicMock(spec=[])  # spec=[] prevents auto-attribute creation
    mock_file.aread = AsyncMock(return_value=content.encode("utf-8"))
    return mock_file


def _make_batch_choice(arguments: str = "{}") -> dict:
    """Build a minimal successful Mistral batch response body with a tool call."""
    return {
        "choices": [{"message": {"tool_calls": [{"function": {"arguments": arguments}}]}}],
    }


# ---------------------------------------------------------------------------
# Tests — helper functions
# ---------------------------------------------------------------------------


class TestIsSet:
    """Tests for _is_set sentinel guard."""

    def test_none_is_not_set(self) -> None:
        assert _is_set(None) is False

    def test_unset_sentinel_is_not_set(self) -> None:
        """The Mistral SDK UNSET sentinel is falsy but not None."""

        class _Unset:
            def __bool__(self) -> bool:
                return False

        assert _is_set(_Unset()) is False

    def test_valid_string_is_set(self) -> None:
        assert _is_set("file-abc-123") is True

    def test_empty_string_is_not_set(self) -> None:
        assert _is_set("") is False


class TestFormatBatchErrors:
    """Tests for _format_batch_errors."""

    def test_empty_list(self) -> None:
        assert _format_batch_errors([]) == ""

    def test_single_error_count_one(self) -> None:
        err = MagicMock()
        err.message = "rate limit exceeded"
        err.count = 1
        assert _format_batch_errors([err]) == "rate limit exceeded"

    def test_single_error_count_greater_than_one(self) -> None:
        err = MagicMock()
        err.message = "context length exceeded"
        err.count = 5
        assert _format_batch_errors([err]) == "context length exceeded (x5)"

    def test_multiple_errors(self) -> None:
        err1 = MagicMock()
        err1.message = "error A"
        err1.count = 1
        err2 = MagicMock()
        err2.message = "error B"
        err2.count = 3
        result = _format_batch_errors([err1, err2])
        assert result == "error A; error B (x3)"

    def test_fallback_to_str_when_no_message_attr(self) -> None:
        """When error objects lack .message, fall back to str()."""
        result = _format_batch_errors(["plain string error"])
        assert result == "plain string error"


class TestParseErrorFileEntries:
    """Tests for _parse_error_file_entries."""

    def test_empty_content(self) -> None:
        assert _parse_error_file_entries("") == []

    def test_blank_lines_skipped(self) -> None:
        assert _parse_error_file_entries("\n\n  \n") == []

    def test_standard_error_format(self) -> None:
        line = json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "body": {
                        "error": {"type": "invalid_request", "message": "bad input"},
                    }
                },
            }
        )
        entries = _parse_error_file_entries(line)
        assert len(entries) == 1
        assert entries[0].custom_id == "req-1"
        assert entries[0].succeeded is False
        assert "bad input" in entries[0].error

    def test_top_level_error_key(self) -> None:
        line = json.dumps(
            {
                "custom_id": "req-2",
                "error": {"message": "server error"},
            }
        )
        entries = _parse_error_file_entries(line)
        assert len(entries) == 1
        assert entries[0].custom_id == "req-2"
        assert "server error" in entries[0].error

    def test_malformed_json_skipped(self, caplog: pytest.LogCaptureFixture) -> None:
        content = "not valid json\n" + json.dumps(
            {
                "custom_id": "req-ok",
                "error": {"message": "real error"},
            }
        )
        with caplog.at_level(logging.WARNING):
            entries = _parse_error_file_entries(content)
        assert len(entries) == 1
        assert entries[0].custom_id == "req-ok"
        assert "malformed" in caplog.text.lower()

    def test_missing_custom_id_defaults_to_unknown(self) -> None:
        line = json.dumps({"error": {"message": "oops"}})
        entries = _parse_error_file_entries(line)
        assert entries[0].custom_id == "unknown"

    def test_multiple_lines(self) -> None:
        lines = "\n".join(
            [
                json.dumps({"custom_id": "r1", "error": {"message": "e1"}}),
                json.dumps({"custom_id": "r2", "error": {"message": "e2"}}),
            ]
        )
        entries = _parse_error_file_entries(lines)
        assert len(entries) == 2
        assert entries[0].custom_id == "r1"
        assert entries[1].custom_id == "r2"


class TestDownloadFileContent:
    """Tests for _download_file_content."""

    @pytest.mark.asyncio
    async def test_file_like_object(self) -> None:
        client = MagicMock()
        file_obj = BytesIO(b"hello world")
        client.files.download_async = AsyncMock(return_value=file_obj)

        result = await _download_file_content(client, "file-123")

        assert result == "hello world"

    @pytest.mark.asyncio
    async def test_plain_string(self) -> None:
        client = MagicMock()
        client.files.download_async = AsyncMock(return_value="raw string content")

        result = await _download_file_content(client, "file-456")

        assert result == "raw string content"


# ---------------------------------------------------------------------------
# Tests — MistralProvider
# ---------------------------------------------------------------------------


class TestBuildRequestParams:
    """Tests for MistralProvider.build_request_params."""

    def test_builds_mistral_format(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = provider.build_request_params(
            messages=text_messages("You are helpful.", "Do something."),
            output_type=TestOutput,
            model="mistral-large-latest",
            max_tokens=1024,
        )

        assert params["model"] == "mistral-large-latest"
        assert params["max_tokens"] == 1024
        assert params["tool_choice"] == "any"

        # Messages: system + user
        assert len(params["messages"]) == 2
        assert params["messages"][0]["role"] == "system"
        assert params["messages"][1]["role"] == "user"

        # Tool definition in function format
        tool = params["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "test_output"

    def test_no_cache_control(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = provider.build_request_params(
            messages=text_messages("sys", "user"),
            output_type=TestOutput,
            model="mistral-large-latest",
            max_tokens=1024,
            cache_instructions=True,
            cache_tool_definitions=True,
        )

        # No cache_control on messages or tools
        for msg in params["messages"]:
            assert "cache_control" not in msg
        assert "cache_control" not in params["tools"][0]

    def test_thinking_budget_silently_ignored(self) -> None:
        from pydantic import BaseModel

        class TestOutput(BaseModel):
            """Test output model."""

            value: str

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = provider.build_request_params(
            messages=text_messages("sys", "user"),
            output_type=TestOutput,
            model="mistral-large-latest",
            max_tokens=1024,
            thinking_budget_tokens=5000,
        )

        # No thinking param in output
        assert "thinking" not in params

    def test_output_type_none_omits_tools(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = provider.build_request_params(
            messages=text_messages("sys", "user"),
            output_type=None,
            model="mistral-large-latest",
            max_tokens=1024,
        )

        assert "tools" not in params
        assert "tool_choice" not in params

    def test_multimodal_image_content(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

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
        params = provider.build_request_params(
            messages=messages,
            output_type=None,
            model="pixtral-large-latest",
            max_tokens=4096,
        )

        user_msg = params["messages"][1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "image_url"
        assert "data:image/png;base64," in user_msg["content"][0]["image_url"]
        assert user_msg["content"][1]["type"] == "text"

    def test_multimodal_document_content(self) -> None:
        """DocumentContent (e.g. PDF) should use Mistral's document_url type."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        messages = [
            Message(role="system", content="Extract text from this document."),
            Message(
                role="user",
                content=[
                    DocumentContent(media_type="application/pdf", data="JVBERi0="),
                    TextContent(text="Summarize."),
                ],
            ),
        ]
        params = provider.build_request_params(
            messages=messages,
            output_type=None,
            model="mistral-large-latest",
            max_tokens=4096,
        )

        user_msg = params["messages"][1]
        assert isinstance(user_msg["content"], list)
        assert user_msg["content"][0]["type"] == "document_url"
        assert user_msg["content"][0]["document_url"] == "data:application/pdf;base64,JVBERi0="
        assert user_msg["content"][1]["type"] == "text"


class TestCall:
    """Tests for MistralProvider.call."""

    @pytest.mark.asyncio
    async def test_extracts_tool_input(self) -> None:
        tool_input = {"value": "hello"}
        mock_response = _make_mock_response(tool_input)

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call(
            {
                "model": "mistral-large-latest",
                "tools": [{"type": "function"}],
            }
        )

        assert result.tool_input == tool_input

    @pytest.mark.asyncio
    async def test_maps_usage_tokens(self) -> None:
        mock_response = _make_mock_response(
            {"value": "ok"}, prompt_tokens=120, completion_tokens=300
        )

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call(
            {
                "model": "test",
                "tools": [{"type": "function"}],
            }
        )

        assert result.input_tokens == 120
        assert result.output_tokens == 300

    @pytest.mark.asyncio
    async def test_cache_tokens_are_zero(self) -> None:
        mock_response = _make_mock_response({"value": "ok"})

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call(
            {
                "model": "test",
                "tools": [{"type": "function"}],
            }
        )

        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0

    @pytest.mark.asyncio
    async def test_model_name_from_response(self) -> None:
        mock_response = _make_mock_response({"value": "ok"}, model="mistral-large-2")

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=mock_response)

        result = await provider.call(
            {
                "model": "mistral-large-latest",
                "tools": [{"type": "function"}],
            }
        )

        assert result.model_name == "mistral-large-2"

    @pytest.mark.asyncio
    async def test_no_tools_extracts_text_content(self) -> None:
        """When no tools are in params, extract text content from response."""
        message = MagicMock()
        message.content = "Extracted text from image"
        message.tool_calls = None

        choice = MagicMock()
        choice.message = message

        usage = MagicMock()
        usage.prompt_tokens = 50
        usage.completion_tokens = 30

        response = MagicMock()
        response.choices = [choice]
        response.usage = usage
        response.model = "pixtral-large-latest"
        response.model_dump = MagicMock(return_value={"model": "pixtral-large-latest"})

        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        provider._client.chat.complete_async = AsyncMock(return_value=response)

        result = await provider.call({"model": "pixtral-large-latest"})

        assert result.text_content == "Extracted text from image"
        assert result.tool_input == {}


class TestSupportsBatch:
    """Tests for batch support flag."""

    def test_supports_batch_is_true(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()
        assert provider.supports_batch is True


class TestBuildBatchRequest:
    """Tests for MistralProvider.build_batch_request."""

    def test_wraps_params_with_custom_id_and_body(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = {
            "model": "mistral-large-latest",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1024,
        }
        result = provider.build_batch_request("req-456", params)

        assert result["custom_id"] == "req-456"
        assert result["body"]["messages"] == [{"role": "user", "content": "hi"}]
        assert result["body"]["max_tokens"] == 1024

    def test_strips_model_from_body(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = {"model": "mistral-large-latest", "max_tokens": 512}
        result = provider.build_batch_request("req-789", params)

        assert "model" not in result["body"]

    def test_does_not_mutate_original_params(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        params = {"model": "mistral-large-latest", "max_tokens": 512}
        provider.build_batch_request("req-abc", params)

        assert params["model"] == "mistral-large-latest"


# ---------------------------------------------------------------------------
# submit_batch
# ---------------------------------------------------------------------------


class TestSubmitBatch:
    """Tests for MistralProvider.submit_batch."""

    @pytest.mark.asyncio
    async def test_returns_job_id(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.id = "batch-job-123"
        provider._client.batch.jobs.create_async = AsyncMock(return_value=mock_job)

        requests = [{"custom_id": "r1", "body": {"max_tokens": 512}}]
        result = await provider.submit_batch(requests, "mistral-large-latest")

        assert result == "batch-job-123"

    @pytest.mark.asyncio
    async def test_passes_correct_args(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.id = "batch-job-456"
        provider._client.batch.jobs.create_async = AsyncMock(return_value=mock_job)

        requests = [{"custom_id": "r1", "body": {}}]
        await provider.submit_batch(requests, "codestral-latest")

        call_kwargs = provider._client.batch.jobs.create_async.call_args.kwargs
        assert call_kwargs["model"] == "codestral-latest"
        assert str(call_kwargs["endpoint"]) == "/v1/chat/completions"
        assert len(call_kwargs["requests"]) == 1
        assert call_kwargs["requests"][0].custom_id == "r1"

    @pytest.mark.asyncio
    async def test_ocr_endpoint_uses_file_upload(self) -> None:
        """OCR endpoint uses file-based upload instead of inline requests."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_upload = MagicMock()
        mock_upload.id = "file-upload-123"
        provider._client.files.upload_async = AsyncMock(return_value=mock_upload)

        mock_job = MagicMock()
        mock_job.id = "batch-job-789"
        provider._client.batch.jobs.create_async = AsyncMock(return_value=mock_job)

        requests = [{"custom_id": "r1", "body": {"document": {"type": "document_url"}}}]
        result = await provider.submit_batch(requests, "pixtral-large-latest", endpoint="/v1/ocr")

        assert result == "batch-job-789"

        # Verify file was uploaded with purpose="batch"
        provider._client.files.upload_async.assert_called_once()
        upload_kwargs = provider._client.files.upload_async.call_args.kwargs
        assert upload_kwargs["purpose"] == "batch"
        assert upload_kwargs["file"]["file_name"] == "batch.jsonl"

        # Verify batch job was created with input_files (not inline requests)
        create_kwargs = provider._client.batch.jobs.create_async.call_args.kwargs
        assert create_kwargs["input_files"] == ["file-upload-123"]
        assert create_kwargs["model"] == "pixtral-large-latest"
        assert str(create_kwargs["endpoint"]) == "/v1/ocr"
        assert "requests" not in create_kwargs

    @pytest.mark.asyncio
    async def test_empty_endpoint_uses_default(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = MagicMock()
        mock_job.id = "batch-job-000"
        provider._client.batch.jobs.create_async = AsyncMock(return_value=mock_job)

        requests = [{"custom_id": "r1", "body": {}}]
        await provider.submit_batch(requests, "mistral-large-latest", endpoint="")

        call_kwargs = provider._client.batch.jobs.create_async.call_args.kwargs
        assert call_kwargs["model"] == "mistral-large-latest"
        assert str(call_kwargs["endpoint"]) == "/v1/chat/completions"
        assert len(call_kwargs["requests"]) == 1


# ---------------------------------------------------------------------------
# poll_batch
# ---------------------------------------------------------------------------


class TestPollBatch:
    """Tests for MistralProvider.poll_batch."""

    @pytest.mark.asyncio
    async def test_queued_returns_pending(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("QUEUED", output_file=None)
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.PENDING
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_running_returns_in_progress(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("RUNNING", output_file=None)
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.IN_PROGRESS

    @pytest.mark.asyncio
    async def test_failed_returns_failed(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("FAILED", output_file=None)
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.FAILED
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_timeout_exceeded_returns_expired(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("TIMEOUT_EXCEEDED", output_file=None)
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_cancelled_returns_canceled(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("CANCELLED", output_file=None)
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.CANCELED

    @pytest.mark.asyncio
    async def test_success_parses_jsonl_results(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("SUCCESS", output_file="file-output-123")
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        body_1 = {
            **_make_batch_choice(),
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            "model": "mistral-large-latest",
        }
        body_2 = {
            **_make_batch_choice(),
            "usage": {"prompt_tokens": 15, "completion_tokens": 25},
            "model": "mistral-large-latest",
        }
        jsonl = "\n".join(
            [
                json.dumps({"custom_id": "req-1", "response": {"body": body_1}}),
                json.dumps({"custom_id": "req-2", "response": {"body": body_2}}),
            ]
        )
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 2
        assert result.entries[0].custom_id == "req-1"
        assert result.entries[0].succeeded is True
        assert result.entries[0].raw_response_json is not None
        assert result.entries[1].custom_id == "req-2"
        assert result.entries[1].succeeded is True

    @pytest.mark.asyncio
    async def test_success_with_errored_entry(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("SUCCESS", output_file="file-output-456")
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        jsonl = json.dumps(
            {
                "custom_id": "req-bad",
                "response": {
                    "body": {
                        "error": {"type": "invalid_request", "message": "bad input"},
                    }
                },
            }
        )
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 1
        assert result.entries[0].custom_id == "req-bad"
        assert result.entries[0].succeeded is False
        assert result.entries[0].error is not None

    @pytest.mark.asyncio
    async def test_success_downloads_from_output_file(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("SUCCESS", output_file="file-abc-789")
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        jsonl = json.dumps(
            {
                "custom_id": "req-1",
                "response": {"body": {"choices": [{"message": {"tool_calls": []}}]}},
            }
        )
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))

        await provider.poll_batch("batch-1")

        provider._client.files.download_async.assert_called_once_with(file_id="file-abc-789")

    @pytest.mark.asyncio
    async def test_success_with_string_output_file(self) -> None:
        """When download returns a string instead of a file-like object."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("SUCCESS", output_file="file-str-1")
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        jsonl = json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "body": {
                        "choices": [
                            {"message": {"tool_calls": [{"function": {"arguments": "{}"}}]}}
                        ],
                        "model": "mistral-large-latest",
                    }
                },
            }
        )
        # Return a plain string (no .read method)
        provider._client.files.download_async = AsyncMock(return_value=jsonl)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 1
        assert result.entries[0].succeeded is True

    @pytest.mark.asyncio
    async def test_success_with_null_output_file_returns_failed(self) -> None:
        """When Mistral reports SUCCESS but output_file is None, return FAILED."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("SUCCESS", output_file=None)
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.FAILED
        assert result.entries == []
        provider._client.files.download_async.assert_not_called()

    @pytest.mark.asyncio
    async def test_skips_blank_lines_in_jsonl(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("SUCCESS", output_file="file-blank")
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        entry = {
            "custom_id": "req-1",
            "response": {"body": _make_batch_choice()},
        }
        jsonl = "\n" + json.dumps(entry) + "\n\n"
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))

        result = await provider.poll_batch("batch-1")

        assert len(result.entries) == 1

    # -----------------------------------------------------------------------
    # New tests: error logging, error_file merging, UNSET sentinel
    # -----------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_failed_status_logs_errors(self, caplog: pytest.LogCaptureFixture) -> None:
        """FAILED batch with errors logs WARNING with error details."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        err = MagicMock()
        err.message = "context length exceeded"
        err.count = 3

        mock_job = _make_mock_batch_job(
            "FAILED",
            output_file=None,
            errors=[err],
            failed_requests=3,
        )
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        with caplog.at_level(logging.WARNING):
            result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.FAILED
        assert "context length exceeded (x3)" in caplog.text
        assert "3 failed request" in caplog.text

    @pytest.mark.asyncio
    async def test_success_with_errors_logs_warning(self, caplog: pytest.LogCaptureFixture) -> None:
        """SUCCESS with errors and failed_requests still returns ENDED but logs."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        err = MagicMock()
        err.message = "partial failure"
        err.count = 1

        mock_job = _make_mock_batch_job(
            "SUCCESS",
            output_file="file-ok",
            errors=[err],
            failed_requests=2,
        )
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        jsonl = json.dumps(
            {
                "custom_id": "req-1",
                "response": {"body": _make_batch_choice()},
            }
        )
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))

        with caplog.at_level(logging.WARNING):
            result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.ENDED
        assert "partial failure" in caplog.text
        assert "2 failed request" in caplog.text

    @pytest.mark.asyncio
    async def test_success_with_error_file_merges_entries(self) -> None:
        """output_file (1 ok) + error_file (1 fail) = 2 entries."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job(
            "SUCCESS",
            output_file="file-output",
            error_file="file-errors",
        )
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        output_jsonl = json.dumps(
            {
                "custom_id": "req-1",
                "response": {"body": _make_batch_choice()},
            }
        )
        error_jsonl = json.dumps(
            {
                "custom_id": "req-2",
                "response": {
                    "body": {"error": {"message": "context too long"}},
                },
            }
        )

        output_mock = _make_mock_file(output_jsonl)
        error_mock = _make_mock_file(error_jsonl)

        async def _download(file_id: str):
            if file_id == "file-output":
                return output_mock
            return error_mock

        provider._client.files.download_async = AsyncMock(side_effect=_download)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 2
        ids = {e.custom_id for e in result.entries}
        assert ids == {"req-1", "req-2"}
        # req-1 succeeded, req-2 failed
        by_id = {e.custom_id: e for e in result.entries}
        assert by_id["req-1"].succeeded is True
        assert by_id["req-2"].succeeded is False

    @pytest.mark.asyncio
    async def test_success_with_error_file_deduplicates(self) -> None:
        """Same custom_id in both files — output_file entry wins."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job(
            "SUCCESS",
            output_file="file-output",
            error_file="file-errors",
        )
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        # Both files contain req-1
        output_jsonl = json.dumps(
            {
                "custom_id": "req-1",
                "response": {"body": _make_batch_choice()},
            }
        )
        error_jsonl = json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "body": {"error": {"message": "should be ignored"}},
                },
            }
        )

        output_mock = _make_mock_file(output_jsonl)
        error_mock = _make_mock_file(error_jsonl)

        async def _download(file_id: str):
            if file_id == "file-output":
                return output_mock
            return error_mock

        provider._client.files.download_async = AsyncMock(side_effect=_download)

        result = await provider.poll_batch("batch-1")

        assert len(result.entries) == 1
        assert result.entries[0].custom_id == "req-1"
        assert result.entries[0].succeeded is True  # output_file wins

    @pytest.mark.asyncio
    async def test_success_with_error_file_download_failure(self) -> None:
        """Error file download raises — OSError propagates to caller."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job(
            "SUCCESS",
            output_file="file-output",
            error_file="file-errors",
        )
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        async def _download(file_id: str):
            if file_id == "file-errors":
                raise OSError("download failed")
            return _make_mock_file("")  # pragma: no cover

        provider._client.files.download_async = AsyncMock(side_effect=_download)

        with pytest.raises(OSError, match="download failed"):
            await provider.poll_batch("batch-1")

    @pytest.mark.asyncio
    async def test_success_with_unset_output_file_returns_failed(self) -> None:
        """output_file=UNSET (falsy sentinel, not None) detected as missing."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        class _Unset:
            """Mimic Mistral SDK UNSET sentinel."""

            def __bool__(self) -> bool:
                return False

        mock_job = _make_mock_batch_job("SUCCESS", output_file=_Unset())
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.FAILED
        assert result.entries == []

    @pytest.mark.asyncio
    async def test_success_with_null_output_file_logs_errors(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """When output_file is None and errors are present, log the errors."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        err = MagicMock()
        err.message = "all requests failed"
        err.count = 10

        mock_job = _make_mock_batch_job(
            "SUCCESS",
            output_file=None,
            errors=[err],
            failed_requests=10,
        )
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        with caplog.at_level(logging.WARNING):
            result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.FAILED
        assert "all requests failed (x10)" in caplog.text

    @pytest.mark.asyncio
    async def test_error_file_downloaded_when_output_file_missing(self) -> None:
        """Error file entries are returned even when output_file is None."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job(
            "SUCCESS",
            output_file=None,
            error_file="file-errors",
            errors=[],
            failed_requests=1,
        )
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        error_jsonl = json.dumps(
            {
                "custom_id": "req-1",
                "response": {
                    "body": {"error": {"message": "invalid request body"}},
                },
            }
        )
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(error_jsonl))

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.FAILED
        assert len(result.entries) == 1
        assert result.entries[0].custom_id == "req-1"
        assert result.entries[0].succeeded is False
        assert "invalid request body" in result.entries[0].error
        provider._client.files.download_async.assert_called_once_with(file_id="file-errors")

    @pytest.mark.asyncio
    async def test_error_file_unset_not_downloaded(self) -> None:
        """UNSET error_file does not trigger download."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        class _Unset:
            def __bool__(self) -> bool:
                return False

        mock_job = _make_mock_batch_job("SUCCESS", output_file="file-ok", error_file=_Unset())
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        jsonl = json.dumps(
            {
                "custom_id": "req-1",
                "response": {"body": _make_batch_choice()},
            }
        )
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.ENDED
        # Only one download call — for output_file, not error_file
        provider._client.files.download_async.assert_called_once_with(file_id="file-ok")

    @pytest.mark.asyncio
    async def test_error_file_none_not_downloaded(self) -> None:
        """None error_file does not trigger download."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("SUCCESS", output_file="file-ok", error_file=None)
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        jsonl = json.dumps(
            {
                "custom_id": "req-1",
                "response": {"body": _make_batch_choice()},
            }
        )
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))

        result = await provider.poll_batch("batch-1")

        assert result.status == BatchPollStatus.ENDED
        provider._client.files.download_async.assert_called_once_with(file_id="file-ok")

    @pytest.mark.asyncio
    async def test_ocr_response_with_pages_succeeds(self) -> None:
        """OCR batch responses have 'pages' instead of 'choices'."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        mock_job = _make_mock_batch_job("SUCCESS", output_file="file-ocr-1")
        provider._client.batch.jobs.get_async = AsyncMock(return_value=mock_job)

        ocr_body = {
            "pages": [
                {"markdown": "Page one."},
                {"markdown": "Page two."},
            ],
            "model": "mistral-ocr-latest",
            "usage_info": {"pages_processed": 2, "doc_size_bytes": 5000},
        }
        jsonl = json.dumps(
            {
                "custom_id": "req-ocr-1",
                "response": {"body": ocr_body},
            }
        )
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(jsonl))

        result = await provider.poll_batch("batch-ocr")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 1
        assert result.entries[0].custom_id == "req-ocr-1"
        assert result.entries[0].succeeded is True
        assert result.entries[0].raw_response_json is not None
        body = json.loads(result.entries[0].raw_response_json)
        assert len(body["pages"]) == 2


# ---------------------------------------------------------------------------
# parse_batch_result
# ---------------------------------------------------------------------------


class TestParseBatchResult:
    """Tests for MistralProvider.parse_batch_result."""

    def setup_method(self) -> None:
        # parse_batch_result only checks that the name is registered (tool_input
        # passes through unvalidated), so any model under these keys will do.
        from pydantic import BaseModel

        from sax_llm import register_output_type
        from sax_llm.registry import reset_output_type_registry

        class BatchOutput(BaseModel):
            pass

        reset_output_type_registry()
        register_output_type("LLMResponse", BatchOutput)
        register_output_type("Plan", BatchOutput)

    def teardown_method(self) -> None:
        from sax_llm.registry import reset_output_type_registry

        reset_output_type_registry()

    def test_parses_successful_response(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        raw = json.dumps(
            {
                "model": "mistral-large-latest",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "llm_response",
                                        "arguments": json.dumps(
                                            {
                                                "files": [],
                                                "edits": [],
                                                "explanation": "Done.",
                                            }
                                        ),
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 200,
                },
            }
        )

        result = provider.parse_batch_result(raw, "LLMResponse")

        assert result.tool_input["explanation"] == "Done."
        assert result.model_name == "mistral-large-latest"
        assert result.input_tokens == 100
        assert result.output_tokens == 200
        assert result.cache_creation_input_tokens == 0
        assert result.cache_read_input_tokens == 0
        assert result.raw_response_json == raw

    def test_parses_plan(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        plan_data = {
            "task_id": "t1",
            "steps": [{"step_id": "s1", "description": "Do it.", "target_files": ["a.py"]}],
            "explanation": "Single step.",
        }
        raw = json.dumps(
            {
                "model": "mistral-large-latest",
                "choices": [
                    {
                        "message": {
                            "tool_calls": [
                                {
                                    "function": {
                                        "name": "plan",
                                        "arguments": json.dumps(plan_data),
                                    }
                                }
                            ]
                        }
                    }
                ],
                "usage": {"prompt_tokens": 50, "completion_tokens": 100},
            }
        )

        result = provider.parse_batch_result(raw, "Plan")

        assert result.tool_input["task_id"] == "t1"
        assert len(result.tool_input["steps"]) == 1

    def test_raises_for_unknown_output_type(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        with pytest.raises(KeyError, match="Unknown output type"):
            provider.parse_batch_result("{}", "NonExistentType")

    def test_missing_tool_calls_returns_empty_dict(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        raw = json.dumps(
            {
                "model": "mistral-large-latest",
                "choices": [{"message": {"content": "text only"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            }
        )

        result = provider.parse_batch_result(raw, "LLMResponse")

        assert result.tool_input == {}

    def test_missing_usage_defaults_to_zero(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        raw = json.dumps(
            {
                "model": "mistral-large-latest",
                "choices": [{"message": {"tool_calls": [{"function": {"arguments": "{}"}}]}}],
            }
        )

        result = provider.parse_batch_result(raw, "LLMResponse")

        assert result.input_tokens == 0
        assert result.output_tokens == 0

    def test_dict_arguments_not_double_parsed(self) -> None:
        """If arguments is already a dict (not a string), it should pass through."""
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        raw = json.dumps(
            {
                "model": "mistral-large-latest",
                "choices": [
                    {"message": {"tool_calls": [{"function": {"arguments": {"key": "value"}}}]}}
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3},
            }
        )

        result = provider.parse_batch_result(raw, "LLMResponse")

        assert result.tool_input == {"key": "value"}

    def test_output_type_name_none_returns_text_content(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        raw = json.dumps(
            {
                "model": "pixtral-large-latest",
                "choices": [{"message": {"content": "Extracted OCR text"}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 200},
            }
        )

        result = provider.parse_batch_result(raw, None)

        assert result.text_content == "Extracted OCR text"
        assert result.tool_input == {}
        assert result.model_name == "pixtral-large-latest"
        assert result.input_tokens == 100


# ---------------------------------------------------------------------------
# Tests — _extract_images_from_response
# ---------------------------------------------------------------------------


class TestExtractImagesFromResponse:
    """Tests for _extract_images_from_response helper."""

    def test_extracts_images_and_strips_base64(self) -> None:
        response_body = {
            "pages": [
                {
                    "markdown": "Page 1 text",
                    "images": [
                        {
                            "id": "img-0.jpeg",
                            "image_base64": "aW1hZ2UtZGF0YQ==",
                            "top_left_x": 10,
                            "top_left_y": 20,
                            "bottom_right_x": 100,
                            "bottom_right_y": 200,
                        },
                    ],
                },
            ],
            "model": "mistral-ocr-latest",
        }

        extracted = _extract_images_from_response(response_body)

        assert len(extracted) == 1
        assert extracted[0].original_image_id == "img-0.jpeg"
        assert extracted[0].image_base64 == "aW1hZ2UtZGF0YQ=="
        assert extracted[0].page_index == 0
        assert extracted[0].top_left_x == 10
        assert extracted[0].bottom_right_y == 200

        # Verify base64 was stripped from response body
        assert "image_base64" not in response_body["pages"][0]["images"][0]
        # But image id still present
        assert response_body["pages"][0]["images"][0]["id"] == "img-0.jpeg"

    def test_multiple_pages_multiple_images(self) -> None:
        response_body = {
            "pages": [
                {
                    "markdown": "Page 1",
                    "images": [
                        {"id": "img-0.jpeg", "image_base64": "data0"},
                        {"id": "img-1.jpeg", "image_base64": "data1"},
                    ],
                },
                {
                    "markdown": "Page 2",
                    "images": [
                        {"id": "img-0.jpeg", "image_base64": "data2"},
                    ],
                },
            ],
        }

        extracted = _extract_images_from_response(response_body)

        assert len(extracted) == 3
        assert extracted[0].page_index == 0
        assert extracted[1].page_index == 0
        assert extracted[2].page_index == 1

    def test_no_images_returns_empty(self) -> None:
        response_body = {
            "pages": [
                {"markdown": "No images here", "images": []},
            ],
        }

        extracted = _extract_images_from_response(response_body)
        assert extracted == []

    def test_no_pages_returns_empty(self) -> None:
        response_body = {"choices": [{"message": {"content": "text"}}]}
        extracted = _extract_images_from_response(response_body)
        assert extracted == []

    def test_skips_images_without_base64(self) -> None:
        response_body = {
            "pages": [
                {
                    "markdown": "Page 1",
                    "images": [
                        {"id": "img-0.jpeg"},  # no image_base64
                        {"id": "img-1.jpeg", "image_base64": "data1"},
                    ],
                },
            ],
        }

        extracted = _extract_images_from_response(response_body)
        assert len(extracted) == 1
        assert extracted[0].original_image_id == "img-1.jpeg"

    def test_no_bounding_box_fields(self) -> None:
        response_body = {
            "pages": [
                {
                    "markdown": "Page 1",
                    "images": [
                        {"id": "img-0.jpeg", "image_base64": "data0"},
                    ],
                },
            ],
        }

        extracted = _extract_images_from_response(response_body)
        assert len(extracted) == 1
        assert extracted[0].top_left_x is None
        assert extracted[0].bottom_right_y is None

    def test_detects_png_mime_type_from_data_uri(self) -> None:
        """MIME type is parsed from data-URI prefix, not hardcoded to JPEG."""
        response_body = {
            "pages": [
                {
                    "markdown": "Page 1",
                    "images": [
                        {
                            "id": "img-0.png",
                            "image_base64": "data:image/png;base64,iVBORw0KGgo=",
                        },
                    ],
                },
            ],
        }

        extracted = _extract_images_from_response(response_body)
        assert len(extracted) == 1
        assert extracted[0].mime_type == "image/png"

    def test_detects_webp_mime_type_from_data_uri(self) -> None:
        response_body = {
            "pages": [
                {
                    "markdown": "Page 1",
                    "images": [
                        {
                            "id": "img-0.webp",
                            "image_base64": "data:image/webp;base64,UklGR...",
                        },
                    ],
                },
            ],
        }

        extracted = _extract_images_from_response(response_body)
        assert len(extracted) == 1
        assert extracted[0].mime_type == "image/webp"

    def test_defaults_to_jpeg_without_data_uri_prefix(self) -> None:
        """When no data-URI prefix, falls back to image/jpeg."""
        response_body = {
            "pages": [
                {
                    "markdown": "Page 1",
                    "images": [
                        {
                            "id": "img-0.jpeg",
                            "image_base64": "aW1hZ2UtZGF0YQ==",
                        },
                    ],
                },
            ],
        }

        extracted = _extract_images_from_response(response_body)
        assert len(extracted) == 1
        assert extracted[0].mime_type == "image/jpeg"


# ---------------------------------------------------------------------------
# Tests — poll_batch with image extraction
# ---------------------------------------------------------------------------


class TestPollBatchWithImages:
    """Test that poll_batch extracts images from OCR responses."""

    @pytest.mark.asyncio
    async def test_poll_batch_extracts_images(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        # Build OCR response with images
        ocr_response = {
            "pages": [
                {
                    "markdown": "![img-0.jpeg](img-0.jpeg)",
                    "images": [
                        {"id": "img-0.jpeg", "image_base64": "aW1hZ2VieXRlcw=="},
                    ],
                },
            ],
            "model": "mistral-ocr-latest",
            "usage_info": {"pages_processed": 1},
        }
        output_line = json.dumps(
            {
                "custom_id": "req-1",
                "response": {"body": ocr_response},
            }
        )

        job = _make_mock_batch_job("SUCCESS", output_file="file-out-1")
        provider._client.batch.jobs.get_async = AsyncMock(return_value=job)
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(output_line))

        result = await provider.poll_batch("batch-img-1")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 1
        entry = result.entries[0]
        assert entry.succeeded is True
        assert len(entry.extracted_images) == 1
        assert entry.extracted_images[0].original_image_id == "img-0.jpeg"
        assert entry.extracted_images[0].image_base64 == "aW1hZ2VieXRlcw=="

        # Verify base64 was stripped from raw_response_json
        raw = json.loads(entry.raw_response_json)
        assert "image_base64" not in json.dumps(raw)

    @pytest.mark.asyncio
    async def test_poll_batch_no_images_backward_compat(self) -> None:
        provider = MistralProvider.__new__(MistralProvider)
        provider._client = MagicMock()

        # Build OCR response without images
        ocr_response = {
            "pages": [{"markdown": "Just text"}],
            "model": "mistral-ocr-latest",
        }
        output_line = json.dumps(
            {
                "custom_id": "req-2",
                "response": {"body": ocr_response},
            }
        )

        job = _make_mock_batch_job("SUCCESS", output_file="file-out-2")
        provider._client.batch.jobs.get_async = AsyncMock(return_value=job)
        provider._client.files.download_async = AsyncMock(return_value=_make_mock_file(output_line))

        result = await provider.poll_batch("batch-no-img")

        assert result.status == BatchPollStatus.ENDED
        assert len(result.entries) == 1
        assert result.entries[0].extracted_images == []
