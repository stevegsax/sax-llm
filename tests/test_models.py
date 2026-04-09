"""Tests for sax_llm.models — message types and provider response."""

from __future__ import annotations

from sax_llm.models import (
    BatchPollResult,
    BatchPollStatus,
    ProviderResponse,
    text_messages,
)


class TestTextMessages:
    def test_creates_system_and_user(self):
        msgs = text_messages("system prompt", "user prompt")
        assert len(msgs) == 2
        assert msgs[0].role == "system"
        assert msgs[0].content == "system prompt"
        assert msgs[0].cache_control is True
        assert msgs[1].role == "user"
        assert msgs[1].content == "user prompt"

    def test_cache_disabled(self):
        msgs = text_messages("sys", "usr", cache_system=False)
        assert msgs[0].cache_control is False


class TestProviderResponse:
    def test_defaults(self):
        resp = ProviderResponse(
            model_name="test",
            input_tokens=10,
            output_tokens=5,
            raw_response_json="{}",
        )
        assert resp.tool_input == {}
        assert resp.text_content is None
        assert resp.cache_creation_input_tokens == 0

    def test_with_tool_input(self):
        resp = ProviderResponse(
            tool_input={"key": "value"},
            model_name="claude",
            input_tokens=100,
            output_tokens=50,
            raw_response_json='{"test": true}',
        )
        assert resp.tool_input == {"key": "value"}


class TestBatchPollResult:
    def test_pending(self):
        result = BatchPollResult(status=BatchPollStatus.PENDING)
        assert result.status == "pending"
        assert result.entries == []

    def test_ended_with_entries(self):
        from sax_llm.models import BatchResultEntry

        result = BatchPollResult(
            status=BatchPollStatus.ENDED,
            entries=[
                BatchResultEntry(
                    custom_id="req-1",
                    succeeded=True,
                    raw_response_json="{}",
                ),
            ],
        )
        assert len(result.entries) == 1
        assert result.entries[0].succeeded is True
