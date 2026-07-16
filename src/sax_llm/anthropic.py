"""Anthropic LLM provider — wraps sax_llm.client functions."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from sax_llm.client import (
    build_batch_request,
    build_system_param,
    build_thinking_param,
    build_tool_definition,
    extract_usage,
)
from sax_llm.models import (
    BatchPollResult,
    BatchPollStatus,
    BatchResultEntry,
    DocumentContent,
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
)

if TYPE_CHECKING:
    from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Message translation helpers
# ---------------------------------------------------------------------------


def _content_block_to_anthropic(block: TextContent | ImageContent | DocumentContent) -> dict:
    """Convert a ContentBlock to Anthropic API format."""
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageContent):
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": block.media_type,
                "data": block.data,
            },
        }
    if isinstance(block, DocumentContent):
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": block.media_type,
                "data": block.data,
            },
        }
    msg = f"Unknown content block type: {type(block)}"
    raise TypeError(msg)


def _content_to_anthropic(
    content: str | list[TextContent | ImageContent | DocumentContent],
    cache_control: bool = False,
) -> str | list[dict]:
    """Convert message content to Anthropic format."""
    if isinstance(content, str):
        if cache_control:
            return [{"type": "text", "text": content, "cache_control": {"type": "ephemeral"}}]
        return content

    blocks = [_content_block_to_anthropic(b) for b in content]
    if cache_control and blocks:
        blocks[-1]["cache_control"] = {"type": "ephemeral"}
    return blocks


def _extract_system_and_messages(
    messages: list[Message],
    cache_instructions: bool = True,
) -> tuple[str | list[dict], list[dict]]:
    """Split messages into Anthropic's top-level system param and message array."""
    system_parts: list[str] = []
    system_cache = False
    conversation: list[dict] = []

    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str):
                system_parts.append(msg.content)
            else:
                system_parts.append(
                    " ".join(b.text for b in msg.content if isinstance(b, TextContent))
                )
            if msg.cache_control:
                system_cache = True
        else:
            conversation.append({
                "role": msg.role,
                "content": _content_to_anthropic(msg.content),
            })

    system_text = "\n".join(system_parts)
    system = build_system_param(system_text, cache_control=cache_instructions and system_cache)

    return system, conversation


class AnthropicProvider:
    """LLM provider backed by the Anthropic Messages API."""

    def __init__(self) -> None:
        from sax_llm.client import get_anthropic_client

        self._get_client = get_anthropic_client

    def build_request_params(
        self,
        messages: list[Message],
        output_type: type[BaseModel] | None,
        model: str,
        max_tokens: int,
        *,
        cache_instructions: bool = True,
        cache_tool_definitions: bool = True,
        thinking_budget_tokens: int = 0,
    ) -> dict:
        """Build Anthropic messages.create kwargs."""
        system, conversation = _extract_system_and_messages(messages, cache_instructions)

        params: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": conversation,
        }

        if output_type is not None:
            tool_def = build_tool_definition(output_type, cache_control=cache_tool_definitions)
            tool_name = tool_def["name"]
            params["tools"] = [tool_def]
            params["tool_choice"] = {"type": "tool", "name": tool_name}

        thinking = build_thinking_param(model, thinking_budget_tokens)
        if thinking is not None:
            params["thinking"] = thinking
            if "tool_choice" in params:
                params["tool_choice"] = {"type": "auto"}
            params["max_tokens"] = max(max_tokens, thinking_budget_tokens + max_tokens)

        return params

    async def call(self, params: dict) -> ProviderResponse:
        """Call the Anthropic API and return a normalized response."""
        client = self._get_client()
        message = await client.messages.create(**params)

        tool_input: dict = {}
        text_content: str | None = None
        has_tools = "tools" in params and params["tools"]

        if has_tools:
            for block in message.content:
                if block.type == "tool_use":
                    tool_input = block.input
                    break
        else:
            text_parts: list[str] = []
            for block in message.content:
                if block.type == "text":
                    text_parts.append(block.text)
            text_content = "\n".join(text_parts) if text_parts else None

        in_tok, out_tok, cache_create, cache_read = extract_usage(message)

        return ProviderResponse(
            tool_input=tool_input,
            text_content=text_content,
            model_name=message.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
            raw_response_json=message.model_dump_json(),
        )

    @property
    def supports_batch(self) -> bool:
        return True

    def build_batch_request(self, request_id: str, params: dict) -> dict:
        return build_batch_request(request_id, params)

    async def submit_batch(self, requests: list[dict], model: str, *, endpoint: str = "") -> str:
        client = self._get_client()
        batch = await client.messages.batches.create(requests=requests)
        return batch.id

    async def poll_batch(self, batch_id: str) -> BatchPollResult:
        client = self._get_client()
        batch = await client.messages.batches.retrieve(batch_id)

        if batch.processing_status != "ended":
            return BatchPollResult(status=BatchPollStatus.IN_PROGRESS)

        results_iter = await client.messages.batches.results(batch_id)
        entries: list[BatchResultEntry] = []
        async for entry in results_iter:
            result_type = entry.result.type
            if result_type == "succeeded":
                entries.append(
                    BatchResultEntry(
                        custom_id=entry.custom_id,
                        succeeded=True,
                        raw_response_json=entry.result.message.model_dump_json(),
                    )
                )
            else:
                error_msg = _format_batch_error(entry)
                entries.append(
                    BatchResultEntry(
                        custom_id=entry.custom_id,
                        succeeded=False,
                        error=error_msg,
                    )
                )

        return BatchPollResult(status=BatchPollStatus.ENDED, entries=entries)

    def parse_batch_result(
        self,
        raw_json: str,
        output_type_name: str | None,
    ) -> ProviderResponse:
        if output_type_name is None:
            return self._parse_batch_text_result(raw_json)

        from sax_llm.client import parse_batch_response_json

        parsed, model_name, in_tok, out_tok, cache_create, cache_read = (
            parse_batch_response_json(raw_json, output_type_name)
        )

        tool_input = json.loads(parsed.model_dump_json())

        return ProviderResponse(
            tool_input=tool_input,
            model_name=model_name,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
            raw_response_json=raw_json,
        )

    def _parse_batch_text_result(self, raw_json: str) -> ProviderResponse:
        from anthropic.types import Message as AnthropicMessage

        data = json.loads(raw_json)
        message = AnthropicMessage.model_validate(data)

        text_parts: list[str] = []
        for block in message.content:
            if block.type == "text":
                text_parts.append(block.text)

        in_tok, out_tok, cache_create, cache_read = extract_usage(message)

        return ProviderResponse(
            text_content="\n".join(text_parts) if text_parts else None,
            model_name=message.model,
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_creation_input_tokens=cache_create,
            cache_read_input_tokens=cache_read,
            raw_response_json=raw_json,
        )


def _format_batch_error(entry: object) -> str:
    result = getattr(entry, "result", None)
    result_type = getattr(result, "type", "unknown")

    if result_type == "errored":
        return f"Batch error: {getattr(result, 'error', 'unknown')}"
    if result_type == "expired":
        return "Batch request expired (24h limit)"
    if result_type == "canceled":
        return "Batch request was canceled"
    return f"Unknown result type: {result_type}"
