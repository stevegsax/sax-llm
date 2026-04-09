"""Mistral LLM provider implementation."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import TYPE_CHECKING, cast

from sax_llm.models import (
    BatchPollResult,
    BatchPollStatus,
    BatchResultEntry,
    DocumentContent,
    ExtractedImage,
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
)

if TYPE_CHECKING:
    from pydantic import BaseModel

logger = logging.getLogger(__name__)


def _snake_case(name: str) -> str:
    """Convert CamelCase class name to snake_case tool name."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


# ---------------------------------------------------------------------------
# Message translation helpers
# ---------------------------------------------------------------------------


def _content_block_to_mistral(block: TextContent | ImageContent | DocumentContent) -> dict:
    """Convert a ContentBlock to Mistral API format."""
    if isinstance(block, TextContent):
        return {"type": "text", "text": block.text}
    if isinstance(block, ImageContent):
        data_uri = f"data:{block.media_type};base64,{block.data}"
        return {"type": "image_url", "image_url": data_uri}
    if isinstance(block, DocumentContent):
        data_uri = f"data:{block.media_type};base64,{block.data}"
        return {"type": "document_url", "document_url": data_uri}
    msg = f"Unknown content block type: {type(block)}"
    raise TypeError(msg)


def _content_to_mistral(
    content: str | list[TextContent | ImageContent | DocumentContent],
) -> str | list[dict]:
    """Convert message content to Mistral format."""
    if isinstance(content, str):
        return content
    return [_content_block_to_mistral(b) for b in content]


def _messages_to_mistral(messages: list[Message]) -> list[dict]:
    """Convert Message list to Mistral format (system messages stay in array)."""
    return [
        {"role": msg.role, "content": _content_to_mistral(msg.content)}
        for msg in messages
    ]


# ---------------------------------------------------------------------------
# Batch error / file helpers
# ---------------------------------------------------------------------------


def _is_set(value: object) -> bool:
    """Return True only if *value* is a usable, non-empty string."""
    return bool(value)


def _format_batch_errors(errors: list) -> str:
    """Format a list of BatchError objects into a human-readable string."""
    parts: list[str] = []
    for err in errors:
        message = getattr(err, "message", str(err))
        count = getattr(err, "count", None)
        count = count if isinstance(count, int) else 1
        part = message if count <= 1 else f"{message} (x{count})"
        parts.append(part)
    return "; ".join(parts)


async def _download_file_content(client: object, file_id: str) -> str:
    """Download a Mistral file and return its decoded text content."""
    output_file = await client.files.download_async(file_id=file_id)
    if hasattr(output_file, "aread"):
        content = await output_file.aread()
        return content.decode("utf-8")
    if hasattr(output_file, "read"):
        return output_file.read().decode("utf-8")
    return str(output_file)


def _parse_error_file_entries(content: str) -> list[BatchResultEntry]:
    """Parse error-file JSONL content into BatchResultEntry objects."""
    entries: list[BatchResultEntry] = []
    for line in content.strip().split("\n"):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("Skipping malformed error-file line: %.120s", line)
            continue

        custom_id = data.get("custom_id", "unknown")
        error_detail = (
            data.get("response", {}).get("body", {}).get("error")
            or data.get("error")
        )
        error_str = json.dumps(error_detail) if error_detail else "Unknown error"

        entries.append(
            BatchResultEntry(
                custom_id=custom_id,
                succeeded=False,
                error=error_str,
            )
        )
    return entries


def _extract_images_from_response(response_body: dict) -> list[ExtractedImage]:
    """Extract images from an OCR response and strip base64 data from the body."""
    extracted: list[ExtractedImage] = []
    pages = response_body.get("pages", [])
    for page_index, page in enumerate(pages):
        for img in page.get("images", []):
            image_base64 = img.get("image_base64")
            if not image_base64:
                continue
            original_id = img.get("id", f"img-{page_index}.jpeg")

            mime_type = "image/jpeg"
            if isinstance(image_base64, str) and image_base64.startswith("data:"):
                header = image_base64.split(",", 1)[0]
                mime_type = header.split(":")[1].split(";")[0]

            top_left_x = None
            top_left_y = None
            bottom_right_x = None
            bottom_right_y = None
            top_left = img.get("top_left_x"), img.get("top_left_y")
            bottom_right = img.get("bottom_right_x"), img.get("bottom_right_y")
            if top_left[0] is not None:
                top_left_x = top_left[0]
            if top_left[1] is not None:
                top_left_y = top_left[1]
            if bottom_right[0] is not None:
                bottom_right_x = bottom_right[0]
            if bottom_right[1] is not None:
                bottom_right_y = bottom_right[1]

            extracted.append(
                ExtractedImage(
                    original_image_id=original_id,
                    page_index=page_index,
                    image_base64=image_base64,
                    mime_type=mime_type,
                    top_left_x=top_left_x,
                    top_left_y=top_left_y,
                    bottom_right_x=bottom_right_x,
                    bottom_right_y=bottom_right_y,
                )
            )
            del img["image_base64"]

    return extracted


_DEFAULT_MISTRAL_ENDPOINT = "/v1/chat/completions"


class MistralProvider:
    """LLM provider backed by the Mistral API."""

    def __init__(self) -> None:
        from mistralai import Mistral

        api_key = os.environ.get("MISTRAL_API_KEY", "")
        self._client = Mistral(api_key=api_key)

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
        """Build Mistral chat.complete kwargs."""
        params: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": _messages_to_mistral(messages),
        }

        if output_type is not None:
            schema = output_type.model_json_schema()
            tool_name = _snake_case(output_type.__name__)
            description = (
                (output_type.__doc__ or "").strip() or f"Structured output: {tool_name}"
            )

            tool_def = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description,
                    "parameters": schema,
                },
            }
            params["tools"] = [tool_def]
            params["tool_choice"] = "any"

        return params

    async def call(self, params: dict) -> ProviderResponse:
        """Call the Mistral API and return a normalized response."""
        response = await self._client.chat.complete_async(**params)

        tool_input: dict = {}
        text_content: str | None = None
        has_tools = "tools" in params and params["tools"]

        if has_tools:
            if response.choices and response.choices[0].message.tool_calls:
                args_str = response.choices[0].message.tool_calls[0].function.arguments
                tool_input = json.loads(args_str) if isinstance(args_str, str) else args_str
        else:
            if response.choices and response.choices[0].message.content:
                content = response.choices[0].message.content
                text_content = content if isinstance(content, str) else str(content)

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        input_tokens = input_tokens or 0
        output_tokens = output_tokens or 0

        return ProviderResponse(
            tool_input=tool_input,
            text_content=text_content,
            model_name=response.model or params.get("model", ""),
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            raw_response_json=json.dumps(response.model_dump(), default=str),
        )

    @property
    def supports_batch(self) -> bool:
        return True

    def build_batch_request(self, request_id: str, params: dict) -> dict:
        body = {k: v for k, v in params.items() if k != "model"}
        return {"custom_id": request_id, "body": body}

    async def submit_batch(
        self, requests: list[dict], model: str, *, endpoint: str = ""
    ) -> str:
        from mistralai.types.basemodel import UnrecognizedStr

        resolved_endpoint = endpoint or _DEFAULT_MISTRAL_ENDPOINT

        if resolved_endpoint == "/v1/ocr":
            return await self._submit_batch_via_file(requests, model, resolved_endpoint)

        from mistralai.models import BatchRequest

        typed_requests = [BatchRequest(**r) for r in requests]
        job = await self._client.batch.jobs.create_async(
            requests=typed_requests,
            model=model,
            endpoint=UnrecognizedStr(resolved_endpoint),
        )
        return job.id

    async def _submit_batch_via_file(
        self, requests: list[dict], model: str, endpoint: str
    ) -> str:
        from mistralai.types.basemodel import UnrecognizedStr

        lines = [json.dumps(r) for r in requests]
        jsonl_bytes = ("\n".join(lines) + "\n").encode("utf-8")

        upload_result = await self._client.files.upload_async(
            file={"file_name": "batch.jsonl", "content": jsonl_bytes},
            purpose="batch",
        )

        job = await self._client.batch.jobs.create_async(
            input_files=[upload_result.id],
            model=model,
            endpoint=UnrecognizedStr(endpoint),
        )
        return job.id

    async def poll_batch(self, batch_id: str) -> BatchPollResult:
        job = await self._client.batch.jobs.get_async(job_id=batch_id)

        status_map = {
            "QUEUED": BatchPollStatus.PENDING,
            "RUNNING": BatchPollStatus.IN_PROGRESS,
            "SUCCESS": BatchPollStatus.ENDED,
            "FAILED": BatchPollStatus.FAILED,
            "TIMEOUT_EXCEEDED": BatchPollStatus.EXPIRED,
            "CANCELLATION_REQUESTED": BatchPollStatus.CANCELED,
            "CANCELLED": BatchPollStatus.CANCELED,
        }
        poll_status = status_map.get(job.status, BatchPollStatus.IN_PROGRESS)

        if getattr(job, "errors", None):
            logger.warning(
                "Batch %s errors: %s", batch_id, _format_batch_errors(job.errors),
            )
        if getattr(job, "failed_requests", None) and job.failed_requests > 0:
            logger.warning(
                "Batch %s has %d failed request(s)", batch_id, job.failed_requests,
            )

        if poll_status != BatchPollStatus.ENDED:
            return BatchPollResult(status=poll_status)

        entries: list[BatchResultEntry] = []
        if _is_set(getattr(job, "error_file", None)):
            error_content = await _download_file_content(
                self._client, cast("str", job.error_file)
            )
            error_entries = _parse_error_file_entries(error_content)
            entries.extend(error_entries)

        if not _is_set(job.output_file):
            error_detail = (
                _format_batch_errors(job.errors)
                if getattr(job, "errors", None)
                else "no output file"
            )
            logger.warning(
                "Batch %s succeeded but output_file is not set: %s",
                batch_id, error_detail,
            )
            return BatchPollResult(status=BatchPollStatus.FAILED, entries=entries)

        content = await _download_file_content(self._client, cast("str", job.output_file))
        error_ids = {e.custom_id for e in entries}
        for line in content.strip().split("\n"):
            if not line.strip():
                continue
            entry_data = json.loads(line)
            custom_id = entry_data.get("custom_id", "")
            response_body = entry_data.get("response", {}).get("body", {})
            if custom_id in error_ids:
                entries = [e for e in entries if e.custom_id != custom_id]
                error_ids.discard(custom_id)
            if response_body.get("choices") or response_body.get("pages"):
                extracted = _extract_images_from_response(response_body)
                entries.append(
                    BatchResultEntry(
                        custom_id=custom_id,
                        succeeded=True,
                        raw_response_json=json.dumps(response_body),
                        extracted_images=extracted,
                    )
                )
            else:
                entries.append(
                    BatchResultEntry(
                        custom_id=custom_id,
                        succeeded=False,
                        error=json.dumps(response_body.get("error", "Unknown error")),
                    )
                )

        return BatchPollResult(status=BatchPollStatus.ENDED, entries=entries)

    @property
    def supports_sync_ocr(self) -> bool:
        return True

    async def call_ocr(
        self,
        *,
        document_data_uri: str,
        model: str,
        include_image_base64: bool = True,
    ) -> dict:
        from mistralai.models import DocumentURLChunk, ImageURLChunk

        mime_type = document_data_uri.split(":")[1].split(";")[0]
        if mime_type.startswith("image/"):
            document = ImageURLChunk(image_url=document_data_uri)
        else:
            document = DocumentURLChunk(document_url=document_data_uri)

        response = await self._client.ocr.process_async(
            model=model,
            document=document,
            include_image_base64=include_image_base64,
        )

        return response.model_dump(mode="json")

    def parse_batch_result(
        self,
        raw_json: str,
        output_type_name: str | None,
    ) -> ProviderResponse:
        data = json.loads(raw_json)
        usage = data.get("usage", {})
        model_name = data.get("model", "")

        if output_type_name is None:
            choices = data.get("choices", [])
            text_content: str | None = None
            if choices:
                message = choices[0].get("message", {})
                content = message.get("content", "")
                text_content = content if isinstance(content, str) else str(content)

            return ProviderResponse(
                text_content=text_content,
                model_name=model_name,
                input_tokens=usage.get("prompt_tokens", 0),
                output_tokens=usage.get("completion_tokens", 0),
                cache_creation_input_tokens=0,
                cache_read_input_tokens=0,
                raw_response_json=raw_json,
            )

        from sax_llm.registry import get_output_type_registry

        registry = get_output_type_registry()
        if output_type_name not in registry:
            msg = f"Unknown output type: {output_type_name!r}. Registered: {list(registry)}"
            raise KeyError(msg)

        choices = data.get("choices", [])
        tool_input: dict = {}
        if choices:
            message = choices[0].get("message", {})
            tool_calls = message.get("tool_calls", [])
            if tool_calls:
                args = tool_calls[0].get("function", {}).get("arguments", "{}")
                tool_input = json.loads(args) if isinstance(args, str) else args

        return ProviderResponse(
            tool_input=tool_input,
            model_name=model_name,
            input_tokens=usage.get("prompt_tokens", 0),
            output_tokens=usage.get("completion_tokens", 0),
            cache_creation_input_tokens=0,
            cache_read_input_tokens=0,
            raw_response_json=raw_json,
        )
