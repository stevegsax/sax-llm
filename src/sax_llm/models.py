"""Normalized response types and structured message models for LLM providers."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Structured message models
# ---------------------------------------------------------------------------


class TextContent(BaseModel):
    """Plain text content block."""

    type: Literal["text"] = "text"
    text: str


class ImageContent(BaseModel):
    """Base64-encoded image content block."""

    type: Literal["image"] = "image"
    media_type: str  # "image/png", "image/jpeg"
    data: str  # base64-encoded


class DocumentContent(BaseModel):
    """Base64-encoded document content block (e.g. PDF)."""

    type: Literal["document"] = "document"
    media_type: str  # "application/pdf"
    data: str  # base64-encoded


ContentBlock = TextContent | ImageContent | DocumentContent


class Message(BaseModel):
    """A single message in a conversation."""

    role: Literal["system", "user", "assistant"]
    content: str | list[ContentBlock]
    cache_control: bool = False  # hint for Anthropic cache breakpoints


def text_messages(
    system_prompt: str, user_prompt: str, *, cache_system: bool = True
) -> list[Message]:
    """Build standard system + user message pair from plain text strings."""
    return [
        Message(role="system", content=system_prompt, cache_control=cache_system),
        Message(role="user", content=user_prompt),
    ]


# ---------------------------------------------------------------------------
# Provider response
# ---------------------------------------------------------------------------


class ProviderResponse(BaseModel):
    """Normalized response from any LLM provider."""

    tool_input: dict = Field(
        default_factory=dict,
        description="Parsed structured output (tool call arguments).",
    )
    text_content: str | None = Field(
        default=None,
        description="Plain text response when output_type is None.",
    )
    model_name: str = Field(description="Actual model that responded.")
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    raw_response_json: str = Field(
        description="Serialized original response for message logging.",
    )


class BatchPollStatus(StrEnum):
    """Normalized batch poll statuses across providers."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    ENDED = "ended"
    FAILED = "failed"
    CANCELED = "canceled"
    EXPIRED = "expired"


class ExtractedImage(BaseModel):
    """An image extracted from an OCR response, before storage."""

    original_image_id: str
    page_index: int
    image_base64: str
    mime_type: str = "image/jpeg"
    top_left_x: int | None = None
    top_left_y: int | None = None
    bottom_right_x: int | None = None
    bottom_right_y: int | None = None


class BatchResultEntry(BaseModel):
    """A single result entry from a batch response."""

    custom_id: str
    succeeded: bool
    raw_response_json: str | None = None
    error: str | None = None
    extracted_images: list[ExtractedImage] = Field(default_factory=list)


class BatchPollResult(BaseModel):
    """Result of polling a batch job."""

    status: BatchPollStatus
    entries: list[BatchResultEntry] = Field(default_factory=list)
