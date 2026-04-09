"""sax-llm — Shared LLM provider abstraction.

Public API:

- parse_model_id: Split "provider:model" strings
- get_provider: Factory returning cached provider singletons
- ProviderResponse: Normalized response from any provider
- LLMProvider: Protocol for provider implementations
- Message, ContentBlock, TextContent, ImageContent, DocumentContent: Structured messages
- text_messages: Helper to build text-only message pairs
- register_output_type / get_output_type_registry: Pluggable type registry for batch parsing
"""

from __future__ import annotations

from sax_llm.models import (
    ContentBlock,
    DocumentContent,
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
    text_messages,
)
from sax_llm.protocol import LLMProvider
from sax_llm.registry import (
    get_output_type_registry,
    get_provider,
    get_provider_by_name,
    parse_model_id,
    register_output_type,
)

__all__ = [
    "ContentBlock",
    "DocumentContent",
    "ImageContent",
    "LLMProvider",
    "Message",
    "ProviderResponse",
    "TextContent",
    "get_output_type_registry",
    "get_provider",
    "get_provider_by_name",
    "parse_model_id",
    "register_output_type",
    "text_messages",
]
