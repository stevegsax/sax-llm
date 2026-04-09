"""Shared LLM client utilities.

Pure functions for Anthropic SDK request construction and response parsing.
get_anthropic_client provides client management.

Design follows Function Core / Imperative Shell:

- Pure functions: build_tool_definition, build_system_param, build_thinking_param,
  build_messages_params, extract_tool_result, extract_usage
- Imperative shell: get_anthropic_client
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from anthropic.types import Message
    from pydantic import BaseModel


def _snake_case(name: str) -> str:
    """Convert CamelCase class name to snake_case tool name."""
    s = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", name)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return s.lower()


def build_tool_definition(
    output_type: type[BaseModel],
    *,
    cache_control: bool = True,
) -> dict:
    """Build an Anthropic tool definition from a Pydantic model."""
    schema = output_type.model_json_schema()
    tool_name = _snake_case(output_type.__name__)
    description = (output_type.__doc__ or "").strip() or f"Structured output: {tool_name}"

    tool: dict = {
        "name": tool_name,
        "description": description,
        "input_schema": schema,
    }
    if cache_control:
        tool["cache_control"] = {"type": "ephemeral"}
    return tool


def build_system_param(
    system_prompt: str,
    *,
    cache_control: bool = True,
) -> list[dict] | str:
    """Build the system parameter for messages.create."""
    if cache_control:
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ]
    return system_prompt


def build_thinking_param(
    model_name: str,
    budget_tokens: int,
) -> dict | None:
    """Build the thinking parameter for messages.create.

    Returns a thinking dict for Anthropic models, or None for non-Anthropic/Haiku.
    """
    if budget_tokens <= 0:
        return None

    if "haiku" in model_name:
        return None

    if "opus" in model_name:
        return {"type": "enabled", "budget_tokens": budget_tokens}

    if "sonnet" in model_name or "claude" in model_name:
        return {"type": "enabled", "budget_tokens": budget_tokens}

    return None


def build_messages_params(
    system_prompt: str,
    user_prompt: str,
    output_type: type[BaseModel],
    model: str,
    max_tokens: int,
    *,
    cache_instructions: bool = True,
    cache_tool_definitions: bool = True,
    thinking_budget_tokens: int = 0,
) -> dict:
    """Build the full kwargs dict for client.messages.create."""
    tool_def = build_tool_definition(output_type, cache_control=cache_tool_definitions)
    tool_name = tool_def["name"]
    system = build_system_param(system_prompt, cache_control=cache_instructions)

    params: dict = {
        "model": model,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user_prompt}],
        "tools": [tool_def],
        "tool_choice": {"type": "tool", "name": tool_name},
    }

    thinking = build_thinking_param(model, thinking_budget_tokens)
    if thinking is not None:
        params["thinking"] = thinking
        params["tool_choice"] = {"type": "auto"}
        params["max_tokens"] = max(max_tokens, thinking_budget_tokens + max_tokens)

    return params


def extract_tool_result(message: Message, output_type: type[BaseModel]) -> BaseModel:
    """Extract and validate the tool_use result from an Anthropic Message."""
    for block in message.content:
        if block.type == "tool_use":
            return output_type.model_validate(block.input)

    msg = f"No tool_use block found in message. Content types: {[b.type for b in message.content]}"
    raise ValueError(msg)


def extract_usage(message: Message) -> tuple[int, int, int, int]:
    """Extract usage statistics from an Anthropic Message.

    Returns (input_tokens, output_tokens, cache_creation, cache_read).
    """
    usage = message.usage
    return (
        usage.input_tokens,
        usage.output_tokens,
        getattr(usage, "cache_creation_input_tokens", 0) or 0,
        getattr(usage, "cache_read_input_tokens", 0) or 0,
    )


# ---------------------------------------------------------------------------
# Batch processing helpers
# ---------------------------------------------------------------------------


def build_batch_request(custom_id: str, params: dict) -> dict:
    """Wrap messages.create params into a batch request item."""
    return {"custom_id": custom_id, "params": params}


def parse_batch_response_json(
    raw_json: str,
    output_type_name: str,
) -> tuple:
    """Deserialize a raw Anthropic Message JSON from a batch response.

    Returns (parsed_model, model_name, input_tokens, output_tokens,
             cache_creation_input_tokens, cache_read_input_tokens).

    Uses the output type registry — consumers must register their types
    via ``register_output_type()`` before calling this.
    """
    from anthropic.types import Message as AnthropicMessage

    from sax_llm.registry import get_output_type_registry

    registry = get_output_type_registry()
    if output_type_name not in registry:
        msg = f"Unknown output type: {output_type_name!r}. Registered: {list(registry)}"
        raise KeyError(msg)

    output_type = registry[output_type_name]
    data = json.loads(raw_json)
    message = AnthropicMessage.model_validate(data)

    parsed = extract_tool_result(message, output_type)
    in_tok, out_tok, cache_create, cache_read = extract_usage(message)

    return (parsed, message.model, in_tok, out_tok, cache_create, cache_read)


# ---------------------------------------------------------------------------
# Imperative shell
# ---------------------------------------------------------------------------

_client = None


def get_anthropic_client():
    """Get or create a shared AsyncAnthropic client."""
    global _client
    if _client is None:
        from anthropic import AsyncAnthropic

        _client = AsyncAnthropic()
    return _client


def reset_client() -> None:
    """Clear the cached client. Intended for testing."""
    global _client
    _client = None
