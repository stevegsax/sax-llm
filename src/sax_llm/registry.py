"""Provider registry, model ID parsing, and output type registration.

The output type registry uses a plugin pattern: consumers call
``register_output_type()`` at startup to register their Pydantic models
for batch response parsing.  This replaces the hardcoded import list
from the original Forge implementation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from sax_llm.protocol import LLMProvider

_DEFAULT_PROVIDER = "anthropic"

_provider_cache: dict[str, LLMProvider] = {}

# ---------------------------------------------------------------------------
# Output type registry (plugin pattern)
# ---------------------------------------------------------------------------

_output_type_registry: dict[str, type[BaseModel]] = {}


def register_output_type(name: str, output_type: type[BaseModel]) -> None:
    """Register a Pydantic model for batch response parsing.

    Call this at worker startup to register your models::

        from sax_llm import register_output_type
        from myapp.models import MyResponse
        register_output_type("MyResponse", MyResponse)
    """
    _output_type_registry[name] = output_type


def get_output_type_registry() -> dict[str, type[BaseModel]]:
    """Return the registered output type mapping."""
    return _output_type_registry


def reset_output_type_registry() -> None:
    """Clear the output type registry. Intended for testing."""
    _output_type_registry.clear()


# ---------------------------------------------------------------------------
# Model ID parsing
# ---------------------------------------------------------------------------


def parse_model_id(model_id: str) -> tuple[str, str]:
    """Parse a model ID into (provider, model) tuple.

    Supports explicit ``provider:model`` syntax. Bare names without ``:``
    default to the ``"anthropic"`` provider for backward compatibility.

    Examples:
        >>> parse_model_id("anthropic:claude-sonnet-4-5-20250929")
        ('anthropic', 'claude-sonnet-4-5-20250929')
        >>> parse_model_id("mistral:mistral-large-latest")
        ('mistral', 'mistral-large-latest')
        >>> parse_model_id("claude-sonnet-4-5-20250929")
        ('anthropic', 'claude-sonnet-4-5-20250929')
    """
    if ":" in model_id:
        provider, model = model_id.split(":", 1)
        return provider, model
    return _DEFAULT_PROVIDER, model_id


# ---------------------------------------------------------------------------
# Provider factory
# ---------------------------------------------------------------------------


def get_provider_by_name(provider_name: str) -> LLMProvider:
    """Return a cached provider instance for a bare provider name."""
    if provider_name in _provider_cache:
        return _provider_cache[provider_name]

    if provider_name == "anthropic":
        from sax_llm.anthropic import AnthropicProvider

        instance: LLMProvider = AnthropicProvider()
    elif provider_name == "mistral":
        from sax_llm.mistral import MistralProvider

        instance = MistralProvider()
    else:
        msg = f"Unknown LLM provider: {provider_name!r}"
        raise ValueError(msg)

    _provider_cache[provider_name] = instance
    return instance


def get_provider(model_id: str) -> LLMProvider:
    """Return a cached provider instance for the given model ID."""
    provider_name, _ = parse_model_id(model_id)
    return get_provider_by_name(provider_name)


def reset_provider_cache() -> None:
    """Clear the provider cache. Intended for testing."""
    _provider_cache.clear()
