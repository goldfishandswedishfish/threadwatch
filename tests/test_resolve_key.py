"""Tests for ProviderClient._resolve_key in threadwatch/provider.py.

_resolve_key is a pure static method with two precedence levels:
  1. Provider-specific env var: THREADWATCH_<NAME_UPPER>_KEY
  2. URL-based well-known env var (OPENAI_API_KEY, GROQ_API_KEY, …)
  3. Empty-string fallback
"""
from __future__ import annotations

import os

import pytest

from threadwatch.provider import ProviderClient


def resolve(name: str, base_url: str, env: dict[str, str] | None = None) -> str:
    """Call _resolve_key with a clean environment (no leakage from the test runner)."""
    env = env or {}
    # Patch os.environ for the duration of this call only.
    original = os.environ.copy()
    # Remove ALL env vars that _resolve_key might read, then inject only ours.
    for k in list(os.environ.keys()):
        del os.environ[k]
    os.environ.update(env)
    try:
        return ProviderClient._resolve_key(name, base_url)
    finally:
        os.environ.clear()
        os.environ.update(original)


# ---------------------------------------------------------------------------
# Fallback — empty string when no env var is set
# ---------------------------------------------------------------------------

class TestResolveKeyFallback:
    def test_no_env_vars_returns_empty_string(self):
        result = resolve("someprovider", "https://example.com/v1", {})
        assert result == ""

    def test_unknown_url_no_specific_key_returns_empty_string(self):
        result = resolve("myco", "https://myco.internal/api/v1", {})
        assert result == ""


# ---------------------------------------------------------------------------
# Provider-specific key takes highest precedence
# ---------------------------------------------------------------------------

class TestResolveKeyProviderSpecific:
    def test_provider_specific_key_returned(self):
        env = {"THREADWATCH_GROQ_KEY": "specific-key"}
        result = resolve("groq", "https://api.groq.com/openai/v1", env)
        assert result == "specific-key"

    def test_provider_specific_key_beats_url_based_key(self):
        env = {
            "THREADWATCH_OPENAI_KEY": "specific-key",
            "OPENAI_API_KEY": "generic-key",
        }
        result = resolve("openai", "https://api.openai.com/v1", env)
        assert result == "specific-key"

    def test_provider_name_is_uppercased_for_lookup(self):
        # Name "Groq" (mixed case) should map to THREADWATCH_GROQ_KEY
        env = {"THREADWATCH_GROQ_KEY": "my-key"}
        result = resolve("Groq", "https://api.groq.com/openai/v1", env)
        assert result == "my-key"

    def test_provider_specific_key_for_custom_name(self):
        env = {"THREADWATCH_MYCORP_KEY": "corp-key"}
        result = resolve("mycorp", "https://mycorp.ai/v1", env)
        assert result == "corp-key"


# ---------------------------------------------------------------------------
# URL-based well-known env vars (second priority)
# ---------------------------------------------------------------------------

class TestResolveKeyUrlFallbacks:
    @pytest.mark.parametrize("base_url,env_var,expected", [
        ("https://api.openai.com/v1",    "OPENAI_API_KEY",     "openai-key"),
        ("https://api.groq.com/openai/v1", "GROQ_API_KEY",    "groq-key"),
        ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "or-key"),
        ("https://api.anthropic.com/v1", "ANTHROPIC_API_KEY",  "anth-key"),
        ("https://api.together.xyz/v1",  "TOGETHER_API_KEY",   "together-key"),
        ("https://api.mistral.ai/v1",    "MISTRAL_API_KEY",    "mistral-key"),
    ])
    def test_url_based_key_returned(self, base_url: str, env_var: str, expected: str):
        result = resolve("generic", base_url, {env_var: expected})
        assert result == expected

    def test_url_based_key_absent_returns_empty(self):
        # The URL matches openai, but the env var isn't set
        result = resolve("generic", "https://api.openai.com/v1", {})
        assert result == ""

    def test_url_based_key_does_not_fire_for_wrong_url(self):
        result = resolve("x", "https://api.groq.com/openai/v1", {"OPENAI_API_KEY": "key"})
        # OPENAI_API_KEY should NOT be returned for a Groq URL
        assert result == ""


# ---------------------------------------------------------------------------
# Precedence ordering: specific > url-based > empty
# ---------------------------------------------------------------------------

class TestResolveKeyPrecedence:
    def test_specific_beats_url_based(self):
        env = {
            "THREADWATCH_GROQ_KEY": "specific",
            "GROQ_API_KEY": "url-based",
        }
        assert resolve("groq", "https://api.groq.com/openai/v1", env) == "specific"

    def test_url_based_beats_empty(self):
        env = {"GROQ_API_KEY": "url-based"}
        assert resolve("groq", "https://api.groq.com/openai/v1", env) == "url-based"

    def test_empty_when_neither_is_set(self):
        assert resolve("groq", "https://api.groq.com/openai/v1", {}) == ""
