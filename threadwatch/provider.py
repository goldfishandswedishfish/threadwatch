from __future__ import annotations

import os
import json
import time
import httpx
from typing import Any


class ProviderError(Exception):
    pass


class ProviderClient:
    def __init__(self, name: str, base_url: str, model: str):
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = self._resolve_key(name, base_url)

    @staticmethod
    def _resolve_key(name: str, base_url: str) -> str:
        # 1. Provider-specific key always wins
        specific = os.environ.get(f"THREADWATCH_{name.upper()}_KEY")
        if specific:
            return specific
        # 2. Fall back to well-known env vars based on the base URL
        _URL_FALLBACKS = {
            "openrouter.ai": "OPENROUTER_API_KEY",
            "api.openai.com": "OPENAI_API_KEY",
            "api.groq.com": "GROQ_API_KEY",
            "api.anthropic.com": "ANTHROPIC_API_KEY",
            "api.together.xyz": "TOGETHER_API_KEY",
            "api.mistral.ai": "MISTRAL_API_KEY",
        }
        for domain, env_var in _URL_FALLBACKS.items():
            if domain in base_url:
                val = os.environ.get(env_var)
                if val:
                    return val
        return ""

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self._api_key:
            h["Authorization"] = f"Bearer {self._api_key}"
        if "openrouter.ai" in self.base_url:
            h["HTTP-Referer"] = "https://github.com/threadwatch"
            h["X-Title"] = "threadwatch"
        return h

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict],
        timeout: float = 120,
    ) -> tuple[dict[str, Any], float]:
        payload = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
        }
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    f"{self.base_url}/chat/completions",
                    headers=self._headers(),
                    json=payload,
                )
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as e:
            body = e.response.text[:300]
            raise ProviderError(f"HTTP {e.response.status_code}: {body}") from e
        except httpx.TimeoutException:
            raise ProviderError(f"Request timed out after {timeout}s")
        except Exception as e:
            raise ProviderError(str(e)) from e

        latency_ms = (time.perf_counter() - t0) * 1000
        return data, latency_ms


def parse_tool_calls(message: dict) -> list[dict]:
    """Return a list of {id, name, arguments_dict} from a response message."""
    raw = message.get("tool_calls") or []
    result = []
    for tc in raw:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments", "{}"))
        except json.JSONDecodeError:
            args = {"_raw": fn.get("arguments", "")}
        result.append({"id": tc.get("id", ""), "name": fn.get("name", ""), "arguments": args})
    return result
