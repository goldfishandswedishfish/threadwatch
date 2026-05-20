"""
Keyless web search via DuckDuckGo Instant Answers API.

Drop-in replacement for web_search.py — no API key required.

To use: copy this file to ~/.threadwatch/tools/ and remove (or rename)
the existing web_search.py so only one web_search tool is active at a time.

    cp examples/tools/web_search_ddg.py ~/.threadwatch/tools/web_search.py

Limitations: DuckDuckGo Instant Answers returns summaries and related topics,
not full search results. Good for factual lookups; less useful for deep research.
"""
from __future__ import annotations

import httpx

DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for information using DuckDuckGo. Returns a summary and related topics. No API key required.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                }
            },
            "required": ["query"],
        },
    },
}


async def web_search(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                "https://api.duckduckgo.com/",
                params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
                headers={"User-Agent": "threadwatch/1.0"},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        return f"Error: {e}"

    parts: list[str] = []

    if data.get("AbstractText"):
        parts.append(data["AbstractText"])
        if data.get("AbstractURL"):
            parts.append(f"Source: {data['AbstractURL']}")

    if data.get("Answer"):
        parts.append(f"Answer: {data['Answer']}")

    topics = data.get("RelatedTopics", [])[:5]
    if topics:
        parts.append("\nRelated:")
        for t in topics:
            if isinstance(t, dict) and t.get("Text"):
                parts.append(f"- {t['Text'][:200]}")

    return "\n".join(parts) if parts else "No results found. Try a more specific query."
