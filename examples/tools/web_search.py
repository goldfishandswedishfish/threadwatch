from __future__ import annotations

import os
import httpx


TAVILY_URL = "https://api.tavily.com/search"


async def web_search(query: str, max_results: int = 5) -> str:
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return "web_search is unavailable: TAVILY_API_KEY is not set. Add it to ~/.threadwatch.env to enable web search."
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                TAVILY_URL,
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": True,
                },
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPStatusError as e:
        return f"Error: HTTP {e.response.status_code} from Tavily"
    except Exception as e:
        return f"Error: {e}"

    parts: list[str] = []
    if data.get("answer"):
        parts.append(f"Summary: {data['answer']}\n")
    for r in data.get("results", [])[:max_results]:
        parts.append(f"[{r.get('title', 'No title')}]({r.get('url', '')})")
        if r.get("content"):
            parts.append(r["content"][:400])
        parts.append("")
    return "\n".join(parts) if parts else "No results found."


DEFINITION = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for current information using Tavily. Returns a summary and top results.",
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
