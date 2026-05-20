from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Awaitable
from typing import Any, Optional

from .provider import ProviderClient, ProviderError, parse_tool_calls
from .trace import ProviderTrace, Step, ToolCall
from .tools import TOOL_DEFINITIONS, TOOL_NAMES, execute_tool
from . import pathology

UpdateCallback = Callable[[ProviderTrace], Optional[Awaitable[None]]]

SYSTEM_PROMPT = (
    "You are a capable research and analysis assistant. "
    "Use the provided tools as needed to complete the task thoroughly. "
    "When you have gathered enough information, write a final answer directly."
)


async def run_provider(
    name: str,
    base_url: str,
    model: str,
    task: str,
    max_steps: int,
    on_update: UpdateCallback | None = None,
) -> ProviderTrace:
    client = ProviderClient(name, base_url, model)
    trace = ProviderTrace(provider_name=name, model=model, task=task)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task},
    ]

    async def _notify():
        if on_update:
            result = on_update(trace)
            if asyncio.iscoroutine(result):
                await result

    for step_num in range(1, max_steps + 1):
        try:
            response, latency_ms = await client.chat(messages, TOOL_DEFINITIONS)
        except ProviderError as e:
            trace.error = str(e)
            trace.finished_at = time.time()
            await _notify()
            return trace

        choice = response.get("choices", [{}])[0]
        message = choice.get("message", {})
        usage = response.get("usage", {})

        tokens_in = usage.get("prompt_tokens", 0)
        tokens_out = usage.get("completion_tokens", 0)
        content = message.get("content") or ""
        finish_reason = choice.get("finish_reason", "")

        tool_call_dicts = parse_tool_calls(message)
        step = Step(
            step_num=step_num,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_ms=latency_ms,
            content=content,
        )

        # Append assistant turn to conversation
        assistant_msg: dict = {"role": "assistant", "content": content}
        if tool_call_dicts:
            import json as _json
            assistant_msg["tool_calls"] = [
                {
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])},
                }
                for tc in tool_call_dicts
            ]
        messages.append(assistant_msg)

        if tool_call_dicts:
            tool_results: list[dict] = []
            for tc in tool_call_dicts:
                t0 = time.perf_counter()
                result = await execute_tool(tc["name"], tc["arguments"])
                tool_latency = (time.perf_counter() - t0) * 1000

                step.tool_calls.append(ToolCall(
                    name=tc["name"],
                    arguments=tc["arguments"],
                    response=result,
                    latency_ms=tool_latency,
                ))
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result,
                })

            messages.extend(tool_results)

        trace.steps.append(step)
        await _notify()

        if finish_reason == "stop" or not tool_call_dicts:
            trace.final_response = content
            trace.completed = True
            break

        if step_num == max_steps:
            trace.final_response = content
            trace.error = f"Reached max_steps ({max_steps}) without finishing"

    trace.pathologies = pathology.detect(trace, TOOL_NAMES)
    trace.finished_at = time.time()
    await _notify()
    return trace


async def run_providers(
    task: str,
    providers: list[tuple[str, str, str]],
    max_steps: int,
    on_update: UpdateCallback | None = None,
) -> list[ProviderTrace]:
    tasks = [
        run_provider(name, base_url, model, task, max_steps, on_update)
        for name, base_url, model in providers
    ]
    return list(await asyncio.gather(*tasks))
