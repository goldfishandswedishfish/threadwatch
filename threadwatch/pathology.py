from __future__ import annotations

import json
from .trace import ProviderTrace, Step


def detect(trace: ProviderTrace, known_tools: set[str]) -> list[str]:
    findings: list[str] = []

    seen_calls: list[tuple[str, str]] = []
    tool_sequences: list[list[str]] = []

    for step in trace.steps:
        step_tools = []
        for tc in step.tool_calls:
            args_key = json.dumps(tc.arguments, sort_keys=True)
            pair = (tc.name, args_key)

            if pair in seen_calls:
                findings.append(
                    f"repetition: '{tc.name}' called again with identical args at step {step.step_num}"
                )
            else:
                seen_calls.append(pair)

            if tc.name not in known_tools:
                findings.append(
                    f"hallucinated_tool: '{tc.name}' is not in the provided tool list (step {step.step_num})"
                )

            step_tools.append(tc.name)

        if step_tools:
            tool_sequences.append(step_tools)

    # Backtracking: same sequence of tool names appears more than once
    seq_strs = [",".join(s) for s in tool_sequences]
    seen_seqs: set[str] = set()
    for s in seq_strs:
        if s in seen_seqs:
            findings.append(f"backtracking: tool sequence [{s}] repeated")
        seen_seqs.add(s)

    # Context bloat: token count more than triples with no tool calls
    idle_steps = [s for s in trace.steps if not s.tool_calls and not s.content.strip()]
    if len(idle_steps) >= 3:
        findings.append(
            f"context_bloat: {len(idle_steps)} steps produced no tool calls and no content"
        )

    if len(trace.steps) >= 4:
        early_tokens = sum(s.tokens_in for s in trace.steps[:2])
        late_tokens = sum(s.tokens_in for s in trace.steps[-2:])
        if early_tokens > 0 and late_tokens / early_tokens > 4:
            findings.append(
                f"context_bloat: prompt token count grew {late_tokens / early_tokens:.1f}x by final steps"
            )

    return findings
