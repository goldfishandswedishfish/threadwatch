from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import time


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]
    response: str
    latency_ms: float


@dataclass
class Step:
    step_num: int
    tokens_in: int
    tokens_out: int
    latency_ms: float
    tool_calls: list[ToolCall] = field(default_factory=list)
    content: str = ""
    error: str | None = None


@dataclass
class ProviderTrace:
    provider_name: str
    model: str
    task: str
    steps: list[Step] = field(default_factory=list)
    pathologies: list[str] = field(default_factory=list)
    final_response: str = ""
    completed: bool = False
    error: str | None = None
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    @property
    def total_tokens_in(self) -> int:
        return sum(s.tokens_in for s in self.steps)

    @property
    def total_tokens_out(self) -> int:
        return sum(s.tokens_out for s in self.steps)

    @property
    def total_latency_ms(self) -> float:
        return sum(s.latency_ms for s in self.steps)

    @property
    def total_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def elapsed_s(self) -> float:
        end = self.finished_at or time.time()
        return end - self.started_at
