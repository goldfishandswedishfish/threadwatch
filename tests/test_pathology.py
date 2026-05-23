"""Tests for pathology.detect in threadwatch/pathology.py."""
from __future__ import annotations

import pytest

from threadwatch.pathology import detect
from threadwatch.trace import ProviderTrace, Step, ToolCall


KNOWN_TOOLS = {"calculator", "web_search", "word_count"}


def _trace(*steps: Step) -> ProviderTrace:
    t = ProviderTrace(provider_name="test", model="m", task="t")
    t.steps = list(steps)
    return t


def _step(num: int, tools: list[tuple[str, dict]] | None = None, content: str = "") -> Step:
    """Build a Step, optionally with tool calls."""
    s = Step(step_num=num, tokens_in=100, tokens_out=50, latency_ms=200, content=content)
    for name, args in (tools or []):
        s.tool_calls.append(ToolCall(name=name, arguments=args, response="ok", latency_ms=10))
    return s


# ---------------------------------------------------------------------------
# Clean trace — no pathologies
# ---------------------------------------------------------------------------

class TestDetectClean:
    def test_empty_trace_no_findings(self):
        assert detect(_trace(), KNOWN_TOOLS) == []

    def test_single_step_no_tools_no_findings(self):
        t = _trace(_step(1, content="Done."))
        assert detect(t, KNOWN_TOOLS) == []

    def test_two_different_tool_calls_no_findings(self):
        t = _trace(
            _step(1, [("calculator", {"expression": "1+1"})]),
            _step(2, [("web_search",  {"query": "python"})]),
        )
        assert detect(t, KNOWN_TOOLS) == []


# ---------------------------------------------------------------------------
# Repetition detection
# ---------------------------------------------------------------------------

class TestRepetition:
    def test_same_tool_same_args_flagged(self):
        args = {"expression": "2+2"}
        t = _trace(
            _step(1, [("calculator", args)]),
            _step(2, [("calculator", args)]),
        )
        findings = detect(t, KNOWN_TOOLS)
        assert any("repetition" in f for f in findings)

    def test_same_tool_different_args_not_flagged(self):
        t = _trace(
            _step(1, [("calculator", {"expression": "1+1"})]),
            _step(2, [("calculator", {"expression": "2+2"})]),
        )
        assert not any("repetition" in f for f in detect(t, KNOWN_TOOLS))

    def test_repetition_message_includes_tool_name(self):
        args = {"q": "test"}
        t = _trace(
            _step(1, [("web_search", args)]),
            _step(2, [("web_search", args)]),
        )
        findings = detect(t, KNOWN_TOOLS)
        assert any("web_search" in f for f in findings if "repetition" in f)

    def test_repetition_message_includes_step_number(self):
        args = {"expression": "3"}
        t = _trace(
            _step(1, [("calculator", args)]),
            _step(5, [("calculator", args)]),
        )
        findings = detect(t, KNOWN_TOOLS)
        repeat_findings = [f for f in findings if "repetition" in f]
        assert any("5" in f for f in repeat_findings)


# ---------------------------------------------------------------------------
# Hallucinated-tool detection
# ---------------------------------------------------------------------------

class TestHallucinatedTool:
    def test_unknown_tool_flagged(self):
        t = _trace(_step(1, [("fly_to_moon", {"dest": "moon"})]))
        findings = detect(t, KNOWN_TOOLS)
        assert any("hallucinated_tool" in f for f in findings)

    def test_known_tool_not_flagged_as_hallucination(self):
        t = _trace(_step(1, [("calculator", {"expression": "1"})]))
        assert not any("hallucinated_tool" in f for f in detect(t, KNOWN_TOOLS))

    def test_hallucination_includes_tool_name(self):
        t = _trace(_step(1, [("ghost_tool", {})]))
        findings = detect(t, KNOWN_TOOLS)
        assert any("ghost_tool" in f for f in findings)


# ---------------------------------------------------------------------------
# Backtracking detection
# ---------------------------------------------------------------------------

class TestBacktracking:
    def test_repeated_sequence_flagged(self):
        tools = [("calculator", {"expression": "1"}), ("web_search", {"query": "x"})]
        t = _trace(_step(1, tools), _step(2, tools))
        findings = detect(t, KNOWN_TOOLS)
        assert any("backtracking" in f for f in findings)

    def test_different_sequences_not_flagged(self):
        t = _trace(
            _step(1, [("calculator", {"expression": "1"}), ("web_search", {"query": "a"})]),
            _step(2, [("web_search", {"query": "a"}), ("calculator", {"expression": "1"})]),
        )
        # Order matters for sequences, so these are different sequences
        assert not any("backtracking" in f for f in detect(t, KNOWN_TOOLS))

    def test_single_tool_sequence_repeated(self):
        t = _trace(
            _step(1, [("calculator", {"expression": "2"})]),
            _step(2, [("calculator", {"expression": "3"})]),
            _step(3, [("calculator", {"expression": "2"})]),
        )
        # step 1 and step 3 have the same single-tool sequence; step 2 is different
        findings = detect(t, KNOWN_TOOLS)
        assert any("backtracking" in f for f in findings)


# ---------------------------------------------------------------------------
# Context bloat — blank steps
# ---------------------------------------------------------------------------

class TestContextBloatBlankSteps:
    def test_two_blank_steps_not_enough(self):
        t = _trace(
            _step(1, content=""),
            _step(2, content="   "),
        )
        findings = detect(t, KNOWN_TOOLS)
        assert not any("context_bloat" in f and "no tool calls" in f for f in findings)

    def test_three_blank_steps_triggers_bloat(self):
        """context_bloat fires when >= 3 steps have no tool calls and no content."""
        t = _trace(
            _step(1, content=""),
            _step(2, content=""),
            _step(3, content=""),
        )
        findings = detect(t, KNOWN_TOOLS)
        assert any("context_bloat" in f and "no tool calls" in f for f in findings)

    def test_blank_step_count_in_message(self):
        t = _trace(
            _step(1, content=""),
            _step(2, content=""),
            _step(3, content=""),
            _step(4, content=""),
        )
        findings = detect(t, KNOWN_TOOLS)
        bloat = [f for f in findings if "context_bloat" in f and "no tool calls" in f]
        assert any("4" in f for f in bloat)

    def test_step_with_content_not_counted_as_blank(self):
        t = _trace(
            _step(1, content=""),
            _step(2, content="thinking…"),  # has content — not blank
            _step(3, content=""),
        )
        # Only 2 blank steps, should not trigger
        findings = detect(t, KNOWN_TOOLS)
        assert not any("no tool calls" in f for f in findings)

    def test_step_with_tool_calls_not_counted_as_blank(self):
        t = _trace(
            _step(1, content=""),
            _step(2, [("calculator", {"expression": "1"})], content=""),  # has tool call
            _step(3, content=""),
        )
        # Only 2 blank steps
        findings = detect(t, KNOWN_TOOLS)
        assert not any("no tool calls" in f for f in findings)

    def test_whitespace_only_content_counts_as_blank(self):
        """Steps with only whitespace should count as blank."""
        t = _trace(
            _step(1, content="   "),
            _step(2, content="\t\n"),
            _step(3, content="  "),
        )
        findings = detect(t, KNOWN_TOOLS)
        assert any("context_bloat" in f and "no tool calls" in f for f in findings)


# ---------------------------------------------------------------------------
# Context bloat — token growth
# ---------------------------------------------------------------------------

def _step_with_tokens(num: int, tokens_in: int, content: str = "ok") -> Step:
    return Step(step_num=num, tokens_in=tokens_in, tokens_out=50, latency_ms=10, content=content)


class TestContextBloatTokenGrowth:
    def test_fewer_than_four_steps_no_token_bloat_check(self):
        # Token bloat check requires >= 4 steps
        t = _trace(
            _step_with_tokens(1, 100),
            _step_with_tokens(2, 1000),
        )
        assert not any("grew" in f for f in detect(t, KNOWN_TOOLS))

    def test_no_token_bloat_when_growth_is_moderate(self):
        t = _trace(
            _step_with_tokens(1, 100),
            _step_with_tokens(2, 150),
            _step_with_tokens(3, 200),
            _step_with_tokens(4, 250),
        )
        assert not any("grew" in f for f in detect(t, KNOWN_TOOLS))

    def test_token_bloat_triggered_when_ratio_exceeds_4x(self):
        # early = 100 + 100 = 200; late = 900 + 900 = 1800; ratio = 9x > 4
        t = _trace(
            _step_with_tokens(1, 100),
            _step_with_tokens(2, 100),
            _step_with_tokens(3, 900),
            _step_with_tokens(4, 900),
        )
        findings = detect(t, KNOWN_TOOLS)
        assert any("grew" in f for f in findings)

    def test_token_bloat_not_triggered_at_exactly_4x(self):
        # early = 200; late = 800; ratio = 4.0 — NOT > 4
        t = _trace(
            _step_with_tokens(1, 100),
            _step_with_tokens(2, 100),
            _step_with_tokens(3, 400),
            _step_with_tokens(4, 400),
        )
        findings = detect(t, KNOWN_TOOLS)
        assert not any("grew" in f for f in findings)

    def test_zero_early_tokens_skips_division(self):
        # early_tokens == 0 should not cause ZeroDivisionError
        t = _trace(
            _step_with_tokens(1, 0),
            _step_with_tokens(2, 0),
            _step_with_tokens(3, 1000),
            _step_with_tokens(4, 1000),
        )
        # Should not raise; condition `early_tokens > 0` guards the division
        findings = detect(t, KNOWN_TOOLS)
        assert not any("grew" in f for f in findings)
