"""Tests for parse_tool_calls in threadwatch/provider.py."""
from __future__ import annotations

import json

import pytest

from threadwatch.provider import parse_tool_calls


def _make_message(tool_calls: list[dict]) -> dict:
    """Helper: wrap raw tool_call dicts in a message envelope."""
    return {"role": "assistant", "content": None, "tool_calls": tool_calls}


def _make_tc(id_: str, name: str, arguments: str) -> dict:
    """Helper: create a single raw tool_call dict as returned by the API."""
    return {
        "id": id_,
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


# ---------------------------------------------------------------------------
# Basic happy-path tests
# ---------------------------------------------------------------------------

class TestParseToolCallsBasic:
    def test_empty_message_returns_empty_list(self):
        assert parse_tool_calls({}) == []

    def test_null_tool_calls_returns_empty_list(self):
        assert parse_tool_calls({"tool_calls": None}) == []

    def test_empty_tool_calls_list_returns_empty_list(self):
        assert parse_tool_calls({"tool_calls": []}) == []

    def test_single_valid_tool_call(self):
        msg = _make_message([_make_tc("call_1", "calculator", '{"expression": "2+2"}')])
        result = parse_tool_calls(msg)
        assert len(result) == 1
        assert result[0]["id"] == "call_1"
        assert result[0]["name"] == "calculator"
        assert result[0]["arguments"] == {"expression": "2+2"}

    def test_multiple_valid_tool_calls(self):
        msg = _make_message([
            _make_tc("call_1", "calculator", '{"expression": "1+1"}'),
            _make_tc("call_2", "web_search", '{"query": "python"}'),
        ])
        result = parse_tool_calls(msg)
        assert len(result) == 2
        assert result[0]["name"] == "calculator"
        assert result[1]["name"] == "web_search"

    def test_empty_arguments_string_parses_to_empty_dict(self):
        msg = _make_message([_make_tc("call_1", "word_count", "{}")])
        result = parse_tool_calls(msg)
        assert result[0]["arguments"] == {}

    def test_missing_arguments_field_defaults_to_empty_dict(self):
        # API sometimes omits the arguments key entirely
        msg = {"tool_calls": [{"id": "c1", "function": {"name": "noop"}}]}
        result = parse_tool_calls(msg)
        assert result[0]["arguments"] == {}

    def test_missing_id_defaults_to_empty_string(self):
        msg = {"tool_calls": [{"function": {"name": "noop", "arguments": "{}"}}]}
        result = parse_tool_calls(msg)
        assert result[0]["id"] == ""

    def test_missing_name_defaults_to_empty_string(self):
        msg = {"tool_calls": [{"id": "c1", "function": {"arguments": "{}"}}]}
        result = parse_tool_calls(msg)
        assert result[0]["name"] == ""

    def test_result_keys_are_id_name_arguments(self):
        msg = _make_message([_make_tc("c1", "foo", '{"x": 1}')])
        result = parse_tool_calls(msg)
        assert set(result[0].keys()) == {"id", "name", "arguments"}


# ---------------------------------------------------------------------------
# Malformed JSON handling
# ---------------------------------------------------------------------------

class TestParseToolCallsMalformedJSON:
    def test_completely_invalid_json_falls_back_to_raw(self):
        msg = _make_message([_make_tc("c1", "tool", "not json at all")])
        result = parse_tool_calls(msg)
        assert result[0]["arguments"] == {"_raw": "not json at all"}

    def test_truncated_json_falls_back_to_raw(self):
        msg = _make_message([_make_tc("c1", "tool", '{"expression": "2+')])
        result = parse_tool_calls(msg)
        assert result[0]["arguments"] == {"_raw": '{"expression": "2+'}

    def test_bare_string_falls_back_to_raw(self):
        msg = _make_message([_make_tc("c1", "tool", "hello")])
        result = parse_tool_calls(msg)
        assert result[0]["arguments"] == {"_raw": "hello"}

    def test_array_json_does_not_raise(self):
        # A JSON array is valid JSON but not an object — parse succeeds and
        # returns the list (not the _raw fallback).
        msg = _make_message([_make_tc("c1", "tool", "[1, 2, 3]")])
        result = parse_tool_calls(msg)
        # The function calls json.loads and stores whatever comes back.
        assert result[0]["arguments"] == [1, 2, 3]

    def test_malformed_one_of_two_tool_calls(self):
        """Only the broken call should fall back; the good one parses normally."""
        msg = _make_message([
            _make_tc("c1", "calc", '{"expression": "3*3"}'),
            _make_tc("c2", "bad",  "oops{"),
        ])
        result = parse_tool_calls(msg)
        assert result[0]["arguments"] == {"expression": "3*3"}
        assert result[1]["arguments"] == {"_raw": "oops{"}

    def test_raw_value_preserved_exactly(self):
        raw = "  {bad json} "
        msg = _make_message([_make_tc("c1", "t", raw)])
        result = parse_tool_calls(msg)
        assert result[0]["arguments"]["_raw"] == raw
