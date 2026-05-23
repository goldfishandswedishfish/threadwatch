"""Tests for _fmt_args in threadwatch/display.py."""
from __future__ import annotations

import pytest

from threadwatch.display import _fmt_args


class TestFmtArgsBasic:
    def test_empty_dict_returns_empty_string(self):
        assert _fmt_args({}) == ""

    def test_single_short_value_returned_as_is(self):
        assert _fmt_args({"expression": "2+2"}) == "2+2"

    def test_uses_first_value_only(self):
        # Only the first dict value should be reflected in output.
        # Python 3.7+ guarantees insertion order.
        result = _fmt_args({"first": "hello", "second": "world"})
        assert result == "hello"

    def test_value_converted_to_string(self):
        assert _fmt_args({"count": 42}) == "42"

    def test_none_value_converted_to_string(self):
        assert _fmt_args({"key": None}) == "None"

    def test_bool_value_converted_to_string(self):
        assert _fmt_args({"flag": True}) == "True"


class TestFmtArgsTruncation:
    def test_value_exactly_40_chars_not_truncated(self):
        val = "a" * 40
        result = _fmt_args({"k": val})
        assert result == val
        assert not result.endswith("…")

    def test_value_41_chars_truncated_to_40_plus_ellipsis(self):
        val = "a" * 41
        result = _fmt_args({"k": val})
        assert result == "a" * 40 + "…"

    def test_value_100_chars_truncated(self):
        val = "x" * 100
        result = _fmt_args({"k": val})
        assert result == "x" * 40 + "…"

    def test_truncation_on_first_value_of_multi_key_dict(self):
        val = "b" * 50
        result = _fmt_args({"first": val, "second": "short"})
        assert result == "b" * 40 + "…"

    def test_value_39_chars_not_truncated(self):
        val = "z" * 39
        result = _fmt_args({"k": val})
        assert result == val


class TestFmtArgsEdgeCases:
    def test_list_value_displayed_as_string(self):
        result = _fmt_args({"items": [1, 2, 3]})
        assert result == "[1, 2, 3]"

    def test_dict_value_displayed_as_string(self):
        result = _fmt_args({"nested": {"a": 1}})
        assert "a" in result

    def test_long_list_value_truncated(self):
        val = list(range(100))  # str(val) >> 40 chars
        result = _fmt_args({"k": val})
        assert result.endswith("…")

    def test_whitespace_value_preserved(self):
        assert _fmt_args({"k": "   "}) == "   "

    def test_newline_in_value_preserved_up_to_40_chars(self):
        val = "line1\nline2"
        result = _fmt_args({"k": val})
        assert result == val
