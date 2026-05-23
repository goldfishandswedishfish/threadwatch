"""Tests for the calculator tool in examples/tools/calculator.py."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

# The calculator lives outside the installable package — add the examples dir to sys.path.
sys.path.insert(0, str(Path(__file__).parent.parent / "examples" / "tools"))
from calculator import calculator  # noqa: E402


# ---------------------------------------------------------------------------
# Basic arithmetic
# ---------------------------------------------------------------------------

class TestCalculatorBasicArithmetic:
    def test_addition(self):
        assert calculator("1 + 2") == "3"

    def test_subtraction(self):
        assert calculator("10 - 4") == "6"

    def test_multiplication(self):
        assert calculator("3 * 4") == "12"

    def test_division(self):
        result = float(calculator("7 / 2"))
        assert result == pytest.approx(3.5)

    def test_floor_division(self):
        assert calculator("7 // 2") == "3"

    def test_modulo(self):
        assert calculator("10 % 3") == "1"

    def test_exponentiation(self):
        assert calculator("2 ** 10") == "1024"

    def test_unary_negation(self):
        result = float(calculator("-5"))
        assert result == pytest.approx(-5)

    def test_unary_positive(self):
        assert calculator("+3") == "3"

    def test_chained_operations(self):
        result = float(calculator("2 + 3 * 4"))
        assert result == pytest.approx(14)  # operator precedence preserved

    def test_parentheses_respected(self):
        result = float(calculator("(2 + 3) * 4"))
        assert result == pytest.approx(20)


# ---------------------------------------------------------------------------
# Math functions and constants
# ---------------------------------------------------------------------------

class TestCalculatorMathFunctions:
    def test_sqrt(self):
        result = float(calculator("sqrt(144)"))
        assert result == pytest.approx(12.0)

    def test_log_natural(self):
        result = float(calculator("log(e)"))
        assert result == pytest.approx(1.0)

    def test_log10(self):
        result = float(calculator("log10(1000)"))
        assert result == pytest.approx(3.0)

    def test_sin_zero(self):
        result = float(calculator("sin(0)"))
        assert result == pytest.approx(0.0)

    def test_cos_zero(self):
        result = float(calculator("cos(0)"))
        assert result == pytest.approx(1.0)

    def test_tan_zero(self):
        result = float(calculator("tan(0)"))
        assert result == pytest.approx(0.0)

    def test_pi_constant(self):
        result = float(calculator("pi"))
        assert result == pytest.approx(math.pi)

    def test_e_constant(self):
        result = float(calculator("e"))
        assert result == pytest.approx(math.e)

    def test_abs_negative(self):
        assert calculator("abs(-7)") == "7"

    def test_round(self):
        assert calculator("round(2.7)") == "3"

    def test_min_two_args(self):
        assert calculator("min(3, 7)") == "3"

    def test_max_two_args(self):
        assert calculator("max(3, 7)") == "7"

    def test_sqrt_of_expression(self):
        result = float(calculator("sqrt(3 ** 2 + 4 ** 2)"))
        assert result == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Whitespace tolerance
# ---------------------------------------------------------------------------

class TestCalculatorWhitespace:
    def test_leading_trailing_whitespace_stripped(self):
        assert calculator("  2 + 2  ") == "4"

    def test_no_spaces(self):
        assert calculator("2+2") == "4"


# ---------------------------------------------------------------------------
# Unsafe expression rejection (AST whitelist)
# ---------------------------------------------------------------------------

class TestCalculatorUnsafeExpressions:
    def _is_error(self, expr: str) -> bool:
        result = calculator(expr)
        return result.startswith("Error:")

    def test_eval_rejected(self):
        assert self._is_error("eval('1+1')")

    def test_exec_rejected(self):
        assert self._is_error("exec('x=1')")

    def test_import_rejected(self):
        assert self._is_error("__import__('os')")

    def test_dunder_rejected(self):
        assert self._is_error("__class__")

    def test_open_rejected(self):
        # 'open' is not in _SAFE_NAMES
        assert self._is_error("open('file.txt')")

    def test_string_literal_rejected(self):
        # Unsupported constant type (str)
        assert self._is_error("'hello'")

    def test_attribute_access_rejected(self):
        # Attribute nodes are not handled by _eval_node
        assert self._is_error("math.sqrt(4)")

    def test_list_literal_rejected(self):
        assert self._is_error("[1, 2, 3]")

    def test_unknown_name_rejected(self):
        assert self._is_error("x + 1")

    def test_unknown_function_rejected(self):
        assert self._is_error("os_listdir()")

    def test_lambda_rejected(self):
        assert self._is_error("lambda x: x")

    def test_comprehension_rejected(self):
        assert self._is_error("[x for x in range(10)]")


# ---------------------------------------------------------------------------
# Edge cases / error paths
# ---------------------------------------------------------------------------

class TestCalculatorEdgeCases:
    def test_division_by_zero_returns_error(self):
        result = calculator("1 / 0")
        assert result.startswith("Error:")

    def test_empty_string_returns_error(self):
        result = calculator("")
        assert result.startswith("Error:")

    def test_garbage_input_returns_error(self):
        result = calculator("???")
        assert result.startswith("Error:")

    def test_large_exponent(self):
        # Should evaluate without crashing (Python handles big ints)
        result = calculator("2 ** 64")
        assert result == str(2 ** 64)
