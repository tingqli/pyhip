"""测试 ``hiptools._parse_kernel_demangle_func_sig``（demangle 串 → 函数名与参数类型列表）。
python -m pytest tests/core/test_hiptools.py -v
"""
from __future__ import annotations

from pyhip.core.hiptools import _parse_kernel_demangle_func_sig


def test_strip_void_prefix_matches_python_func_name():
    """demangle 常带 ``void foo(...)``；pyhip 用 ``getattr(mod, \"foo\")``，必须去掉 ``void ``。"""
    assert _parse_kernel_demangle_func_sig("void bar(int, float)") == ("bar", ["int", "float"])


def test_no_paren_returns_none():
    assert _parse_kernel_demangle_func_sig("plain_symbol") is None


def test_split_first_open_paren_only():
    """只用第一个 ``(`` 分开函数名与参数；避免无上限 split 把模板里括号拆碎。"""
    r = _parse_kernel_demangle_func_sig("void baz(int, float)")
    assert r is not None
    fname, args = r
    assert fname == "baz"
    assert args == ["int", "float"]


def test_empty_arg_list():
    assert _parse_kernel_demangle_func_sig("void qux()") == ("qux", [])


def test_comma_in_types_simple():
    """参数类型里逗号分隔；若类型名含逗号（模板）会误切，属已知限制。"""
    r = _parse_kernel_demangle_func_sig("void k(void const*, int)")
    assert r == ("k", ["void const*", "int"])
