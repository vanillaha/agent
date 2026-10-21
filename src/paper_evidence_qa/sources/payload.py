"""上游负载取值助手。

四个数据源的 JSON 结构差别很大，但"安全地取一个值"这件事是一样的：
字段可能是 ``None``、可能是字符串、可能是数字、也可能整段不见了。
把这些小函数集中在一处，是为了让 adapter 里只剩"这个源的字段叫什么"，
而不是每个 adapter 都塞满 ``isinstance`` 判断。

统一约定：**类型不对一律返回 ``None``，绝不猜、不兜默认值。**
真的需要区分"缺失"与"类型错误"的地方，由调用方显式检查。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def as_mapping(raw: Any) -> Mapping[str, Any]:
    """确保拿到可当字典用的对象，否则返回空字典。"""
    return raw if isinstance(raw, Mapping) else {}


def as_sequence(raw: Any) -> list[Any]:
    """确保拿到列表。字符串不算列表（否则会被逐字符遍历，是个经典 bug）。"""
    if isinstance(raw, (list, tuple)):
        return list(raw)
    return []


def as_str(raw: Any) -> str | None:
    """取字符串。数字会被转成字符串（上游常把 ID 给成数字），其余返回 ``None``。"""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, str):
        return raw or None
    if isinstance(raw, (int, float)):
        return str(raw)
    return None


def as_bool(raw: Any) -> bool | None:
    """只有真正的布尔值才算布尔值。``"true"`` 一律视为未知，不猜。"""
    return raw if isinstance(raw, bool) else None


def first_present(mapping: Mapping[str, Any], *keys: str) -> Any:
    """按顺序返回第一个存在且非 ``None`` 的字段值。"""
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def first_string(mapping: Mapping[str, Any], *keys: str) -> str | None:
    """按顺序返回第一个能取到字符串的字段值。"""
    return as_str(first_present(mapping, *keys))


def first_mapping_in(raw: Any) -> Mapping[str, Any]:
    """从列表里取第一个字典元素。Crossref 的 ``title``/``license`` 都是单元素数组。"""
    for item in as_sequence(raw):
        if isinstance(item, Mapping):
            return item
    return {}


def first_string_in(raw: Any) -> str | None:
    """从列表里取第一个字符串元素。"""
    for item in as_sequence(raw):
        if isinstance(item, str):
            return item or None
    return None


def strings_in(raw: Any) -> list[str]:
    """从列表里取出所有非空字符串。"""
    return [item for item in (as_str(value) for value in as_sequence(raw)) if item]


def nested_str(raw: Mapping[str, Any], *path: str) -> str | None:
    """按路径取嵌套字段的字符串值，例如 ``nested_str(item, "primary", "name")``。"""
    current: Any = raw
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return as_str(current)


def join_names(parts: Sequence[str | None], separator: str = " ") -> str | None:
    """把姓与名拼成可读的人名，跳过空片段。"""
    cleaned = [part.strip() for part in parts if part and part.strip()]
    if not cleaned:
        return None
    return separator.join(cleaned)


__all__ = [
    "as_bool",
    "as_mapping",
    "as_sequence",
    "as_str",
    "first_mapping_in",
    "first_present",
    "first_string",
    "first_string_in",
    "join_names",
    "nested_str",
    "strings_in",
]
