"""标识生成（M2 起共用）。

本项目有两类标识，用途不同，不得混用：

- **随机标识**：事件、检索记录等只要求在库内唯一即可的对象。用 ``random_id``。
- **确定性标识**：文献等要求"跨运行一致"的对象。同一篇论文无论从哪个来源、
  哪一轮检索得到，都必须派生出同一个 ``ppr_`` 标识；否则检索记录、证据链与
  评估结果无法跨运行对齐。用 ``deterministic_id``。

约定：标识一律形如 ``<前缀>_<摘要片段>``，与 M1 的前缀规则一致，不暴露自增主键。
"""

from __future__ import annotations

import hashlib
import uuid

RANDOM_DIGEST_LENGTH = 16
"""随机标识的摘要长度。16 位十六进制足以避免库内碰撞。"""

DETERMINISTIC_DIGEST_LENGTH = 16
"""确定性标识的摘要长度。截短只为可读，碰撞概率在文献量级下可忽略。"""

_PARTS_SEPARATOR = "\x1f"
"""组成部件的分隔符。用不可见控制符，避免"a|b"与"a/b"被算成同一个键。"""


def random_id(prefix: str) -> str:
    """生成一次性随机标识，例如 ``rcd_9f2c1d4e6a8b0c3f``。"""
    if not prefix:
        raise ValueError("标识前缀不得为空")
    return f"{prefix}_{uuid.uuid4().hex[:RANDOM_DIGEST_LENGTH]}"


def deterministic_id(prefix: str, *parts: str) -> str:
    """由若干字符串部件派生稳定标识。相同部件必然得到相同结果。

    部件按顺序参与摘要，且不允许为空——空部件会让不同对象退化成同一个键。
    """
    if not prefix:
        raise ValueError("标识前缀不得为空")
    if not parts:
        raise ValueError("确定性标识至少需要一个组成部件")
    cleaned: list[str] = []
    for part in parts:
        value = part.strip()
        if not value:
            raise ValueError("确定性标识的组成部件不得为空")
        cleaned.append(value)
    payload = _PARTS_SEPARATOR.join(cleaned).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:DETERMINISTIC_DIGEST_LENGTH]
    return f"{prefix}_{digest}"


def sha256_hex(text: str) -> str:
    """返回 64 位十六进制摘要。用于缓存键等需要完整摘要的场合。"""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
