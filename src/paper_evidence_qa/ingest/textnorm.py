"""PDF 文本规整（M5）。

出版 PDF 是排版格式：每个视觉行约 60-90 字符，pypdf 提取时在行末
插入 ``\\n``。这些"视觉换行"不是段落结构，若不处理会污染分块切点
（chunk 被提前切短）与语义（句子中间断开）。

规整规则（顺序敏感）：
1. 统一换行符：``\\r\\n`` / ``\\r`` → ``\\n``；
2. 修复行尾断词：``predic-\\ntion`` → ``prediction``（仅限段落内，
   跨段落的 ``-\\n\\n`` 是列表/缩写等真实内容，不合并）；
3. 段落内视觉换行合并为空格；段落边界（``\\n\\n``）保留——
   分块器 ``fixed-v1`` 策略依赖它作为首选切点；
4. 折叠连续空白，去除行首尾空白。

纯函数、零依赖、确定性：同样输入永远得到同样输出。
"""

from __future__ import annotations

import re

_CRLF = re.compile(r"\r\n?")

# 行尾连字符 + 段落内换行 + 行首字母 → 断词，删除连字符与换行
_HYPHEN_BREAK = re.compile(r"(?<=[A-Za-z0-9])-\n(?=[a-z])")

# 两个及以上连续换行视为段落边界（分块器的首选切点）
_PARA_RUN = re.compile(r"\n{2,}")

# 空白（空格/制表符）折叠为单个空格
_SPACE_RUN = re.compile(r"[ \t]+")


def normalize_text(raw: str) -> str:
    """把 pypdf 提取的原始页文本规整为"段落边界 + 行内空格"的规范形态。"""
    if not raw.strip():
        return ""
    text = _CRLF.sub("\n", raw)
    text = _HYPHEN_BREAK.sub("", text)

    paragraphs: list[str] = []
    for para in _PARA_RUN.split(text):
        # 段落内：视觉换行 → 空格；折叠空白；去首尾
        joined = " ".join(line.strip() for line in para.split("\n") if line.strip())
        paragraphs.append(_SPACE_RUN.sub(" ", joined).strip())
    return "\n\n".join(p for p in paragraphs if p)
