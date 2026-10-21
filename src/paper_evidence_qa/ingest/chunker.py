"""固定长度分块器（M5，策略名 ``fixed-v1``）。

纯函数、无 I/O、无模型参与——只负责"把带页码的文本切成 ChunkDraft"。
这是全模块最该被穷举测试的一层：它错了，页码引用会一路错到最终回答。

切分规则（确定性，可复现）：
1. 基准文本是 ``ParsedDocument.full_text``（各页以 ``\\n\\n`` 连接）。
2. 优先在段落边界（``\\n\\n``）切，其次换行符、空格；都找不到则硬切。
3. 相邻 chunk 重叠 ``chunk_overlap`` 个字符；``overlap=0`` 时首尾相接。
4. 空白 chunk 直接丢弃；推进不了就强制前进 1 字符，保证终止。
"""

from __future__ import annotations

from dataclasses import dataclass

from paper_evidence_qa.ingest.errors import IngestError
from paper_evidence_qa.ingest.models import IngestOptions, ParsedDocument
from paper_evidence_qa.schemas.common import ErrorCode


@dataclass(frozen=True)
class ChunkDraft:
    """分块中间产物：文本 + 全文偏移 + 页码区间。由服务层组装成 ``Chunk``。"""

    text: str
    char_start: int
    char_end: int
    page_from: int
    page_to: int


def chunk_document(doc: ParsedDocument, options: IngestOptions) -> list[ChunkDraft]:
    """按 ``fixed-v1`` 策略切分解析结果。空文档抛 ``PARSE_FAILED``。"""
    full = doc.full_text
    if not full.strip():
        raise IngestError(
            ErrorCode.PARSE_FAILED,
            "解析结果无可提取文本（可能为扫描件或空文档）",
            {"page_count": len(doc.pages)},
        )

    page_spans = _page_spans(doc)
    size = options.chunk_size
    overlap = options.chunk_overlap
    total = len(full)

    drafts: list[ChunkDraft] = []
    start = 0
    while start < total:
        # 尾部已完全落在上一块的 overlap 区内时，不再单独成块（避免碎片）
        if drafts and total - start <= overlap:
            break
        end = min(start + size, total)
        if end < total:
            end = _find_cut(full, start, size, overlap)
        text = full[start:end]
        if text.strip():
            page_from = _page_of(page_spans, start)
            page_to = _page_of(page_spans, end - 1)
            drafts.append(
                ChunkDraft(
                    text=text,
                    char_start=start,
                    char_end=end,
                    page_from=page_from,
                    page_to=page_to,
                )
            )
        if end >= total:
            break
        nxt = end - overlap
        start = nxt if nxt > start else start + 1
    return drafts


def _page_spans(doc: ParsedDocument) -> list[tuple[int, int, int, str]]:
    """返回 ``[(page_no, start, end, section_path)]``。页间分隔符 ``\\n\\n`` 不属于任何页。"""
    spans: list[tuple[int, int, int, str]] = []
    cursor = 0
    for page in doc.pages:
        spans.append((page.page_no, cursor, cursor + len(page.text), "\n".join(page.section_path)))
        cursor += len(page.text) + 2
    return spans


def _page_of(spans: list[tuple[int, int, int, str]], offset: int) -> int:
    for page_no, span_start, span_end, _ in spans:
        if span_start <= offset < span_end:
            return page_no
    return spans[-1][0]


def _find_cut(full: str, start: int, size: int, overlap: int) -> int:
    """在 ``[start, start+size)`` 内找切点：段落边界 > 换行 > 空格 > 硬切。

    边界切点必须落在 ``start + overlap`` 之后：否则会围绕同一个页边界
    反复出产出只差几个字符的重复小 chunk。
    """
    end = min(start + size, len(full))
    if end >= len(full):
        return end
    segment = full[start:end]
    for needle in ("\n\n", "\n", " "):
        cut = segment.rfind(needle)
        if cut > overlap:
            return start + cut
    return end
