"""生成脱敏样例 PDF（fixtures/pdfs/sample_paper.pdf）。

不依赖任何第三方库：直接构造最小合法 PDF（3 页文本层，标准字体）。
文本为自造的英文摘要体内容，不含任何真实论文数据。
运行：python fixtures/pdfs/make_sample_pdf.py
"""

from __future__ import annotations

from pathlib import Path

PAGES: list[list[str]] = [
    [
        "Sample Paper: Retrieval Quality in Long-Context RAG",
        "Abstract. This synthetic fixture studies how chunk size affects",
        "retrieval quality when context grows long. We build a controlled",
        "corpus and measure recall at fixed budgets. The corpus contains",
        "twelve topical clusters with gradual vocabulary drift. Each",
        "cluster holds thirty synthetic paragraphs with unique markers.",
        "1. Introduction. Retrieval augmented generation grounds model",
        "output in external evidence. Long documents challenge fixed",
        "budget retrievers because relevant evidence spreads thinly.",
        "Prior work reports degraded recall as document length grows.",
        "We hypothesize that smaller chunks sharpen boundary precision,",
        "while larger chunks preserve local cohesion. Both effects",
        "matter for answer support rate in evidence question answering.",
    ],
    [
        "2. Method. We index every paragraph with a fixed encoder and",
        "query each cluster with paraphrased questions. Retrieval uses",
        "dense cosine similarity over normalized vectors. We sweep",
        "chunk sizes from one hundred to two thousand characters with",
        "a fixed overlap ratio of twelve percent. Metrics include",
        "recall at five, boundary precision, and duplicate rate.",
        "2.1 Corpus construction. Every paragraph embeds six marker",
        "tokens that never appear outside its cluster. Markers allow",
        "exact attribution of retrieved chunks without human labels.",
        "2.2 Controls. Encoding batch order is fixed. Random seeds",
        "are pinned for all encoders. The evaluation harness replays",
        "cached model outputs to keep runs deterministic.",
    ],
    [
        "3. Results. Recall at five peaks near medium chunk sizes and",
        "declines at both extremes. Small chunks fragment evidence and",
        "raise duplicate retrieval. Large chunks dilute similarity and",
        "pull in irrelevant neighbors. Boundary precision improves",
        "monotonically with smaller chunks. Answer support rate follows",
        "recall closely across all settings.",
        "4. Discussion. The tradeoff suggests adaptive strategies that",
        "respect section boundaries. Limitations. Our corpus is synthetic,",
        "so conclusions may not transfer to scanned scientific papers.",
        "5. Conclusion. Chunk size is a first-order factor for long",
        "context retrieval quality and must be reported in evaluations.",
    ],
]


def _escape(text: str) -> str:
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _content_stream(lines: list[str]) -> bytes:
    parts: list[str] = []
    y = 760
    for line in lines:
        parts.append(f"BT /F1 11 Tf 72 {y} Td ({_escape(line)}) Tj ET")
        y -= 18
    return "\n".join(parts).encode("latin-1", errors="replace")


def build_pdf(pages: list[list[str]]) -> bytes:
    """构造最小合法多页 PDF（目录/页树/字体/每页内容流），xref 偏移精确计算。"""
    total_objects = 3 + 2 * len(pages)  # 1 目录, 2 页树, 3 字体, 之后每页 2 个对象
    objects: list[bytes | None] = [None] * total_objects

    objects[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objects[1] = f"<< /Type /Pages /Kids [4 0 R 6 0 R 8 0 R] /Count {len(pages)} >>".encode()
    objects[2] = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

    for index, lines in enumerate(pages):
        page_id = 4 + 2 * index
        content_id = page_id + 1
        stream = _content_stream(lines)
        objects[page_id - 1] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> "
            f"/Contents {content_id} 0 R >>".encode()
        )
        objects[content_id - 1] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream"
        )

    kids = " ".join(f"{4 + 2 * i} 0 R" for i in range(len(pages)))
    objects[1] = f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode()

    out = bytearray(b"%PDF-1.4\n")
    offsets: list[int] = []
    for obj_id, body in enumerate(objects, start=1):
        assert body is not None, f"对象 {obj_id} 未填充"
        offsets.append(len(out))
        out += f"{obj_id} 0 obj\n".encode() + body + b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(objects) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets:
        out += f"{offset:010d} 00000 n \n".encode()
    out += (
        f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
        f"startxref\n{xref_pos}\n%%EOF\n"
    ).encode()
    return bytes(out)


if __name__ == "__main__":
    target = Path(__file__).with_name("sample_paper.pdf")
    target.write_bytes(build_pdf(PAGES))
    print(f"已生成 {target}（{target.stat().st_size} 字节，{len(PAGES)} 页）")
