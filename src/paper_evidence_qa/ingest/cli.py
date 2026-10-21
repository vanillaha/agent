"""入库命令行入口（M5 演示与日常验证用）。

用法：
    python -m paper_evidence_qa.ingest.cli <pdf路径...> [--chunk-size N] [--chunk-overlap N]

输出每篇论文的分块摘要与每个 chunk 的页码定位，便于人工核对
"页码可回溯"这一验收要求。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from paper_evidence_qa.ingest.errors import IngestError
from paper_evidence_qa.ingest.models import IngestOptions, IngestRequest
from paper_evidence_qa.ingest.pypdf_parser import PypdfParser
from paper_evidence_qa.ingest.service import IngestService


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="本地 PDF 入库（解析 + 固定长度分块）")
    parser.add_argument("files", nargs="+", type=Path, help="PDF 文件路径")
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--chunk-overlap", type=int, default=150)
    args = parser.parse_args(argv)

    service = IngestService(parser=PypdfParser())
    request = IngestRequest(
        files=args.files,
        options=IngestOptions(chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap),
    )
    try:
        result = service.process(request)
    except IngestError as error:
        print(f"[失败] {error.code.value}: {error.message}", file=sys.stderr)
        return 1

    kb = result.kb_version
    print(f"知识库版本 {kb.kb_version}  策略={kb.chunk_strategy}  "
          f"chunk_size={kb.chunk_size}  overlap={kb.chunk_overlap}")
    for paper in result.papers:
        print(f"\n[{paper.source_path}]")
        print(f"  paper_id={paper.paper_id}  页数={paper.page_count}  分块数={paper.chunk_count}")
    for failure in result.failures:
        print(f"\n[失败] {failure.source_path}: {failure.code.value} {failure.message}")
    print("\n分块明细（chunk_id | 页码 | 偏移 | 首行文本）：")
    for chunk in result.chunks:
        locator = chunk.locator
        first_line = chunk.text.strip().splitlines()[0][:40] if chunk.text.strip() else ""
        print(f"  {chunk.chunk_id} | p{locator.page_from}-{locator.page_to} "
              f"| {chunk.char_start}-{chunk.char_end} | {first_line}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
