"""ingest 契约模型（M5）。

边界：本模块只定义解析与入库的数据形状，不做任何 I/O。
``ParsedDocument`` 是解析层的对外产物；``IngestRequest/IngestResult``
是入库服务的输入输出。字段含义冻结在 ``doc/接口与字段.md`` 第 9 节。
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, model_validator

from paper_evidence_qa.schemas.common import (
    BaseSchema,
    ErrorCode,
    Sha256,
    ShortText,
)
from paper_evidence_qa.schemas.domain import Chunk, KBVersion


class ParsedPage(BaseSchema):
    """解析出的一页。``text`` 是该页文本层内容；无章节归属时留空列表。"""

    page_no: int = Field(ge=1)
    text: str
    section_path: list[ShortText] = Field(default_factory=list)


class ParsedDocument(BaseSchema):
    """一篇文献的解析结果。``full_text`` 是分块与偏移定位的基准。"""

    pages: list[ParsedPage] = Field(min_length=1)
    parser_name: ShortText
    parser_version: ShortText
    source_file_sha256: Sha256

    @model_validator(mode="after")
    def _check_pages(self) -> "ParsedDocument":
        for index, page in enumerate(self.pages):
            if page.page_no != index + 1:
                raise ValueError("页码必须从 1 开始且连续递增")
        return self

    @property
    def full_text(self) -> str:
        """各页文本以空行连接；页与页之间消耗 2 个字符（``\\n\\n``）。"""
        return "\n\n".join(page.text for page in self.pages)


class IngestOptions(BaseSchema):
    """分块参数。策略固定 ``fixed-v1``，只暴露尺寸与重叠两个参数位。"""

    chunk_size: int = Field(default=1200, ge=1)
    chunk_overlap: int = Field(default=150, ge=0)

    @model_validator(mode="after")
    def _check_overlap(self) -> "IngestOptions":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        return self


class IngestRequest(BaseSchema):
    """入库请求：本地文件路径列表 + 分块参数。"""

    files: list[Path] = Field(min_length=1)
    options: IngestOptions = Field(default_factory=IngestOptions)


class PaperIngestResult(BaseSchema):
    """单篇文献入库成功的结果摘要。"""

    paper_id: str
    source_path: str
    source_file_sha256: Sha256
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)


class PaperIngestFailure(BaseSchema):
    """单篇文献入库失败记录。失败被隔离，不中断其余文件。"""

    source_path: str
    code: ErrorCode
    message: str
    details: dict = Field(default_factory=dict)


class IngestResult(BaseSchema):
    """一次入库请求的完整产出：成功摘要、失败清单、全部 chunk 与知识库版本。"""

    papers: list[PaperIngestResult] = Field(default_factory=list)
    failures: list[PaperIngestFailure] = Field(default_factory=list)
    chunks: list[Chunk] = Field(default_factory=list)
    kb_version: KBVersion
