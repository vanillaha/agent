"""本地知识库 SQLite 存储（M5-5）。

职责：把 ``IngestResult`` 的知识库版本、论文、chunk 三类记录持久化，
并支持按版本原样读回——"检索结果绑定知识库版本、可复现"的落点。

设计要点：
- 幂等：``chunk_id`` / ``paper_id`` 确定性派生，重复入库走 upsert，不产生重复行；
- 隔离：chunk 以 ``(kb_version, chunk_index)`` 为主键，不同参数的知识库版本互不覆盖；
- 零依赖：只用标准库 ``sqlite3``；schema 由代码创建，不依赖外部迁移工具。
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from paper_evidence_qa.schemas.domain import Chunk, KBVersion

_SCHEMA = """
CREATE TABLE IF NOT EXISTS kb_versions (
    kb_version     TEXT PRIMARY KEY,
    built_at       TEXT NOT NULL,
    paper_count    INTEGER NOT NULL,
    chunk_count    INTEGER NOT NULL,
    parser_name    TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    chunk_strategy TEXT NOT NULL,
    chunk_size     INTEGER,
    chunk_overlap  INTEGER,
    embedding_model TEXT,
    notes          TEXT
);

CREATE TABLE IF NOT EXISTS papers (
    paper_id           TEXT PRIMARY KEY,
    source_file_sha256 TEXT NOT NULL UNIQUE,
    page_count         INTEGER NOT NULL,
    chunk_count        INTEGER NOT NULL,
    latest_source_path TEXT,
    first_ingested_at  TEXT NOT NULL,
    last_ingested_at   TEXT NOT NULL,
    kb_version         TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    kb_version         TEXT NOT NULL REFERENCES kb_versions(kb_version),
    chunk_index        INTEGER NOT NULL,
    chunk_id           TEXT NOT NULL,
    paper_id           TEXT NOT NULL,
    text               TEXT NOT NULL,
    page_from          INTEGER,
    page_to            INTEGER,
    char_start         INTEGER NOT NULL,
    char_end           INTEGER NOT NULL,
    token_count        INTEGER,
    source_file_sha256 TEXT NOT NULL,
    PRIMARY KEY (kb_version, chunk_index)
);
"""


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class KBStore:
    """本地知识库存储入口。``db_path`` 指向 SQLite 文件（不存在则创建）。"""

    def __init__(self, db_path: Path | str) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    # ---------- 写入 ----------

    def save_ingest(self, result) -> None:  # noqa: ANN001 — IngestResult，避免环导入
        conn = self._conn
        kb = result.kb_version
        now = _utc_now()
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO kb_versions VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (
                    kb.kb_version, kb.built_at.isoformat(), kb.paper_count,
                    kb.chunk_count, kb.parser_name, kb.parser_version,
                    kb.chunk_strategy, kb.chunk_size, kb.chunk_overlap,
                    kb.embedding_model, kb.notes,
                ),
            )
            for paper in result.papers:
                conn.execute(
                    """
                    INSERT INTO papers (paper_id, source_file_sha256, page_count,
                                        chunk_count, latest_source_path,
                                        first_ingested_at, last_ingested_at, kb_version)
                    VALUES (?,?,?,?,?,?,?,?)
                    ON CONFLICT(paper_id) DO UPDATE SET
                        page_count = excluded.page_count,
                        chunk_count = excluded.chunk_count,
                        latest_source_path = excluded.latest_source_path,
                        last_ingested_at = excluded.last_ingested_at,
                        kb_version = excluded.kb_version
                    """,
                    (
                        paper.paper_id, paper.source_file_sha256, paper.page_count,
                        paper.chunk_count, paper.source_path, now, now,
                        kb.kb_version,
                    ),
                )
            for chunk in result.chunks:
                page_from, page_to = self._page_bounds(chunk.locator)
                conn.execute(
                    "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        chunk.kb_version, chunk.chunk_index, chunk.chunk_id,
                        chunk.paper_id, chunk.text,
                        page_from, page_to,
                        chunk.char_start, chunk.char_end, chunk.token_count,
                        chunk.source_file_sha256,
                    ),
                )

    # ---------- 读取 ----------

    def load_kb(self, kb_version: str) -> tuple[KBVersion, list[Chunk]]:
        row = self._conn.execute(
            "SELECT * FROM kb_versions WHERE kb_version = ?", (kb_version,)
        ).fetchone()
        if row is None:
            raise KeyError(f"知识库版本不存在：{kb_version}")
        kb = KBVersion(
            kb_version=row["kb_version"],
            built_at=row["built_at"],
            paper_count=row["paper_count"],
            chunk_count=row["chunk_count"],
            parser_name=row["parser_name"],
            parser_version=row["parser_version"],
            chunk_strategy=row["chunk_strategy"],
            chunk_size=row["chunk_size"],
            chunk_overlap=row["chunk_overlap"],
            embedding_model=row["embedding_model"],
            notes=row["notes"],
        )
        chunks = [
            self._row_to_chunk(r)
            for r in self._conn.execute(
                "SELECT * FROM chunks WHERE kb_version = ? ORDER BY chunk_index",
                (kb_version,),
            )
        ]
        return kb, chunks

    def list_kb_versions(self) -> list[str]:
        return [
            r["kb_version"]
            for r in self._conn.execute(
                "SELECT kb_version FROM kb_versions ORDER BY built_at"
            )
        ]

    def list_papers(self) -> list[dict[str, Any]]:
        return [
            dict(r)
            for r in self._conn.execute("SELECT * FROM papers ORDER BY paper_id")
        ]

    def get_paper(self, source_file_sha256: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM papers WHERE source_file_sha256 = ?", (source_file_sha256,)
        ).fetchone()
        return dict(row) if row else None

    # ---------- 内部 ----------

    @staticmethod
    def _page_bounds(locator: Any) -> tuple[int | None, int | None]:
        """兼容 ``Locator`` 模型与 dict 两种形态，取出页码范围。"""
        if isinstance(locator, dict):
            return locator.get("page_from"), locator.get("page_to")
        return getattr(locator, "page_from", None), getattr(locator, "page_to", None)

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        locator: dict[str, int] = {}
        if row["page_from"] is not None:
            locator["page_from"] = row["page_from"]
        if row["page_to"] is not None:
            locator["page_to"] = row["page_to"]
        return Chunk(
            chunk_id=row["chunk_id"],
            paper_id=row["paper_id"],
            kb_version=row["kb_version"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            locator=locator,
            char_start=row["char_start"],
            char_end=row["char_end"],
            token_count=row["token_count"],
            source_file_sha256=row["source_file_sha256"],
        )
