"""响应缓存（M2）。

缓存解决三个具体问题：

1. **限流**：同一检索式反复执行时不重复打上游（arXiv 间隔 3 秒，重跑一次很贵）。
2. **可复现**：评估时若每次都取到新数据，"结果变化"分不清是代码改了还是上游变了。
   ``from_cache`` 字段让评估可以显式排除缓存命中的样本。
3. **离线重放**：没有网络时仍能用已落盘的响应把整条链路跑通，
   这样"关闭联网"的行为验证不必依赖公网。

只缓存成功响应。错误响应不缓存——否则一次偶发 500 会被固化下来，
后续所有请求都被它污染。
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated

from pydantic import Field

from paper_evidence_qa.schemas import BaseSchema, Sha256, UtcDatetime

_SCHEMA = """
CREATE TABLE IF NOT EXISTS http_cache (
    cache_key    TEXT PRIMARY KEY,
    url          TEXT NOT NULL,
    status_code  INTEGER NOT NULL,
    content_type TEXT NOT NULL,
    body         TEXT NOT NULL,
    fetched_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_http_cache_fetched_at ON http_cache (fetched_at);
"""


class CachedResponse(BaseSchema):
    """一条缓存记录。"""

    cache_key: Sha256
    url: str
    status_code: int
    content_type: str
    body: str
    fetched_at: UtcDatetime

    def age_seconds(self, now: datetime) -> float:
        return (now - self.fetched_at).total_seconds()

    def is_fresh(self, ttl_seconds: int, now: datetime) -> bool:
        """是否在有效期内。``ttl_seconds <= 0`` 表示永不过期。"""
        if ttl_seconds <= 0:
            return True
        return self.age_seconds(now) <= ttl_seconds


class ResponseCache:
    """基于 SQLite 的响应缓存。

    使用 SQLite 而不是内存字典，是因为知识库、检索记录都要落盘，
    多一个存储介质不如复用同一个，也便于人工查看某次检索到底取回了什么。
    """

    def __init__(self, path: str | Path) -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.Lock()

    def get(self, cache_key: str) -> CachedResponse | None:
        with self._lock:
            cursor = self._conn.execute(
                "SELECT cache_key, url, status_code, content_type, body, fetched_at "
                "FROM http_cache WHERE cache_key = ?",
                (cache_key,),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        return CachedResponse(
            cache_key=row[0],
            url=row[1],
            status_code=row[2],
            content_type=row[3],
            body=row[4],
            fetched_at=datetime.fromisoformat(row[5]),
        )

    def put(
        self,
        *,
        cache_key: str,
        url: str,
        status_code: int,
        content_type: str,
        body: str,
        fetched_at: datetime | None = None,
    ) -> None:
        """写入缓存。只应缓存成功响应，判断由调用方负责。"""
        stamp = fetched_at or datetime.now(timezone.utc)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO http_cache "
                "(cache_key, url, status_code, content_type, body, fetched_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cache_key,
                    url,
                    status_code,
                    content_type,
                    body,
                    stamp.isoformat(),
                ),
            )
            self._conn.commit()

    def delete(self, cache_key: str) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM http_cache WHERE cache_key = ?", (cache_key,)
            )
            self._conn.commit()

    def purge_expired(self, ttl_seconds: int, now: datetime | None = None) -> int:
        """清理过期条目，返回删除条数。长驻任务可定期调用。"""
        if ttl_seconds <= 0:
            return 0
        moment = now or datetime.now(timezone.utc)
        threshold = (moment - timedelta(seconds=ttl_seconds)).isoformat()
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM http_cache WHERE fetched_at < ?", (threshold,)
            )
            self._conn.commit()
            return cursor.rowcount

    def count(self) -> int:
        with self._lock:
            cursor = self._conn.execute("SELECT COUNT(1) FROM http_cache")
            row = cursor.fetchone()
        return int(row[0]) if row else 0

    def clear(self) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM http_cache")
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "ResponseCache":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class CachePolicy(BaseSchema):
    """缓存策略。单独成模型，便于在不同任务里换策略而不用改客户端。"""

    ttl_seconds: Annotated[int, Field(ge=0, le=2592000)] = 86400
    enabled: bool = True
    replay_only: bool = False
    """只读缓存、不发起网络请求。用于离线重放与"关闭联网"的行为验证。"""


__all__ = ["CachePolicy", "CachedResponse", "ResponseCache"]
