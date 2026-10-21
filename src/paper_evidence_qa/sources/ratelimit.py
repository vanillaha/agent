"""限流器（M2）。

课程规范要求"遵守数据源限流规则，禁止高频抓取"。四个数据源的容忍度差别很大：
arXiv 明确要求调用间隔不短于 3 秒，Semantic Scholar 匿名访问极易 429，
OpenAlex 与 Crossref 则靠 polite pool 获得较宽松的配额。

**限流必须在发请求之前生效**，而不是被 429 打回来再退避——后者属于"先把人家
打疼了再道歉"。所以这里的实现是"最小间隔闸门"：同源两次请求之间强制等待。

为了可测试，时钟和睡眠函数都可注入。测试用假时钟断言间隔，
不真的等 3 秒，否则测试会慢到没人愿意跑。
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Sequence

DEFAULT_SLEEP = time.sleep


class RateLimiter:
    """按最小间隔放行的限流器。线程安全。

    语义：``acquire()`` 返回前保证"距离上一次放行已过去 ``min_interval_seconds``"。
    首次调用不等待。
    """

    def __init__(
        self,
        min_interval_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
        max_wait_seconds: float | None = None,
    ) -> None:
        if min_interval_seconds < 0:
            raise ValueError("最小间隔不得为负数")
        self.min_interval_seconds = float(min_interval_seconds)
        self._clock = clock or time.monotonic
        self._sleeper = sleeper or DEFAULT_SLEEP
        self._max_wait = max_wait_seconds
        self._last_release: float | None = None
        self._lock = threading.Lock()
        self._waits: list[float] = []

    def acquire(self) -> float:
        """等待到允许发请求为止，返回本次实际等待秒数。"""
        with self._lock:
            now = self._clock()
            wait = 0.0
            if self._last_release is not None:
                elapsed = now - self._last_release
                wait = max(0.0, self.min_interval_seconds - elapsed)
            if self._max_wait is not None:
                wait = min(wait, self._max_wait)
            if wait > 0:
                self._sleeper(wait)
            self._last_release = self._clock()
            self._waits.append(wait)
            return wait

    @property
    def recorded_waits(self) -> Sequence[float]:
        """已发生的等待序列，用于测试与观测（只读语义）。"""
        with self._lock:
            return tuple(self._waits)

    @property
    def call_count(self) -> int:
        with self._lock:
            return len(self._waits)

    def reset(self) -> None:
        """清空状态。仅用于测试与长驻进程的批次隔离。"""
        with self._lock:
            self._last_release = None
            self._waits.clear()


class NullRateLimiter(RateLimiter):
    """不等待的限流器。用于明确知道无需限流的场合（如纯缓存重放）。"""

    def __init__(self) -> None:
        super().__init__(0.0)


__all__ = ["DEFAULT_SLEEP", "NullRateLimiter", "RateLimiter"]
