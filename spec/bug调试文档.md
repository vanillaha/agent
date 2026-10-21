# Bug 调试文档

每条记录必须包含：触发场景、根因、修复、回归测试、遗留风险。

当前状态：**6 条记录**（M2 阶段，均已修复并有回归测试）。

---

## BUG-001 `BaseSchema` 未导出，M2 全部模型无法定义

- 发现时间：2026-10-11
- 发现来源：首次导入 `sources.base` 时的 `ImportError`
- 影响范围：M2 全部新增模型（`SourceQuery`、`SourceCandidate`、`SourceFetchResult`、`AdapterSettings`、`SearchOutcome` 等）
- 触发场景：`from paper_evidence_qa.schemas import BaseSchema`
- 期望行为：M2 模型与 M1 模型继承同一个基类，保证 `extra=forbid`、`validate_assignment`、`str_strip_whitespace` 等约束一致
- 实际行为：`ImportError: cannot import name 'BaseSchema' from 'paper_evidence_qa.schemas'`
- 根因：M1 的 `common.py` 定义了 `BaseSchema`，但 `schemas/__init__.py` 的导入与 `__all__` 未包含它。M1 阶段所有模型都在同一包内部使用，没有暴露给外部，所以未被发现
- 修复：在 `schemas/__init__.py` 中补上 `BaseSchema` 的导入与 `__all__` 条目。属**纯增量修改**，不改变任何已有字段或校验规则
- 回归测试：M1 的 76 个契约测试全部重跑通过，证明无破坏性影响；M2 的 134 个用例依赖此导入，全部通过
- 是否产生回归：否。改动只增不减
- 遗留风险：无。但暴露了一个纪律问题——**M1 阶段没有"外部消费者视角"的导入测试**，后续应补充"包级公开接口可导入"的用例

---

## BUG-002 `registry.py` 缺少 `Field` 导入

- 发现时间：2026-10-11
- 发现来源：导入 `sources.registry` 时的 `NameError`
- 影响范围：`SearchOutcome` 模型定义
- 触发场景：加载 `sources/registry.py` 即触发，因为 `Field(default_factory=list)` 在**类定义时**求值
- 期望行为：模块可正常导入
- 实际行为：`NameError: name 'Field' is not defined`
- 根因：清理导入语句时误删了 `from pydantic import Field, model_validator` 中的 `Field`
- 修复：恢复 `from pydantic import Field, model_validator`
- 回归测试：`tests/test_sources_service.py` 全部 24 个用例；该模块导入即覆盖
- 是否产生回归：否
- 遗留风险：无。此类问题会在导入期立即暴露，不会潜伏到运行期

---

## BUG-003 `_raise_for_status` 中用 `and` 串联构造异常，产生语义错误

- 发现时间：2026-10-11
- 发现来源：代码自查（在补测试用例前发现）
- 影响范围：所有上游错误状态的翻译逻辑
- 触发场景：

  ```python
  raise UpstreamError(...) .to_error_info() and _as_rate_limited(...)
  ```

- 期望行为：429 抛 `RateLimitedError`，其余非 2xx 抛 `UpstreamError`
- 实际行为：该表达式本身没有语法错误，但语义完全错乱——`and` 会对第一个操作数的**真值**求值，且 `to_error_info()` 返回的是 `ErrorInfo` 而不是异常对象，最终抛出的是被 `and` 短路后的表达式结果而非设计中的异常类型
- 根因：把"构造对象"与"条件分支"两种意图混写在一行，靠 `and` 短路表达分支
- 修复：改写为显式的 `raise RateLimitedError(...)` / `raise UpstreamError(...)` 两个分支；同时删除因此变得多余的 `_as_rate_limited` / `_as_upstream_error` 两个辅助函数
- 回归测试：`tests/test_sources_infra.py` 中 429 与 5xx 的状态码翻译用例；`tests/test_sources_adapters.py` 中"上游返回 500 时记为 `upstream_unavailable`"用例
- 是否产生回归：否
- 遗留风险：无。但这是一类易复发的问题——**异常构造必须用显式 `raise`**，不得用布尔运算符短路

---

## BUG-004 测试中缓存键用占位符，违反 `Sha256` 类型约束

- 发现时间：2026-10-11
- 发现来源：单元测试失败（5 个用例）
- 影响范围：`tests/test_sources_infra.py`、`tests/test_sources_service.py`
- 触发场景：测试中用 `"k1"` 之类的字符串作 `CachedResponse.cache_key`
- 期望行为：测试构造合法的缓存项
- 实际行为：`ValidationError`——`cache_key` 类型为 `Sha256`，要求 64 位小写十六进制
- 根因：**测试数据不满足被测模型自身的约束**。写测试时为了简短用了占位字符串，而 `extra=forbid` 与强类型约束在这里正确地把它们拒了
- 修复：改用 `sha256_hex(...)` 生成真实摘要常量（`KEY_A`、`KEY_MISSING`、`KEY_OLD`、`KEY_NEW`）；同时把"重开数据库后缓存仍可读"的用例中遗留的 `reopened.get("k1")` 一并改为 `reopened.get(KEY_A)`
- 回归测试：`tests/test_sources_infra.py` 的缓存全部用例；`test_cache_persists_across_reopen`
- 是否产生回归：否
- 遗留风险：无。**这个失败是好事**：说明模型约束真的在生效，而不是形同虚设。反之若测试能塞进非法键，就说明约束没起作用

---

## BUG-005 OpenAlex 摘要重建的测试期望字符串与脱敏样本不一致

- 发现时间：2026-10-11
- 发现来源：单元测试失败
- 影响范围：`test_openalex_normalizes_core_fields`
- 触发场景：断言重建后的摘要等于 `"...no real content."`
- 期望行为：断言与 fixture 一致
- 实际行为：实际值为 `"...no real fixture content."`，断言失败
- 根因：断言是先按预期写的，之后才生成脱敏样本（倒排索引）并改了措辞，没回头同步断言
- 修复：把断言改为与 fixture 实际一致的值
- 回归测试：`tests/test_sources_adapters.py` 的 OpenAlex 摘要重建用例
- 是否产生回归：否
- 遗留风险：**fixture 与断言之间的手工同步很脆弱**。当前做法是"fixture 是事实源，断言跟随 fixture"。后续若样本频繁调整，应考虑让断言直接从 fixture 派生期望值，而不是硬编码字符串

---

## BUG-006 `internal_error` 的异常类名断言位置写错

- 发现时间：2026-10-11
- 发现来源：单元测试失败
- 影响范围：`test_unexpected_adapter_defect_is_labelled_internal_error`
- 触发场景：断言 `"RuntimeError" in failed.error.message`
- 期望行为：验证适配器内部缺陷被如实标记为 `internal_error`
- 实际行为：断言失败——异常类名不在 `message` 里
- 根因：实现中异常类名放在 `error.details["exception"]`，而 `message` 只放可读描述。这是有意的设计：`message` 面向人，`details` 面向程序
- 修复：断言改为 `failed.error.details["exception"] == "RuntimeError"`
- 回归测试：同用例
- 是否产生回归：否
- 遗留风险：无。同时确认了"失败必须可归因"这一契约真的落地了——代码缺陷不会被伪装成上游故障

---

## 记录模板

```markdown
## BUG-<序号> <一句话标题>

- 发现时间：YYYY-MM-DD
- 发现来源：（单元测试 / 评估失败样例 / 人工复核 / 教师反馈）
- 影响范围：
- 触发场景：（可复现的最小输入与步骤）
- 期望行为：
- 实际行为：
- 根因：
- 修复：
- 回归测试：（文件与用例名）
- 是否产生回归：
- 遗留风险：
```

---

## 待观察问题（尚未构成 Bug）

| 观察点 | 说明 | 处理计划 |
| --- | --- | --- |
| 解析层文本层缺失 | 扫描件 PDF 无文本层时，若直接空解析会被误判为"成功但无证据" | M5 任务中明确报错，不产出空 Chunk |
| 预印本与期刊版共存 | 同一论文多版本入库可能导致同一事实被重复计为"多源支持" | M4 任务中先建立版本族，再统计支持度 |
| 无 DOI 时的弱键误合 | `paper_id` 在缺 DOI 与 arXiv ID 时退化为"标题+年份"，同名综述可能被误合 | M4 复核；必要时引入作者列表作为附加键 |
| 上游字段改名 | 四个上游均可能在无通知的情况下改字段名，`ResponseShapeError` 能捕获结构不符，但不能捕获"字段还在、含义变了" | M11 建立"字段覆盖率"监控：定期实弹跑一次，比对关键字段的非空率 |
| 多进程限流叠加 | 限流器按进程计数，多进程并发会叠加出高于礼仪的请求频率 | 生产化时外部化为独立限流组件；当前单进程使用不构成问题 |
| fixture 与断言的同步成本 | BUG-005 暴露了手工同步的脆弱性 | 若样本调整频率上升，改为从 fixture 派生期望值 |
