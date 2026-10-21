# 科研文献搜集与证据问答 Agent（S01）

课程课题编号 07，AI for Science 类，形态：Agent。

## 一句话说明

系统维护一个本地科研文献知识库，同时可以按需联网搜集合法开放论文。面对一个研究问题时，Agent 先看本地库够不够用，不够就联网补充，最后给出**每条事实都能追溯到原文页码/章节**的回答；证据不足时明确拒答，而不是编一个答案。

## 两条独立路径

| 路径 | 说明 |
| --- | --- |
| 本地文献库检索 | 管理已入库 PDF，解析、分块、索引、召回、重排，产出可定位的全文证据 |
| 联网文献搜集 | 检索论文元数据、摘要、引用关系与开放全文，去重、识别版本、下载入库 |

两条路径的证据**不得混写**：回答中的每条事实必须标明来源类型（全文证据 / 摘要信息 / 模型推断）。

## 项目位置

- 项目根目录：`E:\agent_project`（PyCharm 工程，`.idea/` 已就位）。
- 解释器：`E:\Develop\Anaconda\envs\python_3.13\python.exe`（Python 3.13.16）。

### PyCharm 首次打开要做两件事

1. **配置解释器**：`File → Settings → Project: agent_project → Python Interpreter → Add Interpreter → Conda Environment → Existing environment`，选择 `E:\Develop\Anaconda\envs\python_3.13\python.exe`。
2. **标记源码根目录**：右键 `src` 文件夹 → `Mark Directory as → Sources Root`（变蓝）。这样 `import paper_evidence_qa` 才能被识别，右键运行测试也不会报导入错误。

## 目录结构

```text
.
├── src/paper_evidence_qa/      业务代码
│   ├── ids.py                  标识生成（M2 起共用）
│   ├── schemas/                统一数据模型与接口契约（M1）
│   └── sources/                多源在线检索（M2）
│       ├── base.py             适配器模板方法与 M2 对外模型
│       ├── errors.py           适配器异常 → ErrorCode 映射
│       ├── normalize.py        上游字段归一化纯函数
│       ├── payload.py          上游 JSON 取值辅助
│       ├── query.py            检索式构造与结构化筛选
│       ├── http.py             HTTP 客户端（协议 + httpx 实现 + 缓存包装）
│       ├── cache.py            SQLite 响应缓存
│       ├── ratelimit.py        按源最小请求间隔
│       ├── openalex.py         OpenAlex 适配器
│       ├── crossref.py         Crossref 适配器
│       ├── arxiv.py            arXiv 适配器
│       ├── semantic_scholar.py Semantic Scholar 适配器
│       └── registry.py         SourceRegistry / SearchService / SearchOutcome
├── tests/                      单元测试与契约测试
│   └── fakes.py                可注入的假 HTTP 客户端（测试零网络）
├── evals/                      评估脚本与评估记录（当前仅规划，不放数据）
├── fixtures/                   固定脱敏测试数据
│   └── sources/                四个上游的脱敏响应样本
├── spec/                       任务规格与状态
│   ├── 当前开发任务.md
│   ├── 未来规划.md
│   ├── 已完成任务.md
│   └── bug调试文档.md
├── doc/                        长期说明
│   ├── 架构设计.md
│   ├── 接口与字段.md
│   └── 评估设计.md
├── AGENTS.md                   项目规则（协作与开发纪律）
├── pyproject.toml              依赖与工具配置
└── README.md
```

`spec/` 回答"当前做什么、什么算完成"；`doc/` 回答"系统如何设计、如何接入和使用"。

## 环境要求

- `requires-python = ">=3.11,<3.14"`（见 `pyproject.toml`）。课程规范原写 3.11，本课题实际使用 3.13，属于该范围之内。
- 依赖锁定见 `pyproject.toml`。
- 使用独立虚拟环境，不在 Anaconda base 环境上直接升级 Python。

### 本项目实际使用的环境（已验证）

| 项 | 值 |
| --- | --- |
| 解释器 | `E:\Develop\Anaconda\envs\python_3.13\python.exe` |
| Python | 3.13.16 |
| pydantic | 2.14.0（pydantic-core 2.50.0） |
| pytest | 8.4.2 |
| httpx | 0.28.1 |

该环境已跑通全部 210 个用例（M1 76 例 + M2 134 例，`210 passed`）。换环境后必须重跑，详见 `spec/已完成任务.md` 的验证记录。

M2 的全部测试**不依赖公网**：HTTP 客户端通过注入假实现替换（`tests/fakes.py`），响应使用 `fixtures/sources/` 的脱敏样本。因此断网也能跑全量测试。

## 运行测试

```bash
# 用本项目环境的解释器（Windows）
"E:\Develop\Anaconda\envs\python_3.13\python.exe" -m pytest

# 或先激活环境
conda activate python_3.13
python -m pytest
```

首次在新环境安装依赖：

```bash
python -m pip install -e ".[dev]"
```

## 当前进度

- 已完成：M0 项目规范与文档体系、M1 统一数据模型与接口契约、M2 多源在线检索适配器。
- 进行中：无。
- 下一步见 `spec/未来规划.md`（下一项 M5 解析与分块索引，但与 M3 下载层之间需先做一次取舍确认，见 `spec/当前开发任务.md` 第 3 节）。
