"""归一化规则（M2）。

四个数据源对同一件事的写法都不一样：DOI 有的带 URL 前缀有的不带，
arXiv ID 有的带版本号有的不带，标题里混着换行与连字符，开放获取状态
各家口径也不一致。**这一层是纯函数，不含任何网络与业务判断**，
所以它最容易被穷举测试，也最应该被穷举测试。

三条硬规则：

1. **确定性**：同一篇论文无论来自哪个源，都要派生出同一个 ``paper_id``。
   否则检索记录、证据链和评估结果无法跨运行对齐。
2. **保守**：开放获取状态只在"上游明确说开放"且"拿到可下载链接"时才写 ``open``。
   只有标记没有链接时写 ``unknown``——因为拿不到全文就不能当全文证据。
3. **不静默**：任何截断或降级都要能被上层看见（返回状态或告警），不得悄悄改数据。
"""

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Iterable, Sequence

from paper_evidence_qa.ids import deterministic_id
from paper_evidence_qa.schemas import BaseSchema, DocType, OAStatus, ShortText

DOI_RESOLVER_PREFIXES = (
    "https://doi.org/",
    "http://doi.org/",
    "https://dx.doi.org/",
    "http://dx.doi.org/",
    "doi:",
)

ARXIV_URL_MARKERS = ("arxiv.org/abs/", "arxiv.org/pdf/", "arxiv.org/format/")

_ARXIV_NEW_STYLE = re.compile(r"(?P<id>\d{4}\.\d{4,5})(?:v(?P<version>\d+))?")
_ARXIV_OLD_STYLE = re.compile(
    r"(?P<id>[a-z-]+(?:\.[A-Z]{2})?/\d{7})(?:v(?P<version>\d+))?"
)

_WHITESPACE = re.compile(r"\s+")
_MARKUP_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&(#x?[0-9A-Fa-f]+|[A-Za-z]+);")

_NAMED_ENTITIES = {
    "amp": "&",
    "lt": "<",
    "gt": ">",
    "quot": '"',
    "apos": "'",
    "nbsp": " ",
    "ndash": "-",
    "mdash": "-",
}

_TITLE_PUNCTUATION = re.compile(r"^[\s\W_]+|[\s\W_]+$", re.UNICODE)

MIN_YEAR = 1800
MAX_YEAR = 2200


class OAResolution(BaseSchema):
    """开放获取状态的归一化结论。``oa_evidence`` 必须能说明"凭什么这么判"。"""

    oa_status: OAStatus
    oa_url: ShortText | None = None
    oa_evidence: ShortText


def collapse_whitespace(value: str | None) -> str | None:
    """把换行、制表符、连续空格压成单个空格。标题与摘要里这类噪声极多。"""
    if value is None:
        return None
    collapsed = _WHITESPACE.sub(" ", value).strip()
    return collapsed or None


def strip_markup(value: str | None) -> str | None:
    """去掉 XML/HTML 标记并还原常见实体。

    Crossref 的摘要字段是 JATS XML（``<jats:p>...</jats:p>``），
    直接入库会把标记当正文引用，所以必须先剥掉。
    """
    if value is None:
        return None

    def _replace_entity(match: re.Match[str]) -> str:
        token = match.group(1)
        if token.lower().startswith("#x"):
            return _safe_chr(int(token[2:], 16))
        if token.startswith("#"):
            return _safe_chr(int(token[1:]))
        return _NAMED_ENTITIES.get(token.lower(), match.group(0))

    without_tags = _MARKUP_TAG.sub(" ", value)
    decoded = _HTML_ENTITY.sub(_replace_entity, without_tags)
    return collapse_whitespace(decoded)


def _safe_chr(codepoint: int) -> str:
    try:
        return chr(codepoint)
    except (ValueError, OverflowError):  # pragma: no cover - 极端畸形实体
        return ""


def fit_text(value: str | None, limit: int) -> tuple[str | None, bool]:
    """按长度上限裁剪文本，并**返回是否发生了裁剪**。

    契约字段有长度上限（如摘要 2000 字符）。宁可截断并明确告知，
    也不要因为一条超长摘要丢掉整篇文献；但绝不允许悄悄截断。
    """
    if value is None:
        return None, False
    if len(value) <= limit:
        return value, False
    return value[:limit], True


def normalize_doi(raw: str | None) -> str | None:
    """统一 DOI 写法：去解析前缀、去空白、转小写。

    ``https://doi.org/10.1234/ABC`` 与 ``10.1234/abc`` 必须归一到同一个值，
    否则同一篇论文会在库里出现两条记录。
    """
    if raw is None:
        return None
    value = collapse_whitespace(raw)
    if value is None:
        return None
    lowered = value.lower()
    for prefix in DOI_RESOLVER_PREFIXES:
        if lowered.startswith(prefix):
            value = value[len(prefix) :]
            break
    value = value.strip().strip(".")
    return value.lower() or None


def normalize_arxiv_id(raw: str | None) -> str | None:
    """抽出规范化的 arXiv ID（**不含版本号**）。

    可识别 ``arXiv:2401.00001v2``、``2401.00001``、``http://arxiv.org/abs/2401.00001v2``
    以及旧式 ``cs/0701001v1``。
    """
    if raw is None:
        return None
    value = collapse_whitespace(raw)
    if value is None:
        return None
    value = value.replace("arXiv:", " ").replace("arxiv:", " ")
    for pattern in (_ARXIV_NEW_STYLE, _ARXIV_OLD_STYLE):
        match = pattern.search(value)
        if match:
            return match.group("id")
    return None


def arxiv_version(raw: str | None) -> int | None:
    """抽出 arXiv 版本号。版本信息在 M4 判定版本关系时有用，但不进 ``arxiv_id``。"""
    if raw is None:
        return None
    value = collapse_whitespace(raw)
    if value is None:
        return None
    for pattern in (_ARXIV_NEW_STYLE, _ARXIV_OLD_STYLE):
        match = pattern.search(value)
        if match and match.group("version"):
            return int(match.group("version"))
    return None


def extract_arxiv_id_from_urls(urls: Iterable[str | None]) -> str | None:
    """从一组 URL 里找出 arXiv ID。

    OpenAlex 不给 arXiv ID，但它的 location 链接里常常含 ``arxiv.org/abs/...``。
    这是**有据可查的字段提取**，不是猜测。
    """
    for url in urls:
        if not url:
            continue
        lowered = url.lower()
        if any(marker in lowered for marker in ARXIV_URL_MARKERS):
            found = normalize_arxiv_id(url)
            if found:
                return found
    return None


def normalize_title(raw: str | None) -> str | None:
    """标题归一化：压空白、转小写、去掉首尾标点。用于标题级去重与键派生。"""
    cleaned = collapse_whitespace(strip_markup(raw))
    if cleaned is None:
        return None
    stripped = _TITLE_PUNCTUATION.sub("", cleaned).casefold()
    return stripped or None


def parse_year(raw: Any) -> int | None:
    """从多种写法里解析出版年份。

    可处理 ``2024``、``"2024"``、``"2024-05-01"``、``[[2024, 5, 1]]``（Crossref 的
    ``date-parts``）。越界或无法解析一律返回 ``None``，不猜测、不取当年。
    """
    candidate: Any = raw
    if isinstance(candidate, (list, tuple)):
        for item in candidate:
            parsed = parse_year(item)
            if parsed is not None:
                return parsed
        return None
    if isinstance(candidate, dict):
        parts = candidate.get("date-parts")
        if parts is not None:
            return parse_year(parts)
        for key in ("year", "published", "issued"):
            if key in candidate:
                parsed = parse_year(candidate[key])
                if parsed is not None:
                    return parsed
        return None
    if isinstance(candidate, (datetime, date)):
        return _clamp_year(candidate.year)
    if isinstance(candidate, bool):
        return None
    if isinstance(candidate, int):
        return _clamp_year(candidate)
    if isinstance(candidate, float):
        return _clamp_year(int(candidate))
    if isinstance(candidate, str):
        match = re.search(r"(1[89]\d{2}|2[0-2]\d{2})", candidate)
        if match:
            return _clamp_year(int(match.group(1)))
    return None


def _clamp_year(value: int) -> int | None:
    if MIN_YEAR <= value <= MAX_YEAR:
        return value
    return None


def coerce_int(raw: Any) -> int | None:
    """把上游可能给成字符串/浮点的计数值转成整数。不合法一律返回 ``None``。"""
    if raw is None or isinstance(raw, bool):
        return None
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    if isinstance(raw, str):
        match = re.search(r"-?\d+", raw)
        if match:
            return int(match.group(0))
    return None


def dedupe_authors(names: Sequence[str | None]) -> list[str]:
    """按出现顺序去重作者，保持原始顺序。

    大小写不同但实为同一人的（``"Zhang San"`` / ``"zhang san"``）只保留首次出现。
    """
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        cleaned = collapse_whitespace(name)
        if cleaned is None:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def canonical_key(
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    title: str | None = None,
    year: int | None = None,
) -> str:
    """派生跨来源一致的规范键。优先级：DOI > arXiv ID > 标题+年份。

    DOI 是最强的同一性证据；没有 DOI 时 arXiv ID 次之；两者都没有才退到
    标题+年份（弱键，可能把同名综述误合，但比"无标识"好）。
    """
    normalized_doi = normalize_doi(doi)
    if normalized_doi:
        return f"doi:{normalized_doi}"

    normalized_arxiv = normalize_arxiv_id(arxiv_id)
    if normalized_arxiv:
        return f"arxiv:{normalized_arxiv}"

    normalized_title = normalize_title(title)
    if normalized_title:
        return f"title:{normalized_title}|{year if year is not None else 'na'}"

    raise ValueError(
        "文献缺少 DOI、arXiv ID 与标题，无法派生稳定标识；"
        "不接受无标识记录入库"
    )


def derive_paper_id(
    *,
    doi: str | None = None,
    arxiv_id: str | None = None,
    title: str | None = None,
    year: int | None = None,
) -> str:
    """由规范键派生 ``ppr_`` 标识。同一篇论文必然得到同一个标识。"""
    key = canonical_key(doi=doi, arxiv_id=arxiv_id, title=title, year=year)
    return deterministic_id("ppr", key)


def resolve_oa_status(
    *,
    is_open: bool | None,
    oa_url: str | None,
    source: str,
) -> OAResolution:
    """归一化开放获取状态。**这是全项目"能不能下载"的第一道闸门。**

    - 只有"上游明确开放"且"拿到可下载链接"才判定 ``open``。
    - 上游说开放但没给链接 → ``unknown``：没有链接就下不到全文，
      下不到全文就不能作为全文证据，所以不能写 ``open`` 骗自己。
    - 上游说封闭却给了链接（自相矛盾）→ ``unknown``：宁可保守，也不凭猜测下载。
    """
    url = collapse_whitespace(oa_url)
    if url and not url.lower().startswith(("http://", "https://")):
        url = None

    if is_open is True:
        if url:
            return OAResolution(
                oa_status=OAStatus.OPEN,
                oa_url=url,
                oa_evidence=f"{source}.open_access=true, url={url}",
            )
        return OAResolution(
            oa_status=OAStatus.UNKNOWN,
            oa_url=None,
            oa_evidence=(
                f"{source}.open_access=true 但未提供可下载链接，按未核查处理（缺链接不得下载）"
            ),
        )

    if is_open is False:
        if url:
            return OAResolution(
                oa_status=OAStatus.UNKNOWN,
                oa_url=None,
                oa_evidence=(
                    f"{source}.open_access=false 却提供了链接，上游自相矛盾，按未核查处理"
                ),
            )
        return OAResolution(
            oa_status=OAStatus.CLOSED,
            oa_url=None,
            oa_evidence=f"{source}.open_access=false",
        )

    if url:
        return OAResolution(
            oa_status=OAStatus.UNKNOWN,
            oa_url=None,
            oa_evidence=(
                f"{source} 未提供开放获取标记，仅见链接 {url}，保守按未核查处理"
            ),
        )

    return OAResolution(
        oa_status=OAStatus.UNKNOWN,
        oa_url=None,
        oa_evidence=f"{source} 未提供开放获取信息",
    )


_OPENALEX_DOC_TYPES = {
    "article": DocType.JOURNAL_ARTICLE,
    "preprint": DocType.PREPRINT,
    "review": DocType.REVIEW,
    "proceedings-article": DocType.CONFERENCE_PAPER,
    "book-chapter": DocType.OTHER,
    "dissertation": DocType.OTHER,
    "report": DocType.OTHER,
    "paratext": DocType.OTHER,
    "editorial": DocType.OTHER,
    "letter": DocType.OTHER,
    "erratum": DocType.OTHER,
}

_CROSSREF_DOC_TYPES = {
    "journal-article": DocType.JOURNAL_ARTICLE,
    "proceedings-article": DocType.CONFERENCE_PAPER,
    "posted-content": DocType.PREPRINT,
    "book-chapter": DocType.OTHER,
    "report": DocType.OTHER,
    "dissertation": DocType.OTHER,
    "reference-entry": DocType.OTHER,
    "dataset": DocType.OTHER,
}

_S2_DOC_TYPE_PRIORITY = (
    ("Review", DocType.REVIEW),
    ("MetaAnalysis", DocType.REVIEW),
    ("Conference", DocType.CONFERENCE_PAPER),
    ("JournalArticle", DocType.JOURNAL_ARTICLE),
    ("Book", DocType.OTHER),
    ("BookSection", DocType.OTHER),
    ("Dataset", DocType.OTHER),
)


def classify_openalex_type(raw: str | None) -> DocType:
    """映射 OpenAlex 的 ``type`` 字段。未知取值归 ``other``，不猜测。"""
    if not raw:
        return DocType.OTHER
    return _OPENALEX_DOC_TYPES.get(raw.strip().lower(), DocType.OTHER)


def classify_crossref_type(raw: str | None) -> DocType:
    """映射 Crossref 的 ``type`` 字段。"""
    if not raw:
        return DocType.OTHER
    return _CROSSREF_DOC_TYPES.get(raw.strip().lower(), DocType.OTHER)


def classify_semantic_scholar_types(raw: Sequence[str] | None) -> DocType:
    """映射 Semantic Scholar 的 ``publicationTypes`` 数组。

    一篇文献可能同时标 ``Review`` 与 ``JournalArticle``，按"信息量更大者优先"
    的固定顺序取第一个命中项，保证结果可复现。
    """
    if not raw:
        return DocType.OTHER
    present = {str(item).strip() for item in raw if item}
    for label, doc_type in _S2_DOC_TYPE_PRIORITY:
        if label in present:
            return doc_type
    return DocType.OTHER


def doc_types_to_crossref_filter(doc_types: Sequence[DocType]) -> list[str]:
    """把内部 ``DocType`` 反查成 Crossref 的 ``type`` 取值，用于服务端筛选。"""
    mapping = {
        DocType.JOURNAL_ARTICLE: "journal-article",
        DocType.CONFERENCE_PAPER: "proceedings-article",
        DocType.PREPRINT: "posted-content",
        # Crossref 无独立的"综述"类型，综述按期刊论文检索，落到本地后再细分。
        DocType.REVIEW: "journal-article",
    }
    result: list[str] = []
    for doc_type in doc_types:
        value = mapping.get(doc_type)
        if value and value not in result:
            result.append(value)
    return result


def doc_types_to_openalex_filter(doc_types: Sequence[DocType]) -> list[str]:
    """把内部 ``DocType`` 反查成 OpenAlex 的 ``type`` 取值。"""
    mapping = {
        DocType.JOURNAL_ARTICLE: "article",
        DocType.CONFERENCE_PAPER: "proceedings-article",
        DocType.PREPRINT: "preprint",
        DocType.REVIEW: "review",
    }
    result: list[str] = []
    for doc_type in doc_types:
        value = mapping.get(doc_type)
        if value and value not in result:
            result.append(value)
    return result


def coalesce(*values: Any) -> Any:
    """返回第一个非 ``None`` 的值。用于"多个候选字段取第一个可用的"。"""
    for value in values:
        if value is not None:
            return value
    return None


class TextFit(BaseSchema):
    """裁剪结果的显式表达。避免调用方忘了检查"是否被截断"。"""

    text: str | None = None
    truncated: bool = False

    @classmethod
    def of(cls, value: str | None, limit: int) -> "TextFit":
        text, truncated = fit_text(value, limit)
        return cls(text=text, truncated=truncated)


class NormalizedFields(BaseSchema):
    """一次性归一化后的常用字段组合。

    仅在 adapter 内部使用，用于减少参数传递；它不是对外契约。
    """

    doi: str | None = None
    arxiv_id: str | None = None
    arxiv_version: int | None = None
    title: str | None = None
    year: int | None = None
    abstract: str | None = None

    @classmethod
    def build(
        cls,
        *,
        doi: str | None = None,
        arxiv_raw: str | None = None,
        title: str | None = None,
        year: Any = None,
        abstract: str | None = None,
        abstract_limit: int = 2000,
    ) -> "NormalizedFields":
        abstract_text, _ = fit_text(strip_markup(abstract), abstract_limit)
        return cls(
            doi=normalize_doi(doi),
            arxiv_id=normalize_arxiv_id(arxiv_raw),
            arxiv_version=arxiv_version(arxiv_raw),
            title=collapse_whitespace(strip_markup(title)),
            year=parse_year(year),
            abstract=abstract_text,
        )


__all__ = [
    "MIN_YEAR",
    "MAX_YEAR",
    "NormalizedFields",
    "OAResolution",
    "TextFit",
    "arxiv_version",
    "canonical_key",
    "classify_crossref_type",
    "classify_openalex_type",
    "classify_semantic_scholar_types",
    "coalesce",
    "coerce_int",
    "collapse_whitespace",
    "dedupe_authors",
    "derive_paper_id",
    "doc_types_to_crossref_filter",
    "doc_types_to_openalex_filter",
    "extract_arxiv_id_from_urls",
    "fit_text",
    "normalize_arxiv_id",
    "normalize_doi",
    "normalize_title",
    "parse_year",
    "resolve_oa_status",
    "strip_markup",
]
