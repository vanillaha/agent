"""文本规整（textnorm）测试（M5）。

真实出版 PDF 经 pypdf 提取后，每个视觉行末尾都带 \\n，单词粘连、
行尾连字符断词等问题会污染分块切点。规整层必须把"视觉换行"
还原为空格、保留段落边界、修复行尾连词，且不引入新内容。
"""

from __future__ import annotations

from pathlib import Path

import sys as _sys

_sys.path.insert(0, str(Path(__file__).parent.parent / "fixtures" / "pdfs"))

from make_sample_pdf import PAGES, build_pdf  # noqa: E402

from paper_evidence_qa.ingest.pypdf_parser import PypdfParser  # noqa: E402
from paper_evidence_qa.ingest.textnorm import normalize_text  # noqa: E402


class TestNormalizeText:
    def test_visual_line_breaks_become_spaces(self) -> None:
        raw = "Retrieval quality\ndegrades as context\ngrows long."
        assert normalize_text(raw) == "Retrieval quality degrades as context grows long."

    def test_paragraph_breaks_are_preserved(self) -> None:
        raw = "First paragraph ends here.\n\nSecond paragraph begins."
        normalized = normalize_text(raw)
        assert "\n\n" in normalized, "段落边界（空行）必须保留，分块器靠它找切点"
        assert normalized.split("\n\n")[0].strip() == "First paragraph ends here."

    def test_dehyphenation_joins_broken_word(self) -> None:
        raw = "The model uses predic-\ntion heads for drafting."
        assert "prediction heads" in normalize_text(raw)
        assert "-\n" not in normalize_text(raw)

    def test_dehyphenation_not_across_paragraph(self) -> None:
        # 连字符 + 段落边界：不是断词，是列表/缩写等真实内容，不得合并
        raw = "end of section A-\n\nStart of section B"
        assert "A-\n\nStart" in normalize_text(raw)

    def test_crlf_and_cr_normalized(self) -> None:
        assert normalize_text("line one\r\nline two\rline three") == (
            "line one line two line three"
        )

    def test_space_runs_collapsed(self) -> None:
        raw = "too   many\t\tspaces   here"
        assert normalize_text(raw) == "too many spaces here"

    def test_no_single_newline_survives(self) -> None:
        raw = "para one line one\npara one line two\n\npara two line one\npara two line two"
        normalized = normalize_text(raw)
        for segment in normalized.split("\n\n"):
            assert "\n" not in segment, f"段落内不允许残留单换行：{segment!r}"

    def test_empty_and_whitespace_only(self) -> None:
        assert normalize_text("") == ""
        assert normalize_text("   \n  \n\n  ") == ""


class TestPypdfParserAppliesNormalization:
    """端到端：构造"每行一个 Tj 文本操作"的真实 PDF（模拟出版排版），
    解析输出必须已完成视觉换行合并。"""

    def test_extracted_page_has_no_visual_breaks(self, tmp_path: Path) -> None:
        lines = [line for page in PAGES[:1] for line in page]
        pdf = tmp_path / "layout.pdf"
        pdf.write_bytes(build_pdf([lines]))
        pages, _, _ = PypdfParser().parse(pdf)
        text = pages[0][0]
        assert "chunk size affects retrieval quality" in text, (
            "跨视觉行的句子应被合并：受影响的原文是 "
            "'...affects'（行尾）+ 'retrieval quality...'（行首）"
        )
        for segment in text.split("\n\n"):
            assert "\n" not in segment, f"段落内残留单换行：{segment!r}"
