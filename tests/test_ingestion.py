from agentic_rag.ingestion.chunking import chunk_section
from agentic_rag.ingestion.metadata import coerce_doctype, meta_from_filename
from agentic_rag.ingestion.parsers import parse_html, parse_markdown, parse_research
from agentic_rag.schema import DocType

_MD = """<!-- META {"company":"山河酒业","ticker":"600011","period":"2022","doc_type":"annual_report"} -->
# 山河酒业 2022年年度报告
## 公司简介
公司主营白酒。
## 主要财务数据
| 指标 | 2022年 | 2021年 |
|---|---|---|
| 营业收入 | 138.0 | 120.0 |
"""

_HTML = ('<!-- META {"company":"X","ticker":"600011","period":"2022","doc_type":"announcement"} -->'
         "<html><body><h1>业绩快报</h1><h2>一、财务数据</h2>"
         "<table><tr><td>营业收入</td><td>138.0</td></tr></table></body></html>")

_TXT = ('<!-- META {"company":"Y","doc_type":"research_report"} -->\n'
        "【投资评级】买入\n【核心逻辑】营收增长强劲\n")


def test_parse_markdown_sections_and_meta():
    p = parse_markdown(_MD)
    assert p.meta["ticker"] == "600011"
    titles = [s.title for s in p.sections]
    assert "公司简介" in titles and "主要财务数据" in titles
    fin = next(s for s in p.sections if s.title == "主要财务数据")
    assert "138.0" in fin.body and "|" in fin.body


def test_parse_html_table_text():
    p = parse_html(_HTML)
    assert p.meta["doc_type"] == "announcement"
    assert any("营业收入" in s.body and "138.0" in s.body for s in p.sections)


def test_parse_research_brackets():
    p = parse_research(_TXT)
    titles = [s.title for s in p.sections]
    assert "投资评级" in titles and "核心逻辑" in titles


def test_chunking_keeps_table_atomic():
    table = "| a | b |\n|---|---|\n| 1 | 2 |\n| 3 | 4 |"
    assert len(chunk_section("财务", table, max_chars=10)) == 1


def test_metadata_helpers():
    assert coerce_doctype("annual_report") == DocType.annual_report
    assert coerce_doctype("nonsense") == DocType.other
    m = meta_from_filename("600011_2022_annual_report.md")
    assert m["ticker"] == "600011" and m["period"] == "2022"
