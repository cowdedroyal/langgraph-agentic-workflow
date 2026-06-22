"""Multi-hop QA generators (roadmap P3).

Answers are computed directly from ``facts.json`` (never hallucinated by an LLM) and
``evidence_doc_ids`` are resolved to real corpus chunks, so each sample carries a
verifiable ground truth for retrieval metrics and citation checking.
"""

from __future__ import annotations

from ..ingestion.synthetic import SECTION_FINANCIALS, SECTION_SUPPLY
from ..schema import Document, MultiHopQA
from .samplers import EvidenceChain

METRIC_ZH = {
    "revenue": "营业收入", "net_profit": "归母净利润", "gross_margin": "销售毛利率",
    "rd_expense": "研发费用", "debt_ratio": "资产负债率", "ocf": "经营活动现金流量净额",
}
METRIC_UNIT = {
    "revenue": "亿元", "net_profit": "亿元", "rd_expense": "亿元", "ocf": "亿元",
    "gross_margin": "%", "debt_ratio": "%",
}


def build_evidence_index(docs: list[Document]) -> dict[tuple[str, str, str], str]:
    """(ticker, period, section) -> doc_id, using annual-report chunks (unique per key)."""
    idx: dict[tuple[str, str, str], str] = {}
    for d in docs:
        md = d.metadata
        if md.doc_type.value == "annual_report" and md.ticker and md.period and md.section:
            idx[(md.ticker, md.period, md.section)] = d.doc_id
    return idx


def _name(facts: dict, ticker: str) -> str:
    return facts["companies"].get(ticker, {}).get("name", ticker)


def _fval(facts: dict, ticker: str, year: int, metric: str):
    try:
        return facts["financials"][ticker][str(year)][metric]
    except (KeyError, TypeError):
        return None


def _ev(idx, ticker, year, section):
    return idx.get((ticker, str(year), section))


def _gen_cross_year(c: EvidenceChain, facts, idx) -> MultiHopQA | None:
    t, (y1, y2), m = c.primary, c.years, c.metric
    v1, v2 = _fval(facts, t, y1, m), _fval(facts, t, y2, m)
    e1, e2 = _ev(idx, t, y1, SECTION_FINANCIALS), _ev(idx, t, y2, SECTION_FINANCIALS)
    if None in (v1, v2, e1, e2):
        return None
    name, mz, unit = _name(facts, t), METRIC_ZH[m], METRIC_UNIT[m]
    growth = round((v2 - v1) / v1 * 100, 1) if v1 else 0.0
    word = "增长" if growth >= 0 else "下降"
    return MultiHopQA(
        qa_id=Document.make_id("cross_year", t, m, str(y1), str(y2)),
        question=f"{name}（{t}）{y2}年的{mz}较{y1}年{word}了百分之多少？",
        answer=f"{name}{y2}年{mz}为{v2}{unit}，{y1}年为{v1}{unit}，{word}约{abs(growth)}%。",
        evidence_doc_ids=[e1, e2],
        reasoning_steps=[
            f"查询{name}{y1}年{mz}：{v1}{unit}",
            f"查询{name}{y2}年{mz}：{v2}{unit}",
            f"计算同比变化：({v2}-{v1})/{v1}≈{growth}%",
        ],
        hop_type="cross_year", companies=[name],
        metadata={"metric": m, "years": [y1, y2]},
    )


def _gen_metric_change(c: EvidenceChain, facts, idx) -> MultiHopQA | None:
    t, (y1, y2), m = c.primary, c.years, c.metric
    v1, v2 = _fval(facts, t, y1, m), _fval(facts, t, y2, m)
    e1, e2 = _ev(idx, t, y1, SECTION_FINANCIALS), _ev(idx, t, y2, SECTION_FINANCIALS)
    if None in (v1, v2, e1, e2):
        return None
    name, mz, unit = _name(facts, t), METRIC_ZH[m], METRIC_UNIT[m]
    delta = round(v2 - v1, 2)
    word = "上升" if delta > 0 else ("下降" if delta < 0 else "基本持平")
    change_unit = "个百分点" if unit == "%" else unit
    return MultiHopQA(
        qa_id=Document.make_id("metric_change", t, m, str(y1), str(y2)),
        question=f"{name}的{mz}从{y1}年到{y2}年是上升还是下降？变化了多少？",
        answer=f"{name}{mz}由{y1}年的{v1}{unit}变为{y2}年的{v2}{unit}，{word}约{abs(delta)}{change_unit}。",
        evidence_doc_ids=[e1, e2],
        reasoning_steps=[
            f"查询{name}{y1}年{mz}：{v1}{unit}",
            f"查询{name}{y2}年{mz}：{v2}{unit}",
            f"比较两期：{word}{abs(delta)}{change_unit}",
        ],
        hop_type="metric_change", companies=[name],
        metadata={"metric": m, "years": [y1, y2]},
    )


def _gen_supply_chain(c: EvidenceChain, facts, idx) -> MultiHopQA | None:
    t, sup, year, mat = c.primary, c.secondary, c.years[0], c.material
    v = _fval(facts, sup, year, "net_profit")
    e1, e2 = _ev(idx, t, year, SECTION_SUPPLY), _ev(idx, sup, year, SECTION_FINANCIALS)
    if None in (v, e1, e2):
        return None
    cn, sn = _name(facts, t), _name(facts, sup)
    return MultiHopQA(
        qa_id=Document.make_id("supply_chain", t, sup, str(year)),
        question=f"{cn}的{mat}供应商在{year}年的归母净利润是多少？",
        answer=f"{cn}的{mat}供应商为{sn}（{sup}），其{year}年归母净利润为{v}亿元。",
        evidence_doc_ids=[e1, e2],
        reasoning_steps=[
            f"在{cn}年报『主要供应商与客户』中查到{mat}供应商为{sn}",
            f"查询{sn}{year}年归母净利润：{v}亿元",
        ],
        hop_type="supply_chain", companies=[cn, sn],
        metadata={"material": mat, "year": year},
    )


def _gen_deep_chain(c: EvidenceChain, facts, idx) -> MultiHopQA | None:
    t, sup, grand, mat = c.primary, c.secondary, c.extra.get("grand"), c.material
    e1, e2 = _ev(idx, t, 2022, SECTION_SUPPLY), _ev(idx, sup, 2022, SECTION_SUPPLY)
    if None in (e1, e2) or grand is None:
        return None
    cn, sn, gn = _name(facts, t), _name(facts, sup), _name(facts, grand)
    return MultiHopQA(
        qa_id=Document.make_id("deep_chain", t, sup, grand),
        question=f"{cn}的{mat}供应商，其上游供应商是哪家公司？",
        answer=f"{cn}的{mat}供应商为{sn}（{sup}），而{sn}的上游供应商为{gn}（{grand}）。",
        evidence_doc_ids=[e1, e2],
        reasoning_steps=[
            f"查到{cn}的{mat}供应商为{sn}",
            f"再查{sn}年报，其上游供应商为{gn}",
        ],
        hop_type="deep_chain", companies=[cn, sn, gn],
        metadata={"material": mat},
    )


_DISPATCH = {
    "cross_year": _gen_cross_year,
    "metric_change": _gen_metric_change,
    "supply_chain": _gen_supply_chain,
    "deep_chain": _gen_deep_chain,
}


def generate(chain: EvidenceChain, facts: dict, idx: dict) -> MultiHopQA | None:
    fn = _DISPATCH.get(chain.hop_type)
    return fn(chain, facts, idx) if fn else None
