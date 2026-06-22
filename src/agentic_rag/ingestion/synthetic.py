"""Synthetic Chinese-financial corpus generator (roadmap P1).

Produces a small, fully reproducible corpus of *fictional* listed companies so the
whole pipeline (ingestion -> hybrid retrieval -> multi-hop QA synthesis -> eval ->
SFT/GRPO) runs end-to-end with no copyright or network dependency.

Design goals that make downstream multi-hop QA *answerable with a ground truth*:
  * deterministic financials with clean year-over-year growth (cross-year hops),
  * an explicit supply chain so "X's supplier's net profit" is traceable (supply-chain hops),
  * distinct margin trends per company (metric-change hops).

Section titles below are stable constants and are reused by the synthesis layer (P3)
to locate evidence chunks by (company, period, section).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ..config import Settings, get_settings
from ..log import get_logger

logger = get_logger(__name__)

# -- stable section titles (referenced by synthesis P3) --
SECTION_PROFILE = "公司简介"
SECTION_OPERATION = "经营情况讨论与分析"
SECTION_FINANCIALS = "主要财务数据"
SECTION_SUPPLY = "主要供应商与客户"
SECTION_RISK = "风险因素"

REPORT_YEARS = [2021, 2022, 2023]
_ALL_YEARS = [2020, 2021, 2022, 2023]  # 2020 only used as the prior-year comparison


@dataclass
class CompanyProfile:
    ticker: str
    name: str
    sector: str
    chain: str            # supply-chain group label
    role: str             # upstream | midstream | downstream
    base_rev: float       # 2021 revenue, 亿元
    rev_growth: float     # annual revenue growth
    base_gm: float        # 2021 gross margin, %
    gm_delta: float       # gross-margin change per year, pct points
    net_margin: float     # net profit / revenue
    rd_ratio: float       # R&D expense / revenue
    base_debt: float      # 2021 debt ratio, %
    debt_delta: float
    ocf_ratio: float      # operating cash flow / net profit
    intro: str


# Two three-tier supply chains -> 6 fictional companies.
COMPANIES: list[CompanyProfile] = [
    # --- 白酒 / 消费链 ---
    CompanyProfile("600011", "山河酒业", "白酒", "白酒链", "downstream",
                   base_rev=120.0, rev_growth=0.15, base_gm=78.0, gm_delta=0.5,
                   net_margin=0.36, rd_ratio=0.008, base_debt=22.0, debt_delta=-1.0, ocf_ratio=1.05,
                   intro="公司是一家以高端酱香型白酒研发、生产与销售为主的消费品企业，旗下核心品牌定位中高端市场。"),
    CompanyProfile("600012", "黔源包装", "包装材料", "白酒链", "upstream",
                   base_rev=18.0, rev_growth=0.10, base_gm=28.0, gm_delta=-1.0,
                   net_margin=0.09, rd_ratio=0.030, base_debt=45.0, debt_delta=1.5, ocf_ratio=0.85,
                   intro="公司主营高端玻璃瓶及酒类包装材料，是区域内主要的白酒包装供应商。"),
    CompanyProfile("600013", "丰收农业", "农产品", "白酒链", "upstream",
                   base_rev=25.0, rev_growth=0.06, base_gm=18.0, gm_delta=0.8,
                   net_margin=0.06, rd_ratio=0.015, base_debt=38.0, debt_delta=0.5, ocf_ratio=0.9,
                   intro="公司从事优质高粱等酿酒原粮的种植、收储与供应，服务于白酒产业链上游。"),
    # --- 新能源链 ---
    CompanyProfile("300021", "长江动力", "动力电池", "新能源链", "downstream",
                   base_rev=200.0, rev_growth=0.40, base_gm=20.0, gm_delta=-1.5,
                   net_margin=0.08, rd_ratio=0.060, base_debt=55.0, debt_delta=2.0, ocf_ratio=0.7,
                   intro="公司是国内领先的动力电池电芯制造商，产品广泛应用于新能源汽车及储能领域。"),
    CompanyProfile("300022", "极锋材料", "正极材料", "新能源链", "midstream",
                   base_rev=60.0, rev_growth=0.55, base_gm=25.0, gm_delta=-2.0,
                   net_margin=0.11, rd_ratio=0.045, base_debt=50.0, debt_delta=1.0, ocf_ratio=0.6,
                   intro="公司专注于锂电池正极材料的研发与制造，是动力电池厂商的核心材料供应商。"),
    CompanyProfile("300023", "云岭矿业", "锂矿", "新能源链", "upstream",
                   base_rev=40.0, rev_growth=0.70, base_gm=45.0, gm_delta=3.0,
                   net_margin=0.30, rd_ratio=0.020, base_debt=33.0, debt_delta=-2.0, ocf_ratio=0.95,
                   intro="公司主营锂矿开采与锂盐加工，资源储量丰富，受益于新能源材料需求增长。"),
]

# supplier -> customer relationships (enables supply-chain multi-hop questions)
SUPPLY_LINKS: list[dict[str, str]] = [
    {"supplier": "600013", "customer": "600011", "material": "酿酒高粱"},
    {"supplier": "600012", "customer": "600011", "material": "玻璃瓶及包装"},
    {"supplier": "300023", "customer": "300022", "material": "锂矿石/锂盐"},
    {"supplier": "300022", "customer": "300021", "material": "正极材料"},
]

_BROKERS = ["中信建源", "国泰君平", "华兴证券"]


def by_ticker() -> dict[str, CompanyProfile]:
    return {p.ticker: p for p in COMPANIES}


def suppliers_of(ticker: str) -> list[tuple[str, str]]:
    """Return (supplier_ticker, material) feeding the given company."""
    return [(l["supplier"], l["material"]) for l in SUPPLY_LINKS if l["customer"] == ticker]


def customers_of(ticker: str) -> list[str]:
    return [l["customer"] for l in SUPPLY_LINKS if l["supplier"] == ticker]


def compute_financials(p: CompanyProfile) -> dict[int, dict]:
    """Deterministic per-year financials (亿元 / %)."""
    out: dict[int, dict] = {}
    for y in _ALL_YEARS:
        n = y - 2021
        rev = round(p.base_rev * (1 + p.rev_growth) ** n, 2)
        gm = round(p.base_gm + p.gm_delta * n, 2)
        net = round(rev * p.net_margin, 2)
        rd = round(rev * p.rd_ratio, 2)
        debt = round(p.base_debt + p.debt_delta * n, 2)
        ocf = round(net * p.ocf_ratio, 2)
        out[y] = dict(revenue=rev, net_profit=net, gross_margin=gm,
                      rd_expense=rd, debt_ratio=debt, ocf=ocf)
    return out


def yoy(cur: float, prev: float) -> float:
    return round((cur - prev) / prev * 100, 1) if prev else 0.0


def _meta_line(meta: dict) -> str:
    return f"<!-- META {json.dumps(meta, ensure_ascii=False)} -->"


# --------------------------------------------------------------------------- #
# Renderers
# --------------------------------------------------------------------------- #
def render_annual_report(p: CompanyProfile, fins: dict[int, dict], year: int) -> str:
    f, pf = fins[year], fins[year - 1]
    bt = by_ticker()
    sup = "；".join(f"{bt[s].name}（{s}，供应{m}）" for s, m in suppliers_of(p.ticker)) or "无重大依赖的单一供应商"
    cus = "；".join(bt[c].name for c in customers_of(p.ticker)) or "下游为经销商及终端客户"

    rows = [
        ("营业收入（亿元）", f["revenue"], pf["revenue"], f"{yoy(f['revenue'], pf['revenue'])}%"),
        ("归母净利润（亿元）", f["net_profit"], pf["net_profit"], f"{yoy(f['net_profit'], pf['net_profit'])}%"),
        ("销售毛利率（%）", f["gross_margin"], pf["gross_margin"], f"{round(f['gross_margin'] - pf['gross_margin'], 2)}pct"),
        ("研发费用（亿元）", f["rd_expense"], pf["rd_expense"], f"{yoy(f['rd_expense'], pf['rd_expense'])}%"),
        ("资产负债率（%）", f["debt_ratio"], pf["debt_ratio"], f"{round(f['debt_ratio'] - pf['debt_ratio'], 2)}pct"),
        ("经营活动现金流量净额（亿元）", f["ocf"], pf["ocf"], f"{yoy(f['ocf'], pf['ocf'])}%"),
    ]
    table = ["| 指标 | {}年 | {}年 | 同比变动 |".format(year, year - 1), "|---|---|---|---|"]
    table += [f"| {n} | {c} | {p2} | {d} |" for n, c, p2, d in rows]

    meta = dict(company=p.name, ticker=p.ticker, period=str(year), doc_type="annual_report",
                publish_date=f"{year + 1}-03-28", source=f"{p.ticker}_{year}_annual_report.md")
    return "\n".join([
        _meta_line(meta),
        f"# {p.name}（{p.ticker}）{year}年年度报告（摘要）",
        "",
        f"## {SECTION_PROFILE}",
        f"{p.intro}公司所属行业为{p.sector}，处于{p.chain}的{ {'upstream':'上游','midstream':'中游','downstream':'下游'}[p.role] }环节。",
        "",
        f"## {SECTION_OPERATION}",
        f"报告期内，公司实现营业收入{f['revenue']}亿元，同比{'增长' if f['revenue'] >= pf['revenue'] else '下降'}"
        f"{abs(yoy(f['revenue'], pf['revenue']))}%；实现归属于母公司股东的净利润{f['net_profit']}亿元。"
        f"公司销售毛利率为{f['gross_margin']}%，研发费用{f['rd_expense']}亿元。",
        "",
        f"## {SECTION_FINANCIALS}",
        *table,
        "",
        f"## {SECTION_SUPPLY}",
        f"报告期内，公司主要供应商包括：{sup}。主要客户/下游包括：{cus}。",
        "",
        f"## {SECTION_RISK}",
        f"公司面临的主要风险包括{p.sector}行业景气波动、原材料价格波动以及市场竞争加剧等风险。",
        "",
    ])


def render_announcement(p: CompanyProfile, fins: dict[int, dict], year: int) -> str:
    f, pf = fins[year], fins[year - 1]
    meta = dict(company=p.name, ticker=p.ticker, period=str(year), doc_type="announcement",
                publish_date=f"{year + 1}-01-20", source=f"{p.ticker}_{year}_announcement.html")
    return "\n".join([
        _meta_line(meta),
        "<html><head><meta charset=\"utf-8\">",
        f"<title>{p.name}{year}年度业绩快报</title></head><body>",
        f"<h1>{p.name}（{p.ticker}）{year}年度业绩快报公告</h1>",
        f"<p>公告编号：{p.ticker}-{year + 1}-001</p>",
        "<h2>一、本报告期主要财务数据</h2>",
        "<table border=\"1\">",
        "<tr><th>项目</th><th>本年度</th><th>上年同期</th><th>同比</th></tr>",
        f"<tr><td>营业收入（亿元）</td><td>{f['revenue']}</td><td>{pf['revenue']}</td><td>{yoy(f['revenue'], pf['revenue'])}%</td></tr>",
        f"<tr><td>归母净利润（亿元）</td><td>{f['net_profit']}</td><td>{pf['net_profit']}</td><td>{yoy(f['net_profit'], pf['net_profit'])}%</td></tr>",
        f"<tr><td>销售毛利率（%）</td><td>{f['gross_margin']}</td><td>{pf['gross_margin']}</td><td>{round(f['gross_margin'] - pf['gross_margin'], 2)}pct</td></tr>",
        "</table>",
        "<h2>二、经营业绩说明</h2>",
        f"<p>报告期内公司{p.sector}业务保持稳健，营业收入达到{f['revenue']}亿元，"
        f"归母净利润{f['net_profit']}亿元。</p>",
        "</body></html>",
        "",
    ])


def render_research_report(p: CompanyProfile, fins: dict[int, dict], year: int) -> str:
    f = fins[year]
    bt = by_ticker()
    broker = _BROKERS[(int(p.ticker) + year) % len(_BROKERS)]
    sup = "、".join(f"{bt[s].name}（{m}）" for s, m in suppliers_of(p.ticker)) or "无"
    cus = "、".join(bt[c].name for c in customers_of(p.ticker)) or "下游经销商"
    target = round(f["revenue"] * (0.6 if p.chain == "白酒链" else 0.3), 1)
    meta = dict(company=p.name, ticker=p.ticker, period=str(year), doc_type="research_report",
                publish_date=f"{year + 1}-04-10", source=f"{p.ticker}_{year}_research_report.txt")
    return "\n".join([
        _meta_line(meta),
        f"【{broker} 行业研究】{p.name}（{p.ticker}）{year}年报点评",
        f"【投资评级】{'买入' if p.rev_growth > 0.2 else '增持'}（维持）",
        f"【目标价】{target}元",
        f"【核心逻辑】{p.name}{year}年实现营业收入{f['revenue']}亿元、归母净利润{f['net_profit']}亿元，"
        f"毛利率{f['gross_margin']}%。公司受益于{p.chain}景气度，主营{p.sector}。",
        f"【上下游】上游主要供应商：{sup}；下游主要客户：{cus}。",
        f"【风险提示】{p.sector}需求不及预期、原材料价格大幅波动、行业竞争加剧。",
        "",
    ])


# --------------------------------------------------------------------------- #
# Build / load
# --------------------------------------------------------------------------- #
def facts_path(settings: Settings) -> Path:
    return settings.data_dir / "facts.json"


def build_corpus(settings: Settings | None = None) -> dict:
    """Render raw files into data/raw/ and write the structured facts.json. Returns a summary."""
    settings = (settings or get_settings()).ensure_dirs()
    raw = settings.raw_dir
    raw.mkdir(parents=True, exist_ok=True)

    facts: dict = {"companies": {}, "supply_links": SUPPLY_LINKS, "financials": {}}
    n_files = 0
    for p in COMPANIES:
        fins = compute_financials(p)
        facts["companies"][p.ticker] = {
            "ticker": p.ticker, "name": p.name, "sector": p.sector,
            "chain": p.chain, "role": p.role,
        }
        facts["financials"][p.ticker] = {str(y): fins[y] for y in REPORT_YEARS}
        for year in REPORT_YEARS:
            (raw / f"{p.ticker}_{year}_annual_report.md").write_text(
                render_annual_report(p, fins, year), encoding="utf-8")
            (raw / f"{p.ticker}_{year}_announcement.html").write_text(
                render_announcement(p, fins, year), encoding="utf-8")
            (raw / f"{p.ticker}_{year}_research_report.txt").write_text(
                render_research_report(p, fins, year), encoding="utf-8")
            n_files += 3

    fp = facts_path(settings)
    fp.write_text(json.dumps(facts, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Synthetic corpus: %d companies x %d years -> %d raw files; facts -> %s",
                len(COMPANIES), len(REPORT_YEARS), n_files, fp)
    return {"companies": len(COMPANIES), "years": len(REPORT_YEARS), "raw_files": n_files, "facts_path": str(fp)}


def load_facts(settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    return json.loads(facts_path(settings).read_text(encoding="utf-8"))
