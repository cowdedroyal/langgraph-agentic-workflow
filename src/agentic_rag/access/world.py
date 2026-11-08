"""Scalable synthetic worlds: the 6 core companies plus N distractor companies.

The question set is always generated from the core companies, so it stays fixed while
the corpus grows. Distractors are built to be *hard*, not just numerous:

  * they share sectors and supply-chain vocabulary with the core companies,
  * some of them buy from the same suppliers as core companies (so "X's supplier" is
    ambiguous for a retriever that ignores X),
  * their financial tables look the same, only the numbers differ.

Everything is seeded, so a world of a given size is byte-for-byte reproducible.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..ingestion.synthetic import (
    COMPANIES,
    REPORT_YEARS,
    SUPPLY_LINKS,
    CompanyProfile,
    World,
    compute_financials,
    render_annual_report,
    render_announcement,
    render_research_report,
)

_PREFIX = list("东南西北中华金银宏远长盛恒泰安康鼎鑫瑞丰嘉润鸿达新天海川明光永兴")
_SUFFIX = {
    "白酒": ["酒业", "酒厂", "酿酒"], "包装材料": ["包装", "印务", "玻璃"],
    "农产品": ["农业", "粮业", "农发"], "动力电池": ["动力", "电池", "储能"],
    "正极材料": ["材料", "新材", "锂电"], "锂矿": ["矿业", "锂业", "资源"],
}
_INTRO = {
    "白酒": "公司从事白酒的研发、生产与销售，产品覆盖中高端及大众价位段。",
    "包装材料": "公司主营酒类及食品包装材料，为下游品牌提供瓶体与外包装。",
    "农产品": "公司从事酿酒原粮的种植、收储与供应。",
    "动力电池": "公司从事动力电池电芯及模组的研发制造，产品用于新能源汽车与储能。",
    "正极材料": "公司专注锂电池正极材料的研发与制造。",
    "锂矿": "公司主营锂矿开采与锂盐加工。",
}
_CHAIN = {"白酒": "白酒链", "包装材料": "白酒链", "农产品": "白酒链",
          "动力电池": "新能源链", "正极材料": "新能源链", "锂矿": "新能源链"}
_ROLE = {"白酒": "downstream", "包装材料": "upstream", "农产品": "upstream",
         "动力电池": "downstream", "正极材料": "midstream", "锂矿": "upstream"}
_MATERIAL = {"包装材料": "玻璃瓶及包装", "农产品": "酿酒高粱",
             "正极材料": "正极材料", "锂矿": "锂矿石/锂盐"}
_BUYER_OF = {"包装材料": "白酒", "农产品": "白酒", "正极材料": "动力电池", "锂矿": "正极材料"}


@dataclass
class ScaledWorld:
    world: World
    n_distractors: int
    seed: int

    @property
    def core_tickers(self) -> list[str]:
        return [p.ticker for p in COMPANIES]


def _unique_names(rng: random.Random, sector: str, taken: set[str], n: int) -> list[str]:
    out = []
    while len(out) < n:
        name = rng.choice(_PREFIX) + rng.choice(_PREFIX) + rng.choice(_SUFFIX[sector])
        if name not in taken:
            taken.add(name)
            out.append(name)
    return out


def build_world(n_distractors: int, seed: int = 7) -> ScaledWorld:
    """Core companies + ``n_distractors`` look-alike companies with their own supply links."""
    rng = random.Random(seed)
    companies = list(COMPANIES)
    links = list(SUPPLY_LINKS)
    taken = {p.name for p in companies}
    sectors = list(_SUFFIX)
    new: list[CompanyProfile] = []
    for i in range(n_distractors):
        sector = sectors[i % len(sectors)]
        name = _unique_names(rng, sector, taken, 1)[0]
        board = "6" if _CHAIN[sector] == "白酒链" else "3"
        ticker = f"{board}{(10000 + i):05d}"
        p = CompanyProfile(
            ticker, name, sector, _CHAIN[sector], _ROLE[sector],
            base_rev=round(rng.uniform(10, 220), 1), rev_growth=round(rng.uniform(-0.05, 0.6), 2),
            base_gm=round(rng.uniform(15, 80), 1), gm_delta=round(rng.uniform(-2, 2), 1),
            net_margin=round(rng.uniform(0.04, 0.35), 2), rd_ratio=round(rng.uniform(0.005, 0.06), 3),
            base_debt=round(rng.uniform(20, 60), 1), debt_delta=round(rng.uniform(-2, 2), 1),
            ocf_ratio=round(rng.uniform(0.6, 1.1), 2), intro=_INTRO[sector],
        )
        new.append(p)
    companies += new
    # supply links among distractors, and distractors that buy from *core* suppliers
    by_sector: dict[str, list[CompanyProfile]] = {}
    for p in companies:
        by_sector.setdefault(p.sector, []).append(p)
    for p in new:
        if p.sector in _BUYER_OF:  # p is a supplier: give it a buyer
            buyers = by_sector[_BUYER_OF[p.sector]]
            buyer = rng.choice([b for b in buyers if b.ticker not in {c.ticker for c in COMPANIES}] or buyers)
            links.append({"supplier": p.ticker, "customer": buyer.ticker, "material": _MATERIAL[p.sector]})
        else:  # p is a buyer: half of them share a core supplier (makes "X's supplier" ambiguous)
            sup_sectors = [s for s, b in _BUYER_OF.items() if b == p.sector]
            if sup_sectors and rng.random() < 0.5:
                s = rng.choice(sup_sectors)
                core_sup = [c for c in COMPANIES if c.sector == s]
                if core_sup:
                    links.append({"supplier": core_sup[0].ticker, "customer": p.ticker, "material": _MATERIAL[s]})
    return ScaledWorld(World(companies, links), n_distractors, seed)


@dataclass
class RawDoc:
    doc_id: str        # e.g. 600011_2023_annual_report
    ticker: str
    year: int
    doc_type: str
    text: str


def render_corpus(sw: ScaledWorld) -> list[RawDoc]:
    """Render every company-year into three documents. Order is shuffled with the seed so
    the core companies are not always at the start of a long context."""
    docs: list[RawDoc] = []
    w = sw.world
    for p in w.companies:
        fins = compute_financials(p)
        for y in REPORT_YEARS:
            docs.append(RawDoc(f"{p.ticker}_{y}_annual_report", p.ticker, y, "annual_report",
                               render_annual_report(p, fins, y, w)))
            docs.append(RawDoc(f"{p.ticker}_{y}_announcement", p.ticker, y, "announcement",
                               render_announcement(p, fins, y)))
            docs.append(RawDoc(f"{p.ticker}_{y}_research_report", p.ticker, y, "research_report",
                               render_research_report(p, fins, y, w)))
    random.Random(sw.seed + 1).shuffle(docs)
    return docs


def facts_of(sw: ScaledWorld) -> dict:
    """Ground-truth facts in the same shape as ``facts.json``."""
    w = sw.world
    return {
        "companies": {p.ticker: {"ticker": p.ticker, "name": p.name, "sector": p.sector,
                                 "chain": p.chain, "role": p.role} for p in w.companies},
        "supply_links": w.links,
        "financials": {p.ticker: {str(y): v for y, v in compute_financials(p).items() if y in REPORT_YEARS}
                       for p in w.companies},
    }
