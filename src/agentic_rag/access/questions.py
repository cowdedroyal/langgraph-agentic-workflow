"""Question set for the access-path study.

Lookup-style questions (cross_year, metric_change, supply_chain, deep_chain) are generated
from the 6 core companies only, so they stay identical at every corpus size. Aggregate
questions ("which ... is highest", "how many ...") range over *all* companies of a sector,
so their answers change with the corpus: that is the case a top-k index cannot serve.

Gold answers are computed from the facts, never by an LLM. Each question carries a typed
gold value so grading is a deterministic comparison, not an LLM judge.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..ingestion.synthetic import COMPANIES
from .world import ScaledWorld, facts_of

METRIC_ZH = {"revenue": "营业收入", "net_profit": "归母净利润", "gross_margin": "销售毛利率"}
METRIC_UNIT = {"revenue": "亿元", "net_profit": "亿元", "gross_margin": "%"}


@dataclass
class Question:
    qid: str
    hop_type: str
    question: str
    answer_kind: str               # number | entity | count
    gold: float | str | int
    gold_text: str
    evidence: list[tuple[str, int]] = field(default_factory=list)   # (ticker, year) pairs
    tolerance: float = 0.0          # absolute tolerance for numbers

    def to_dict(self) -> dict:
        return asdict(self)


def _pct(v2: float, v1: float) -> float:
    return round((v2 - v1) / v1 * 100, 1)


def build_questions(sw: ScaledWorld) -> list[Question]:
    facts = facts_of(sw)
    fin, comp = facts["financials"], facts["companies"]
    w = sw.world
    qs: list[Question] = []
    core = [p.ticker for p in COMPANIES]

    # cross_year: growth rate between two years
    for t in core:
        n = comp[t]["name"]
        for m in ("revenue", "net_profit"):
            for y1, y2 in ((2021, 2023), (2022, 2023)):
                v1, v2 = fin[t][str(y1)][m], fin[t][str(y2)][m]
                g = _pct(v2, v1)
                qs.append(Question(
                    f"cy-{t}-{m}-{y1}{y2}", "cross_year",
                    f"{n}（{t}）{y2}年的{METRIC_ZH[m]}较{y1}年变化了百分之多少？（增长为正、下降为负）",
                    "number", g, f"{g}%", [(t, y1), (t, y2)], tolerance=0.6))

    # metric_change: absolute change
    for t in core:
        n = comp[t]["name"]
        for m in ("gross_margin", "revenue", "net_profit"):
            v1, v2 = fin[t]["2021"][m], fin[t]["2023"][m]
            d = round(v2 - v1, 2)
            unit = "个百分点" if m == "gross_margin" else "亿元"
            qs.append(Question(
                f"mc-{t}-{m}", "metric_change",
                f"{n}的{METRIC_ZH[m]}从2021年到2023年变化了多少{unit}？（上升为正、下降为负）",
                "number", d, f"{d}{unit}", [(t, 2021), (t, 2023)],
                tolerance=max(0.06, abs(d) * 0.01)))

    # supply_chain: a core company's supplier's net profit
    for t in core:
        for sup, mat in w.suppliers_of(t):
            if sup not in core:
                continue
            for y in (2022, 2023):
                v = fin[sup][str(y)]["net_profit"]
                qs.append(Question(
                    f"sc-{t}-{sup}-{y}", "supply_chain",
                    f"{comp[t]['name']}的{mat}供应商在{y}年的归母净利润是多少亿元？",
                    "number", v, f"{comp[sup]['name']} {v}亿元", [(t, y), (sup, y)],
                    tolerance=max(0.06, v * 0.01)))

    # deep_chain: supplier of a supplier
    for t in core:
        for sup, mat in w.suppliers_of(t):
            if sup not in core:
                continue
            for g, _ in w.suppliers_of(sup):
                if g not in core:
                    continue
                qs.append(Question(
                    f"dc-{t}-{sup}-{g}", "deep_chain",
                    f"{comp[t]['name']}的{mat}供应商，它自己的上游供应商是哪家公司？",
                    "entity", comp[g]["name"], comp[g]["name"], [(t, 2022), (sup, 2022)]))

    # aggregate: range over every company of a sector (answers change with corpus size)
    sectors = sorted({comp[t]["sector"] for t in core})
    for s in sectors:
        members = [t for t in comp if comp[t]["sector"] == s]
        best = max(members, key=lambda t: fin[t]["2023"]["gross_margin"])
        qs.append(Question(
            f"ag-max-{s}", "aggregate",
            f"在语料中所有主营{s}的公司里，2023年销售毛利率最高的是哪家公司？",
            "entity", comp[best]["name"], f"{comp[best]['name']} {fin[best]['2023']['gross_margin']}%",
            [(t, 2023) for t in members]))
        thr = sorted(fin[t]["2023"]["revenue"] for t in members)[len(members) // 2]
        thr = max(10, int(round(thr / 10.0)) * 10)
        cnt = sum(1 for t in members if fin[t]["2023"]["revenue"] > thr)
        qs.append(Question(
            f"ag-cnt-{s}", "aggregate",
            f"在语料中所有主营{s}的公司里，2023年营业收入超过{thr}亿元的有几家？",
            "count", cnt, str(cnt), [(t, 2023) for t in members]))
    return qs


# --------------------------------------------------------------------------- beyond the view's schema
# Questions whose answer is in the corpus but NOT in the fact table the view extracts
# (company, year, sector, revenue, net_profit, gross_margin, suppliers, customers).
# They test whether a planner knows when a materialized view cannot answer: the view schema
# was designed from the base question set, which is the usual way a view is designed.
OOS_HOPS = ("oos_lookup", "oos_report", "oos_prior", "oos_join", "oos_aggregate")
OOS_METRIC = {"rd_expense": ("研发费用", "亿元"), "debt_ratio": ("资产负债率", "%"),
              "ocf": ("经营活动现金流量净额", "亿元")}


def build_oos_questions(sw: ScaledWorld) -> list[Question]:
    from ..ingestion.synthetic import _BROKERS, compute_financials

    w = sw.world
    bt = w.by_ticker()
    core = [p.ticker for p in COMPANIES]
    fins = {p.ticker: compute_financials(p) for p in w.companies}
    qs: list[Question] = []

    # oos_lookup: annual-report-only metrics of a named company (2023)
    for t in core:
        n = bt[t].name
        for m, (zh, unit) in OOS_METRIC.items():
            v = fins[t][2023][m]
            tol = 0.06 if unit == "%" else max(0.06, abs(v) * 0.01)
            qs.append(Question(f"ol-{t}-{m}", "oos_lookup", f"{n}2023年的{zh}是多少{unit}？",
                               "number", v, f"{v}{unit}", [(t, 2023)], tolerance=tol))

    # oos_report: fields only in the research report (2023 report)
    for t in core:
        p = bt[t]
        f = fins[t][2023]
        rating = "买入" if p.rev_growth > 0.2 else "增持"
        target = round(f["revenue"] * (0.6 if p.chain == "白酒链" else 0.3), 1)
        broker = _BROKERS[(int(p.ticker) + 2023) % len(_BROKERS)]
        qs.append(Question(f"or-{t}-rating", "oos_report", f"券商对{p.name}2023年报点评给出的投资评级是什么？",
                           "entity", rating, rating, [(t, 2023)]))
        qs.append(Question(f"or-{t}-target", "oos_report", f"券商对{p.name}2023年报点评给出的目标价是多少元？",
                           "number", target, f"{target}元", [(t, 2023)], tolerance=0.06))
        qs.append(Question(f"or-{t}-broker", "oos_report", f"{p.name}2023年报点评研报是哪家券商出具的？",
                           "entity", broker, broker, [(t, 2023)]))

    # oos_prior: 2020 figures only appear as the prior-year column of 2021 documents
    for t in core:
        v = fins[t][2020]["revenue"]
        qs.append(Question(f"op-{t}-revenue2020", "oos_prior", f"{bt[t].name}2020年的营业收入是多少亿元？",
                           "number", v, f"{v}亿元", [(t, 2021)], tolerance=max(0.06, v * 0.01)))

    # oos_join: join key is in the schema (supplier), the fetched attribute is not
    for t in core:
        for sup, mat in w.suppliers_of(t):
            if sup not in core:
                continue
            v = fins[sup][2023]["debt_ratio"]
            qs.append(Question(f"oj-{t}-{sup}-debt", "oos_join",
                               f"{bt[t].name}的{mat}供应商在2023年的资产负债率是多少%？",
                               "number", v, f"{bt[sup].name} {v}%", [(t, 2023), (sup, 2023)], tolerance=0.06))

    # oos_aggregate: sector-wide ranking on an attribute outside the schema
    for s in sorted({bt[t].sector for t in core}):
        members = [p.ticker for p in w.companies if p.sector == s]
        vals = sorted(((fins[m][2023]["rd_expense"], m) for m in members), reverse=True)
        assert len(vals) < 2 or vals[0][0] != vals[1][0], f"tie for max rd_expense in {s}"
        best = vals[0][1]
        qs.append(Question(f"oa-max-rd-{s}", "oos_aggregate",
                           f"在语料中所有主营{s}的公司里，2023年研发费用最高的是哪家公司？",
                           "entity", bt[best].name, f"{bt[best].name} {vals[0][0]}亿元",
                           [(m, 2023) for m in members]))
    return qs
