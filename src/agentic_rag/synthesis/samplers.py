"""Evidence-chain samplers for multi-hop QA synthesis (roadmap P3).

Each sampler emits ``EvidenceChain`` specs describing *which* facts a question must
combine. The three hop types map to the roadmap's evidence sources:
  * cross_year    — same company, two periods (year-over-year reasoning),
  * supply_chain  — a company -> its supplier (entity-link reasoning),
  * metric_change — same company/metric trend across periods.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..ingestion.synthetic import SUPPLY_LINKS, by_ticker, suppliers_of


@dataclass
class EvidenceChain:
    hop_type: str
    primary: str                       # primary company ticker
    years: tuple[int, ...]
    metric: str | None = None
    secondary: str | None = None       # e.g. supplier ticker
    material: str | None = None
    extra: dict = field(default_factory=dict)


_CROSS_YEAR_METRICS = ["revenue", "net_profit"]
_CROSS_YEAR_SPANS = [(2021, 2023), (2022, 2023)]
_METRIC_CHANGE_METRICS = ["gross_margin", "revenue", "net_profit"]


def sample_cross_year() -> list[EvidenceChain]:
    out = []
    for ticker in by_ticker():
        for metric in _CROSS_YEAR_METRICS:
            for y1, y2 in _CROSS_YEAR_SPANS:
                out.append(EvidenceChain("cross_year", ticker, (y1, y2), metric=metric))
    return out


def sample_metric_change() -> list[EvidenceChain]:
    out = []
    for ticker in by_ticker():
        for metric in _METRIC_CHANGE_METRICS:
            out.append(EvidenceChain("metric_change", ticker, (2021, 2023), metric=metric))
    return out


def sample_supply_chain() -> list[EvidenceChain]:
    """2-hop: a company's supplier's net profit in a given year."""
    out = []
    for link in SUPPLY_LINKS:
        customer, supplier, material = link["customer"], link["supplier"], link["material"]
        for year in (2022, 2023):
            out.append(EvidenceChain("supply_chain", customer, (year,), metric="net_profit",
                                     secondary=supplier, material=material))
    return out


def sample_deep_chain() -> list[EvidenceChain]:
    """3-hop: a company's supplier's upstream supplier (entity -> entity -> entity)."""
    out = []
    bt = by_ticker()
    for ticker in bt:
        sups = suppliers_of(ticker)
        for sup_ticker, material in sups:
            grand = suppliers_of(sup_ticker)
            for g_ticker, g_material in grand:
                out.append(EvidenceChain("deep_chain", ticker, (2022,), secondary=sup_ticker,
                                         material=material, extra={"grand": g_ticker, "g_material": g_material}))
    return out


def sample_all() -> list[EvidenceChain]:
    return sample_cross_year() + sample_metric_change() + sample_supply_chain() + sample_deep_chain()
