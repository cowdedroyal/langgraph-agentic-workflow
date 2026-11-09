"""Paired join / lookup questions.

For a buyer B whose only supplier of material M is S, the pair is

  join:    "B 的 M 供应商在 y 年的归母净利润是多少亿元？"   (must find S through B's report)
  lookup:  "S 在 y 年的归母净利润是多少亿元？"              (names S directly)

Both have the same gold value and the same evidence document (S's report for year y). The
accuracy gap between the two is the cost of the join alone; ranking difficulty, document
position and arithmetic are identical inside a pair.

Pairs come from every supply link in the world, not only the six core companies, so the
number of pairs grows with the corpus (both years per link; capped, core links first, then a
seeded sample of the others).
"""

from __future__ import annotations

import random

from ..access.questions import Question
from ..access.world import ScaledWorld, facts_of


def build_pairs(sw: ScaledWorld, max_pairs: int = 32, seed: int = 11) -> list[Question]:
    facts = facts_of(sw)
    fin, comp = facts["financials"], facts["companies"]
    core = set(sw.core_tickers)

    # (buyer, material) -> suppliers; keep only unambiguous ones
    sup_by: dict[tuple[str, str], list[str]] = {}
    for link in facts["supply_links"]:
        sup_by.setdefault((link["customer"], link["material"]), []).append(link["supplier"])
    links = [(b, m, s[0]) for (b, m), s in sup_by.items() if len(s) == 1 and s[0] in fin and b in comp]

    core_links = sorted(l for l in links if l[0] in core)
    other = sorted(l for l in links if l[0] not in core)
    random.Random(seed).shuffle(other)

    picked: list[tuple[str, str, str, int]] = []
    for b, m, s in core_links + other:              # both report years, core links first
        picked += [(b, m, s, 2022), (b, m, s, 2023)]
    picked = picked[:max_pairs]

    qs: list[Question] = []
    for b, m, s, y in picked:
        v = fin[s][str(y)]["net_profit"]
        tol = max(0.06, v * 0.01)
        pid = f"{b}-{s}-{y}"
        qs.append(Question(f"pj-{pid}", "join",
                           f"{comp[b]['name']}的{m}供应商在{y}年的归母净利润是多少亿元？",
                           "number", v, f"{comp[s]['name']} {v}亿元", [(b, y), (s, y)], tolerance=tol))
        qs.append(Question(f"pl-{pid}", "join_lookup",
                           f"{comp[s]['name']}在{y}年的归母净利润是多少亿元？",
                           "number", v, f"{comp[s]['name']} {v}亿元", [(s, y)], tolerance=tol))
    return qs


def pair_id(qid: str) -> str | None:
    return qid[3:] if qid.startswith(("pj-", "pl-")) else None
