"""Offline tests for the paired join/lookup questions (no GPU, no LLM)."""

from agentic_rag.access.world import build_world, facts_of
from agentic_rag.kvpaths.questions import build_pairs, pair_id


def test_pairs_share_gold_and_second_hop():
    sw = build_world(18)
    qs = build_pairs(sw)
    by = {}
    for q in qs:
        by.setdefault(pair_id(q.qid), {})[q.hop_type] = q
    assert by and all(set(v) == {"join", "join_lookup"} for v in by.values())
    for v in by.values():
        j, l = v["join"], v["join_lookup"]
        assert j.gold == l.gold and j.tolerance == l.tolerance
        assert j.evidence[-1] == l.evidence[-1]          # same second-hop document
        assert len(j.evidence) == 2 and len(l.evidence) == 1


def test_join_question_does_not_name_the_supplier():
    sw = build_world(54)
    comp = facts_of(sw)["companies"]
    for q in build_pairs(sw):
        sup = comp[q.evidence[-1][0]]["name"]
        if q.hop_type == "join":
            assert sup not in q.question
        else:
            assert sup in q.question


def test_supplier_is_unambiguous_for_the_buyer():
    sw = build_world(54)
    links = facts_of(sw)["supply_links"]
    for q in build_pairs(sw):
        if q.hop_type != "join":
            continue
        (buyer, _), (sup, _) = q.evidence
        mats = [l["material"] for l in links if l["customer"] == buyer and l["supplier"] == sup]
        assert len(mats) == 1
        assert sum(1 for l in links if l["customer"] == buyer and l["material"] == mats[0]) == 1


def test_pairs_grow_with_corpus_and_are_deterministic():
    n = [len(build_pairs(build_world(s))) for s in (0, 18, 54)]
    assert n[0] < n[1] < n[2]
    assert [q.qid for q in build_pairs(build_world(18))] == [q.qid for q in build_pairs(build_world(18))]


def test_rare_pairs_ask_single_table_metrics_present_in_the_annual_report():
    from agentic_rag.access.world import render_corpus
    from agentic_rag.kvpaths.questions import build_pairs as bp
    import re
    sw = build_world(18)
    docs = render_corpus(sw)
    qs = bp(sw, pair_set="rare")
    assert qs and {q.qid.rsplit("-", 1)[-1] for q in qs} == {"ocf", "debt_ratio"}
    for q in qs:
        t, y = q.evidence[-1]
        hits = [d.doc_id for d in docs if d.ticker == t and d.year == y
                and re.search(r"(?<![0-9.])" + re.escape(str(q.gold)) + r"(?![0-9])", d.text)]
        assert hits and all("annual_report" in h for h in hits)
    # net_profit pair texts are unchanged (earlier runs stay comparable)
    assert build_pairs(sw)[0].question.endswith("的归母净利润是多少亿元？")


def test_lenient_parse_reads_final_statement_only():
    from agentic_rag.kvpaths.grading import parse_lenient
    assert parse_lenient('推理\n{"answer": "10.23"}') == "10.23"
    assert parse_lenient("……为10.26亿元。\n答案：10.23") == "10.23"
    assert parse_lenient("净利润是**10.23**亿元。") == "10.23"
    assert parse_lenient('{"answer": null}') is None
    assert parse_lenient("资料中未提及。") is None


def test_cluster_bootstrap_counts_links():
    from agentic_rag.kvpaths.run import bootstrap_gap
    g = bootstrap_gap([("a-b", 1), ("a-b", 1), ("c-d", 0), ("e-f", 0)])
    assert g["n_pairs"] == 4 and g["n_links"] == 3 and g["gap"] == 0.5
    assert g["lo"] <= g["gap"] <= g["hi"]
