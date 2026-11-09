"""Offline tests for the access-path study (no LLM, no embedding model)."""

from agentic_rag.access.grade import grade
from agentic_rag.access.paths import parse_answer
from agentic_rag.access.questions import build_questions
from agentic_rag.access.world import build_world, facts_of, render_corpus
from agentic_rag.ingestion import synthetic as S


def test_core_world_renders_like_original():
    p = S.COMPANIES[0]
    f = S.compute_financials(p)
    assert S.render_annual_report(p, f, 2023) == S.render_annual_report(p, f, 2023, S.CORE_WORLD)


def test_world_is_deterministic_and_grows():
    a, b = build_world(54), build_world(54)
    assert [c.name for c in a.world.companies] == [c.name for c in b.world.companies]
    assert [d.doc_id for d in render_corpus(a)] == [d.doc_id for d in render_corpus(b)]
    assert len(a.world.companies) == 60
    names = [c.name for c in build_world(540).world.companies]
    assert len(set(names)) == len(names)


def test_lookup_questions_fixed_across_sizes():
    lookup = lambda sw: [(q.qid, q.gold) for q in build_questions(sw) if q.hop_type != "aggregate"]
    assert lookup(build_world(0)) == lookup(build_world(540))


def test_gold_matches_facts():
    sw = build_world(0)
    fin = facts_of(sw)["financials"]
    q = next(q for q in build_questions(sw) if q.qid == "cy-600011-revenue-20212023")
    v1, v2 = fin["600011"]["2021"]["revenue"], fin["600011"]["2023"]["revenue"]
    assert q.gold == round((v2 - v1) / v1 * 100, 1)
    dc = next(q for q in build_questions(sw) if q.hop_type == "deep_chain")
    assert dc.gold == "云岭矿业"


def test_distractors_share_core_suppliers():
    w = build_world(54).world
    core = {c.ticker for c in S.COMPANIES}
    assert any(l["supplier"] in core and l["customer"] not in core for l in w.links)


def test_grade_and_parse():
    q = next(q for q in build_questions(build_world(0)) if q.answer_kind == "number")
    assert grade(q, q.gold) and grade(q, str(q.gold) + "%") and not grade(q, q.gold + 5)
    assert not grade(q, None)
    e = next(q for q in build_questions(build_world(0)) if q.answer_kind == "entity")
    assert grade(e, e.gold + "（300023）") and not grade(e, "长江动力")
    assert parse_answer('推理……\n{"answer": -3.5}') == -3.5
    assert parse_answer('{"answer": 长江动力}') == "长江动力"
    assert parse_answer('{"answer": null}') is None


def test_planner_replay_and_cascade():
    from agentic_rag.access.llm import Usage
    from agentic_rag.access.planner import evaluate, oracle, p_coverage, p_scan_or_view, p_sql_escalate

    def rec(ok, ans, usd):
        return {"correct": ok, "answer": ans, "usage": {"usd": usd, "latency_s": 1.0}}
    recs = {"index": {"a": rec(True, 1, 0.1), "b": rec(False, None, 0.1)},
            "view_sql": {"a": rec(True, 1, 0.2), "b": rec(False, None, 0.2)},
            "scan": {"a": rec(True, 1, 1.0), "b": rec(True, 2, 1.0)},
            "agent": {"a": rec(True, 1, 0.5), "b": rec(True, 2, 0.5)}}
    zero = {"a": Usage(), "b": Usage()}
    look = {"class": "lookup", "covered": True}
    # a is lookup -> index; b is a join and scan fits -> scan
    r = evaluate(p_scan_or_view, ["a", "b"], {"a": look, "b": {"class": "join", "covered": True}},
                 zero, recs, True, 0.0, 2)
    assert r["accuracy"] == 1.0 and abs(r["query_usd_per_q"] - 0.55) < 1e-9 and r["build_usd"] == 0
    # view_sql answers null on b -> escalate to agent (scan does not fit); build charged once
    r = evaluate(p_sql_escalate, ["a", "b"], {"a": look, "b": {"class": "aggregate", "covered": True}},
                 zero, recs, False, 0.4, 2, hop={"a": "x", "b": "y"})
    assert r["accuracy"] == 1.0 and abs(r["query_usd_per_q"] - (0.1 + 0.2 + 0.5) / 2) < 1e-9
    assert r["build_usd"] == 0.4 and r["paths_used"] == {"index": 1, "view_sql": 1, "agent": 1}
    assert abs(r["usd_per_q_at"]["1000"] - (0.4 + 0.4 / 1000)) < 1e-9 and r["per_hop"] == {"x": 1.0, "y": 1.0}
    # uncovered aggregate skips the view entirely -> no build cost
    r = evaluate(p_coverage, ["a", "b"], {"a": look, "b": {"class": "aggregate", "covered": False}},
                 zero, recs, False, 0.4, 2)
    assert r["build_usd"] == 0 and r["paths_used"] == {"index": 1, "agent": 1}
    o = oracle(recs, ["a", "b"], 0.4, 2)
    assert o["accuracy"] == 1.0 and abs(o["query_usd_per_q"] - 0.3) < 1e-9 and o["build_usd"] == 0


def test_oos_questions_are_outside_the_view_and_grounded():
    from agentic_rag.access.questions import build_oos_questions
    sw = build_world(54)
    by = {}
    for d in render_corpus(sw):
        by.setdefault((d.ticker, d.year), []).append(d.text)
    qs = build_oos_questions(sw)
    assert len(qs) == 52
    for q in qs:
        if q.hop_type != "oos_aggregate":  # the gold value is written in the evidence document
            t, y = q.evidence[-1]
            assert str(q.gold) in " ".join(by[(t, y)]), q.qid


def test_model_tag():
    from agentic_rag.access.run import model_tag
    assert model_tag("gateway/provider/claude-deepseek-v4.1-flash[1m]") == "deepseek-v4.1-flash"
    assert model_tag("deepseek/deepseek-v4-flash") == "deepseek-v4-flash"


def test_lazy_row_selection_is_not_capped_by_display_limit():
    import sqlite3
    import threading

    from agentic_rag.access.paths import ViewSQLPath
    p = ViewSQLPath.__new__(ViewSQLPath)
    p._lock = threading.Lock()
    p.db = sqlite3.connect(":memory:")
    p.db.execute("CREATE TABLE facts(ticker TEXT, year INTEGER)")
    p.db.executemany("INSERT INTO facts VALUES (?, 2023)", [(str(i),) for i in range(91)])
    assert len(p._exec("SELECT ticker, year FROM facts")[1]) == 50          # display cap
    assert len(p._exec("SELECT ticker, year FROM facts", limit=201)[1]) == 91  # row selection
