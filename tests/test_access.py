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
