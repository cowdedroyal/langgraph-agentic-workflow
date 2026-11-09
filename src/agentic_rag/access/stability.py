"""Repeatability probe: temperature 0 is not deterministic on the gateway, so a single run
per question can make a path look better or worse than it is. This re-asks the chosen
questions R more times per path and reports per-question hit rates.

The response cache keys on the full request body, so each repeat changes ``max_tokens`` by
one token (4096, 4095, ...) to force a fresh call without changing the budget materially.

Run: ``python -m agentic_rag.access.stability --size 180 --paths scan view --hop aggregate --repeats 3``
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from .grade import grade
from .llm import LLM, MODEL, Usage
from .paths import ANSWER_RULE, doc_block, parse_answer
from .questions import build_questions
from .run import OUT, ROOT, model_tag
from .world import build_world, render_corpus


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--paths", nargs="+", default=["scan", "view"])
    ap.add_argument("--hop", default="aggregate")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    tag = a.tag or model_tag(MODEL)
    llm = LLM(model=MODEL, cache_dir=OUT / "cache")
    sw = build_world(a.size)
    qs = [q for q in build_questions(sw) if q.hop_type == a.hop]
    d = OUT / tag / str(a.size)
    ctx = {}
    if "scan" in a.paths:
        ctx["scan"] = ("你是金融分析助手。下面是完整的资料库，只能依据资料回答。\n\n"
                       + "\n".join(doc_block(x) for x in render_corpus(sw)))
    if "view" in a.paths:  # the table built in the main run, so extraction is held fixed
        ctx["view"] = ("你是金融分析助手。下面是从资料库抽取出的事实表（每行一家公司一个年份，金额单位亿元，毛利率单位%），"
                       "只能依据此表回答，需要计算时自己算。\n\n" + (d / "view_table.txt").read_text())

    if "view_sql" in a.paths:  # same extraction (cached), SQL path; repeats perturb both calls' budget
        from .paths import ViewSQLPath
        ctx["view_sql"] = ViewSQLPath(llm, render_corpus(sw), workers=8)

    out = {"model": model_tag(MODEL), "size": a.size, "hop": a.hop, "repeats_extra": a.repeats, "paths": {}}
    for p, sys_msg in ctx.items():
        first = {r["qid"]: r for r in map(json.loads, (d / f"records_{p}.jsonl").read_text().splitlines())}

        def one(job):
            q, k = job
            if p == "view_sql":
                pr = sys_msg.answer(q.question, max_tokens=4096 - k)
                return q.qid, pr.answer, grade(q, pr.answer), pr.usage
            r = llm.chat([{"role": "system", "content": sys_msg},
                          {"role": "user", "content": f"{q.question}\n\n{ANSWER_RULE}"}], max_tokens=4096 - k)
            ans = parse_answer(r.text)
            return q.qid, ans, grade(q, ans), r.usage

        jobs = [(q, k) for k in range(1, a.repeats + 1) for q in qs]
        # warm the prefix cache with one call, then fan out
        res = [one(jobs[0])]
        with ThreadPoolExecutor(a.workers) as ex:
            res += list(ex.map(one, jobs[1:]))
        tot = Usage()
        per_q = {q.qid: {"gold": q.gold, "answers": [first[q.qid]["answer"]], "correct": [first[q.qid]["correct"]]}
                 for q in qs}
        for qid, ans, ok, u in res:
            per_q[qid]["answers"].append(ans)
            per_q[qid]["correct"].append(ok)
            tot.add(u)
        runs = a.repeats + 1
        accs = [sum(v["correct"][i] for v in per_q.values()) / len(qs) for i in range(runs)]
        out["paths"][p] = {
            "runs": runs, "accuracy_per_run": [round(x, 4) for x in accs],
            "accuracy_mean": round(sum(accs) / runs, 4), "accuracy_min": round(min(accs), 4),
            "accuracy_max": round(max(accs), 4),
            "questions_always_right": sum(all(v["correct"]) for v in per_q.values()),
            "questions_flaky": sum(0 < sum(v["correct"]) < runs for v in per_q.values()),
            "questions_always_wrong": sum(not any(v["correct"]) for v in per_q.values()),
            "extra_usd": round(tot.usd, 5), "per_question": per_q}
        s = out["paths"][p]
        print(f"{p}: per-run acc {s['accuracy_per_run']} mean {s['accuracy_mean']} "
              f"always/flaky/never {s['questions_always_right']}/{s['questions_flaky']}/{s['questions_always_wrong']} "
              f"extra ${s['extra_usd']}")
    f = ROOT / "docs" / "metrics" / f"stability_{tag}_{a.size}_{a.hop}.json"
    if f.exists():  # merge: runs for different paths share one file
        old = json.loads(f.read_text())
        old["paths"].update(out["paths"])
        out = {**out, "paths": old["paths"]}
    f.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", f)


if __name__ == "__main__":
    main()
