"""Sensitivity check: are scan's misses on large corpora a context failure or an output-budget
failure? Re-asks the questions scan got wrong with a larger max_tokens and records both runs.

Run: ``python -m agentic_rag.access.budget_probe --tag <tag> --size 180 --max-tokens 16384``
"""

from __future__ import annotations

import argparse
import json

from .grade import grade
from .llm import LLM, MODEL
from .paths import ANSWER_RULE, ScanPath, parse_answer
from .questions import build_questions
from .run import OUT, ROOT, model_tag
from .world import build_world, render_corpus


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None)
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--max-tokens", type=int, default=16384)
    a = ap.parse_args(argv)
    tag = a.tag or model_tag(MODEL)
    llm = LLM(model=MODEL, cache_dir=OUT / "cache")
    sw = build_world(a.size)
    qs = {q.qid: q for q in build_questions(sw)}
    scan = ScanPath(llm, render_corpus(sw))
    sys_msg = "你是金融分析助手。下面是完整的资料库，只能依据资料回答。\n\n" + scan.corpus
    recs = [json.loads(l) for l in (OUT / tag / str(a.size) / "records_scan.jsonl").read_text().splitlines()]
    rows = []
    for r in recs:
        if r["correct"]:
            continue
        q = qs[r["qid"]]
        rep = llm.chat([{"role": "system", "content": sys_msg},
                        {"role": "user", "content": f"{q.question}\n\n{ANSWER_RULE}"}], max_tokens=a.max_tokens)
        ans = parse_answer(rep.text)
        rows.append({"qid": q.qid, "gold": q.gold,
                     "at_4096": {"answer": r["answer"], "completion": r["usage"]["completion"], "correct": r["correct"]},
                     f"at_{a.max_tokens}": {"answer": ans, "completion": rep.usage.completion,
                                           "finish": rep.finish, "correct": grade(q, ans)}})
        print(json.dumps(rows[-1], ensure_ascii=False))
    p = ROOT / "docs" / "metrics" / f"scan_budget_probe_{tag}_{a.size}.json"
    p.write_text(json.dumps({"model": model_tag(MODEL), "size": a.size, "max_tokens": a.max_tokens,
                             "n_wrong_at_4096": len(rows), "fixed": sum(x[f"at_{a.max_tokens}"]["correct"] for x in rows),
                             "rows": rows}, ensure_ascii=False, indent=2))
    print("wrote", p)


if __name__ == "__main__":
    main()
