"""Experiment runner: ``python -m agentic_rag.access.run --sizes 0 54 --paths scan index view agent``.

For each corpus size (number of distractor companies) it builds the world, the questions
and every requested path, answers all questions, grades them, and writes:

  outputs/access/<size>/records.jsonl   one line per (path, question)
  outputs/access/summary.json           aggregate per (size, path) and per hop type
  docs/metrics/access_paths.md          the table that the README cites

LLM responses are cached under ``outputs/access/cache`` so a re-run is free.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .grade import grade
from .llm import LLM, Usage
from .paths import AgentPath, IndexJoinPath, IndexPath, ScanPath, ViewPath
from .questions import build_questions
from .retrieval import DocSearch, HybridIndex, chunk_docs
from .world import build_world, render_corpus

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "outputs" / "access"


def _run_path(path, questions, workers: int, sequential: bool):
    def one(q):
        t0 = time.time()
        try:
            r = path.answer(q.question)
            err = None
        except Exception as e:  # keep going; record the failure
            r, err = None, repr(e)[:300]
        ok = grade(q, r.answer) if r else False
        return {"qid": q.qid, "hop_type": q.hop_type, "gold": q.gold, "answer": r.answer if r else None,
                "correct": ok, "usage": r.usage.to_dict() if r else None, "trace": r.trace if r else None,
                "wall_s": round(time.time() - t0, 2), "error": err,
                "text_tail": (r.text[-400:] if r else "")}
    with ThreadPoolExecutor(workers) as ex:
        if sequential:  # first call warms the prefix cache, then the rest reuse it
            first = one(questions[0])
            return [first] + list(ex.map(one, questions[1:]))
        return list(ex.map(one, questions))


def summarize(records: list[dict], build: Usage | None = None) -> dict:
    n = len(records)
    tot = Usage()
    for r in records:
        if r["usage"]:
            u = {k: v for k, v in r["usage"].items() if k != "usd"}
            tot.add(Usage(**u))
    by_hop = defaultdict(list)
    for r in records:
        by_hop[r["hop_type"]].append(r["correct"])
    out = {
        "n": n, "accuracy": round(sum(r["correct"] for r in records) / n, 4) if n else 0,
        "errors": sum(1 for r in records if r["error"]),
        "per_hop": {h: round(sum(v) / len(v), 4) for h, v in sorted(by_hop.items())},
        "query_usage": tot.to_dict(),
        "usd_per_q": round(tot.usd / n, 6) if n else 0,
        "prompt_tokens_per_q": round(tot.prompt / n) if n else 0,
        "cache_hit_rate": round(tot.hit / tot.prompt, 4) if tot.prompt else 0,
        # latency = model time recorded on the first (uncached) call, so cached re-runs report the same number
        "p50_latency_s": round(sorted((r["usage"] or {}).get("latency_s", 0) for r in records)[n // 2], 2) if n else 0,
    }
    if build is not None:
        out["build_usage"] = build.to_dict()
        out["usd_per_q_amortized"] = round((tot.usd + build.usd) / n, 6) if n else 0
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[0])
    ap.add_argument("--paths", nargs="+", default=["scan", "index", "view", "agent"])
    ap.add_argument("--limit", type=int, default=0, help="only the first N questions (smoke test)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--model", default=None)
    ap.add_argument("--context-limit", type=int, default=1_000_000)
    args = ap.parse_args(argv)

    llm = LLM(model=args.model, cache_dir=OUT / "cache") if args.model else LLM(cache_dir=OUT / "cache")
    summary_path = OUT / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

    for size in args.sizes:
        sw = build_world(size)
        docs = render_corpus(sw)
        qs = build_questions(sw)
        if args.limit:
            qs = qs[: args.limit]
        chars = sum(len(d.text) for d in docs)
        print(f"\n=== size={size} companies={len(sw.world.companies)} docs={len(docs)} chars={chars} questions={len(qs)}")
        d = OUT / str(size)
        d.mkdir(parents=True, exist_ok=True)
        (d / "questions.jsonl").write_text("\n".join(json.dumps(q.to_dict(), ensure_ascii=False) for q in qs))
        entry = summary.setdefault(str(size), {"companies": len(sw.world.companies), "docs": len(docs),
                                                 "chars": chars, "paths": {}})
        index = None
        for name in args.paths:
            t0 = time.time()
            build = None
            if name == "scan":
                # 0.534 prompt tokens per corpus char (measured: 176,588 tokens for 331,101 chars)
                est = int(chars * 0.534)
                if est > args.context_limit:
                    entry["paths"][name] = {"infeasible": True,
                                            "reason": f"corpus ~{est} tokens > context {args.context_limit}"}
                    print(f"  scan   infeasible: corpus ~{est} tokens")
                    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
                    continue
                path, seq = ScanPath(llm, docs), True
            elif name in ("index", "index_join"):
                if index is None:
                    index = HybridIndex(chunk_docs(docs), OUT / "emb")
                cls = IndexPath if name == "index" else IndexJoinPath
                path, seq = cls(llm, index, args.topk), False
            elif name == "view":
                path, seq = ViewPath(llm, docs, workers=args.workers * 2), True
                build = path.build_usage
                (d / "view_table.txt").write_text(path.table)
                print(f"  view: {path.n_rows} rows, {path.failed} failed extractions, build ${build.usd:.4f}")
            elif name == "agent":
                path, seq = AgentPath(llm, docs, DocSearch(docs)), False
            else:
                raise SystemExit(f"unknown path {name}")
            recs = _run_path(path, qs, args.workers, seq)
            with (d / f"records_{name}.jsonl").open("w") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            s = summarize(recs, build)
            s["wall_s"] = round(time.time() - t0, 1)
            entry["paths"][name] = s
            print(f"  {name:6s} acc={s['accuracy']:.3f} ${s['usd_per_q']:.5f}/q "
                  f"prompt={s['prompt_tokens_per_q']} hit={s['cache_hit_rate']:.2f} p50={s['p50_latency_s']}s "
                  f"err={s['errors']} {s['per_hop']}")
            summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    write_table(summary)


def write_table(summary: dict) -> None:
    hops = ["cross_year", "metric_change", "supply_chain", "deep_chain", "aggregate"]
    lines = ["# Access paths for multi-hop financial QA", "",
             "Generated by `python -m agentic_rag.access.run`. Model: deepseek-v4-flash, temperature 0. "
             "Cost uses list prices (input miss $0.15/M, cache hit $0.003/M, output $0.6/M).", "",
             "| distractors | companies | corpus chars | path | accuracy | " + " | ".join(hops)
             + " | prompt tok/q | cache hit | $/q | $/q incl. build | p50 s |",
             "|---|---|---|---|---|" + "---|" * len(hops) + "---|---|---|---|---|"]
    for size in sorted(summary, key=int):
        e = summary[size]
        for name in ("scan", "index", "index_join", "view", "agent"):
            s = e["paths"].get(name)
            if not s:
                continue
            if s.get("infeasible"):
                lines.append(f"| {size} | {e['companies']} | {e['chars']} | {name} | infeasible: {s['reason']} |"
                             + " |" * (len(hops) + 5))
                continue
            ph = " | ".join(f"{s['per_hop'].get(h, float('nan')):.2f}" if h in s["per_hop"] else "-" for h in hops)
            lines.append(f"| {size} | {e['companies']} | {e['chars']} | {name} | {s['accuracy']:.3f} | {ph} | "
                         f"{s['prompt_tokens_per_q']} | {s['cache_hit_rate']:.2f} | {s['usd_per_q']:.5f} | "
                         f"{s.get('usd_per_q_amortized', s['usd_per_q']):.5f} | {s['p50_latency_s']} |")
    p = ROOT / "docs" / "metrics" / "access_paths.md"
    p.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {p}")


if __name__ == "__main__":
    import os
    import sys

    main()
    sys.stdout.flush()
    os._exit(0)  # torch / tokenizer threads can keep the interpreter alive after main returns
