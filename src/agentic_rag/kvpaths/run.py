"""KV-level access paths: ``python -m agentic_rag.kvpaths.run --model <dir> --sizes 0 18``.

Same corpus, questions, answer rule and grader as ``agentic_rag.access``; the paired
join/lookup questions of ``kvpaths.questions`` are added. Writes

  <out>/<tag>/<size>/records.jsonl   one line per question (all policies and budgets)
  <out>/<tag>/summary.json           accuracy per (size, policy, budget), per hop, paired gaps

Records are appended per question, so an interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import torch

from ..access.grade import grade
from ..access.paths import ANSWER_RULE, doc_block, parse_answer
from ..access.questions import build_questions
from ..access.retrieval import DocSearch
from ..access.world import build_world, render_corpus
from .engine import KVEngine
from .questions import build_pairs, pair_id

PREFIX = "<|im_start|>system\n你是金融分析助手。下面是完整的资料库，只能依据资料回答。" + ANSWER_RULE + "\n\n"
SUFFIX = "<|im_end|>\n"
POLICIES = ["full", "window", "keydiff", "query", "join", "oracle"]


def q_text(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def build_context(eng: KVEngine, docs):
    """Tokenize piece by piece so every document has an exact token span."""
    ids = eng.encode(PREFIX)
    spans = []
    for d in docs:
        a = len(ids)
        ids += eng.encode(doc_block(d) + "\n")
        spans.append((a, len(ids)))
    n_prefix = spans[0][0]
    ids += eng.encode(SUFFIX)
    return ids, spans, (n_prefix, len(ids) - spans[-1][1])


def rec(q, g, policy, budget):
    ans = parse_answer(g.text)
    return {"policy": policy, "budget": budget, "answer": ans, "correct": grade(q, ans),
            "kept_frac": round(g.kept_frac, 4), "gen_tokens": g.n_tokens, "gen_s": round(g.seconds, 3),
            "text_tail": g.text[-300:]}


def run_size(eng: KVEngine, size: int, args, out_dir: Path) -> None:
    sw = build_world(size)
    docs = render_corpus(sw)
    qs = build_questions(sw) + build_pairs(sw)
    if args.only_pairs:
        qs = [q for q in qs if pair_id(q.qid)]
    if args.limit:
        qs = qs[: args.limit]
    ctx_ids, spans, protect = build_context(eng, docs)
    d = out_dir / str(size)
    d.mkdir(parents=True, exist_ok=True)
    rpath = d / "records.jsonl"
    done = set()
    if rpath.exists():
        done = {json.loads(l)["qid"] for l in rpath.read_text().splitlines() if l.strip()}
    todo = [q for q in qs if q.qid not in done]
    print(f"=== size={size} docs={len(docs)} ctx_tokens={len(ctx_ids)} questions={len(qs)} todo={len(todo)}", flush=True)
    if not todo:
        return
    torch.cuda.reset_peak_memory_stats()
    t_pre = eng.prefill(ctx_ids, protect)
    meta = {"size": size, "docs": len(docs), "ctx_tokens": len(ctx_ids), "protected_tokens": list(protect),
            "prefill_s": round(t_pre, 2),
            "kv_gb_full": round(eng.kv_bytes(len(ctx_ids)) / 1e9, 2),
            "mem_after_prefill_gb": round(torch.cuda.memory_allocated() / 1e9, 2)}
    print("   prefill", meta, flush=True)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))

    budgets = args.budgets
    kd = eng.keydiff_scores()
    sel_keydiff = {b: eng.select_topk(kd, b) for b in budgets}
    sel_window = {b: eng.select_window(b) for b in budgets}
    del kd
    doc_index = {(dd.ticker, dd.year): [] for dd in docs}
    for i, dd in enumerate(docs):
        doc_index[(dd.ticker, dd.year)].append(i)
    bm25 = DocSearch(docs)
    docpos = {dd.doc_id: i for i, dd in enumerate(docs)}

    with rpath.open("a") as f:
        for qi, q in enumerate(todo):
            t0 = time.time()
            q_ids = eng.encode(q_text(q.question))
            ts = time.time()
            scores, q_win, _ = eng.score_window(q_ids)
            t_score = time.time() - ts
            runs = []
            if "full" in args.policies:
                runs.append(rec(q, eng.answer_full(q_ids, args.max_new), "full", 1.0))
            for b in budgets:
                if "window" in args.policies:
                    runs.append(rec(q, eng.answer_agnostic(sel_window[b], q_ids, args.max_new), "window", b))
                if "keydiff" in args.policies:
                    runs.append(rec(q, eng.answer_agnostic(sel_keydiff[b], q_ids, args.max_new), "keydiff", b))
                if "query" in args.policies or "join" in args.policies:
                    gq = eng.answer_aware(eng.select_topk(scores, b), q_ids, q_win, args.max_new)
                    if "query" in args.policies:
                        runs.append(rec(q, gq, "query", b))
                    if "join" in args.policies:
                        # re-score with the question + first-pass answer (names the join key), keep
                        # the question KV from the plain pass so only the selection differs
                        probe = q_ids + eng.encode(gq.text)[: args.max_probe]
                        js, _, _ = eng.score_window(probe)
                        runs.append(rec(q, eng.answer_aware(eng.select_topk(js, b), q_ids, q_win, args.max_new),
                                        "join", b))
                        del js
            ev = [i for pair in q.evidence for i in doc_index.get(tuple(pair), [])]
            if "oracle" in args.policies:
                runs.append(rec(q, eng.answer_agnostic(eng.select_spans([spans[i] for i in ev]), q_ids,
                                                       args.max_new), "oracle", None))
            analysis = None
            if pair_id(q.qid):                       # where does the second hop rank, per layer?
                tgt = doc_index[tuple(q.evidence[-1])]
                order = [docpos[x.doc_id] for x in bm25(q.question, len(docs))]
                bm25_rank = min((order.index(t) + 1 for t in tgt if t in order), default=None)
                analysis = {"layer_rank": eng.doc_ranks(scores, spans, tgt), "bm25_rank": bm25_rank,
                            "n_docs": len(docs)}
            row = {"qid": q.qid, "hop_type": q.hop_type, "gold": q.gold, "pair": pair_id(q.qid),
                   "q_tokens": len(q_ids), "score_s": round(t_score, 3), "wall_s": round(time.time() - t0, 2),
                   "runs": runs, "analysis": analysis}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()
            del scores, q_win
            acc = {f"{r['policy']}@{r['budget']}": int(r["correct"]) for r in runs}
            print(f"   [{qi + 1}/{len(todo)}] {q.qid} {q.hop_type} wall={row['wall_s']}s {acc}", flush=True)
    meta["peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))


def bootstrap_gap(pairs: list[tuple[int, int]], n: int = 4000, seed: int = 0):
    """Mean of (lookup - join) correctness over pairs, with a 95% paired-bootstrap interval."""
    if not pairs:
        return None
    rng = random.Random(seed)
    diffs = [l - j for j, l in pairs]
    m = sum(diffs) / len(diffs)
    bs = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(n))
    return {"gap": round(m, 4), "lo": round(bs[int(0.025 * n)], 4), "hi": round(bs[int(0.975 * n)], 4),
            "n_pairs": len(diffs)}


def summarize(out_dir: Path) -> dict:
    summary = {}
    for d in sorted((p for p in out_dir.iterdir() if p.is_dir()), key=lambda p: int(p.name)):
        rows = [json.loads(l) for l in (d / "records.jsonl").read_text().splitlines() if l.strip()]
        meta = json.loads((d / "meta.json").read_text()) if (d / "meta.json").exists() else {}
        cells: dict[str, dict] = {}
        by = defaultdict(list)
        for r in rows:
            for x in r["runs"]:
                by[(x["policy"], x["budget"])].append((r, x))
        for (pol, b), lst in sorted(by.items(), key=lambda kv: (POLICIES.index(kv[0][0]), -(kv[0][1] or 0))):
            hop = defaultdict(list)
            for r, x in lst:
                hop[r["hop_type"]].append(x["correct"])
            pj = {}
            for r, x in lst:
                if r["pair"]:
                    pj.setdefault(r["pair"], {})[r["hop_type"]] = int(x["correct"])
            pairs = [(v["join"], v["join_lookup"]) for v in pj.values() if len(v) == 2]
            orig = [x["correct"] for r, x in lst if not r["pair"]]
            cells[f"{pol}@{b}"] = {
                "policy": pol, "budget": b, "n": len(lst),
                "accuracy_original63": round(sum(orig) / len(orig), 4) if orig else None,
                "per_hop": {h: round(sum(v) / len(v), 4) for h, v in sorted(hop.items())},
                "paired_gap_lookup_minus_join": bootstrap_gap(pairs),
                "kept_frac": round(sum(x["kept_frac"] for _, x in lst) / len(lst), 4),
                "gen_s_mean": round(sum(x["gen_s"] for _, x in lst) / len(lst), 3),
            }
        an = [r["analysis"] for r in rows if r["analysis"]]
        layer = {}
        for kind in ("join", "join_lookup"):
            a = [r["analysis"] for r in rows if r["analysis"] and r["hop_type"] == kind]
            if a:
                nl = len(a[0]["layer_rank"])
                layer[kind] = {
                    "median_rank_by_layer": [sorted(x["layer_rank"][i] for x in a)[len(a) // 2] for i in range(nl)],
                    "top3_rate_by_layer": [round(sum(x["layer_rank"][i] <= 3 for x in a) / len(a), 3) for i in range(nl)],
                    "bm25_median_rank": sorted(x["bm25_rank"] or 10**6 for x in a)[len(a) // 2],
                    "bm25_top3_rate": round(sum((x["bm25_rank"] or 10**6) <= 3 for x in a) / len(a), 3),
                }
        summary[d.name] = {"meta": meta, "n_questions": len(rows), "cells": cells, "second_hop_rank": layer,
                           "score_s_mean": round(sum(r["score_s"] for r in rows) / len(rows), 3) if rows else None,
                           "n_analysed": len(an)}
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sizes", type=int, nargs="+", default=[0])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.25, 0.1, 0.05, 0.02])
    ap.add_argument("--policies", nargs="+", default=POLICIES)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only-pairs", action="store_true")
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--max-probe", type=int, default=384)
    ap.add_argument("--out", default="outputs/kv")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args(argv)
    tag = args.tag or Path(args.model).name
    out_dir = Path(args.out) / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        eng = KVEngine(args.model)
        for s in args.sizes:
            run_size(eng, s, args, out_dir)
    summary = summarize(out_dir)
    summary["_meta"] = {"model": tag, "budgets": args.budgets, "policies": args.policies}
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    print("wrote", out_dir / "summary.json")


if __name__ == "__main__":
    main()
