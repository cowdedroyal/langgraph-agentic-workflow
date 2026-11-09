"""Query planner: pick an access path per question, like a database optimizer picks a plan.

Inputs available before answering:
  * question class and the data fields it needs, predicted by ONE cheap LLM call that sees
    only the view's catalog (column list and year range), never the gold labels:
      lookup     every entity it needs is named in the question
      join       it needs an entity that is *not* named (a supplier, a supplier's supplier)
      aggregate  it ranges over every company of a sector
    "covered" = every needed field is a column of the materialized view (query containment,
    as in answering queries using views).
  * corpus size in tokens (decides whether scan fits the window).

The policies are written from the mechanisms found in the single-path runs, not fitted to
the scores (fitting 4 sizes x 115 questions would just memorize the table):
  * top-k retrieval serves lookups (the question names the entity, so ranking works);
  * it cannot serve joins (the join key is not in the question) or aggregates (top-k cannot
    enumerate a sector) -> those go to a whole-corpus path;
  * the view only answers what its schema covers -> uncovered questions go to scan while
    the corpus fits, otherwise to the agent.

Workload: the 63 base questions (the view schema was designed for them) plus the 52
"oos" questions whose answers lie outside that schema.

Evaluation replays each chosen path's recorded answer. Paths run at temperature 0 with a
request cache, so a live planner would issue the same requests and get the same answers;
replay adds the classification call. Note that temperature 0 is not deterministic on the
gateway (see stability_*.json), so a 1-2 question difference between strategies is noise.

Run: ``python -m agentic_rag.access.planner --qsets base oos``.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .llm import LLM, MODEL, Usage
from .run import OUT, ROOT, model_tag

HOP_CLASS = {"cross_year": "lookup", "metric_change": "lookup",
             "supply_chain": "join", "deep_chain": "join", "aggregate": "aggregate",
             "oos_lookup": "lookup", "oos_report": "lookup", "oos_prior": "lookup",
             "oos_join": "join", "oos_aggregate": "aggregate"}
PATHS = ("scan", "index", "index_join", "view", "view_sql", "view_lazy", "agent")
VIEWS = ("view", "view_sql", "view_lazy")  # paths that need the one-off extraction build
VOLUMES = (115, 1000, 10000)  # questions served per corpus build; the view build is amortized over them
SCHEMA_FIELDS = ("company", "year", "sector", "revenue", "net_profit", "gross_margin", "supplier", "customer")

CLASSIFY = (
    "系统里有一张从资料库抽取出的事实表，只包含这些字段：company（公司简称）、year（年份，只有 2021–2023）、"
    "sector（主营行业）、revenue（营业收入）、net_profit（归母净利润）、gross_margin（销售毛利率）、"
    "supplier（供应商及供应物料）、customer（客户）。\n"
    "对下面的问题做两件事，只输出 JSON：{\"class\": \"lookup\" | \"join\" | \"aggregate\", \"fields\": [...]}\n"
    "1. class：\n"
    "- lookup：回答所需的公司都在问题里点名了。\n"
    "- join：需要先找到问题里没点名的另一家公司（例如某公司的供应商、客户、供应商的供应商）再查它的数据。\n"
    "- aggregate：要在一类公司的全体里比较或计数。\n"
    "2. fields：回答需要用到的字段，从上面的字段名里选；需要的信息不在这张表里的（包括表里没有的年份），"
    "加一项 \"OTHER\"。\n\n问题："
)
PARAPHRASE = (
    "把下面的问题换一种说法改写，意思和所需信息完全不变，不要提示问题类型，不要增删公司名。"
    "只输出改写后的问题本身。\n\n问题："
)


def _money(query_usd: float, n_q: int, build_usd: float, used_view: bool) -> dict:
    """Per-question cost split into the per-query part and the one-off view build, then
    amortized at several query volumes (assumes the same question mix as here)."""
    q = query_usd / n_q
    b = build_usd if used_view else 0.0
    return {"query_usd_per_q": round(q, 6), "build_usd": round(b, 4),
            "usd_per_q_at": {str(n): round(q + b / n, 6) for n in VOLUMES}}


# --------------------------------------------------------------------------- classification
def classify(llm: LLM, questions: list[str], workers: int = 6) -> list[tuple[dict, Usage]]:
    """-> ({"class": ..., "covered": bool, "fields": [...]}, usage) per question."""
    def one(q: str):
        total = Usage()
        for attempt in range(2):  # retry once with a doubled budget when the reply is truncated / unparsable
            r = llm.chat([{"role": "user", "content": CLASSIFY + q}], max_tokens=1500 * (2 if attempt else 1))
            total.add(r.usage)
            m = re.search(r"\{.*\}", r.text, re.S)
            try:
                d = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                d = {}
            cls = d.get("class") if d.get("class") in ("lookup", "join", "aggregate") else None
            fields = [str(f) for f in (d.get("fields") or [])]
            if cls and fields:
                break
        covered = bool(fields) and all(f in SCHEMA_FIELDS for f in fields)
        return {"class": cls, "covered": covered, "fields": fields}, total
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, questions))


def paraphrase(llm: LLM, questions: list[str], workers: int = 6) -> list[str]:
    def one(q: str):
        return llm.chat([{"role": "user", "content": PARAPHRASE + q}], max_tokens=800).text.strip()
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(one, questions))


# --------------------------------------------------------------------------- policies
# A policy maps (prediction, scan fits?) -> ordered list of paths. More than one path is a
# cascade: escalate to the next path when the previous one answers null.
def _whole(fits):
    """A path that reads the raw corpus: scan while it fits, otherwise the agent."""
    return "scan" if fits else "agent"


def p_scan_or_view(c, fits):
    """Schema-blind baseline from the base study: lookup -> index; others -> scan if it fits, else view."""
    return ["index"] if c["class"] == "lookup" else (["scan"] if fits else ["view"])


def p_sql_first(c, fits):
    """Schema-blind: lookup -> index; join/aggregate -> view_sql."""
    return ["index"] if c["class"] == "lookup" else ["view_sql"]


def p_sql_escalate(c, fits):
    """Schema-blind, but escalate when view_sql answers null (it found nothing)."""
    return ["index"] if c["class"] == "lookup" else ["view_sql", _whole(fits)]


def p_coverage(c, fits):
    """Coverage-aware: lookup -> index; covered join/aggregate -> view_sql; uncovered -> raw corpus."""
    if c["class"] == "lookup":
        return ["index"]
    return ["view_sql"] if c["covered"] else [_whole(fits)]


def p_coverage_escalate(c, fits):
    """Coverage-aware, plus escalation from view_sql on null."""
    if c["class"] == "lookup":
        return ["index"]
    return ["view_sql", _whole(fits)] if c["covered"] else [_whole(fits)]


def p_coverage_all(c, fits):
    """Coverage decides everything: covered -> view_sql (even lookups); uncovered lookup -> index;
    uncovered join/aggregate -> raw corpus."""
    if c["covered"]:
        return ["view_sql"]
    return ["index"] if c["class"] == "lookup" else [_whole(fits)]


def p_coverage_lazy(c, fits):
    """Coverage-aware with lazy materialization: uncovered join/aggregate -> scan while it fits,
    otherwise extract the missing attribute on demand (view_lazy)."""
    if c["class"] == "lookup":
        return ["index"]
    if c["covered"]:
        return ["view_sql"]
    return ["scan"] if fits else ["view_lazy"]


def p_coverage_lazy_always(c, fits):
    """Same, but uncovered join/aggregate always go to view_lazy (never pay for a full scan)."""
    if c["class"] == "lookup":
        return ["index"]
    return ["view_sql"] if c["covered"] else ["view_lazy"]


def p_posthoc_uncovered_lookup_agent(c, fits):
    """POST-HOC (added after seeing that index loses the financial-table chunk on uncovered
    lookups, finding 7): like coverage_lazy, but uncovered lookups go to the agent."""
    if c["class"] == "lookup":
        return ["index"] if c["covered"] else ["agent"]
    if c["covered"]:
        return ["view_sql"]
    return ["scan"] if fits else ["view_lazy"]


POLICIES = {"scan_or_view": p_scan_or_view, "sql_first": p_sql_first, "sql_escalate": p_sql_escalate,
            "coverage": p_coverage, "coverage_escalate": p_coverage_escalate, "coverage_all": p_coverage_all,
            "coverage_lazy": p_coverage_lazy, "coverage_lazy_always": p_coverage_lazy_always,
            "posthoc_uncovered_lookup_agent": p_posthoc_uncovered_lookup_agent}


# --------------------------------------------------------------------------- replay
def load_records(dirs: list[Path]) -> dict[str, dict[str, dict]]:
    """Merge records of several question sets: {path: {qid: record}}. A path may cover only some
    question sets (view_lazy is only run on the oos set: covered questions never reach it)."""
    per = []
    for d in dirs:
        recs = {}
        for p in PATHS:
            f = d / f"records_{p}.jsonl"
            if f.exists():
                recs[p] = {r["qid"]: r for r in map(json.loads, f.read_text().splitlines())}
        per.append(recs)
    present = set().union(*(set(r) for r in per)) if per else set()
    return {p: {k: v for r in per if p in r for k, v in r[p].items()} for p in PATHS if p in present}


def _cost(r: dict) -> tuple[float, float]:
    u = r.get("usage") or {}
    return u.get("usd", 0.0), u.get("latency_s", 0.0)


def evaluate(plan_fn, qids, pred, cls_usage, recs, fits, build_usd, n_q, hop=None):
    """Replay a policy. Returns accuracy (overall and per hop type), per-query $, $ amortized
    at several volumes, and latency (sum over every call on the chosen chain)."""
    correct, usd, lat, used_view, chosen = 0, 0.0, [], False, {}
    ok_by_hop: dict[str, list[bool]] = {}
    for qid in qids:
        chain = [p for p in plan_fn(pred[qid], fits) if p in recs and qid in recs[p]]
        if not chain:
            chain = ["scan" if fits and "scan" in recs else "agent"]
        q_usd, q_lat, ok = cls_usage[qid].usd, cls_usage[qid].latency_s, False
        taken = []
        for p in chain:
            r = recs[p][qid]
            c, l = _cost(r)
            q_usd, q_lat = q_usd + c, q_lat + l
            taken.append(p)
            used_view |= p in VIEWS
            ok = bool(r["correct"])
            if r.get("answer") is not None:
                break
        correct += ok
        usd += q_usd
        lat.append(q_lat)
        chosen[qid] = taken
        if hop:
            ok_by_hop.setdefault(hop[qid], []).append(ok)
    lat.sort()
    out = {"accuracy": round(correct / len(qids), 4), **_money(usd, n_q, build_usd, used_view),
           "mean_latency_s": round(sum(lat) / len(lat), 2), "p50_latency_s": round(lat[len(lat) // 2], 2),
           "paths_used": {p: sum(p in v for v in chosen.values()) for p in PATHS if any(p in v for v in chosen.values())}}
    if hop:
        out["per_hop"] = {h: round(sum(v) / len(v), 4) for h, v in sorted(ok_by_hop.items())}
    return out


def single(recs, p, qids, build_usd, n_q, hop=None):
    rs = [recs[p][q] for q in qids]
    usd = sum(_cost(r)[0] for r in rs)
    lat = sorted(_cost(r)[1] for r in rs)
    out = {"accuracy": round(sum(r["correct"] for r in rs) / len(rs), 4), **_money(usd, n_q, build_usd, p in VIEWS),
           "mean_latency_s": round(sum(lat) / len(lat), 2), "p50_latency_s": round(lat[len(lat) // 2], 2)}
    if hop:
        by: dict[str, list[bool]] = {}
        for q, r in zip(qids, rs):
            by.setdefault(hop[q], []).append(bool(r["correct"]))
        out["per_hop"] = {h: round(sum(v) / len(v), 4) for h, v in sorted(by.items())}
    return out


def oracle(recs, qids, build_usd, n_q):
    """Per question, the path with the lowest per-query cost that answered correctly (hindsight
    upper bound on accuracy; the view build is then charged once if any question uses it)."""
    correct, usd, lat, used_view = 0, 0.0, [], False
    for q in qids:
        ok = [(p, *_cost(recs[p][q])) for p in recs if q in recs[p] and recs[p][q]["correct"]]
        if ok:
            p, c, l = min(ok, key=lambda x: (x[1], x[2]))
            correct, usd, used_view = correct + 1, usd + c, used_view or p in VIEWS
            lat.append(l)
        else:  # nobody got it: charge the cheapest attempt
            c, l = min(_cost(recs[p][q]) for p in recs if q in recs[p])
            usd += c
            lat.append(l)
    lat.sort()
    return {"accuracy": round(correct / len(qids), 4), **_money(usd, n_q, build_usd, used_view),
            "mean_latency_s": round(sum(lat) / len(lat), 2), "p50_latency_s": round(lat[len(lat) // 2], 2)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=None, help="model tag (default from the model name)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--qsets", nargs="+", default=["base", "oos"])
    ap.add_argument("--no-paraphrase", action="store_true")
    ap.add_argument("--sizes", nargs="*", default=None, help="default: every size present for all qsets")
    args = ap.parse_args(argv)
    model = args.model or MODEL
    tag = args.tag or model_tag(model)
    llm = LLM(model=model, cache_dir=OUT / "cache")
    outs = [OUT / (tag if q == "base" else f"{tag}-{q}") for q in args.qsets]
    summaries = [json.loads((o / "summary.json").read_text()) for o in outs]
    sizes = args.sizes or sorted(set.intersection(*({k for k in s if not k.startswith("_")} for s in summaries)),
                                 key=int)

    qsets = {s: [json.loads(l) for o in outs for l in (o / s / "questions.jsonl").read_text().splitlines()]
             for s in sizes}
    texts = sorted({q["question"] for qs in qsets.values() for q in qs})
    gold = {}
    for qs in qsets.values():
        for q in qs:
            gold[q["question"]] = {"class": HOP_CLASS[q["hop_type"]], "covered": not q["hop_type"].startswith("oos")}

    report = {"model": model_tag(model), "qsets": args.qsets, "classifier": {}, "sizes": {}}
    variants = {"original": texts}
    if not args.no_paraphrase:
        variants["paraphrased"] = paraphrase(llm, texts)
    preds = {}
    for name, qs in variants.items():
        res = classify(llm, qs)
        preds[name] = dict(zip(texts, res))
        conf_cls, conf_cov = {}, {}
        for t, (c, _) in zip(texts, res):
            k = f"{gold[t]['class']}->{c['class']}"
            conf_cls[k] = conf_cls.get(k, 0) + 1
            k = f"{'covered' if gold[t]['covered'] else 'uncovered'}->{'covered' if c['covered'] else 'uncovered'}"
            conf_cov[k] = conf_cov.get(k, 0) + 1
        tot = Usage()
        for _, u in res:
            tot.add(u)
        report["classifier"][name] = {
            "n": len(texts),
            "class_accuracy": round(sum(c["class"] == gold[t]["class"] for t, (c, _) in zip(texts, res)) / len(texts), 4),
            "coverage_accuracy": round(sum(c["covered"] == gold[t]["covered"] for t, (c, _) in zip(texts, res)) / len(texts), 4),
            "class_confusion": conf_cls, "coverage_confusion": conf_cov,
            "usd_per_q": round(tot.usd / len(texts), 7),
            "coverage_errors": [{"q": t, "fields": c["fields"], "gold_covered": gold[t]["covered"]}
                                for t, (c, _) in zip(texts, res) if c["covered"] != gold[t]["covered"]][:20]}
        if name == "paraphrased":
            report["classifier"][name]["examples"] = [
                {"original": t, "paraphrased": p} for t, p in list(zip(texts, qs))[::12]]

    for s in sizes:
        recs = load_records([o / s for o in outs])
        qs = qsets[s]
        qids = [q["qid"] for q in qs]
        if not recs or not all(q in recs.get(p, {}) for p in ("index", "view_sql") for q in qids):
            print(f"size {s}: records incomplete, skipped")
            continue
        hop = {q["qid"]: q["hop_type"] for q in qs}
        n_q = len(qids)
        e0 = summaries[0][s]
        fits = "scan" in recs and not e0["paths"].get("scan", {}).get("infeasible")
        build_usd = max(((sm[s]["paths"].get(v, {}).get("build_usage") or {}).get("usd", 0.0)
                         for sm in summaries for v in VIEWS), default=0.0)  # same extraction, built once
        rows = {"single": {p: single(recs, p, qids, build_usd, n_q, hop) for p in recs
                           if all(q in recs[p] for q in qids)},
                "oracle": oracle(recs, qids, build_usd, n_q), "policies": {}}
        for vname, cmap in preds.items():
            pred = {q["qid"]: cmap[q["question"]][0] for q in qs}
            cu = {q["qid"]: cmap[q["question"]][1] for q in qs}
            for pname, fn in POLICIES.items():
                rows["policies"][f"{pname}@{vname}"] = evaluate(fn, qids, pred, cu, recs, fits, build_usd, n_q, hop)
        gpred = {q["qid"]: gold[q["question"]] for q in qs}
        zero = {q: Usage() for q in qids}
        for pname, fn in POLICIES.items():
            rows["policies"][f"{pname}@gold"] = evaluate(fn, qids, gpred, zero, recs, fits, build_usd, n_q, hop)
        report["sizes"][s] = {"companies": e0["companies"], "chars": e0["chars"], "n_questions": n_q,
                              "scan_fits": fits, "view_build_usd": round(build_usd, 4), **rows}

    name = "planner_" + tag + ("" if args.qsets == ["base"] else "_" + "+".join(args.qsets))
    (ROOT / "docs" / "metrics" / f"{name}.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    write_table(report, name)


def write_table(rep: dict, name: str) -> None:
    L = [f"# Query planner vs single access paths ({name})", "",
         "Generated by `python -m agentic_rag.access.planner`. Question sets: " + " + ".join(rep["qsets"])
         + ". Per-query $ includes the classification call. `$/q @N` adds the one-off view build amortized "
           "over N questions served from the same corpus (same question mix), for strategies that use the view. "
           "Latency is the sum over every call on the chosen chain (classification included). "
           "Prices: deepseek-v4-flash list. Temperature 0 is not deterministic on this gateway: "
           "differences of 1-2 questions are noise.", "",
         "## Classifier", "",
         "| question text | n | class acc | coverage acc | coverage confusion (gold->pred) | $/q |",
         "|---|---|---|---|---|---|"]
    for k, v in rep["classifier"].items():
        L.append(f"| {k} | {v['n']} | {v['class_accuracy']:.3f} | {v['coverage_accuracy']:.3f} | "
                 f"{', '.join(f'{a}: {b}' for a, b in sorted(v['coverage_confusion'].items()))} | {v['usd_per_q']:.6f} |")
    for s, r in rep["sizes"].items():
        hops = sorted(next(iter(r["single"].values()))["per_hop"])
        L += ["", f"## {s} distractors ({r['companies']} companies, {r['chars']} chars, {r['n_questions']} questions, "
                  f"scan fits: {r['scan_fits']}, view build ${r['view_build_usd']})", "",
              "| strategy | accuracy | " + " | ".join(hops) + " | query $/q | "
              + " | ".join(f"$/q @{n}" for n in VOLUMES) + " | mean s | p50 s | paths used |",
              "|---|---|" + "---|" * len(hops) + "---|" + "---|" * len(VOLUMES) + "---|---|---|"]

        def row(label, v, used=""):
            ph = " | ".join(f"{v['per_hop'][h]:.2f}" for h in hops) if "per_hop" in v else " | ".join("" for _ in hops)
            at = " | ".join(f"{v['usd_per_q_at'][str(n)]:.5f}" for n in VOLUMES)
            return (f"| {label} | {v['accuracy']:.3f} | {ph} | {v['query_usd_per_q']:.5f} | {at} | "
                    f"{v['mean_latency_s']} | {v['p50_latency_s']} | {used} |")
        for p, v in r["single"].items():
            L.append(row(f"single: {p}", v))
        for p, v in r["policies"].items():
            L.append(row(f"planner: {p}", v, ", ".join(f"{a} {b}" for a, b in v["paths_used"].items())))
        L.append(row("oracle (hindsight)", r["oracle"]))
    p = ROOT / "docs" / "metrics" / f"{name}.md"
    p.write_text("\n".join(L) + "\n")
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
