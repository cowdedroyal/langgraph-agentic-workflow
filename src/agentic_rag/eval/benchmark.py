"""Benchmark harness (roadmap P4).

Two reports, both writing real numbers to docs/results.md:
  * retrieval — compares dense / BM25 / hybrid(/+rerank) on Recall@k, MRR, nDCG, multi-hop Hit@k;
  * end-to-end — runs the agentic graph and records answer accuracy, retries, latency,
    citation coverage (+ optional LLM-judge faithfulness / relevance).
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from statistics import mean

from ..config import Settings, get_settings
from ..log import get_logger
from .generation_metrics import answer_accuracy, citation_coverage
from .judge import LLMJudge
from .retrieval_metrics import aggregate, hit_at_k, mrr, ndcg_at_k, recall_at_k

logger = get_logger(__name__)

_METHOD_LABEL = {
    "dense": "Dense (FAISS · bge-zh)",
    "bm25": "BM25 (jieba)",
    "hybrid": "Hybrid (RRF)",
    "hybrid+rerank": "Hybrid + Reranker",
}


def run_retrieval_benchmark(settings: Settings, qas, methods=("dense", "bm25", "hybrid"),
                            ks=(5, 10), with_rerank: bool = False) -> list[dict]:
    from ..retrieval import load_retriever

    retr = load_retriever(settings)
    maxk = max(ks)
    configs = [(m, False) for m in methods] + ([("hybrid", True)] if with_rerank else [])
    rows = []
    for method, rr in configs:
        per = []
        for qa in qas:
            ids = [h.document.doc_id for h in
                   retr.search(qa.question, topk=maxk, method=method, use_reranker=rr)]
            row = {f"Recall@{k}": recall_at_k(ids, qa.evidence_doc_ids, k) for k in ks}
            row["MRR"] = mrr(ids, qa.evidence_doc_ids)
            row[f"nDCG@{ks[0]}"] = ndcg_at_k(ids, qa.evidence_doc_ids, ks[0])
            row[f"Hit@{ks[0]}"] = hit_at_k(ids, qa.evidence_doc_ids, ks[0])
            per.append(row)
        label = "hybrid+rerank" if rr else method
        rows.append({"method": _METHOD_LABEL.get(label, label), **aggregate(per)})
        logger.info("retrieval[%s]: %s", label, rows[-1])
    return rows


def run_end_to_end_benchmark(settings: Settings, qas, limit: int = 20, judge: bool = False) -> dict | None:
    try:
        from ..graph.build import build_graph
        from ..ingestion.pipeline import load_documents
    except Exception as e:  # graph not ready yet
        logger.warning("Graph unavailable (%s); skipping end-to-end benchmark.", e)
        return None

    graph = build_graph(settings)
    docs = {d.doc_id: d for d in load_documents(settings)}
    jd = LLMJudge(settings) if judge else None

    accs, retries, lats, cites, faiths, rels = [], [], [], [], [], []
    for qa in qas[:limit]:
        t0 = time.time()
        try:
            state = graph.invoke({"question": qa.question}, {"recursion_limit": settings.recursion_limit})
        except Exception as e:
            logger.debug("graph invoke failed: %s", e)
            state = {}
        dt = time.time() - t0
        gen = state.get("generation") or ""
        ev_texts = [docs[e].content for e in qa.evidence_doc_ids if e in docs]
        accs.append(answer_accuracy(qa.answer, gen))
        retries.append(float(state.get("loop_step", 0)))
        lats.append(dt)
        cites.append(citation_coverage(gen, ev_texts))
        if jd:
            faiths.append(1.0 if jd.faithfulness("\n\n".join(ev_texts), gen) else 0.0)
            rels.append(1.0 if jd.answer_relevance(qa.question, gen) else 0.0)

    if not accs:
        return None
    summary = {
        "n": len(accs),
        "answer_accuracy": round(mean(accs), 4),
        "citation_coverage": round(mean(cites), 4),
        "avg_retries": round(mean(retries), 3),
        "avg_latency_s": round(mean(lats), 3),
    }
    if jd:
        summary["faithfulness"] = round(mean(faiths), 4)
        summary["answer_relevance"] = round(mean(rels), 4)
    logger.info("end-to-end: %s", summary)
    return summary


# --------------------------------------------------------------------------- #
def _fmt(v) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def _md_table(rows: list[dict], cols: list[str]) -> str:
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(_fmt(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines)


def render_results_md(settings: Settings, retrieval_rows: list[dict], ks,
                      e2e: dict | None, n_qa: int) -> str:
    cols = ["method"] + [f"Recall@{k}" for k in ks] + ["MRR", f"nDCG@{ks[0]}", f"Hit@{ks[0]}"]
    llm_name = settings.vllm_model if settings.llm_backend in ("vllm", "openai") else settings.local_llm
    out = [
        "# Benchmark Results",
        "",
        f"_Generated {datetime.now():%Y-%m-%d %H:%M} · {n_qa} multi-hop QA · "
        f"embeddings={settings.embeddings_backend} · llm={settings.llm_backend}:{llm_name}_",
        "",
        "## 1. Retrieval (roadmap P2)",
        "",
        f"Ground truth = synthesized `evidence_doc_ids`. `Hit@{ks[0]}` requires the *full* "
        "multi-hop evidence chain in the top-k.",
        "",
        _md_table(retrieval_rows, cols),
        "",
    ]
    if e2e:
        out += [
            "## 2. End-to-end Agentic RAG (roadmap P0/P4)",
            "",
            _md_table([e2e], list(e2e.keys())),
            "",
            "`avg_retries` = mean self-correction loops; lower latency/retries with stable answers is better.",
            "",
        ]
    return "\n".join(out)


def run_benchmark(settings: Settings | None = None, mode: str = "all", limit: int = 20,
                  with_rerank: bool = False, judge: bool = False, ks=(5, 10)) -> dict:
    from ..synthesis.pipeline import load_qa

    settings = (settings or get_settings()).ensure_dirs()
    qas = load_qa(settings)
    if not qas:
        raise RuntimeError(f"No QA at {settings.qa_path}; run `arag corpus` first.")

    retrieval_rows = run_retrieval_benchmark(settings, qas, ks=ks, with_rerank=with_rerank) \
        if mode in ("all", "retrieval") else []
    e2e = run_end_to_end_benchmark(settings, qas, limit=limit, judge=judge) \
        if mode in ("all", "e2e") else None

    md = render_results_md(settings, retrieval_rows, ks, e2e, len(qas))
    results_path = settings.project_root / "docs" / "results.md"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(md, encoding="utf-8")
    logger.info("Results written -> %s", results_path)
    return {"retrieval": retrieval_rows, "e2e": e2e, "results_path": str(results_path)}
