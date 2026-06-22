"""Retrieval metrics (roadmap P4): Recall@k / MRR / nDCG / multi-hop Hit@k.

Ground truth is the synthesized ``evidence_doc_ids`` (a set per QA). For multi-hop QA,
``hit_at_k`` requires *all* evidence chunks to appear in the top-k — a strict measure of
whether the retriever surfaced the full reasoning chain.
"""

from __future__ import annotations

import math
from statistics import mean


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    top = retrieved[:k]
    return len([d for d in top if d in rel]) / len(rel)


def hit_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """1.0 only if every relevant doc is within the top-k (full chain recovered)."""
    return 1.0 if set(relevant) <= set(retrieved[:k]) else 0.0


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    rel = set(relevant)
    for i, d in enumerate(retrieved):
        if d in rel:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    dcg = sum(1.0 / math.log2(i + 2) for i, d in enumerate(retrieved[:k]) if d in rel)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg else 0.0


def aggregate(per_query: list[dict]) -> dict:
    """Average a list of per-query metric dicts."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: round(mean(q[key] for q in per_query), 4) for key in keys}
