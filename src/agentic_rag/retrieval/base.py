"""Retrieval primitives shared by dense / sparse / hybrid (roadmap P2).

* ``rrf_fuse`` — Reciprocal Rank Fusion over multiple ranked lists.
* ``match_filters`` — metadata pre/post filter (company / period / doc_type ...).
"""

from __future__ import annotations

from collections import defaultdict

from ..schema import Document


def rrf_fuse(rankings: list[list[tuple[str, float]]], k: int = 60, topk: int | None = None
             ) -> list[tuple[str, float]]:
    """Fuse ranked (doc_id, score) lists by RRF: sum 1/(k + rank). Rank-based, so the
    very different score scales of dense (cosine) and BM25 (tf-idf) combine cleanly."""
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking):
            fused[doc_id] += 1.0 / (k + rank + 1)
    out = sorted(fused.items(), key=lambda x: x[1], reverse=True)
    return out[:topk] if topk else out


def match_filters(doc: Document, filters: dict | None) -> bool:
    if not filters:
        return True
    md = doc.metadata
    for key, want in filters.items():
        if want is None:
            continue
        val = getattr(md, key, None)
        if hasattr(val, "value"):  # enum -> str
            val = val.value
        if isinstance(want, (list, tuple, set)):
            if val not in want:
                return False
        elif val != want:
            return False
    return True
