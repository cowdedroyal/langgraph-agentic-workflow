"""Hybrid retriever (roadmap P2): dense (FAISS) + sparse (BM25) fused by RRF, with an
optional metadata filter and cross-encoder rerank.

Crucially the public ``invoke(question)`` keeps the *same* contract the original
LangGraph node expected (``retriever.invoke(question) -> list[langchain Document]``),
so swapping the baseline ``SKLearnVectorStore`` for this hybrid index required no
change to the graph nodes — only the implementation behind the interface.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings, get_settings
from ..embeddings import create_embedding_model
from ..log import get_logger
from ..schema import Document, RetrievedDoc
from .base import match_filters, rrf_fuse
from .dense_faiss import DenseIndex
from .rerank import maybe_build_reranker
from .sparse_bm25 import BM25Index

logger = get_logger(__name__)


def build_indices(settings: Settings | None = None) -> dict:
    """Build + persist the dense and sparse indices from documents.jsonl."""
    from ..ingestion.pipeline import load_documents

    settings = (settings or get_settings()).ensure_dirs()
    docs = load_documents(settings)
    if not docs:
        raise RuntimeError(f"No documents at {settings.corpus_path}; run `arag corpus` first.")
    emb = create_embedding_model(settings)
    DenseIndex(emb).build(docs).save(settings.index_dir)
    BM25Index().build(docs).save(settings.index_dir)
    logger.info("Indices built for %d docs -> %s", len(docs), settings.index_dir)
    return {"docs": len(docs), "index_dir": str(settings.index_dir)}


class HybridRetriever:
    def __init__(self, settings: Settings, id2doc: dict[str, Document],
                 dense: DenseIndex, bm25: BM25Index, reranker=None):
        self.settings = settings
        self.id2doc = id2doc
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker

    # -- construction --
    @classmethod
    def load(cls, settings: Settings | None = None) -> "HybridRetriever":
        from ..ingestion.pipeline import load_documents

        settings = settings or get_settings()
        docs = load_documents(settings)
        id2doc = {d.doc_id: d for d in docs}
        emb = create_embedding_model(settings)
        dense = DenseIndex.load(settings.index_dir, emb)
        bm25 = BM25Index.load(settings.index_dir)
        return cls(settings, id2doc, dense, bm25, maybe_build_reranker(settings))

    # -- core search; ``method`` lets the benchmark compare retrievers --
    def search(self, query: str, topk: int | None = None, filters: dict | None = None,
               method: str = "hybrid", use_reranker: bool | None = None) -> list[RetrievedDoc]:
        s = self.settings
        topk = topk or s.retrieval_topk
        use_reranker = s.use_reranker if use_reranker is None else use_reranker

        # When a metadata filter is active, probe wider so filtering does not discard
        # relevant docs that fell outside the default top-N (corpus is small).
        nd, nb = s.dense_topk, s.bm25_topk
        if filters:
            nd = max(nd, len(self.dense.ids))
            nb = max(nb, len(self.bm25.ids))

        dense_hits = self.dense.search(query, nd) if method in ("dense", "hybrid") else []
        bm25_hits = self.bm25.search(query, nb) if method in ("bm25", "hybrid") else []

        if method == "dense":
            fused = dense_hits
        elif method == "bm25":
            fused = bm25_hits
        else:
            fused = rrf_fuse([dense_hits, bm25_hits], k=s.rrf_k)

        # metadata filter
        fused = [(i, sc) for i, sc in fused if i in self.id2doc and match_filters(self.id2doc[i], filters)]

        if use_reranker and self.reranker is not None:
            pool = [self.id2doc[i] for i, _ in fused[: max(topk * 4, 20)]]
            reranked = self.reranker.rerank(query, pool, topk)
            return [RetrievedDoc(document=d, score=sc, rank=r, retriever="rerank")
                    for r, (d, sc) in enumerate(reranked)]

        return [RetrievedDoc(document=self.id2doc[i], score=sc, rank=r, retriever=method)
                for r, (i, sc) in enumerate(fused[:topk])]

    # -- LangGraph-compatible interface (unchanged contract) --
    def invoke(self, question: str, filters: dict | None = None):
        return [rd.document.to_langchain() for rd in self.search(question, filters=filters)]


def load_retriever(settings: Settings | None = None) -> HybridRetriever:
    return HybridRetriever.load(settings)
