"""Retrieval for the access-path study, built directly on the raw documents of a world.

Reuses the project's chunker, BM25 tokenizer, RRF fusion and bge embeddings, so ``index``
in this study is the same hybrid retriever the original pipeline used, just rebuilt per
corpus size. Embeddings are cached on disk per chunk text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..ingestion.chunking import chunk_section
from ..retrieval.base import rrf_fuse
from ..retrieval.sparse_bm25 import tokenize
from .world import RawDoc


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def chunk_docs(docs: list[RawDoc], max_chars: int = 600, overlap: int = 80) -> list[Chunk]:
    """Split on headings / blank lines like the original pipeline and prefix each chunk with
    its document title line, so every chunk names its company and year."""
    out: list[Chunk] = []
    for d in docs:
        lines = [ln for ln in d.text.splitlines() if not ln.startswith("<!-- META")]
        title = next((ln.strip("# ").strip() for ln in lines if ln.strip()), d.doc_id)
        sections, cur, head = [], [], ""
        for ln in lines:
            if ln.startswith("## ") or ln.startswith("<h2>") or ln.startswith("【"):
                if cur:
                    sections.append((head, "\n".join(cur)))
                head, cur = ln, [ln]
            else:
                cur.append(ln)
        if cur:
            sections.append((head, "\n".join(cur)))
        i = 0
        for head, body in sections:
            for piece in chunk_section(head, body, max_chars, overlap):
                out.append(Chunk(f"{d.doc_id}#{i}", d.doc_id, f"[{title}]\n{piece}"))
                i += 1
    return out


class _Embedder:
    def __init__(self, cache_dir: Path, model: str = "BAAI/bge-base-zh-v1.5"):
        self.cache_dir = cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model
        self._model = None
        self.query_instruction = "为这个句子生成表示以用于检索相关文章："

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def docs(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha1(t.encode()).hexdigest() for t in texts]
        store = self.cache_dir / "emb.npz"
        cache = dict(np.load(store)) if store.exists() else {}
        todo = [(k, t) for k, t in zip(keys, texts) if k not in cache]
        if todo:
            vecs = self.model.encode([t for _, t in todo], batch_size=32, normalize_embeddings=True,
                                     show_progress_bar=len(todo) > 500)
            for (k, _), v in zip(todo, vecs):
                cache[k] = v.astype("float32")
            np.savez(store, **cache)
        return np.stack([cache[k] for k in keys])

    def query(self, q: str) -> np.ndarray:
        return self.model.encode([self.query_instruction + q], normalize_embeddings=True)[0]


class HybridIndex:
    """BM25 (jieba) + bge dense over chunks, fused by RRF (k=60)."""

    def __init__(self, chunks: list[Chunk], cache_dir: Path, dense: bool = True):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.bm25 = BM25Okapi([tokenize(c.text) for c in chunks])
        self.emb = _Embedder(cache_dir) if dense else None
        self.mat = self.emb.docs([c.text for c in chunks]) if dense else None

    def search(self, query: str, topk: int, method: str = "hybrid") -> list[Chunk]:
        n = max(topk * 5, 40)
        lists = []
        if method in ("bm25", "hybrid"):
            s = self.bm25.get_scores(tokenize(query))
            order = np.argsort(s)[::-1][:n]
            lists.append([(self.chunks[i].chunk_id, float(s[i])) for i in order if s[i] > 0])
        if method in ("dense", "hybrid") and self.mat is not None:
            s = self.mat @ self.emb.query(query)
            order = np.argsort(s)[::-1][:n]
            lists.append([(self.chunks[i].chunk_id, float(s[i])) for i in order])
        fused = rrf_fuse(lists, k=60, topk=topk) if len(lists) > 1 else lists[0][:topk]
        by_id = {c.chunk_id: c for c in self.chunks}
        return [by_id[i] for i, _ in fused]


class DocSearch:
    """BM25 over whole documents, used as the agent's ``search`` tool."""

    def __init__(self, docs: list[RawDoc]):
        from rank_bm25 import BM25Okapi

        self.docs = docs
        self.bm25 = BM25Okapi([tokenize(d.text) for d in docs])

    def __call__(self, query: str, topk: int) -> list[RawDoc]:
        s = self.bm25.get_scores(tokenize(query))
        order = np.argsort(s)[::-1][:topk]
        return [self.docs[i] for i in order if s[i] > 0]
