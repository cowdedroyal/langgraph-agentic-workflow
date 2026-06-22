"""Sparse BM25 index with jieba tokenization (roadmap P2). Chinese text needs word
segmentation before BM25; numbers/percent tokens are kept since they matter for finance.
"""

from __future__ import annotations

import pickle
from pathlib import Path

from ..log import get_logger
from ..schema import Document

logger = get_logger(__name__)

_BM25_FILE = "bm25.pkl"

_PUNCT = set("，。；、！？：（）()【】[]{}<>《》“”‘’\"'`~!@#$%^&*-_=+|\\/.,?:; \t\n　")


def tokenize(text: str) -> list[str]:
    import jieba

    return [t for t in (tok.strip().lower() for tok in jieba.lcut(text)) if t and t not in _PUNCT]


class BM25Index:
    def __init__(self):
        self.bm25 = None
        self.ids: list[str] = []

    def build(self, docs: list[Document]) -> "BM25Index":
        from rank_bm25 import BM25Okapi

        self.ids = [d.doc_id for d in docs]
        corpus_tokens = [tokenize(d.content) for d in docs]
        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info("BM25Index: %d documents", len(self.ids))
        return self

    def search(self, query: str, topk: int) -> list[tuple[str, float]]:
        import numpy as np

        if not self.ids:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        order = np.argsort(scores)[::-1][:topk]
        return [(self.ids[i], float(scores[i])) for i in order if scores[i] > 0]

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / _BM25_FILE).open("wb") as f:
            pickle.dump({"bm25": self.bm25, "ids": self.ids}, f)

    @classmethod
    def load(cls, index_dir: str | Path) -> "BM25Index":
        with (Path(index_dir) / _BM25_FILE).open("rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.bm25 = data["bm25"]
        obj.ids = data["ids"]
        return obj
