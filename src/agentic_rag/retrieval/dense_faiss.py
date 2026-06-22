"""Dense vector index over FAISS (roadmap P2). Vectors are L2-normalized so the inner
product on ``IndexFlatIP`` equals cosine similarity for any embedding backend.
"""

from __future__ import annotations

import json
from pathlib import Path

from langchain_core.embeddings import Embeddings

from ..log import get_logger
from ..schema import Document

logger = get_logger(__name__)

_FAISS_FILE = "dense.faiss"
_IDS_FILE = "dense_ids.json"


class DenseIndex:
    def __init__(self, embeddings: Embeddings):
        self.emb = embeddings
        self.index = None
        self.ids: list[str] = []

    def build(self, docs: list[Document]) -> "DenseIndex":
        import faiss
        import numpy as np

        self.ids = [d.doc_id for d in docs]
        mat = np.asarray(self.emb.embed_documents([d.content for d in docs]), dtype="float32")
        faiss.normalize_L2(mat)
        self.index = faiss.IndexFlatIP(mat.shape[1])
        self.index.add(mat)
        logger.info("DenseIndex: %d vectors x %d dims", len(self.ids), mat.shape[1])
        return self

    def search(self, query: str, topk: int) -> list[tuple[str, float]]:
        import faiss
        import numpy as np

        if not self.ids:
            return []
        q = np.asarray([self.emb.embed_query(query)], dtype="float32")
        faiss.normalize_L2(q)
        scores, idx = self.index.search(q, min(topk, len(self.ids)))
        return [(self.ids[i], float(s)) for s, i in zip(scores[0], idx[0]) if i >= 0]

    def save(self, index_dir: str | Path) -> None:
        import faiss

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / _FAISS_FILE))
        (index_dir / _IDS_FILE).write_text(json.dumps(self.ids, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, index_dir: str | Path, embeddings: Embeddings) -> "DenseIndex":
        import faiss

        index_dir = Path(index_dir)
        obj = cls(embeddings)
        obj.index = faiss.read_index(str(index_dir / _FAISS_FILE))
        obj.ids = json.loads((index_dir / _IDS_FILE).read_text(encoding="utf-8"))
        return obj
