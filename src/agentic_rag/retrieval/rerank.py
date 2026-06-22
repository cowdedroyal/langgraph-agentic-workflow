"""Optional cross-encoder reranker (roadmap P2). Disabled by default (USE_RERANKER);
when on, re-scores the fused candidates with BAAI/bge-reranker-base for precision.
"""

from __future__ import annotations

from ..log import get_logger
from ..schema import Document

logger = get_logger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device)
        logger.info("Reranker: %s on %s", model_name, device)

    def rerank(self, query: str, docs: list[Document], topk: int) -> list[tuple[Document, float]]:
        import numpy as np

        if not docs:
            return []
        scores = self.model.predict([(query, d.content) for d in docs])
        order = np.argsort(scores)[::-1][:topk]
        return [(docs[i], float(scores[i])) for i in order]


def maybe_build_reranker(settings) -> "CrossEncoderReranker | None":
    if not settings.use_reranker:
        return None
    try:
        return CrossEncoderReranker(settings.reranker_model, settings.resolve_device())
    except Exception as e:  # pragma: no cover - optional download
        logger.warning("Reranker unavailable (%s); continuing without rerank.", e)
        return None
