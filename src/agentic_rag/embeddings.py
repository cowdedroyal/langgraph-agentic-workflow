"""Embedding factory with a three-tier fallback (roadmap baseline + P0):

    bge (sentence-transformers, local BAAI/bge-base-zh-v1.5)  ->  nomic (local)  ->  hashing

The hashing tier guarantees the pipeline (and smoke tests) run even with no model
files or system deps available, while bge gives real Chinese-financial retrieval quality.
All tiers implement the langchain ``Embeddings`` interface so the vector store is agnostic.
"""

from __future__ import annotations

from typing import List

from langchain_core.embeddings import Embeddings

from .config import Settings, get_settings
from .log import get_logger

logger = get_logger(__name__)


class BGEEmbeddings(Embeddings):
    """sentence-transformers BAAI/bge-base-zh-v1.5 with the recommended query instruction."""

    def __init__(self, model_name: str, device: str, batch_size: int, query_instruction: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.query_instruction = query_instruction

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        # bge-zh retrieval benefits from prefixing the query with an instruction.
        emb = self.model.encode(
            self.query_instruction + text,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return emb.tolist()


class HashingVectorizerEmbeddings(Embeddings):
    """Lightweight, dependency-free fallback (carried over from the original baseline)."""

    def __init__(self, n_features: int = 2048):
        from sklearn.feature_extraction.text import HashingVectorizer

        self.vectorizer = HashingVectorizer(n_features=n_features, alternate_sign=False, norm="l2")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.vectorizer.transform(list(texts)).toarray().tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


def create_embedding_model(settings: Settings | None = None) -> Embeddings:
    """Build the embedding model selected by ``settings.embeddings_backend`` with fallback."""
    settings = settings or get_settings()
    backend = settings.embeddings_backend.lower()

    if backend == "hashing":
        logger.info("Embeddings: HashingVectorizer (%d dims)", settings.hashing_features)
        return HashingVectorizerEmbeddings(settings.hashing_features)

    if backend == "nomic":
        try:
            from langchain_nomic.embeddings import NomicEmbeddings

            return NomicEmbeddings(model="nomic-embed-text-v1.5", inference_mode="local")
        except Exception as e:  # pragma: no cover - depends on optional system deps
            logger.warning("Nomic embeddings unavailable (%s). Falling back to hashing.", e)
            return HashingVectorizerEmbeddings(settings.hashing_features)

    # default: bge
    try:
        device = settings.resolve_device()
        logger.info("Embeddings: BGE %s on %s", settings.bge_model_name, device)
        return BGEEmbeddings(
            model_name=settings.bge_model_name,
            device=device,
            batch_size=settings.embed_batch_size,
            query_instruction=settings.bge_query_instruction,
        )
    except Exception as e:
        logger.warning("BGE embeddings unavailable (%s). Falling back to hashing.", e)
        return HashingVectorizerEmbeddings(settings.hashing_features)
