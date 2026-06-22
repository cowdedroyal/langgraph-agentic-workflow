"""Data contracts shared across ingestion -> retrieval -> synthesis -> eval -> training.

These pydantic models are the unified schema referenced throughout the project so
every layer speaks the same ``Document`` / ``MultiHopQA`` language.
"""

from __future__ import annotations

import hashlib
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class DocType(str, Enum):
    annual_report = "annual_report"      # 年报
    announcement = "announcement"        # 业绩/重大事项公告
    research_report = "research_report"  # 券商研报
    web = "web"                          # web search result
    other = "other"


class DocMetadata(BaseModel):
    """Normalized metadata per roadmap P1: company / ticker / period / doc_type / ..."""

    company: Optional[str] = None
    ticker: Optional[str] = None
    period: Optional[str] = None          # "2022" or "2022Q4"
    doc_type: DocType = DocType.other
    publish_date: Optional[str] = None    # ISO date "YYYY-MM-DD"
    source: Optional[str] = None          # origin file / URL
    section: Optional[str] = None         # chunk section title
    extra: dict[str, Any] = Field(default_factory=dict)


class Document(BaseModel):
    """A retrieval unit (a chunk). Shared by the dense (FAISS) and sparse (BM25) indices."""

    doc_id: str
    content: str
    metadata: DocMetadata = Field(default_factory=DocMetadata)

    # -- interop with langchain_core.documents.Document --
    def to_langchain(self):
        from langchain_core.documents import Document as LCDocument

        md = self.metadata.model_dump(mode="json")
        md["doc_id"] = self.doc_id
        return LCDocument(page_content=self.content, metadata=md)

    @classmethod
    def from_langchain(cls, lc, doc_id: Optional[str] = None) -> "Document":
        md = dict(getattr(lc, "metadata", {}) or {})
        did = doc_id or md.pop("doc_id", None) or cls.make_id(lc.page_content)
        known = {k: md.pop(k, None) for k in
                 ("company", "ticker", "period", "doc_type", "publish_date", "source", "section")}
        known = {k: v for k, v in known.items() if v is not None}
        return cls(
            doc_id=did,
            content=lc.page_content,
            metadata=DocMetadata(**known, extra=md),
        )

    @staticmethod
    def make_id(*parts: str) -> str:
        h = hashlib.sha1("||".join(p for p in parts if p).encode("utf-8")).hexdigest()
        return h[:16]


class MultiHopQA(BaseModel):
    """A synthesized multi-hop QA sample (roadmap P3). ``evidence_doc_ids`` are the
    ground-truth chunks used for retrieval metrics and citation checking."""

    qa_id: str
    question: str
    answer: str
    evidence_doc_ids: list[str] = Field(default_factory=list)
    reasoning_steps: list[str] = Field(default_factory=list)
    hop_type: str = "cross_year"   # cross_year | supply_chain | metric_change
    companies: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedDoc(BaseModel):
    document: Document
    score: float
    rank: int
    retriever: str = "hybrid"   # dense | bm25 | hybrid | rerank


# --------------------------------------------------------------------------- #
# Evaluation results (roadmap P4)
# --------------------------------------------------------------------------- #
class RetrievalMetrics(BaseModel):
    method: str
    recall_at_k: float = 0.0
    mrr: float = 0.0
    ndcg: float = 0.0
    k: int = 5
    n: int = 0


class GenerationMetrics(BaseModel):
    faithfulness: float = 0.0
    answer_relevance: float = 0.0
    citation_coverage: float = 0.0
    n: int = 0


class EndToEndMetrics(BaseModel):
    multihop_hit_rate: float = 0.0
    answer_accuracy: float = 0.0
    avg_retries: float = 0.0
    avg_latency_s: float = 0.0
    n: int = 0
