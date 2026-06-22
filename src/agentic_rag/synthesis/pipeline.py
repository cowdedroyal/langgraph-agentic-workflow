"""Synthesis pipeline (roadmap P3): sample evidence chains -> generate verifiable
multi-hop QA -> data/qa/multihop_qa.jsonl (shared by eval P4 and training P5)."""

from __future__ import annotations

from collections import Counter

from ..config import Settings, get_settings
from ..io_utils import write_jsonl
from ..log import get_logger
from .generators import build_evidence_index, generate
from .samplers import sample_all

logger = get_logger(__name__)


def run_synthesis(settings: Settings | None = None) -> dict:
    from ..ingestion.pipeline import load_documents
    from ..ingestion.synthetic import load_facts

    settings = (settings or get_settings()).ensure_dirs()
    docs = load_documents(settings)
    if not docs:
        raise RuntimeError(f"No documents at {settings.corpus_path}; run `arag corpus` first.")
    idx = build_evidence_index(docs)
    facts = load_facts(settings)

    seen: set[str] = set()
    qas = []
    for chain in sample_all():
        qa = generate(chain, facts, idx)
        if qa and qa.qa_id not in seen and len(qa.evidence_doc_ids) >= 2:
            seen.add(qa.qa_id)
            qas.append(qa)

    n = write_jsonl(settings.qa_path, qas)
    dist = dict(Counter(q.hop_type for q in qas))
    logger.info("Synthesis: %d multi-hop QA -> %s | hop types: %s", n, settings.qa_path, dist)
    return {"qa": n, "hop_types": dist, "qa_path": str(settings.qa_path)}


def load_qa(settings: Settings | None = None):
    from ..io_utils import read_jsonl
    from ..schema import MultiHopQA

    settings = settings or get_settings()
    return [MultiHopQA(**row) for row in read_jsonl(settings.qa_path)]
