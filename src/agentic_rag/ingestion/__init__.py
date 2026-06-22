"""Ingestion package (roadmap P1)."""

from .pipeline import ingest_dir, ingest_file, load_documents, run_ingestion
from .synthetic import build_corpus, load_facts

__all__ = [
    "build_corpus",
    "load_facts",
    "run_ingestion",
    "ingest_dir",
    "ingest_file",
    "load_documents",
]
