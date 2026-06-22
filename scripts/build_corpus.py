#!/usr/bin/env python
"""Build the synthetic financial corpus and ingest it into documents.jsonl (roadmap P1).

    python scripts/build_corpus.py

Equivalent to ``arag corpus`` (without the P3 QA synthesis step).
"""

from agentic_rag.config import get_settings
from agentic_rag.ingestion import build_corpus, run_ingestion


def main() -> None:
    settings = get_settings()
    summary = build_corpus(settings)
    ing = run_ingestion(settings)
    print(f"corpus: {summary['companies']} companies x {summary['years']} years "
          f"-> {summary['raw_files']} raw files -> {ing['chunks']} chunks")
    print(f"  documents: {ing['corpus_path']}")


if __name__ == "__main__":
    main()
