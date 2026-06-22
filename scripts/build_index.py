#!/usr/bin/env python
"""Build the FAISS + BM25 indices from documents.jsonl (roadmap P2).

    python scripts/build_index.py     # equivalent to `arag index`
"""

from agentic_rag.config import get_settings
from agentic_rag.retrieval import build_indices


def main() -> None:
    info = build_indices(get_settings())
    print(f"indices built for {info['docs']} docs -> {info['index_dir']}")


if __name__ == "__main__":
    main()
