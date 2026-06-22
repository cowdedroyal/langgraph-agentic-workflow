#!/usr/bin/env python
"""Backward-compatible entry point.

The project is now a package (``src/agentic_rag``) driven by the ``arag`` CLI, but
``python main.py`` still works: it builds the corpus/indices on first run and answers a
few demo questions through the self-correcting agentic graph — mirroring the original
baseline's behavior, now on the financial multi-hop corpus.
"""

from __future__ import annotations

from agentic_rag.config import get_settings
from agentic_rag.log import get_logger

logger = get_logger("main")

_DEMOS = [
    "山河酒业2023年的营业收入较2021年增长了百分之多少？",
    "长江动力的正极材料供应商，其上游供应商是哪家公司？",
    "云岭矿业的销售毛利率从2021年到2023年是上升还是下降？变化了多少？",
]


def ensure_ready(settings) -> None:
    from agentic_rag.ingestion import build_corpus, run_ingestion
    from agentic_rag.retrieval import build_indices
    from agentic_rag.synthesis import run_synthesis

    settings.ensure_dirs()
    if not settings.corpus_path.exists():
        logger.info("First run: building corpus + QA ...")
        build_corpus(settings)
        run_ingestion(settings)
        run_synthesis(settings)
    if not (settings.index_dir / "dense.faiss").exists():
        logger.info("Building indices ...")
        build_indices(settings)


def main() -> None:
    settings = get_settings()
    ensure_ready(settings)

    from agentic_rag.graph import build_graph

    graph = build_graph(settings)
    for q in _DEMOS[: settings.examples_to_run]:
        print(f"\n{'=' * 60}\nQ: {q}")
        state = graph.invoke({"question": q}, {"recursion_limit": settings.recursion_limit})
        print(f"A: {state.get('generation')}")
        print(f"(retries={state.get('loop_step', 0)}, docs={len(state.get('documents') or [])})")


if __name__ == "__main__":
    main()
