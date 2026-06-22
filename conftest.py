"""Session fixture: build a tiny, hashing-embedding project in a tmp dir so tests run
fast and offline (no model downloads, minimal LLM calls)."""

import os

import pytest


@pytest.fixture(scope="session")
def project(tmp_path_factory):
    data_dir = tmp_path_factory.mktemp("arag_data")
    os.environ.update({
        "DATA_DIR": str(data_dir),
        "EMBEDDINGS_BACKEND": "hashing",   # no model files needed
        "ENABLE_WEB_SEARCH": "false",
        "RETRIEVAL_TOPK": "2",             # fewer doc-grader LLM calls in smoke
        "MAX_RETRIES": "1",
    })
    from agentic_rag.config import get_settings, reset_settings_cache

    reset_settings_cache()
    settings = get_settings().ensure_dirs()

    from agentic_rag.ingestion import build_corpus, run_ingestion
    from agentic_rag.retrieval import build_indices
    from agentic_rag.synthesis import run_synthesis

    build_corpus(settings)
    run_ingestion(settings)
    run_synthesis(settings)
    build_indices(settings)
    yield settings
