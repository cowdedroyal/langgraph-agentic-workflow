"""agentic_rag — Financial multi-hop Agentic RAG.

A LangGraph self-correcting RAG workflow over a synthetic Chinese financial corpus,
with hybrid (FAISS + BM25 + RRF) retrieval, an LLM-judge evaluation harness, and a
LoRA/GRPO post-training + vLLM/Ray serving track.

Public surface is intentionally small; import submodules directly, e.g.::

    from agentic_rag.config import get_settings
    from agentic_rag.graph.build import build_graph
"""

__version__ = "0.2.0"

__all__ = ["__version__"]
