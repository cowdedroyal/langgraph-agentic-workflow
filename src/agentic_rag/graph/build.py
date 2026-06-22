"""Assemble the self-correcting agentic RAG graph (roadmap: interface unchanged).

Topology (identical to the baseline) ::

    (route) ─► retrieve ─► grade_documents ─► decide ─► generate ─► grade_generation ─► END
                                   │                        ▲    │
                                   └──► web_search_node ────┘    └─(regenerate / augment)
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

from langgraph.graph import END, StateGraph

from ..config import Settings, get_settings
from ..llm import get_chat_llm, get_json_llm
from ..log import get_logger
from .edges import decide_to_generate, grade_generation_v_documents_and_question, route_question
from .nodes import GraphContext, generate, grade_documents, retrieve, web_search_node
from .state import GraphState

logger = get_logger(__name__)


def build_web_search(settings: Settings):
    if not (settings.enable_web_search and settings.tavily_enabled):
        return None
    try:
        from langchain_tavily import TavilySearch

        return TavilySearch(max_results=3)
    except Exception as e:  # pragma: no cover
        logger.warning("Tavily unavailable (%s); web search disabled.", e)
        return None


def build_context(settings: Settings, retriever=None) -> GraphContext:
    from ..retrieval import load_retriever

    return GraphContext(
        settings=settings,
        retriever=retriever or load_retriever(settings),
        llm=get_chat_llm(settings=settings),
        llm_json=get_json_llm(settings=settings),
        web_search_tool=build_web_search(settings),
        max_retries=settings.max_retries,
    )


def build_graph(settings: Settings | None = None, retriever=None):
    """Compile the graph. ``retriever`` can be injected (tests) to skip index loading."""
    settings = settings or get_settings()
    ctx = build_context(settings, retriever)

    wf = StateGraph(GraphState)
    wf.add_node("web_search_node", partial(web_search_node, ctx=ctx))
    wf.add_node("retrieve", partial(retrieve, ctx=ctx))
    wf.add_node("grade_documents", partial(grade_documents, ctx=ctx))
    wf.add_node("generate", partial(generate, ctx=ctx))

    wf.set_conditional_entry_point(
        partial(route_question, ctx=ctx),
        {"web_search_node": "web_search_node", "retrieve": "retrieve"},
    )
    wf.add_edge("retrieve", "grade_documents")
    wf.add_conditional_edges(
        "grade_documents",
        partial(decide_to_generate, ctx=ctx),
        {"web_search_node": "web_search_node", "generate": "generate"},
    )
    wf.add_conditional_edges(
        "generate",
        partial(grade_generation_v_documents_and_question, ctx=ctx),
        {"__end__": END, "generate": "generate", "web_search_node": "web_search_node"},
    )
    wf.add_edge("web_search_node", "generate")

    graph = wf.compile()
    logger.info("Graph compiled (backend=%s, model=%s, web_search=%s)",
                settings.llm_backend, settings.local_llm, ctx.web_search_tool is not None)
    return graph


def render_graph(out_path: str | Path = "graph.png", settings: Settings | None = None) -> str:
    """Render the workflow to PNG (falls back to a .mmd Mermaid file if rendering fails)."""
    graph = build_graph(settings)
    out_path = Path(out_path)
    try:
        out_path.write_bytes(graph.get_graph().draw_mermaid_png())
        return str(out_path)
    except Exception as e:
        logger.warning("PNG render failed (%s); writing Mermaid source instead.", e)
        mmd = out_path.with_suffix(".mmd")
        mmd.write_text(graph.get_graph().draw_mermaid(), encoding="utf-8")
        return str(mmd)
