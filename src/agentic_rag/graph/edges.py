"""Graph edges (routing + self-correction decisions). Ported from the baseline and made
loop-safe: when web search is unavailable the graph never bounces into an empty
web-search cycle, and missing generations/documents terminate cleanly.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END

from ..llm import parse_json
from ..log import edge
from ..prompts import (
    ANSWER_GRADER_INSTRUCTIONS,
    ANSWER_GRADER_PROMPT,
    HALLUCINATION_GRADER_INSTRUCTIONS,
    HALLUCINATION_GRADER_PROMPT,
    ROUTER_INSTRUCTIONS,
)
from .nodes import GraphContext, format_docs

_VECTORSTORE_HINTS = ("营收", "营业收入", "净利润", "毛利率", "供应商", "年报", "财务", "公司", "股份", "材料")
_WEB_HINTS = ("最新", "实时", "今天", "现在", "股价", "新闻")


def route_question(state, ctx: GraphContext) -> str:
    edge("Route question")
    q = state["question"]
    if any(h in q for h in _VECTORSTORE_HINTS):
        return "retrieve"
    if any(h in q for h in _WEB_HINTS) and ctx.web_search_tool is not None:
        return "web_search_node"
    try:
        r = ctx.llm_json.invoke([SystemMessage(content=ROUTER_INSTRUCTIONS), HumanMessage(content=q)])
        source = parse_json(r.content).get("datasource", "vectorstore").lower()
        if source == "websearch" and ctx.web_search_tool is not None:
            return "web_search_node"
    except Exception:
        pass
    return "retrieve"


def decide_to_generate(state, ctx: GraphContext) -> str:
    edge("Decide: generate vs web search")
    web_search = state.get("web_search", "No")
    filtered = state.get("documents") or []
    if ctx.web_search_tool is not None and (web_search == "Yes" or not filtered):
        return "web_search_node"
    return "generate"


def grade_generation_v_documents_and_question(state, ctx: GraphContext) -> str:
    """Self-correction: regenerate on hallucination, augment on irrelevance, stop on success
    or after max_retries."""
    generation = state.get("generation")
    documents = state.get("documents") or []
    loop_step = state.get("loop_step", 0)
    max_retries = ctx.max_retries

    if generation is None or not documents:
        return "__end__"  # nothing to ground against — stop cleanly

    # 1) grounded in the documents? (faithfulness / hallucination)
    try:
        r = ctx.llm_json.invoke(
            [SystemMessage(content=HALLUCINATION_GRADER_INSTRUCTIONS),
             HumanMessage(content=HALLUCINATION_GRADER_PROMPT.format(
                 documents=format_docs(documents), generation=generation))]
        )
        grounded = parse_json(r.content).get("binary_score", "no").lower() == "yes"
    except Exception:
        return "generate" if loop_step < max_retries else "__end__"

    if not grounded:
        edge(f"Not grounded → regenerate ({loop_step}/{max_retries})")
        return "generate" if loop_step < max_retries else "__end__"

    # 2) does it answer the question?
    try:
        r = ctx.llm_json.invoke(
            [SystemMessage(content=ANSWER_GRADER_INSTRUCTIONS),
             HumanMessage(content=ANSWER_GRADER_PROMPT.format(
                 question=state["question"], generation=generation))]
        )
        useful = parse_json(r.content).get("binary_score", "no").lower() == "yes"
    except Exception:
        useful = True  # grounded but ungradable — accept

    if useful:
        edge("Grounded + answers question → END")
        return "__end__"
    if ctx.web_search_tool is not None and loop_step < max_retries:
        edge("Grounded but not useful → web search")
        return "web_search_node"
    return "__end__"
