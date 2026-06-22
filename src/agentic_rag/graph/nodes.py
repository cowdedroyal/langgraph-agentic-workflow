"""Graph nodes (roadmap: keep the self-correcting RAG loop, swap retrieval implementation).

Dependencies (retriever / LLMs / web tool) are injected via ``GraphContext`` so nodes
stay pure ``(state) -> dict`` callables after ``functools.partial`` binding in build.py.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.documents import Document as LCDocument
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from ..config import Settings
from ..llm import parse_json
from ..log import get_logger, node
from ..prompts import DOC_GRADER_INSTRUCTIONS, DOC_GRADER_PROMPT, RAG_PROMPT

logger = get_logger(__name__)

_NO_CONTEXT = "（基于现有检索资料不足，无法可靠回答该问题。）"


@dataclass
class GraphContext:
    settings: Settings
    retriever: object              # HybridRetriever (duck-typed: .invoke(question))
    llm: BaseChatModel
    llm_json: BaseChatModel
    web_search_tool: object | None
    max_retries: int


def format_docs(docs: list[LCDocument]) -> str:
    return "\n\n".join(d.page_content for d in docs)


def retrieve(state, ctx: GraphContext) -> dict:
    node("Retrieve (hybrid FAISS+BM25)")
    docs = ctx.retriever.invoke(state["question"])
    logger.info("retrieved %d documents", len(docs))
    return {"documents": docs}


def generate(state, ctx: GraphContext) -> dict:
    node("Generate")
    documents = state["documents"]
    if not documents:
        return {"generation": _NO_CONTEXT, "loop_step": 1}
    prompt = RAG_PROMPT.format(context=format_docs(documents), question=state["question"])
    try:
        gen = ctx.llm.invoke([HumanMessage(content=prompt)]).content
    except Exception as e:
        logger.warning("generation failed: %s", e)
        gen = _NO_CONTEXT
    return {"generation": gen, "loop_step": 1}


def grade_documents(state, ctx: GraphContext) -> dict:
    node("Grade Documents")
    question = state["question"]
    documents = state["documents"]
    filtered, web_search = [], "No"
    for d in documents:
        try:
            r = ctx.llm_json.invoke(
                [SystemMessage(content=DOC_GRADER_INSTRUCTIONS),
                 HumanMessage(content=DOC_GRADER_PROMPT.format(document=d.page_content, question=question))]
            )
            if parse_json(r.content).get("binary_score", "no").lower() == "yes":
                filtered.append(d)
            else:
                web_search = "Yes"
        except Exception:
            web_search = "Yes"
    if not filtered:
        web_search = "Yes"
    logger.info("relevant: %d/%d", len(filtered), len(documents))
    return {"documents": filtered, "web_search": web_search}


def web_search_node(state, ctx: GraphContext) -> dict:
    node("Web Search")
    documents = state.get("documents") or []
    if ctx.web_search_tool is None:
        logger.info("web search skipped (disabled or no TAVILY_API_KEY)")
        return {"documents": documents}
    try:
        results = ctx.web_search_tool.invoke({"query": state["question"]})
        contents = []
        if isinstance(results, dict):
            results = results.get("results", [])
        if isinstance(results, list):
            contents = [str(r.get("content", "")) for r in results if isinstance(r, dict) and r.get("content")]
        if contents:
            documents = documents + [LCDocument(page_content="\n\n".join(contents),
                                                metadata={"source": "web_search"})]
    except Exception as e:
        logger.warning("web search error: %s", e)
    return {"documents": documents}
