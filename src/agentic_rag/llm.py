"""LLM factory: a single switch (``LLM_BACKEND``) selects Ollama or an
OpenAI-compatible endpoint (vLLM / OpenAI). Keeps the rest of the codebase backend-agnostic
so the LangGraph nodes never import a concrete client.
"""

from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage

from .config import Settings, get_settings
from .log import get_logger

logger = get_logger(__name__)


def get_chat_llm(json_mode: bool = False, settings: Settings | None = None) -> BaseChatModel:
    """Return a chat model. ``json_mode=True`` constrains output to a JSON object."""
    settings = settings or get_settings()
    backend = settings.llm_backend.lower()

    if backend == "ollama":
        from langchain_ollama import ChatOllama

        kwargs: dict[str, Any] = dict(
            model=settings.local_llm,
            temperature=settings.llm_temperature,
            base_url=settings.ollama_base_url,
            num_predict=settings.ollama_json_num_predict if json_mode else settings.ollama_num_predict,
        )
        if json_mode:
            kwargs["format"] = "json"
        return ChatOllama(**kwargs)

    if backend in ("vllm", "openai"):
        import os

        from langchain_openai import ChatOpenAI

        # langchain-openai 1.x enables custom TCP keepalive socket options that can break
        # connections to a local endpoint; disable to restore default httpx behavior.
        os.environ.setdefault("LANGCHAIN_OPENAI_TCP_KEEPALIVE", "0")
        # never route localhost through an HTTP/SOCKS proxy
        _no = os.environ.get("NO_PROXY", "")
        if "127.0.0.1" not in _no:
            os.environ["NO_PROXY"] = (_no + ",127.0.0.1,localhost,0.0.0.0").lstrip(",")
            os.environ["no_proxy"] = os.environ["NO_PROXY"]

        model_kwargs: dict[str, Any] = {}
        # vLLM 0.6.6 guided-decoding (xgrammar) for response_format is unstable and can crash
        # execute_model; rely on the prompt's "return JSON" instruction + parse_json instead.
        # Real OpenAI keeps native JSON mode.
        if json_mode and backend == "openai":
            model_kwargs["response_format"] = {"type": "json_object"}
        return ChatOpenAI(
            model=settings.vllm_model,
            base_url=settings.vllm_base_url,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
            timeout=settings.request_timeout,
            max_retries=2,
            model_kwargs=model_kwargs,
        )

    raise ValueError(f"Unknown LLM_BACKEND: {settings.llm_backend!r}")


def get_json_llm(settings: Settings | None = None) -> BaseChatModel:
    return get_chat_llm(json_mode=True, settings=settings)


_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_json(content: str) -> dict:
    """Best-effort JSON extraction from an LLM response (handles ```json fences and prose)."""
    content = content.strip()
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    # strip code fences
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass
    m = _JSON_BLOCK.search(content)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise json.JSONDecodeError("no JSON object found", content, 0)


def invoke_json(messages: list[BaseMessage], llm: BaseChatModel | None = None,
                settings: Settings | None = None) -> dict:
    """Invoke a JSON-mode LLM and return the parsed dict (raises on unparseable output)."""
    llm = llm or get_json_llm(settings)
    resp = llm.invoke(messages)
    return parse_json(resp.content if hasattr(resp, "content") else str(resp))
