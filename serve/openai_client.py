"""Minimal OpenAI-compatible client (stdlib only) for the vLLM server (roadmap P7).

Used by serve/healthcheck.py and as a copy-paste example of talking to the endpoint.
The production graph reaches vLLM through langchain_openai (see src/agentic_rag/llm.py);
this client deliberately avoids that dependency so the healthcheck always runs.
"""

from __future__ import annotations

import json
import os
import urllib.request


class VLLMClient:
    def __init__(self, base_url: str | None = None, model: str | None = None, api_key: str = "EMPTY"):
        self.base = (base_url or os.environ.get("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")).rstrip("/")
        self.model = model or os.environ.get("VLLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
        self.api_key = api_key

    def _request(self, path: str, payload: dict | None = None, timeout: int = 120) -> dict:
        url = self.base + path
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = None
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))

    def models(self) -> dict:
        return self._request("/models", timeout=10)

    def chat(self, prompt: str, max_tokens: int = 256, temperature: float = 0.0) -> str:
        r = self._request("/chat/completions", {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature,
        })
        return r["choices"][0]["message"]["content"]
