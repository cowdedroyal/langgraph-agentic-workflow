"""Minimal OpenAI-compatible client with cost accounting and an on-disk response cache.

* Every call returns text plus a ``Usage`` record: prompt tokens split into cache hits and
  misses (DeepSeek reports both), completion tokens, latency and dollars.
* Responses are cached by a hash of the request body, so re-running an experiment costs
  nothing and gives the same numbers. Delete ``cache_dir`` to re-measure latency.

Defaults point at a local magpie gateway (``ACCESS_BASE_URL``, ``ACCESS_API_KEY``,
``ACCESS_MODEL`` override them). Prices are USD per token and only used for reporting.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path

BASE_URL = os.environ.get("ACCESS_BASE_URL", "http://127.0.0.1:3425/v1")
API_KEY = os.environ.get("ACCESS_API_KEY", "magpie")
MODEL = os.environ.get("ACCESS_MODEL", "deepseek/deepseek-v4-flash")
# deepseek-v4-flash list prices (USD / token)
PRICE = {"miss": 0.15e-6, "hit": 0.003e-6, "out": 0.6e-6}


@dataclass
class Usage:
    calls: int = 0
    prompt: int = 0
    hit: int = 0
    miss: int = 0
    completion: int = 0
    reasoning: int = 0
    latency_s: float = 0.0
    cached_response: int = 0

    @property
    def usd(self) -> float:
        return self.miss * PRICE["miss"] + self.hit * PRICE["hit"] + self.completion * PRICE["out"]

    def add(self, other: "Usage") -> "Usage":
        for k in ("calls", "prompt", "hit", "miss", "completion", "reasoning", "cached_response"):
            setattr(self, k, getattr(self, k) + getattr(other, k))
        self.latency_s += other.latency_s
        return self

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usd"] = round(self.usd, 6)
        return d


@dataclass
class Reply:
    text: str
    message: dict
    usage: Usage
    finish: str = ""
    raw: dict = field(default_factory=dict)


class LLM:
    def __init__(self, model: str = MODEL, cache_dir: str | Path | None = None,
                 base_url: str = BASE_URL, api_key: str = API_KEY, timeout: int = 900):
        self.model, self.base_url, self.api_key, self.timeout = model, base_url, api_key, timeout
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _key(self, body: dict) -> str:
        return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             max_tokens: int = 2048, temperature: float = 0.0, **extra) -> Reply:
        body = {"model": self.model, "messages": messages, "max_tokens": max_tokens,
                "temperature": temperature, **extra}
        if tools:
            body["tools"] = tools
        key = self._key(body)
        path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if path and path.exists():
            rec = json.loads(path.read_text())
            u = Usage(**rec["usage"])  # keeps the latency measured on the original call
            u.cached_response = 1
            return Reply(rec["text"], rec["message"], u, rec.get("finish", ""))

        data, dt = self._post(body)
        choice = data["choices"][0]
        msg = choice["message"]
        ud = data.get("usage", {})
        hit = ud.get("prompt_cache_hit_tokens", (ud.get("prompt_tokens_details") or {}).get("cached_tokens", 0)) or 0
        prompt = ud.get("prompt_tokens", 0)
        usage = Usage(calls=1, prompt=prompt, hit=hit,
                      miss=ud.get("prompt_cache_miss_tokens", prompt - hit),
                      completion=ud.get("completion_tokens", 0),
                      reasoning=(ud.get("completion_tokens_details") or {}).get("reasoning_tokens", 0) or 0,
                      latency_s=round(dt, 3))
        reply = Reply(msg.get("content") or "", {k: v for k, v in msg.items() if k != "reasoning_content"},
                      usage, choice.get("finish_reason", ""))
        if path:
            u = asdict(usage)
            path.write_text(json.dumps({"text": reply.text, "message": reply.message, "usage": u,
                                        "finish": reply.finish}, ensure_ascii=False))
        return reply

    def _post(self, body: dict) -> tuple[dict, float]:
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", json.dumps(body).encode(),
            {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"})
        last = None
        for attempt in range(8):
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.load(r), time.time() - t0
            except urllib.error.HTTPError as e:
                if e.code in (400, 401, 403, 404):
                    raise RuntimeError(f"{e.code}: {e.read()[:500]!r}") from e
                last = e  # 429 / 5xx: back off exponentially, capped at 60 s
                time.sleep(min(60, 2 ** attempt * 2))
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = e
                time.sleep(min(60, 2 ** attempt * 2))
        raise RuntimeError(f"LLM call failed after retries: {last}")
