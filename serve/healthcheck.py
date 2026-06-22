#!/usr/bin/env python
"""P7: vLLM server healthcheck + latency/QPS probe.

    python serve/healthcheck.py        # or: arag health

Verifies /models, measures sequential latency (p50/p95) and concurrent throughput (QPS),
and writes outputs/serve_metrics.json.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from openai_client import VLLMClient

_PROMPTS = [
    "用一句话解释什么是市盈率。",
    "白酒企业的毛利率通常受哪些因素影响？",
    "简述动力电池产业链的上下游构成。",
    "什么是资产负债率？",
]


def main() -> None:
    client = VLLMClient()
    try:
        models = client.models()
        served = [m.get("id") for m in models.get("data", [])]
        print(f"[health] /models OK: {served}")
    except Exception as e:
        print(f"[health] server unreachable at {client.base}: {e}")
        print("  start it first:  bash serve/vllm_server.sh")
        raise SystemExit(1)

    # sequential latency
    lats = []
    for i in range(8):
        q = _PROMPTS[i % len(_PROMPTS)]
        t0 = time.time()
        client.chat(q, max_tokens=128)
        lats.append(time.time() - t0)
    p50 = round(statistics.median(lats), 3)
    p95 = round(sorted(lats)[max(0, int(len(lats) * 0.95) - 1)], 3)

    # concurrent throughput
    n_concurrent, total = 16, 32
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_concurrent) as ex:
        list(ex.map(lambda i: client.chat(_PROMPTS[i % len(_PROMPTS)], max_tokens=128), range(total)))
    wall = time.time() - t0
    qps = round(total / wall, 2)

    metrics = {"model": client.model, "latency_p50_s": p50, "latency_p95_s": p95,
               "concurrency": n_concurrent, "requests": total, "wall_s": round(wall, 2), "qps": qps}
    out = Path(os.environ.get("OUTPUT_DIR", "outputs")) / "serve_metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[health] latency p50={p50}s p95={p95}s | QPS@{n_concurrent}={qps} | -> {out}")


if __name__ == "__main__":
    main()
