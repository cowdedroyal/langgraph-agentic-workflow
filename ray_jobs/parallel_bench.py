#!/usr/bin/env python
"""P8: Ray-parallel retrieval benchmark. A pool of Ray actors (each holding a HybridRetriever)
evaluates QA shards concurrently — the distributed-evaluation pattern from roadmap P8.

    python ray_jobs/parallel_bench.py
"""

from __future__ import annotations

import time

from agentic_rag.config import get_settings
from agentic_rag.synthesis import load_qa


def _shards(items, n):
    return [items[i::n] for i in range(n) if items[i::n]]


def main(num_workers: int = 2, method: str = "hybrid", k: int = 5) -> None:
    import ray

    from agentic_rag.eval.retrieval_metrics import aggregate, hit_at_k, mrr, ndcg_at_k, recall_at_k

    settings = get_settings()
    qas = [q.model_dump() for q in load_qa(settings)]
    if not qas:
        raise RuntimeError("No QA; run `arag corpus` first.")

    ray.init(ignore_reinit_error=True, logging_level="ERROR")

    @ray.remote(num_gpus=0)
    class RetrievalWorker:
        def __init__(self):
            from agentic_rag.retrieval import load_retriever

            self.retr = load_retriever()

        def run(self, shard, method, k):
            per = []
            for qa in shard:
                ids = [h.document.doc_id for h in
                       self.retr.search(qa["question"], topk=max(k, 10), method=method)]
                ev = qa["evidence_doc_ids"]
                per.append({f"Recall@{k}": recall_at_k(ids, ev, k), "MRR": mrr(ids, ev),
                            f"nDCG@{k}": ndcg_at_k(ids, ev, k), f"Hit@{k}": hit_at_k(ids, ev, k)})
            return per

    workers = [RetrievalWorker.remote() for _ in range(num_workers)]
    shards = _shards(qas, num_workers)
    t0 = time.time()
    results = ray.get([w.run.remote(sh, method, k) for w, sh in zip(workers, shards)])
    wall = time.time() - t0
    per = [x for r in results for x in r]
    agg = aggregate(per)
    ray.shutdown()

    print(f"[ray-bench] method={method} n={len(per)} workers={num_workers} wall={wall:.2f}s")
    print(f"  {agg}")


if __name__ == "__main__":
    main()
