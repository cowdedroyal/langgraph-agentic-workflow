#!/usr/bin/env python
"""P8: Ray-parallel ingestion. Shards raw files across Ray workers and compares wall-clock
to serial ingestion — the scale-out pattern roadmap P8 calls for (ingestion / index build /
benchmark parallelized with Ray, decoupled from the online RAG path).

    python ray_jobs/parallel_ingest.py
"""

from __future__ import annotations

import time

from agentic_rag.config import get_settings
from agentic_rag.ingestion.pipeline import _RAW_GLOBS, ingest_file
from agentic_rag.io_utils import write_jsonl


def _list_raw(raw_dir):
    files = []
    for pattern in _RAW_GLOBS:
        files.extend(sorted(raw_dir.glob(pattern)))
    return files


def _shards(items, n):
    return [items[i::n] for i in range(n) if items[i::n]]


def main(num_workers: int = 4) -> None:
    import ray

    settings = get_settings().ensure_dirs()
    files = _list_raw(settings.raw_dir)
    if not files:
        raise RuntimeError(f"No raw files in {settings.raw_dir}; run `arag corpus` first.")

    t0 = time.time()
    serial = [d for f in files for d in ingest_file(f)]
    serial_t = time.time() - t0

    ray.init(ignore_reinit_error=True, num_cpus=num_workers, logging_level="ERROR")

    @ray.remote
    def _ingest_batch(paths):
        return [d.model_dump(mode="json") for p in paths for d in ingest_file(p)]

    t0 = time.time()
    results = ray.get([_ingest_batch.remote([str(p) for p in sh]) for sh in _shards(files, num_workers)])
    parallel_t = time.time() - t0
    docs = [d for r in results for d in r]
    write_jsonl(settings.corpus_path, docs)
    ray.shutdown()

    speedup = serial_t / parallel_t if parallel_t else 0.0
    print(f"[ray-ingest] files={len(files)} chunks={len(docs)} (serial={len(serial)}) | "
          f"serial={serial_t:.2f}s  parallel(x{num_workers})={parallel_t:.2f}s  speedup={speedup:.2f}x")


if __name__ == "__main__":
    main()
