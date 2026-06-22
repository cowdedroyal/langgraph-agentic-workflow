"""Tiny JSONL helpers used by every data-producing layer (ingestion, synthesis, eval).

Named ``io_utils`` to avoid shadowing the stdlib ``io`` module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator


def write_jsonl(path: str | Path, rows: Iterable[Any]) -> int:
    """Write rows (dicts or pydantic models) to a JSONL file. Returns the count."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            if hasattr(row, "model_dump"):
                row = row.model_dump(mode="json")
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def read_jsonl(path: str | Path) -> Iterator[dict]:
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_jsonl(path: str | Path) -> list[dict]:
    return list(read_jsonl(path))
