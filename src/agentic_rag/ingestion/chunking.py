"""Section-aware chunking (roadmap P1). Short sections stay whole; long prose is split
with a recursive splitter; table-like blocks are kept atomic so financial tables are
never broken across chunks.
"""

from __future__ import annotations

from .parsers import ParsedDoc


def _is_tableish(body: str) -> bool:
    return sum(1 for ln in body.splitlines() if ln.count("|") >= 2) >= 2


def chunk_section(title: str, body: str, max_chars: int = 600, overlap: int = 80) -> list[str]:
    body = body.strip()
    if not body:
        return []
    if len(body) <= max_chars or _is_tableish(body):
        return [body]
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_chars,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", "。", "；", "，", " ", ""],
        )
        return [c.strip() for c in splitter.split_text(body) if c.strip()]
    except Exception:
        # Dependency-free fallback: fixed-width windows.
        step = max(1, max_chars - overlap)
        return [body[i:i + max_chars] for i in range(0, len(body), step)]


def chunk_parsed(parsed: ParsedDoc, max_chars: int = 600, overlap: int = 80) -> list[tuple[str, str, int]]:
    """Return [(section_title, chunk_text, idx_within_section), ...]."""
    out: list[tuple[str, str, int]] = []
    for sec in parsed.sections:
        for idx, chunk in enumerate(chunk_section(sec.title, sec.body, max_chars, overlap)):
            out.append((sec.title, chunk, idx))
    return out
