"""Ingestion pipeline (roadmap P1): raw files -> unified Document JSONL shared by the
dense (FAISS) and sparse (BM25) indices.
"""

from __future__ import annotations

from pathlib import Path

from ..config import Settings, get_settings
from ..io_utils import write_jsonl
from ..log import get_logger
from ..schema import Document
from .chunking import chunk_parsed
from .metadata import content_header, make_doc_id, meta_from_filename, to_doc_metadata
from .parsers import parse_file

logger = get_logger(__name__)

_RAW_GLOBS = ("*.md", "*.markdown", "*.html", "*.htm", "*.txt", "*.pdf")


def ingest_file(path: str | Path, max_chars: int = 600, overlap: int = 80) -> list[Document]:
    parsed = parse_file(path)
    meta = parsed.meta or meta_from_filename(path)
    docs: list[Document] = []
    for section, chunk, idx in chunk_parsed(parsed, max_chars, overlap):
        md = to_doc_metadata(meta, section)
        header = content_header(md)
        content = f"{header}\n{chunk}" if header else chunk
        docs.append(Document(doc_id=make_doc_id(meta, section, idx), content=content, metadata=md))
    return docs


def ingest_dir(raw_dir: str | Path, max_chars: int = 600, overlap: int = 80) -> list[Document]:
    raw_dir = Path(raw_dir)
    files: list[Path] = []
    for pattern in _RAW_GLOBS:
        files.extend(sorted(raw_dir.glob(pattern)))
    docs: list[Document] = []
    seen: set[str] = set()
    for fp in files:
        for d in ingest_file(fp, max_chars, overlap):
            if d.doc_id in seen:  # guard against id collisions
                d.doc_id = Document.make_id(d.doc_id, str(len(seen)))
            seen.add(d.doc_id)
            docs.append(d)
    return docs


def run_ingestion(settings: Settings | None = None) -> dict:
    settings = (settings or get_settings()).ensure_dirs()
    docs = ingest_dir(settings.raw_dir)
    n = write_jsonl(settings.corpus_path, docs)
    logger.info("Ingestion: %d files -> %d chunks -> %s",
                len(list(settings.raw_dir.glob("*"))), n, settings.corpus_path)
    return {"chunks": n, "corpus_path": str(settings.corpus_path)}


def load_documents(settings: Settings | None = None) -> list[Document]:
    from ..io_utils import read_jsonl

    settings = settings or get_settings()
    return [Document(**row) for row in read_jsonl(settings.corpus_path)]
