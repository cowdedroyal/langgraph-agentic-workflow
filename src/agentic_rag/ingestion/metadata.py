"""Metadata normalization (roadmap P1): coerce raw META dicts into the unified
``DocMetadata`` contract and derive stable chunk ids.
"""

from __future__ import annotations

import re
from pathlib import Path

from ..schema import DocMetadata, DocType, Document

DOCTYPE_ZH = {
    DocType.annual_report: "年报",
    DocType.announcement: "业绩公告",
    DocType.research_report: "研报",
    DocType.web: "网络",
    DocType.other: "文档",
}

_FNAME_RE = re.compile(r"^(?P<ticker>\d{6})_(?P<period>\d{4})_(?P<kind>[a-z_]+)\.")
_KIND_MAP = {
    "annual_report": DocType.annual_report,
    "announcement": DocType.announcement,
    "research_report": DocType.research_report,
}


def coerce_doctype(value) -> DocType:
    if isinstance(value, DocType):
        return value
    try:
        return DocType(str(value))
    except ValueError:
        return DocType.other


def meta_from_filename(path: str | Path) -> dict:
    """Fallback metadata when a file has no META header (e.g. external PDFs)."""
    name = Path(path).name
    m = _FNAME_RE.match(name)
    if not m:
        return {"source": name}
    return {
        "ticker": m.group("ticker"),
        "period": m.group("period"),
        "doc_type": m.group("kind"),
        "source": name,
    }


def to_doc_metadata(meta: dict, section: str) -> DocMetadata:
    return DocMetadata(
        company=meta.get("company"),
        ticker=meta.get("ticker"),
        period=meta.get("period"),
        doc_type=coerce_doctype(meta.get("doc_type")),
        publish_date=meta.get("publish_date"),
        source=meta.get("source"),
        section=section,
    )


def make_doc_id(meta: dict, section: str, idx: int) -> str:
    return Document.make_id(
        str(meta.get("ticker", "")),
        str(meta.get("period", "")),
        str(meta.get("doc_type", "")),
        section,
        str(idx),
    )


def content_header(md: DocMetadata) -> str:
    """A compact, retrieval-friendly prefix embedding company/period/type/section into
    every chunk so both BM25 and dense search see the entity context."""
    parts = [p for p in (md.company, md.period, DOCTYPE_ZH.get(md.doc_type), md.section) if p]
    return "【" + " · ".join(parts) + "】" if parts else ""
