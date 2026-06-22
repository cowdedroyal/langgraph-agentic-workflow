"""Multi-format parsers (roadmap P1): Markdown / HTML / TXT (+ best-effort PDF).

Each raw file carries a ``<!-- META {...} -->`` header injected by the synthetic
generator; metadata is parsed from it and the body is split into titled sections.
HTML is handled with the stdlib ``html.parser`` (no bs4 dependency).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path

_META_RE = re.compile(r"<!--\s*META\s*(\{.*?\})\s*-->", re.DOTALL)
_BRACKET_SEC = re.compile(r"^【([^】]+)】(.*)$")


@dataclass
class Section:
    title: str
    body: str


@dataclass
class ParsedDoc:
    meta: dict
    sections: list[Section] = field(default_factory=list)
    path: str | None = None


def parse_meta(text: str) -> dict:
    m = _META_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    return {}


def _clean_sections(pairs: list[tuple[str, str]]) -> list[Section]:
    return [Section(t.strip(), b.strip()) for t, b in pairs if b and b.strip()]


def parse_markdown(text: str) -> ParsedDoc:
    meta = parse_meta(text)
    pairs: list[tuple[str, str]] = []
    title, buf = "正文", []
    for line in text.splitlines():
        if line.startswith("<!-- META"):
            continue
        if line.startswith("## "):
            if buf:
                pairs.append((title, "\n".join(buf)))
                buf = []
            title = line[3:].strip()
        elif line.startswith("# "):
            continue  # document title — not a retrieval section
        else:
            buf.append(line)
    if buf:
        pairs.append((title, "\n".join(buf)))
    return ParsedDoc(meta, _clean_sections(pairs))


def parse_research(text: str) -> ParsedDoc:
    """Research reports use 【Section】 markers."""
    meta = parse_meta(text)
    pairs: list[tuple[str, str]] = []
    title, buf = "正文", []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("<!-- META"):
            continue
        m = _BRACKET_SEC.match(line)
        if m:
            if buf:
                pairs.append((title, "\n".join(buf)))
                buf = []
            title = m.group(1).strip()
            rest = m.group(2).strip()
            buf = [rest] if rest else []
        else:
            buf.append(line)
    if buf:
        pairs.append((title, "\n".join(buf)))
    return ParsedDoc(meta, _clean_sections(pairs))


class _HTMLSections(HTMLParser):
    HEAD = {"h1", "h2", "h3"}
    SKIP = {"title", "style", "script", "head", "meta"}

    def __init__(self):
        super().__init__()
        self.pairs: list[tuple[str, str]] = []
        self._title = "正文"
        self._buf: list[str] = []
        self._tag: str | None = None
        self._cells: list[str] = []
        self._capture_title = False

    def _flush(self):
        body = "\n".join(x for x in self._buf if x).strip()
        if body:
            self.pairs.append((self._title, body))
        self._buf = []

    def handle_starttag(self, tag, attrs):
        self._tag = tag
        if tag in self.HEAD:
            self._flush()
            self._title = ""
            self._capture_title = True
        elif tag == "tr":
            self._cells = []

    def handle_endtag(self, tag):
        if tag in self.HEAD:
            self._capture_title = False
        elif tag == "tr" and self._cells:
            self._buf.append(" | ".join(self._cells))
            self._cells = []
        self._tag = None

    def handle_data(self, data):
        t = data.strip()
        if not t or self._tag in self.SKIP:
            return
        if self._capture_title:
            self._title = (self._title + t).strip()
        elif self._tag in ("td", "th"):
            self._cells.append(t)
        else:
            self._buf.append(t)

    def result(self) -> list[tuple[str, str]]:
        self._flush()
        return self.pairs


def parse_html(text: str) -> ParsedDoc:
    meta = parse_meta(text)
    p = _HTMLSections()
    p.feed(text)
    return ParsedDoc(meta, _clean_sections(p.result()))


def parse_pdf(path: Path) -> ParsedDoc:
    """Best-effort PDF (pypdf -> docling). Whole text becomes one '正文' section."""
    text = ""
    try:
        from pypdf import PdfReader

        text = "\n".join((pg.extract_text() or "") for pg in PdfReader(str(path)).pages)
    except Exception:
        try:
            from docling.document_converter import DocumentConverter

            text = DocumentConverter().convert(str(path)).document.export_to_markdown()
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"No PDF backend available (install pypdf or docling): {e}")
    return ParsedDoc(parse_meta(text), _clean_sections([("正文", text)]), str(path))


def parse_file(path: str | Path) -> ParsedDoc:
    path = Path(path)
    text = "" if path.suffix.lower() == ".pdf" else path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".md", ".markdown"):
        doc = parse_markdown(text)
    elif path.suffix.lower() in (".html", ".htm"):
        doc = parse_html(text)
    elif path.suffix.lower() == ".pdf":
        doc = parse_pdf(path)
    else:  # .txt and others -> bracket/plain
        doc = parse_research(text)
    doc.path = str(path)
    return doc
