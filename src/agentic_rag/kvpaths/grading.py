"""Lenient answer parsing for the KV runs.

The answer-format rule sits in the system prompt, i.e. before the corpus. At ~200k tokens the
4B model often ends with ``答案：10.23`` instead of the JSON line, which the strict parser of
``agentic_rag.access`` scores as no answer. Strict accuracy then measures instruction following
over distance, not whether the evidence survived compression. Both are reported; the lenient
one is the primary metric and ``format_fail`` (lenient right, strict wrong) is counted per cell.

Fallback order when there is no JSON line: the last ``答案：X`` (or ``答案是X``); for numeric
questions, the last number followed by 亿元 / %. A wrong number stays wrong; the fallback only
reads the model's own final statement.
"""

from __future__ import annotations

import re

from ..access.grade import grade
from ..access.paths import parse_answer

_ANS = re.compile(r"答案\s*(?:是|为)?\s*[:：]?\s*\**\s*([^\n*{}]+)")
_NUM_UNIT = re.compile(r"(-?\d+(?:\.\d+)?)\s*\**\s*(?:亿元|%)")


def parse_lenient(text: str, kind: str = "number"):
    a = parse_answer(text)
    if a is not None:
        return a
    t = text or ""
    if '"answer"' in t and re.search(r'"answer"\s*:\s*null', t):
        return None
    m = _ANS.findall(t)
    if m:
        v = m[-1].strip().strip("。.，, \"'“”")
        if v and v.lower() not in ("null", "none"):
            return v
    if kind in ("number", "count"):
        m = _NUM_UNIT.findall(t)
        if m:
            return m[-1]
    return None


def grade_lenient(q, text: str) -> tuple[object, bool]:
    a = parse_lenient(text, q.answer_kind)
    return a, grade(q, a)
