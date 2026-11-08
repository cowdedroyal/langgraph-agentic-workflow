"""Deterministic grading: compare the parsed answer with the typed gold value."""

from __future__ import annotations

import re

from .questions import Question


def _num(x):
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    m = re.search(r"-?\d+(?:\.\d+)?", str(x).replace(",", ""))
    return float(m.group(0)) if m else None


def _norm(s) -> str:
    s = str(s or "")
    s = re.sub(r"[（(].*?[)）]", "", s)
    return re.sub(r"\s|股份|有限公司|公司", "", s)


def grade(q: Question, answer) -> bool:
    if answer is None:
        return False
    if q.answer_kind == "number":
        v = _num(answer)
        return v is not None and abs(v - float(q.gold)) <= q.tolerance + 1e-9
    if q.answer_kind == "count":
        v = _num(answer)
        return v is not None and int(round(v)) == int(q.gold)
    if q.answer_kind == "entity":
        a, g = _norm(answer), _norm(q.gold)
        return bool(a) and (a == g or g in a)
    return False
