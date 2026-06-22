"""Generation-quality metrics (roadmap P4).

For synthetic finance QA the ground truth is numeric, so we can measure correctness
*without* an LLM via numeric-token matching, and use the LLM judge only for the softer
faithfulness / relevance scores.
"""

from __future__ import annotations

import re

_NUM = re.compile(r"\d+\.?\d*%?")


def numeric_keys(text: str) -> set[str]:
    return set(_NUM.findall(text or ""))


def answer_accuracy(gold: str, generation: str) -> float:
    """Fraction of the gold answer's numeric facts that appear in the generation."""
    g = numeric_keys(gold)
    if not g:
        return 0.0
    return len([k for k in g if k in (generation or "")]) / len(g)


def citation_coverage(generation: str, evidence_texts: list[str]) -> float:
    """Fraction of evidence chunks whose numeric facts are reflected in the generation —
    a proxy for whether the answer actually draws on each piece of evidence."""
    if not evidence_texts:
        return 0.0
    gen_nums = numeric_keys(generation)
    used = 0
    for txt in evidence_texts:
        ev = numeric_keys(txt)
        if ev & gen_nums:
            used += 1
    return used / len(evidence_texts)
