"""GRPO reward functions (roadmap P6): reuse the P4 evaluation signals as rewards.

Each function follows the trl GRPO contract — ``f(completions, **kwargs) -> list[float]`` —
where dataset columns (``answer``, ``evidence``) arrive as keyword lists. Total reward is
the (equally weighted) sum the trainer optimizes:

    correctness  +  citation coverage  +  format validity  -  hallucination penalty
"""

from __future__ import annotations

from agentic_rag.eval.generation_metrics import answer_accuracy, citation_coverage, numeric_keys


def reward_answer_correctness(completions, answer=None, **kwargs) -> list[float]:
    """Numeric-fact overlap with the gold answer (answer correctness)."""
    answer = answer or [""] * len(completions)
    return [answer_accuracy(a, c) for c, a in zip(completions, answer)]


def reward_citation(completions, evidence=None, **kwargs) -> list[float]:
    """Fraction of evidence facts reflected in the answer (citation / evidence coverage)."""
    evidence = evidence or [""] * len(completions)
    out = []
    for c, e in zip(completions, evidence):
        ev = e if isinstance(e, list) else [e]
        out.append(citation_coverage(c, ev))
    return out


def reward_format(completions, **kwargs) -> list[float]:
    """Format validity: rewards a structured answer that states a conclusion with figures."""
    out = []
    for c in completions:
        score = 0.0
        if ("结论" in c) or ("答案" in c):
            score += 0.5
        if numeric_keys(c):
            score += 0.5
        out.append(score)
    return out


def reward_no_hallucination(completions, evidence=None, **kwargs) -> list[float]:
    """Hallucination penalty proxy: fraction of numbers in the answer that ARE grounded in
    the evidence (1.0 = all grounded, lower = more invented figures)."""
    evidence = evidence or [""] * len(completions)
    out = []
    for c, e in zip(completions, evidence):
        ev = numeric_keys(e if isinstance(e, str) else " ".join(e))
        cn = numeric_keys(c)
        out.append(len([n for n in cn if n in ev]) / len(cn) if cn else 0.0)
    return out


ALL_REWARDS = [reward_answer_correctness, reward_citation, reward_format, reward_no_hallucination]
