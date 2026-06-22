"""Evaluation package (roadmap P4)."""

from .benchmark import run_benchmark, run_end_to_end_benchmark, run_retrieval_benchmark
from .generation_metrics import answer_accuracy, citation_coverage
from .judge import LLMJudge
from .retrieval_metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k

__all__ = [
    "run_benchmark",
    "run_retrieval_benchmark",
    "run_end_to_end_benchmark",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
    "hit_at_k",
    "answer_accuracy",
    "citation_coverage",
    "LLMJudge",
]
