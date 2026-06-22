"""Multi-hop QA synthesis package (roadmap P3)."""

from .pipeline import load_qa, run_synthesis
from .samplers import EvidenceChain, sample_all

__all__ = ["run_synthesis", "load_qa", "sample_all", "EvidenceChain"]
