# Benchmark Results

_Generated 2026-06-06 18:47 · 51 multi-hop QA · embeddings=bge (BAAI/bge-base-zh-v1.5) · llm=vllm:Qwen2.5-7B-Instruct_

## 1. Retrieval (roadmap P2)

Ground truth = synthesized `evidence_doc_ids`. `Hit@5` requires the *full* multi-hop evidence chain in the top-k.

| method | Recall@5 | Recall@10 | MRR | nDCG@5 | Hit@5 |
|---|---|---|---|---|---|
| Dense (FAISS · bge-zh) | 0.4412 | 0.6569 | 0.4530 | 0.3386 | 0.0000 |
| BM25 (jieba) | 0.2549 | 0.6765 | 0.1574 | 0.1302 | 0.2157 |
| Hybrid (RRF) | 0.5686 | 0.7843 | 0.5957 | 0.4689 | 0.2549 |
| Hybrid + Reranker (bge-reranker-base) | 0.3039 | 0.7451 | 0.2621 | 0.1984 | 0.0784 |

> **Ablation finding**: a naive cross-encoder reranker *hurts* multi-hop retrieval here
> (Hit@5 0.255 → 0.078, MRR 0.596 → 0.262). It reorders by single query–doc relevance and
> pushes the *indirectly* relevant second-hop evidence (another year / a supplier's report)
> out of the top-k. RRF stays best for multi-hop; multi-hop-aware reranking is future work.

## 2. End-to-end Agentic RAG (roadmap P0/P4)

| n | answer_accuracy | citation_coverage | avg_retries | avg_latency_s | faithfulness | answer_relevance |
|---|---|---|---|---|---|---|
| 10 | 0.6800 | 0.9500 | 1.2000 | 6.8670 | 0.7000 | 0.8000 |

`avg_retries` = mean self-correction loops; lower latency/retries with stable answers is better.
