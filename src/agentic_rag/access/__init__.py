"""Access-path study: which way should a model reach the evidence for multi-hop QA?

The corpus is treated like a database and the question like a join. Four access paths
are compared on the *same* questions while the corpus grows with distractor companies:

  * ``scan``    — full scan: put the whole corpus in the context window (long context).
  * ``index``   — index scan: hybrid BM25 + dense retrieval, then read the top-k chunks (RAG).
  * ``view``    — materialized view: extract a facts table once, answer by lookup + arithmetic.
  * ``agent``   — nested loop: the model drives its own search / read tool calls.

Every run records accuracy, tokens (split into cache hits and misses), dollars and latency,
so the question "which path, at which corpus size" can be answered with numbers.
"""
