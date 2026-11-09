"""Access paths at the KV-cache level (GPU).

The access-path study found that top-k retrieval fails on supply-chain questions because of
the *join*, not the ranking: the question never names the supplier, so a scorer that rates
chunks against the question never reaches the second hop. Query-aware KV compression
(SnapKV) scores the KV cache with the question's attention, which has the same shape. This
package tests whether that failure transfers, using paired questions that differ only in
whether the entity has to be joined or is named outright.

Only ``questions`` is importable without torch; ``engine`` and ``run`` need a GPU.
"""
