from agentic_rag.eval.retrieval_metrics import hit_at_k, mrr, ndcg_at_k, recall_at_k
from agentic_rag.retrieval.base import match_filters, rrf_fuse
from agentic_rag.schema import DocMetadata, Document


def test_rrf_fuse_rewards_agreement():
    a = [("x", 1.0), ("y", 0.5)]
    b = [("y", 2.0), ("z", 1.0)]
    fused = dict(rrf_fuse([a, b], k=60))
    assert fused["y"] > fused["x"] and fused["y"] > fused["z"]


def test_match_filters():
    d = Document(doc_id="1", content="c", metadata=DocMetadata(ticker="600011", period="2022"))
    assert match_filters(d, {"ticker": "600011"})
    assert match_filters(d, None)
    assert not match_filters(d, {"period": "2021"})


def test_retrieval_metrics():
    assert recall_at_k(["a", "b", "c"], ["a", "c"], 3) == 1.0
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5
    assert hit_at_k(["a", "b"], ["a", "b"], 2) == 1.0
    assert hit_at_k(["a", "x"], ["a", "b"], 2) == 0.0
    assert mrr(["x", "a"], ["a"]) == 0.5
    assert ndcg_at_k(["a", "b"], ["a"], 2) == 1.0


def test_hybrid_index_search(project):
    from agentic_rag.retrieval import load_retriever

    retr = load_retriever(project)
    hits = retr.search("山河酒业 营业收入", topk=3, method="hybrid")
    assert len(hits) >= 1
    # LangGraph-compatible interface returns langchain Documents
    docs = retr.invoke("山河酒业 营业收入")
    assert docs and hasattr(docs[0], "page_content")
