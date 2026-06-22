from agentic_rag.schema import DocMetadata, DocType, Document, MultiHopQA


def test_document_langchain_roundtrip():
    d = Document(doc_id="a1", content="营业收入120亿元",
                 metadata=DocMetadata(company="山河酒业", ticker="600011",
                                      period="2022", doc_type=DocType.annual_report))
    lc = d.to_langchain()
    assert lc.page_content == "营业收入120亿元"
    assert lc.metadata["doc_id"] == "a1"
    assert lc.metadata["doc_type"] == "annual_report"

    back = Document.from_langchain(lc)
    assert back.doc_id == "a1"
    assert back.metadata.company == "山河酒业"
    assert back.metadata.doc_type == DocType.annual_report
    assert back.metadata.period == "2022"


def test_make_id_stable_and_unique():
    assert Document.make_id("600011", "2022", "fin") == Document.make_id("600011", "2022", "fin")
    assert Document.make_id("600011", "2022") != Document.make_id("600011", "2023")


def test_multihop_qa_defaults():
    q = MultiHopQA(qa_id="q1", question="?", answer="a", evidence_doc_ids=["e1", "e2"])
    assert q.hop_type == "cross_year"
    assert q.evidence_doc_ids == ["e1", "e2"]
