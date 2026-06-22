"""Smoke test (roadmap P0): the graph compiles and a question flows through
retrieve -> grade -> generate -> grade-generation without raising. Robust even if the
Ollama backend is unreachable (nodes degrade gracefully)."""


def test_graph_compiles(project):
    from agentic_rag.graph import build_graph

    graph = build_graph(project)
    assert graph is not None


def test_end_to_end_runs(project):
    from agentic_rag.graph import build_graph

    graph = build_graph(project)
    state = graph.invoke(
        {"question": "山河酒业2023年的营业收入较2021年增长了百分之多少？"},
        {"recursion_limit": project.recursion_limit},
    )
    assert isinstance(state, dict)
    assert "question" in state
    # generation key is present (may be a fallback string if the LLM is unavailable)
    assert "generation" in state
