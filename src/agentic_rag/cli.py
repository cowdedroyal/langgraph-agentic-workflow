"""Unified CLI (roadmap P0): `arag <command>`.

    arag corpus     # P1+P3: synthetic corpus -> ingest -> multi-hop QA
    arag index      # P2: build FAISS + BM25 indices
    arag demo -q "..."   # run one question through the self-correcting graph
    arag bench           # P4: retrieval + end-to-end benchmark -> docs/results.md
    arag graph           # render the workflow to graph.png
    arag config          # print resolved settings
    arag sft|grpo|serve|health|ray   # P5-P8 entry points (delegate to scripts)
"""

from __future__ import annotations

import json
import subprocess
import sys

import typer

from .config import get_settings

app = typer.Typer(add_completion=False, help="Financial multi-hop Agentic RAG")

_DEMO_Q = "山河酒业2023年的营业收入较2021年增长了百分之多少？"


@app.command()
def corpus():
    """P1+P3: build synthetic corpus, ingest to documents.jsonl, synthesize multi-hop QA."""
    from .ingestion import build_corpus, run_ingestion
    from .synthesis import run_synthesis

    s = get_settings()
    c = build_corpus(s)
    ing = run_ingestion(s)
    qa = run_synthesis(s)
    typer.echo(f"corpus: {c['raw_files']} raw files -> {ing['chunks']} chunks | "
               f"QA: {qa['qa']} ({qa['hop_types']})")


@app.command()
def index():
    """P2: build the FAISS (dense) + BM25 (sparse) indices."""
    from .retrieval import build_indices

    info = build_indices(get_settings())
    typer.echo(f"indices: {info['docs']} docs -> {info['index_dir']}")


@app.command()
def demo(question: str = typer.Option(_DEMO_Q, "-q", "--question")):
    """Run one question through the agentic graph (node trace is logged)."""
    from .graph import build_graph

    s = get_settings()
    graph = build_graph(s)
    state = graph.invoke({"question": question}, {"recursion_limit": s.recursion_limit})
    typer.echo("\n=== Answer ===")
    typer.echo(state.get("generation") or "(no answer)")
    docs = state.get("documents") or []
    cites = sorted({d.metadata.get("doc_id") for d in docs if getattr(d, "metadata", None)})
    typer.echo(f"\nretries={state.get('loop_step', 0)} | docs={len(docs)} | cites={cites[:5]}")


@app.command()
def bench(mode: str = typer.Option("all", help="all | retrieval | e2e"),
          limit: int = typer.Option(20, help="end-to-end QA limit"),
          rerank: bool = typer.Option(False, help="add hybrid+rerank row"),
          judge: bool = typer.Option(False, help="LLM-judge faithfulness/relevance")):
    """P4: run benchmarks and write docs/results.md."""
    from .eval import run_benchmark

    res = run_benchmark(get_settings(), mode=mode, limit=limit, with_rerank=rerank, judge=judge)
    typer.echo(f"results -> {res['results_path']}")


@app.command()
def graph(out: str = typer.Option("graph.png", "-o", "--out")):
    """Render the LangGraph workflow to an image."""
    from .graph import render_graph

    typer.echo(f"graph -> {render_graph(out, get_settings())}")


@app.command()
def config():
    """Print the resolved runtime configuration."""
    typer.echo(json.dumps(get_settings().summary(), ensure_ascii=False, indent=2))


def _run_script(rel_path: str, *args: str) -> None:
    s = get_settings()
    script = s.project_root / rel_path
    if not script.exists():
        typer.echo(f"missing script: {script}", err=True)
        raise typer.Exit(1)
    raise typer.Exit(subprocess.call([sys.executable, str(script), *args]))


@app.command()
def sft(config_path: str = typer.Option("config/sft_lora.yaml", "--config")):
    """P5: LoRA SFT (delegates to training/sft_lora.py)."""
    _run_script("training/sft_lora.py", "--config", config_path)


@app.command()
def grpo(config_path: str = typer.Option("config/grpo.yaml", "--config")):
    """P6: GRPO post-training (delegates to training/grpo_train.py)."""
    _run_script("training/grpo_train.py", "--config", config_path)


@app.command()
def serve():
    """P7: launch the vLLM OpenAI-compatible server."""
    _run_script("serve/vllm_server.sh")


@app.command()
def health():
    """P7: vLLM healthcheck + latency probe."""
    _run_script("serve/healthcheck.py")


@app.command()
def ray():
    """P8: Ray parallel benchmark."""
    _run_script("ray_jobs/parallel_bench.py")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
