# agentic-rag-finance — one-stop commands. Override the interpreter with:
#   make <target> PY=/path/to/python
PY  ?= /data/wtw/envs/langgraph-agentic-workflow/bin/python
PIP := $(PY) -m pip
ARAG := $(PY) -m agentic_rag.cli

.PHONY: help setup install-core install-train install-serve install-ray \
        corpus index demo smoke bench prepare-data sft grpo serve health ray graph clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

setup: install-core  ## Install core deps + the package (editable)

install-core:  ## Install P0-P4 core RAG dependencies
	$(PIP) install -r requirements.txt && $(PIP) install -e .

install-train:  ## Install P5/P6 training stack
	$(PIP) install -r requirements-train.txt

install-serve:  ## Install P7 vLLM serving stack
	$(PIP) install -r requirements-serve.txt

install-ray:  ## Install P8 Ray
	$(PIP) install ray

corpus:  ## P1: build synthetic financial corpus + multi-hop QA
	$(ARAG) corpus

index:  ## P2: build FAISS + BM25 indices
	$(ARAG) index

demo:  ## Run one query through the agentic graph (ARGS='-q "..."')
	$(ARAG) demo $(ARGS)

smoke:  ## Fast smoke test (graph compile + retrieve + generate + grade)
	$(PY) -m pytest tests/test_smoke.py

bench:  ## P4: end-to-end + retrieval benchmark -> docs/results.md
	$(ARAG) bench $(ARGS)

prepare-data:  ## P5: convert multi-hop QA -> ShareGPT/Alpaca SFT data
	$(PY) training/prepare_data.py

sft:  ## P5: LoRA SFT smoke run (trl + peft)
	$(PY) training/sft_lora.py --config config/sft_lora.yaml

grpo:  ## P6: GRPO post-training smoke run (trl)
	$(PY) training/grpo_train.py --config config/grpo.yaml

serve:  ## P7: launch vLLM OpenAI-compatible server (Qwen2.5-7B-Instruct)
	bash serve/vllm_server.sh

health:  ## P7: vLLM healthcheck + latency/QPS probe
	$(PY) serve/healthcheck.py

ray:  ## P8: Ray parallel benchmark
	$(PY) ray_jobs/parallel_bench.py

graph:  ## Render the LangGraph workflow to graph.png
	$(ARAG) graph

clean:  ## Remove generated artifacts
	rm -rf data/index outputs .pytest_cache && find . -name __pycache__ -type d -prune -exec rm -rf {} +
