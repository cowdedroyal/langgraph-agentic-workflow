#!/usr/bin/env bash
# P7: launch a vLLM OpenAI-compatible server for the local Qwen2.5-7B-Instruct.
# The LangGraph LLM client switches onto it via LLM_BACKEND=vllm (see src/agentic_rag/llm.py).
#
#   bash serve/vllm_server.sh            # or: arag serve
#   LLM_BACKEND=vllm arag demo -q "..."  # point the graph at the server
set -euo pipefail

MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct}"
PORT="${VLLM_PORT:-8000}"
GPU="${VLLM_GPU:-0}"
UTIL="${VLLM_GPU_UTIL:-0.55}"
MAXLEN="${VLLM_MAX_LEN:-8192}"

echo "Starting vLLM: model=$MODEL gpu=$GPU port=$PORT util=$UTIL max_len=$MAXLEN"
CUDA_VISIBLE_DEVICES="$GPU" python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$MODEL" \
  --port "$PORT" \
  --gpu-memory-utilization "$UTIL" \
  --max-model-len "$MAXLEN" \
  --dtype bfloat16 \
  ${VLLM_ENFORCE_EAGER:+--enforce-eager} \
  --disable-log-requests \
  --disable-log-stats
