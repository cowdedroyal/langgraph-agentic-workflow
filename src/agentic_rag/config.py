"""Single source of truth for runtime configuration.

Precedence: built-in dataclass defaults  <  environment variables (loaded from
``.env`` if present). ``config/default.yaml`` is a human-readable mirror of the
defaults below and is intentionally NOT parsed here, so the package never depends
on PyYAML and there is exactly one authoritative default per setting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
# .env loading + typed env readers
# --------------------------------------------------------------------------- #
def _load_dotenv() -> None:
    """Load PROJECT_ROOT/.env. Prefer python-dotenv; fall back to a tiny parser."""
    env_path = PROJECT_ROOT / ".env"
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except Exception:
        pass
    if env_path.exists():
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def _s(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _i(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return int(default)


def _f(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _b(key: str, default: bool) -> bool:
    return str(os.environ.get(key, default)).strip().lower() in ("1", "true", "yes", "on")


def _p(key: str, default: Path) -> Path:
    v = os.environ.get(key)
    return Path(v).expanduser() if v else default


# --------------------------------------------------------------------------- #
# Settings
# --------------------------------------------------------------------------- #
@dataclass
class Settings:
    """Resolved runtime settings. Construct via :func:`get_settings`."""

    # -- paths --
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    raw_dir: Path = PROJECT_ROOT / "data" / "raw"
    corpus_path: Path = PROJECT_ROOT / "data" / "corpus" / "documents.jsonl"
    qa_path: Path = PROJECT_ROOT / "data" / "qa" / "multihop_qa.jsonl"
    index_dir: Path = PROJECT_ROOT / "data" / "index"
    cache_dir: Path = PROJECT_ROOT / "agentic-exp" / "cache"
    output_dir: Path = PROJECT_ROOT / "outputs"

    # -- llm --
    llm_backend: str = "ollama"  # ollama | vllm | openai
    local_llm: str = "qwen2.5:0.5b"
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_predict: int = 512
    ollama_json_num_predict: int = 256
    vllm_base_url: str = "http://127.0.0.1:8000/v1"
    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    openai_api_key: str = "EMPTY"
    llm_temperature: float = 0.0
    request_timeout: int = 120

    # -- embeddings (three-tier fallback: bge -> nomic -> hashing) --
    embeddings_backend: str = "bge"
    bge_model_name: str = "BAAI/bge-base-zh-v1.5"
    embed_device: str = "auto"  # auto | cuda | cpu
    embed_batch_size: int = 32
    bge_query_instruction: str = "为这个句子生成表示以用于检索相关文章："
    hashing_features: int = 2048

    # -- retrieval (P2 hybrid) --
    retrieval_topk: int = 5
    dense_topk: int = 20
    bm25_topk: int = 20
    rrf_k: int = 60
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"
    reranker_topk: int = 5

    # -- graph (self-correction loop) --
    max_retries: int = 3
    recursion_limit: int = 50
    enable_web_search: bool = False

    # -- run --
    examples_to_run: int = 3

    # ------------------------------------------------------------------ #
    @property
    def tavily_enabled(self) -> bool:
        v = os.environ.get("TAVILY_API_KEY", "").strip()
        return bool(v and not v.startswith("<insert"))

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        data_dir = _p("DATA_DIR", cls.data_dir)
        return cls(
            data_dir=data_dir,
            raw_dir=_p("RAW_DIR", data_dir / "raw"),
            corpus_path=_p("CORPUS_PATH", data_dir / "corpus" / "documents.jsonl"),
            qa_path=_p("QA_PATH", data_dir / "qa" / "multihop_qa.jsonl"),
            index_dir=_p("INDEX_DIR", data_dir / "index"),
            output_dir=_p("OUTPUT_DIR", cls.output_dir),
            cache_dir=_p("CACHE_DIR", cls.cache_dir),
            llm_backend=_s("LLM_BACKEND", cls.llm_backend),
            local_llm=_s("LOCAL_LLM", cls.local_llm),
            ollama_base_url=_s("OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_num_predict=_i("OLLAMA_NUM_PREDICT", cls.ollama_num_predict),
            ollama_json_num_predict=_i("OLLAMA_JSON_NUM_PREDICT", cls.ollama_json_num_predict),
            vllm_base_url=_s("VLLM_BASE_URL", cls.vllm_base_url),
            vllm_model=_s("VLLM_MODEL", cls.vllm_model),
            openai_api_key=_s("OPENAI_API_KEY", cls.openai_api_key),
            llm_temperature=_f("LLM_TEMPERATURE", cls.llm_temperature),
            request_timeout=_i("REQUEST_TIMEOUT", cls.request_timeout),
            embeddings_backend=_s("EMBEDDINGS_BACKEND", cls.embeddings_backend),
            bge_model_name=_s("BGE_MODEL_NAME", cls.bge_model_name),
            embed_device=_s("EMBED_DEVICE", cls.embed_device),
            embed_batch_size=_i("EMBED_BATCH_SIZE", cls.embed_batch_size),
            bge_query_instruction=_s("BGE_QUERY_INSTRUCTION", cls.bge_query_instruction),
            hashing_features=_i("HASHING_FEATURES", cls.hashing_features),
            retrieval_topk=_i("RETRIEVAL_TOPK", cls.retrieval_topk),
            dense_topk=_i("DENSE_TOPK", cls.dense_topk),
            bm25_topk=_i("BM25_TOPK", cls.bm25_topk),
            rrf_k=_i("RRF_K", cls.rrf_k),
            use_reranker=_b("USE_RERANKER", cls.use_reranker),
            reranker_model=_s("RERANKER_MODEL", cls.reranker_model),
            reranker_topk=_i("RERANKER_TOPK", cls.reranker_topk),
            max_retries=_i("MAX_RETRIES", cls.max_retries),
            recursion_limit=_i("RECURSION_LIMIT", cls.recursion_limit),
            enable_web_search=_b("ENABLE_WEB_SEARCH", cls.enable_web_search),
            examples_to_run=_i("EXAMPLES_TO_RUN", cls.examples_to_run),
        )

    def resolve_device(self) -> str:
        """Resolve ``embed_device='auto'`` to a concrete torch device string."""
        if self.embed_device != "auto":
            return self.embed_device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def ensure_dirs(self) -> "Settings":
        for p in (
            self.data_dir,
            self.raw_dir,
            self.corpus_path.parent,
            self.qa_path.parent,
            self.index_dir,
            self.cache_dir,
            self.output_dir,
        ):
            Path(p).mkdir(parents=True, exist_ok=True)
        return self

    def summary(self) -> dict[str, Any]:
        """Compact view for logging (paths -> str)."""
        out: dict[str, Any] = {}
        for f in fields(self):
            v = getattr(self, f.name)
            out[f.name] = str(v) if isinstance(v, Path) else v
        out["tavily_enabled"] = self.tavily_enabled
        return out


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide cached settings."""
    return Settings.from_env()


def reset_settings_cache() -> None:
    """Drop the cache (used by tests that mutate the environment)."""
    get_settings.cache_clear()
