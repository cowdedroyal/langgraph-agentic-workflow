# Agentic RAG 金融多跳问答技术路线

## 当前基线

- LangGraph 工作流：保留现有 `retrieve -> grade_documents -> generate -> grade_generation` 自纠错闭环。
- 本地推理：通过 Ollama 使用 `LOCAL_LLM` 配置模型，默认仍可指向 `qwen3:30b`。
- 检索缓存：保留现有 parquet vector store 缓存。
- 运行兜底：当 Nomic 本地 embedding 因系统依赖不可用时，使用 `HashingVectorizerEmbeddings` 做 smoke test。

## P0：稳定运行层

- 增加 `EXAMPLES_TO_RUN`、`LOCAL_LLM`、`EMBEDDINGS_BACKEND`、`OLLAMA_NUM_PREDICT` 等运行参数。
- 将示例问题拆成独立脚本或 CLI，避免主流程、demo、benchmark 混在一个文件中。
- 增加最小 smoke test：只验证图编译、检索、生成、评分节点可执行。

## P1：金融数据 Ingestion

- 新增 `data/`：保存年报、公告、研报等原始文件。
- 新增 `src/ingestion/`：
  - PDF/HTML/Markdown 解析。
  - 表格抽取与正文分块。
  - 元数据规范：公司、股票代码、报告期、公告类型、发布日期、来源 URL。
- 输出统一 `Document` JSONL，供向量检索和 BM25 共用。

## P2：FAISS + BM25 混合检索

- 新增 `src/retrieval/`：
  - `dense_faiss.py`：向量索引。
  - `sparse_bm25.py`：BM25 索引。
  - `hybrid.py`：三路融合，建议先实现 dense + BM25 + metadata filter。
- 排序策略先用 Reciprocal Rank Fusion，后续再接 reranker。
- 保留 LangGraph 节点接口不变，只替换 `retriever.invoke(question)` 的实现。

## P3：多跳 QA 数据合成

- 新增 `src/synthesis/`：
  - 从同一公司跨年度、上下游公司、财务指标变化中采样证据链。
  - 生成 `question / evidence_docs / reasoning_steps / answer`。
  - 对每条样本保留可追溯引用，便于 Judge 检查。

## P4：Judge 与 Benchmark

- 新增 `src/eval/`：
  - 检索指标：Recall@k、MRR、nDCG。
  - 生成指标：faithfulness、answer relevance、citation coverage。
  - 端到端指标：多跳命中率、最终答案准确率、平均重试次数。
- Judge 先复用当前答案有效性/幻觉检测 prompt，后续替换为更强本地/远程评审模型。

## P5：LLaMA-Factory LoRA/SFT

- 将 P3 产出的多跳 QA 转成 ShareGPT/Alpaca 格式。
- 训练目标先聚焦：
  - 金融术语问答。
  - 证据引用式回答。
  - JSON 路由/评分稳定性。
- LoRA 产物作为 Ollama/vLLM 推理模型候选。

## P6：veRL / GRPO 后训练

- 奖励函数从 P4 指标复用：
  - 答案正确性。
  - 证据覆盖。
  - 幻觉惩罚。
  - 格式合法性。
- 先做小模型 GRPO smoke run，再扩到主模型。

## P7：vLLM 推理服务

- 新增 `serve/`：
  - vLLM OpenAI-compatible endpoint。
  - LangGraph LLM client 通过环境变量切换 Ollama/vLLM。
- 优先服务生成模型；embedding 和 reranker 可后置拆分。

## P8：FSDP + Ray 调度

- Ray 先用于 ingestion、索引构建、benchmark 并行。
- FSDP 用于 SFT/GRPO 训练阶段，不侵入在线 RAG 工作流。
- 保持训练、评测、在线推理三条路径解耦，避免主流程承担分布式复杂度。
