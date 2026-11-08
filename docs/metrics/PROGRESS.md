# 进度记录：访问路径实验（access-paths 分支）

最后更新：2026-11-08。分支 `access-paths` 已推送到 GitHub，还没有合并进 master。

## 要回答的问题

金融多跳问答里，模型该怎么拿到证据？语料越来越大时，各种方式的分界点在哪？
把语料看成数据库、问题看成连接查询，对比五条访问路径：

| 路径 | 数据库里的对应 | 做法 |
|---|---|---|
| `scan` | 全表扫描 | 整个语料塞进上下文 |
| `index` | 索引扫描 | BM25 + bge 混合检索，读 top-8 个块（原来的 RAG） |
| `index_join` | 索引 + 显式连接 | 先检索一次，让模型说出还缺哪些实体（连接键），再逐个检索 |
| `view` | 物化视图 | 先用 LLM 把每篇文档抽成事实表（一次性成本），再查表回答 |
| `agent` | 嵌套循环 | 模型自己调 search / read 工具 |

问题集固定 63 题（核心 6 家公司的查找题，加按行业的聚合题），只增加“干扰公司”。部分干扰公司和核心公司共用供应商。
标准答案由事实精确计算，判分用代码，不用 LLM 评委。模型 deepseek-v4-flash，temperature 0。

## 已有结果（准确率）

| 干扰公司数（语料 token） | scan | index | index_join | view | agent |
|---|---|---|---|---|---|
| 0（约 1.8 万） | 1.000 | 0.857 | 1.000 | 1.000 | 1.000 |
| 54（约 18 万） | 1.000 | 0.698 | 0.794 | 1.000 | 0.984 |
| 180（约 55 万） | 0.984 | 0.667 | 0.762 | **未跑** | **未跑** |
| 540（约 160 万，超出上下文） | 放不下 | **未跑** | **未跑** | **未跑** | **未跑** |

完整表（分题型、token、缓存命中、每题成本、延迟）：[access_paths.md](access_paths.md)，原始汇总：[access_paths_summary.json](access_paths_summary.json)。

## 目前的发现

1. **长上下文到 55 万 token 仍然最准**，前缀缓存命中 97–98%，所以每题只要 $0.001–0.003，“RAG 更便宜”的理由基本消失；剩下的代价是延迟（55 万 token 时中位数 31 秒，RAG 1.8 秒）。
2. **RAG 在供应链题上所有规模都是 0 分。** 原因是连接，不是排序：问题里没有供应商的名字，检索回来的块全是提问公司自己的。`index_join` 加一次显式连接后，无干扰时修到 1.0，有干扰时只到 0.625（相似的供应商名字把正确报告挤掉）。这也解释了上一轮“重排有害”：重排逐个文档对问题打分，第二跳文档本来就不像问题。
3. **聚合题（行业最高、超过 X 的有几家）top-k 检索天生做不了**：54 家干扰时 index 0.08，180 家时 0。
4. **物化视图性价比最好**：54 家时准确率 1.000，每题 $0.0002（加上建表摊销 $0.0026），建表一次 $0.15。局限是只能答表结构预想到的问题。

## 停在哪里

2026-11-08 晚，规模 180 的 view 建表时 DeepSeek 返回 429。**原因是 API 余额用完，不是限流。** 已停掉所有后台任务。
到停止时一共 2726 次调用，按列表价约 $0.94。

## 换电脑后的准备

```bash
git clone https://github.com/cowdedroyal-lab/langgraph-agentic-workflow.git
cd langgraph-agentic-workflow && git checkout access-paths
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install rank-bm25 jieba pydantic rich numpy faiss-cpu sentence-transformers python-dotenv pytest langchain-core
python -m pytest -q tests/test_access.py      # 6 个离线测试，应全部通过
```

- LLM 接口：默认走本机 magpie（`http://127.0.0.1:3425/v1`，key 随便填）。没有 magpie 就设置
  `ACCESS_BASE_URL=https://api.deepseek.com/v1`、`ACCESS_API_KEY=<DeepSeek key>`、`ACCESS_MODEL=deepseek-v4-flash`（模型名以接口实际返回为准）。
- 嵌入模型：第一次跑 index 路径会下载 `BAAI/bge-base-zh-v1.5`（约 400 MB）。连不上 HuggingFace 时设置 `HF_ENDPOINT=https://hf-mirror.com`。
- 缓存不在仓库里（`outputs/` 被 gitignore）。新电脑上已完成的调用要重跑一遍，约 $1；结果会和这里的数字略有出入（同一模型、temperature 0，但不保证逐字一致）。

## 明天接着做

1. **充值 DeepSeek 后先补跑**（已完成的调用有磁盘缓存，不会重复花钱）：
   ```bash
   cd ~/lab/langgraph-agentic-workflow && source .venv/bin/activate
   export TOKENIZERS_PARALLELISM=false PYTHONPATH=src
   python -W ignore -m agentic_rag.access.run --sizes 180 --paths view agent --workers 3
   python -W ignore -m agentic_rag.access.run --sizes 540 --paths scan index index_join view agent --workers 3
   ```
   预估：540 的 view 建表要抽约 4900 篇文档，约 $1.4；agent 约 $0.1；其余很少。整轮补跑约 $2。
   如果在别的电脑上跑：需要 magpie（默认 `http://127.0.0.1:3425/v1`）或用 `ACCESS_BASE_URL` / `ACCESS_API_KEY` / `ACCESS_MODEL` 指向别的 OpenAI 兼容接口；缓存在 `outputs/access/cache`（被 gitignore，没有它就会重新调用）。
2. **看 540 的结果**：scan 放不下之后，谁接得住？view 的抽取错误会不会随规模增长？agent 的步数和成本怎么变？
3. **做查询规划器**：按题型（查找 / 连接 / 聚合）和语料规模自动选路径，目标是准确率接近 scan、成本和延迟接近 index。这是这个项目真正的“突破点”。
4. **收尾**：写分析文档；README 改成围绕“哪种规模用哪条路径”；删掉玩具规模的 Ray / FSDP 卖点；更新 `.portfolio/project.yaml`（新指标都标 `scale: pilot`）；跑 `profile-atlas check .`；隐私扫描；问过再推送。
5. 之后的新项目：常驻 Agent 的长期记忆（页面置换 + Belady 上界、多版本事实、LSM 式压缩），评测用 LongMemEval 加本机私有的会话记录评测集（不进仓库）。

## 代码位置

- `src/agentic_rag/access/`：`world.py`（可扩展语料）、`questions.py`（题目与标准答案）、`paths.py`（五条路径）、`retrieval.py`（混合检索与文档检索）、`llm.py`（带缓存和成本统计的客户端）、`grade.py`（判分）、`run.py`（实验入口）
- `tests/test_access.py`：6 个离线测试，全部通过
- `src/agentic_rag/ingestion/synthetic.py`：渲染函数改成可传入 `World`，默认行为和原来逐字节一致（有测试）
