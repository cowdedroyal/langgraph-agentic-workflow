#!/usr/bin/env python
"""P5 data prep: convert synthesized multi-hop QA -> SFT datasets (ShareGPT + Alpaca).

Three skills are baked into the training data (per roadmap P5):
  1. evidence-cited financial QA — answer from the retrieved annual-report context,
  2. financial-term QA — the question/answer pairs themselves,
  3. JSON routing/grading stability — small set of strict-JSON routing samples.

    python training/prepare_data.py
"""

from __future__ import annotations

from agentic_rag.config import get_settings
from agentic_rag.ingestion import load_documents
from agentic_rag.io_utils import write_jsonl
from agentic_rag.synthesis import load_qa

SYSTEM = "你是严谨的金融问答助手。请基于给定的财报资料进行多跳推理，引用关键财务数据后再作答。"
ROUTER_SYSTEM = "你是问题路由器。判断问题应使用本地财报向量库(vectorstore)还是联网搜索(websearch)，只返回 JSON。"

_ROUTER_VEC = [
    "山河酒业2022年的净利润是多少？",
    "长江动力的正极材料供应商是谁？",
    "云岭矿业的毛利率变化趋势如何？",
    "极锋材料2023年研发费用是多少？",
    "丰收农业的资产负债率是多少？",
]
_ROUTER_WEB = [
    "今天上证指数收盘是多少？",
    "最新的黄金现货价格是多少？",
    "本周美联储议息会议结果如何？",
]


def _context(qa, docs) -> str:
    return "\n\n".join(docs[e].content for e in qa.evidence_doc_ids if e in docs)


def _assistant(qa) -> str:
    steps = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(qa.reasoning_steps))
    return f"推理过程：\n{steps}\n\n结论：{qa.answer}"


def build(settings) -> dict:
    docs = {d.doc_id: d for d in load_documents(settings)}
    qas = load_qa(settings)

    sharegpt, alpaca = [], []
    for qa in qas:
        ctx = _context(qa, docs)
        user = f"资料：\n{ctx}\n\n问题：{qa.question}"
        assistant = _assistant(qa)
        sharegpt.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]})
        alpaca.append({"instruction": qa.question, "input": ctx, "output": assistant})

    # JSON routing/grading stability samples
    for q in _ROUTER_VEC:
        sharegpt.append({"messages": [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": '{"datasource": "vectorstore"}'},
        ]})
    for q in _ROUTER_WEB:
        sharegpt.append({"messages": [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": '{"datasource": "websearch"}'},
        ]})

    train_dir = settings.data_dir / "train"
    sp = train_dir / "sft_sharegpt.jsonl"
    ap = train_dir / "sft_alpaca.jsonl"
    write_jsonl(sp, sharegpt)
    write_jsonl(ap, alpaca)
    return {"sharegpt": len(sharegpt), "alpaca": len(alpaca), "path": str(sp)}


def main() -> None:
    info = build(get_settings())
    print(f"SFT data: {info['sharegpt']} ShareGPT samples -> {info['path']}")


if __name__ == "__main__":
    main()
