"""The four access paths. Each takes a question and returns an answer plus usage.

Shared rules so the comparison is fair:
  * same model, temperature 0, same answer format (a final JSON line),
  * the corpus is identical across paths; only *how the model reaches it* differs,
  * prompts put the static part (corpus / table) first and the question last, so the
    provider's prefix cache can reuse it across questions — that is part of what is measured.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from .llm import LLM, Usage
from .world import RawDoc

ANSWER_RULE = (
    "回答要求：先简短推理，最后单独一行输出 JSON：{\"answer\": <答案>}。"
    "数值题只填数字（不带单位、不带%），百分比变化填百分数数值（例如下降 3.5% 填 -3.5）；"
    "实体题填公司简称；计数题填整数。若资料中找不到答案，填 null。"
)


@dataclass
class PathResult:
    answer: object
    text: str
    usage: Usage
    trace: list = field(default_factory=list)
    context_tokens_est: int = 0


_JSON_TAIL = re.compile(r"\{[^{}]*\"answer\"[^{}]*\}", re.S)


def parse_answer(text: str):
    m = _JSON_TAIL.findall(text or "")
    if not m:
        return None
    raw = m[-1]
    try:
        return json.loads(raw).get("answer")
    except json.JSONDecodeError:
        # models sometimes emit an unquoted string: {"answer": 长江动力}
        v = re.search(r"\"answer\"\s*:\s*(.+?)\s*\}", raw, re.S)
        if not v:
            return None
        s = v.group(1).strip().strip("\"'“”")
        return None if s.lower() in ("null", "none", "") else s


def doc_block(d: RawDoc) -> str:
    return f"<doc id=\"{d.doc_id}\">\n{d.text.strip()}\n</doc>"


# --------------------------------------------------------------------------- scan
class ScanPath:
    """Full scan: the whole corpus goes into the context window."""

    name = "scan"

    def __init__(self, llm: LLM, docs: list[RawDoc]):
        self.llm = llm
        self.corpus = "\n".join(doc_block(d) for d in docs)

    def answer(self, question: str) -> PathResult:
        sys = ("你是金融分析助手。下面是完整的资料库，只能依据资料回答。\n\n" + self.corpus)
        r = self.llm.chat([{"role": "system", "content": sys},
                           {"role": "user", "content": f"{question}\n\n{ANSWER_RULE}"}], max_tokens=4096)
        return PathResult(parse_answer(r.text), r.text, r.usage, context_tokens_est=r.usage.prompt)


# --------------------------------------------------------------------------- index
class IndexPath:
    """Index scan: hybrid BM25 + dense retrieval over chunks, the model reads the top-k."""

    name = "index"

    def __init__(self, llm: LLM, retriever, topk: int = 8):
        self.llm, self.retriever, self.topk = llm, retriever, topk

    def answer(self, question: str) -> PathResult:
        hits = self.retriever.search(question, self.topk)
        ctx = "\n".join(f"<chunk id=\"{h.chunk_id}\">\n{h.text}\n</chunk>" for h in hits)
        r = self.llm.chat([
            {"role": "system", "content": "你是金融分析助手，只能依据给出的检索片段回答。"},
            {"role": "user", "content": f"检索片段：\n{ctx}\n\n问题：{question}\n\n{ANSWER_RULE}"}],
            max_tokens=4096)
        return PathResult(parse_answer(r.text), r.text, r.usage,
                          trace=[h.chunk_id for h in hits], context_tokens_est=r.usage.prompt)


# --------------------------------------------------------------------------- index + join
class IndexJoinPath(IndexPath):
    """Index scan with an explicit join step. Top-k retrieval scores each chunk against the
    whole question, so the second hop (the supplier's own report) never ranks: the question
    does not contain the supplier's name. This path reads the first hop, lets the model name
    the entities it still needs (the join key), retrieves again for each, then answers."""

    name = "index_join"
    PLAN = ("根据检索片段，判断回答问题还需要查哪些公司的资料（例如片段里提到的供应商）。"
            "只输出 JSON：{\"need\": [\"公司简称 年份 需要的指标\", ...]}，不需要再查则输出 {\"need\": []}。最多 3 条。")

    def answer(self, question: str) -> PathResult:
        hits = self.retriever.search(question, self.topk)
        ctx = "\n".join(f"<chunk id=\"{h.chunk_id}\">\n{h.text}\n</chunk>" for h in hits)
        plan = self.llm.chat([{"role": "user", "content": f"问题：{question}\n\n检索片段：\n{ctx}\n\n{self.PLAN}"}],
                             max_tokens=1500)
        total = Usage().add(plan.usage)
        need = []
        m = re.search(r"\{.*\}", plan.text, re.S)
        if m:
            try:
                need = [str(x) for x in (json.loads(m.group(0)).get("need") or [])][:3]
            except json.JSONDecodeError:
                need = []
        seen = {h.chunk_id for h in hits}
        for sub in need:
            for h in self.retriever.search(sub, max(2, self.topk // 2)):
                if h.chunk_id not in seen:
                    seen.add(h.chunk_id)
                    hits.append(h)
        ctx = "\n".join(f"<chunk id=\"{h.chunk_id}\">\n{h.text}\n</chunk>" for h in hits)
        r = self.llm.chat([
            {"role": "system", "content": "你是金融分析助手，只能依据给出的检索片段回答。"},
            {"role": "user", "content": f"检索片段：\n{ctx}\n\n问题：{question}\n\n{ANSWER_RULE}"}],
            max_tokens=4096)
        total.add(r.usage)
        return PathResult(parse_answer(r.text), r.text, total,
                          trace={"need": need, "chunks": [h.chunk_id for h in hits]}, context_tokens_est=r.usage.prompt)


# --------------------------------------------------------------------------- view
class ViewPath:
    """Materialized view: extract a facts table from every document once (cost amortized
    over all questions), then answer from the table. Extraction is done by the same LLM, so
    extraction errors are part of what is measured."""

    name = "view"
    EXTRACT = (
        "从下面这份文档中抽取结构化事实，只输出一个 JSON 对象，不要解释：\n"
        "{\"company\": 公司简称, \"ticker\": 代码, \"year\": 报告年份(整数), \"sector\": 主营行业或null,\n"
        " \"revenue\": 营业收入(亿元)或null, \"net_profit\": 归母净利润(亿元)或null, \"gross_margin\": 销售毛利率(%)或null,\n"
        " \"suppliers\": [{\"name\": 供应商简称, \"material\": 供应物料}], \"customers\": [客户简称]}\n"
        "只抽取文档里报告年份本身的数据，不要抽上年同期。\n\n"
    )

    def __init__(self, llm: LLM, docs: list[RawDoc], workers: int = 8):
        self.llm = llm
        self.build_usage = Usage()
        self.rows: list[dict] = []
        self.failed = 0
        from concurrent.futures import ThreadPoolExecutor

        def one(d: RawDoc):
            try:
                r = self.llm.chat([{"role": "user", "content": self.EXTRACT + d.text}], max_tokens=1500)
            except RuntimeError:  # gave up after retries: counts as a failed extraction
                return None, Usage(), d
            m = re.search(r"\{.*\}", r.text, re.S)
            try:
                row = json.loads(m.group(0)) if m else None
            except json.JSONDecodeError:
                row = None
            return row, r.usage, d

        with ThreadPoolExecutor(workers) as ex:
            for row, u, d in ex.map(one, docs):
                self.build_usage.add(u)
                if row is None:
                    self.failed += 1
                    continue
                row["_doc"] = d.doc_id
                self.rows.append(row)
        self.table = self._merge()

    def _merge(self) -> str:
        """Merge per-document rows into one row per (ticker, year); later docs fill gaps."""
        merged: dict[tuple, dict] = {}
        for r in self.rows:
            key = (str(r.get("ticker") or r.get("company")), r.get("year"))
            cur = merged.setdefault(key, {"company": r.get("company"), "ticker": r.get("ticker"),
                                          "year": r.get("year"), "suppliers": [], "customers": []})
            for k in ("sector", "revenue", "net_profit", "gross_margin"):
                if cur.get(k) is None and r.get(k) is not None:
                    cur[k] = r[k]
            for s in r.get("suppliers") or []:
                if isinstance(s, dict) and s not in cur["suppliers"]:
                    cur["suppliers"].append(s)
            for c in r.get("customers") or []:
                if c not in cur["customers"]:
                    cur["customers"].append(c)
        lines = ["company|ticker|year|sector|revenue|net_profit|gross_margin|suppliers|customers"]
        for v in sorted(merged.values(), key=lambda x: (str(x["ticker"]), str(x["year"]))):
            sup = ";".join(f"{s.get('name')}({s.get('material')})" for s in v["suppliers"])
            lines.append("|".join(str(x) for x in (v["company"], v["ticker"], v["year"], v.get("sector"),
                                                   v.get("revenue"), v.get("net_profit"), v.get("gross_margin"),
                                                   sup, ",".join(map(str, v["customers"])))))
        self.n_rows = len(merged)
        return "\n".join(lines)

    def answer(self, question: str) -> PathResult:
        sys = ("你是金融分析助手。下面是从资料库抽取出的事实表（每行一家公司一个年份，金额单位亿元，毛利率单位%），"
               "只能依据此表回答，需要计算时自己算。\n\n" + self.table)
        r = self.llm.chat([{"role": "system", "content": sys},
                           {"role": "user", "content": f"{question}\n\n{ANSWER_RULE}"}], max_tokens=4096)
        return PathResult(parse_answer(r.text), r.text, r.usage, context_tokens_est=r.usage.prompt)


# --------------------------------------------------------------------------- agent
class AgentPath:
    """Nested loop: the model plans its own lookups with two tools, ``search`` (BM25 over
    whole documents, returns ids + a snippet) and ``read`` (one full document)."""

    name = "agent"
    TOOLS = [
        {"type": "function", "function": {
            "name": "search", "description": "关键词检索资料库，返回最相关的文档 id 和摘要（最多 8 条）。",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "read", "description": "读取一篇文档的全文。",
            "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}}, "required": ["doc_id"]}}},
    ]

    def __init__(self, llm: LLM, docs: list[RawDoc], doc_search, max_steps: int = 12):
        self.llm, self.search, self.max_steps = llm, doc_search, max_steps
        self.by_id = {d.doc_id: d for d in docs}

    def _run_tool(self, name: str, args: dict) -> str:
        if name == "search":
            hits = self.search(args.get("query", ""), 8)
            return "\n".join(f"{d.doc_id}: {d.text.strip().splitlines()[1][:80] if len(d.text.splitlines()) > 1 else ''}"
                             for d in hits) or "无结果"
        if name == "read":
            d = self.by_id.get(args.get("doc_id", ""))
            return d.text if d else "文档不存在"
        return "未知工具"

    def answer(self, question: str) -> PathResult:
        msgs = [{"role": "system", "content": "你是金融分析助手。用 search 和 read 工具在资料库中查证，"
                                              "不要凭记忆回答。查到足够证据后给出答案。"},
                {"role": "user", "content": f"{question}\n\n{ANSWER_RULE}"}]
        total, trace, text = Usage(), [], ""
        for _ in range(self.max_steps):
            r = self.llm.chat(msgs, tools=self.TOOLS, max_tokens=4096)
            total.add(r.usage)
            calls = r.message.get("tool_calls") or []
            if not calls:
                text = r.text
                break
            msgs.append({"role": "assistant", "content": r.message.get("content") or "", "tool_calls": calls})
            for c in calls:
                try:
                    args = json.loads(c["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                out = self._run_tool(c["function"]["name"], args)
                trace.append((c["function"]["name"], args))
                msgs.append({"role": "tool", "tool_call_id": c["id"], "content": out})
        else:
            msgs.append({"role": "user", "content": "步数已用完，请根据已查到的信息给出最终答案。" + ANSWER_RULE})
            r = self.llm.chat(msgs, tools=self.TOOLS, max_tokens=4096, tool_choice="none")
            total.add(r.usage)
            text = r.text
        return PathResult(parse_answer(text), text, total, trace=trace)
