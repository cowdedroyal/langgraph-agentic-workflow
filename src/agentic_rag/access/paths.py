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
        self._merged = merged
        return "\n".join(lines)

    def answer(self, question: str) -> PathResult:
        sys = ("你是金融分析助手。下面是从资料库抽取出的事实表（每行一家公司一个年份，金额单位亿元，毛利率单位%），"
               "只能依据此表回答，需要计算时自己算。\n\n" + self.table)
        r = self.llm.chat([{"role": "system", "content": sys},
                           {"role": "user", "content": f"{question}\n\n{ANSWER_RULE}"}], max_tokens=4096)
        return PathResult(parse_answer(r.text), r.text, r.usage, context_tokens_est=r.usage.prompt)


# --------------------------------------------------------------------------- view + SQL
class ViewSQLPath(ViewPath):
    """Materialized view queried by code. Same extraction as ``view`` (identical requests, so
    the cache makes the build free when ``view`` ran first), but the rows go into SQLite and
    the model writes SQL instead of reading the table. Counting, max and joins are then done
    by the database, so the remaining errors come from extraction and from SQL generation,
    not from the model miscounting a long table.

    The prompt gets the schema plus the distinct ``sector`` values (the catalog an optimizer
    would have), because extracted sector strings are free text."""

    name = "view_sql"
    SQL = ("下面是一个 SQLite 数据库，内容是从年报、公告、研报抽取出的事实。\n{schema}\n\n"
           "外键：suppliers.supplier 和 customers.customer 与 facts.company 是同一套公司简称，可以直接等值连接。\n"
           "sector 和 material 是抽取出的自由文本，同一含义可能有好几种写法。下面是它们的全部取值；"
           "按这两列筛选时，按含义把所有相关取值都写进 IN (...)，不要用表里没有的取值。\n"
           "sector 全部取值：{sectors}\nmaterial 全部取值：{materials}\n\n各表示例行：\n{samples}\n\n"
           "写一条 SQLite 查询来回答问题。金额单位亿元，毛利率单位%。问题里的公司名就是 company 列的值，直接用等值匹配。"
           "只输出 SQL，放在 ```sql 代码块里。\n\n问题：{q}")
    ANSWER = ("问题：{q}\n\n为回答它执行的 SQL：\n{sql}\n\n结果（最多 50 行）：\n{rows}\n\n"
              "根据结果回答，需要时自己做最后一步计算。" + ANSWER_RULE)
    SCHEMA = ("facts(company TEXT, ticker TEXT, year INTEGER, sector TEXT, revenue REAL, net_profit REAL, "
              "gross_margin REAL)  -- 每行一家公司一个年份\n"
              "suppliers(company TEXT, ticker TEXT, year INTEGER, supplier TEXT, material TEXT)  -- company 的供应商\n"
              "customers(company TEXT, ticker TEXT, year INTEGER, customer TEXT)")

    def __init__(self, llm: LLM, docs: list[RawDoc], workers: int = 8):
        super().__init__(llm, docs, workers)
        import sqlite3
        import threading

        self._lock = threading.Lock()
        self.db = sqlite3.connect(":memory:", check_same_thread=False)
        c = self.db.cursor()
        c.execute("CREATE TABLE facts(company TEXT, ticker TEXT, year INTEGER, sector TEXT, revenue REAL, "
                  "net_profit REAL, gross_margin REAL)")
        c.execute("CREATE TABLE suppliers(company TEXT, ticker TEXT, year INTEGER, supplier TEXT, material TEXT)")
        c.execute("CREATE TABLE customers(company TEXT, ticker TEXT, year INTEGER, customer TEXT)")
        for v in self._merged.values():
            key = (v["company"], v["ticker"], _int(v["year"]))
            c.execute("INSERT INTO facts VALUES (?,?,?,?,?,?,?)",
                      (*key, v.get("sector"), _float(v.get("revenue")), _float(v.get("net_profit")),
                       _float(v.get("gross_margin"))))
            for sp in v["suppliers"]:
                c.execute("INSERT INTO suppliers VALUES (?,?,?,?,?)", (*key, sp.get("name"), sp.get("material")))
            for cu in v["customers"]:
                c.execute("INSERT INTO customers VALUES (?,?,?,?)", (*key, str(cu)))
        self.db.commit()
        self.sectors = [r[0] for r in c.execute("SELECT DISTINCT sector FROM facts WHERE sector IS NOT NULL")]
        self.materials = [r[0] for r in c.execute("SELECT DISTINCT material FROM suppliers WHERE material IS NOT NULL")]
        samples = []
        for t in ("facts", "suppliers", "customers"):
            cur = c.execute(f"SELECT * FROM {t} ORDER BY ticker, year LIMIT 3")
            samples.append(t + ": " + " ; ".join("(" + ", ".join(map(str, r)) + ")" for r in cur.fetchall()))
        self.samples = "\n".join(samples)

    def _exec(self, sql: str, limit: int = 50):
        """Run a query; ``limit`` caps the rows returned (50 is enough to show the model a result)."""
        with self._lock:
            cur = self.db.execute(sql)
            cols = [d[0] for d in cur.description or []]
            return cols, cur.fetchmany(limit)

    def answer(self, question: str, max_tokens: int = 4096) -> PathResult:
        prompt = (self.SQL.replace("{schema}", self.SCHEMA)
                  .replace("{sectors}", "、".join(map(str, self.sectors))).replace("{samples}", self.samples)
                  .replace("{materials}", "、".join(map(str, self.materials)))
                  .replace("{q}", question))
        msgs = [{"role": "user", "content": prompt}]
        total, trace, sql, rows = Usage(), [], "", "（无）"
        for attempt in range(2):  # one retry: with the error message, or with a doubled budget
            # the doubled budget also changes the request body, so the retry is not served
            # from the response cache (an identical retry would return the same empty reply)
            r = self.llm.chat(msgs, max_tokens=max_tokens * (2 if attempt else 1))
            total.add(r.usage)
            m = re.search(r"```(?:sql)?\s*(.+?)```", r.text, re.S | re.I)
            sql = (m.group(1) if m else r.text).strip()
            if not sql:
                trace.append({"sql": "", "error": f"empty reply (finish={r.finish})"})
                continue
            try:
                cols, res = self._exec(sql)
                rows = "\n".join([" | ".join(cols)] + [" | ".join(map(str, x)) for x in res]) if res else "（空）"
                trace.append({"sql": sql, "n_rows": len(res)})
                break
            except Exception as e:  # sqlite errors: give the model one chance to fix the query
                trace.append({"sql": sql, "error": str(e)[:200]})
                msgs += [{"role": "assistant", "content": r.text},
                         {"role": "user", "content": f"执行出错：{e}。请修正，只输出 SQL。"}]
        ans_prompt = self.ANSWER.replace("{q}", question).replace("{sql}", sql).replace("{rows}", rows)
        a = self.llm.chat([{"role": "user", "content": ans_prompt}],
                          max_tokens=max_tokens)
        total.add(a.usage)
        return PathResult(parse_answer(a.text), a.text, total, trace=trace)


class LazyViewPath(ViewSQLPath):
    """Lazy materialization: answer a question the view's schema does not cover by extracting
    the missing attribute on demand, only for the rows that need it.

      1. plan:    the model names the missing attribute and writes SQL over the *existing* view
                  that selects the (ticker, year) rows needing it; the view acts as the index
                  that says which documents to read (a sector, a supplier found by a join, ...);
      2. extract: one call per selected company-year over that company-year's documents;
                  identical requests are cached, so a later question needing the same column
                  reuses the work (the view grows with the workload, like adaptive indexing);
      3. answer:  add the column as table ``extra(ticker, year, value)`` and run the SQL path.

    Cost scales with the rows the question touches, not with the corpus."""

    name = "view_lazy"
    MAX_ROWS = 200
    PLAN_LAZY = ("下面是一个 SQLite 数据库，内容是从资料库抽取出的事实。\n{schema}\n"
                 "外键：suppliers.supplier 与 facts.company 是同一套公司简称。\n"
                 "sector 全部取值：{sectors}\nmaterial 全部取值：{materials}\nyear 全部取值：{years}\n\n"
                 "回答下面的问题需要一个表里没有的属性。请：\n"
                 "1. 给出这个属性的中文描述，包含年份和单位，例如“2023年XX（亿元）”；\n"
                 "2. 写一条 SQL，从现有表里选出需要这个属性的公司年份行，只输出 ticker 和 year 两列"
                 "（year 是存放该属性的报告年份，sector/material 按含义用 IN 选全）。\n"
                 "数据字典：年报和业绩快报都附带上一年的对比数据，所以表里没有的年份 Y，可以从 Y+1 年的报告行里取。\n"
                 "示例（属性与本题无关）：\n"
                 "问：华丰酒业2022年的员工人数是多少？ -> {\"attribute\": \"2022年员工人数（人）\", "
                 "\"rows_sql\": \"SELECT ticker, year FROM facts WHERE company = '华丰酒业' AND year = 2022\"}\n"
                 "问：华丰酒业的包装供应商2023年的员工人数是多少？ -> {\"attribute\": \"2023年员工人数（人）\", "
                 "\"rows_sql\": \"SELECT f.ticker, f.year FROM suppliers s JOIN facts f ON f.company = s.supplier "
                 "WHERE s.company = '华丰酒业' AND s.year = 2023 AND f.year = 2023\"}\n"
                 "只输出 JSON：{\"attribute\": \"...\", \"rows_sql\": \"SELECT ticker, year FROM ...\"}\n\n问题：{q}")
    EXTRACT_ONE = ("从下面这家公司的文档中找出：{attr}。只输出 JSON：{\"value\": 数值或文本}，"
                   "文档里没有就输出 {\"value\": null}。\n\n")

    def __init__(self, llm: LLM, docs: list[RawDoc], workers: int = 8):
        super().__init__(llm, docs, workers)
        self.years = [r[0] for r in self.db.execute("SELECT DISTINCT year FROM facts WHERE year IS NOT NULL ORDER BY year")]
        self.docs_by = {}
        for d in docs:
            self.docs_by.setdefault((d.ticker, d.year), []).append(d)
        self.workers = workers

    def answer(self, question: str, max_tokens: int = 4096) -> PathResult:
        import sqlite3
        from concurrent.futures import ThreadPoolExecutor

        total, trace = Usage(), []
        prompt = (self.PLAN_LAZY.replace("{schema}", self.SCHEMA)
                  .replace("{sectors}", "、".join(map(str, self.sectors)))
                  .replace("{materials}", "、".join(map(str, self.materials)))
                  .replace("{years}", "、".join(map(str, self.years))).replace("{q}", question))
        spec = {}
        for attempt in range(2):  # retry once with a doubled budget on a truncated / unparsable plan
            plan = self.llm.chat([{"role": "user", "content": prompt}], max_tokens=max_tokens * (2 if attempt else 1))
            total.add(plan.usage)
            m = re.search(r"\{.*\}", plan.text, re.S)
            try:
                spec = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                spec = {}
            if spec.get("attribute") and spec.get("rows_sql"):
                break
        attr, rows_sql = str(spec.get("attribute") or ""), str(spec.get("rows_sql") or "")
        try:
            # row selection must see every matching row, not the 50 shown to the model
            _, rows = self._exec(rows_sql, limit=self.MAX_ROWS + 1) if rows_sql else ([], [])
        except Exception as e:
            rows = []
            trace.append({"rows_sql": rows_sql, "error": str(e)[:200]})
        keys = sorted({(str(t), _int(y)) for t, y in rows if (str(t), _int(y)) in self.docs_by})[: self.MAX_ROWS]
        trace.append({"attribute": attr, "rows_sql": rows_sql, "n_rows": len(keys),
                      "truncated": len(rows) > self.MAX_ROWS})
        if not attr or not keys:
            return PathResult(None, plan.text, total, trace=trace)

        def one(k):
            text = "\n\n".join(d.text for d in sorted(self.docs_by[k], key=lambda d: d.doc_id))
            r = self.llm.chat([{"role": "user", "content": self.EXTRACT_ONE.replace("{attr}", attr) + text}],
                              max_tokens=1500)
            mm = re.search(r"\{.*\}", r.text, re.S)
            try:
                v = json.loads(mm.group(0)).get("value") if mm else None
            except json.JSONDecodeError:
                v = None
            return k, v, r.usage
        with ThreadPoolExecutor(self.workers) as ex:
            got = list(ex.map(one, keys))
        for _, _, u in got:
            total.add(u)
        trace.append({"extracted": sum(v is not None for _, v, _ in got)})

        # per-question copy of the view plus the new column
        db = sqlite3.connect(":memory:", check_same_thread=False)
        with self._lock:
            self.db.backup(db)
        db.execute("CREATE TABLE extra(ticker TEXT, year INTEGER, value)")
        db.executemany("INSERT INTO extra VALUES (?,?,?)",
                       [(t, y, _float(v) if _float(v) is not None else v) for (t, y), v, _ in got])
        sub = ViewSQLPath.__new__(ViewSQLPath)
        sub.__dict__.update(self.__dict__)
        sub.db = db
        sub.SCHEMA = self.SCHEMA + (f"\nextra(ticker TEXT, year INTEGER, value)  -- 按需抽取，只覆盖相关公司。"
                                    f"每行的 value 就是“{attr}”；year 是抽取它的报告年份，不一定等于属性所指的年份，"
                                    f"不要按属性年份过滤 extra.year")
        r = sub.answer(question, max_tokens)
        total.add(r.usage)
        return PathResult(r.answer, r.text, total, trace=trace + (r.trace or []))


def _float(x):
    try:
        return float(str(x).replace(",", "").rstrip("%亿元 ")) if x is not None else None
    except ValueError:
        return None


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


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
            # no tools on the last turn: some gateways ignore tool_choice="none" and call a tool anyway
            r = self.llm.chat(msgs, max_tokens=4096)
            total.add(r.usage)
            text = r.text
        return PathResult(parse_answer(text), text, total, trace=trace)
