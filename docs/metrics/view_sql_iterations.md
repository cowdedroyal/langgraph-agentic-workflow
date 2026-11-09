# view_sql: from "LLM reads the table" to "LLM writes SQL over the table"

Model `deepseek-v4.1-flash` (via a local OpenAI-compatible gateway), temperature 0, 63 questions, 2026-11-09.

## Why

The `view` path (LLM reads the extracted fact table) scored 0.984 at 180 distractors in one run, but
the repeatability probe ([stability_deepseek-v4.1-flash_180_aggregate.json](stability_deepseek-v4.1-flash_180_aggregate.json))
showed its aggregate accuracy moves between 0.667 and 1.000 across 4 runs (mean 0.875) **with the table held fixed**:
the errors are the model miscounting a 558-row table, not extraction. Full scan behaves the same on aggregates
(0.75–0.92, mean 0.854). A materialized view is supposed to be queried by the engine, so `view_sql` loads the same
extracted rows into SQLite and has the model write the query.

## Iterations

Numbers for v1 and v2 are from the run output printed during development; their record files were overwritten by
the next version (same output path). v3 is the committed version; its numbers are in
[access_paths_deepseek-v4.1-flash.md](access_paths_deepseek-v4.1-flash.md).

| version | prompt gives the model | 0 distractors | 54 distractors | what still failed |
|---|---|---|---|---|
| v1 | schema only, "company names may be abbreviations, match on company or ticker" | 0.841 | 0.873 | Runaway reasoning: SQL generation used all 4096 output tokens without emitting SQL (the model kept deliberating how to match names it could not see); empty SQL ran as a no-op and the answer became null. deep_chain 0, supply_chain 0.25. |
| v2 | + 3 sample rows per table, + declared foreign key (`suppliers.supplier` = `facts.company`), "use equality on company" | 0.952 | 0.936 | Free-text categorical values: the question says 玻璃瓶及包装供应商, the model wrote `material = '包装材料'`; extraction labelled one company's sector `锂电池正极材料`, so `sector = '正极材料'` undercounted by one. |
| v3 | + the full list of distinct `sector` and `material` values, "put every value with the same meaning in IN (...)" | 1.000 | 1.000 | — |

## Takeaways

* Moving aggregation from the model into the engine removes miscounting, but trades it for **schema-normalization
  brittleness**: equality on free-text values that extraction did not normalize. An LLM reading the table tolerates
  `锂电池正极材料` ≈ `正极材料`; SQL does not.
* The fixes are exactly the catalog a query optimizer has (sample rows, foreign keys, the domain of each categorical
  column), not hints about the answers.
* Cost: two calls per question with reasoning, so median latency is about 2× the `view` path; tokens per question
  stay small because the table itself never enters the prompt.
