# 进度记录：访问路径实验（access-paths 分支）

最后更新：2026-11-09。状态：**完成**（定位为实证研究 + 成本模型）。结论、表格、局限和已有工作见仓库根目录的 [README](../../README.md)。

## 时间线

- 2026-11-08：五条路径（scan / index / index_join / view / agent）在 deepseek-v4-flash 上跑完规模 0、54 和 180 的一部分，
  因 API 余额用完停下。结果保留在 [access_paths_deepseek-v4-flash.md](access_paths_deepseek-v4-flash.md)。
- 2026-11-09：换到 deepseek-v4.1-flash（经本地网关），整张表重跑，不和旧模型混在一张表里。新增：
  - `view_sql`（视图放进 SQLite，模型写 SQL），三版迭代见 [view_sql_iterations.md](view_sql_iterations.md)；
  - 表外题集（52 题，答案不在视图表结构里）和 `view_lazy`（按需物化）；
  - 重复性测试（`stability.py`）和输出预算测试（`budget_probe.py`）；
  - 规划器（`planner.py`）：题型 + 覆盖判断，按记录重放评估。

## 结果文件

| 文件 | 内容 |
|---|---|
| [access_paths_deepseek-v4.1-flash.md](access_paths_deepseek-v4.1-flash.md) / [summary JSON](access_paths_summary_deepseek-v4.1-flash.json) | 基础题集，4 档规模 × 6 条路径 |
| [access_paths_deepseek-v4.1-flash-oos.md](access_paths_deepseek-v4.1-flash-oos.md) / [summary JSON](access_paths_summary_deepseek-v4.1-flash-oos.json) | 表外题集，4 档规模 × 7 条路径 |
| [planner_deepseek-v4.1-flash_base+oos.md](planner_deepseek-v4.1-flash_base+oos.md) / [JSON](planner_deepseek-v4.1-flash_base+oos.json) | 规划器：分类器准确率、各策略的准确率 / 成本 / 延迟 |
| [stability_deepseek-v4.1-flash_180_aggregate.json](stability_deepseek-v4.1-flash_180_aggregate.json)、[540](stability_deepseek-v4.1-flash_540_aggregate.json) | 聚合题重复 4 次 |
| [scan_budget_probe_deepseek-v4.1-flash_180.json](scan_budget_probe_deepseek-v4.1-flash_180.json) | scan 错题放宽输出预算重问 |
| [view_sql_iterations.md](view_sql_iterations.md) | view_sql 的提示迭代 |

## 花费

2026-11-08 约 $0.94（deepseek-v4-flash）。2026-11-09 约 $4.8（缓存里 13,417 次调用 $4.65，另有约 $0.2 直接调接口的探测；按 deepseek-v4-flash 列表价估算，实际走网关），其中 540 家干扰的建表约 $1.34。

## 换设备接着做

缓存在 `outputs/access/cache`（被 gitignore）。没有缓存时重跑全部约 $4.7。环境和命令见 README 的「复现」。
