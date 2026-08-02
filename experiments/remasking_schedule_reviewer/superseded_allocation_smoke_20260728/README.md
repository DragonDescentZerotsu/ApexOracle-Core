# Superseded allocation smoke

本目录不属于正式 3,600-attempt 分析输入。

2026-07-28 首次启动使用了对全部 36 tasks 做简单 round-robin 的 allocation。启动后发现该
allocation 会让 `current`、`narrower` 和 `no_peptide_correction` 等 condition 与 host
不必要地混杂，因此在任何 task 完成前停止。本目录保留两个已产生 3/4 batches 的 partial task
及当时的 queue/log，作为中止血缘：

- `earlier__baa-3170__seed20260729/`
- `earlier__baa-3170__seed20260730/`

这两项不会被 `runs/` 下的正式 evaluator 发现或统计。新的 manifest 固定每个 condition 为
2 个 local tasks 和 4 个 node002 tasks，并保证两个 strain 都有 local 与 node002 generation。
