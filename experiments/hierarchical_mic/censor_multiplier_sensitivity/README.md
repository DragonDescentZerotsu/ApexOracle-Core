# MIC censor-multiplier sensitivity

本 capsule 回答 reviewer 对 DBAASP 右删失 MIC 点值编码（`>V -> 2V`）的质疑。分析只读消费
已经完成的 strain-wise 和 phylum-wise 七成员 ensemble prediction，不重训模型，也不修改 frozen
paper training table。

## Canonical 入口

```bash
PYTHONPATH=src python \
  scripts/audit/analyze_hierarchical_mic_censor_sensitivity.py
```

默认输出到 `analysis/`。已有输出时必须显式传 `--overwrite`；该参数只覆盖本入口登记的派生文件。
可用重复的 `--right-censor-multiplier` 改写默认 `1, 2, 4` 网格，但网格必须包含 paper 使用的
`2`。

纯 label/metric 逻辑位于
`src/apexoracle/evaluation/hierarchical_mic_censor_sensitivity.py`，冻结输入重建、prediction 对齐、
输出契约和 manifest 写入统一位于
`src/apexoracle/evaluation/hierarchical_mic_censor_workflow.py`；CLI 只负责参数解析。为把 raw
DBAASP concentration 血缘附回 frozen prediction，本入口还允许
`prepare_hierarchical_mic_data(..., mic_frame=...)` 在内存中携带审计列；该扩展不改变默认文件入口
或历史训练行为。

## 冻结协议

- 从 `all_peptides_data.json` 按 paper-era `collect_strain_measurements` 和 `select_measurement`
  重建每个 `DBAASP_id × original strain` 的 selected concentration，并与 frozen DBAASP MIC 表按
  行数、顺序、ID、strain 和 MIC 精确核验。
- ordinary right-censored 定义为非 range 的 ASCII `>`/`>=` 或 Unicode `≥`。历史 parser 对
  22,138 条 ASCII 记录使用 `2×`，但对 20 条 Unicode `≥` 记录实际使用 `1×`；这 20 条在
  sensitivity grid 中按统一普通右删失口径重编码。
- 19 条 `>>` 记录的历史 parser 使用 `3×`。由于其语义不等同于普通 `>V`，它们保留在
  `paper_legacy` 基线，但从 `1×/2×/4×` 网格中排除。
- 五种 reviewer-facing sensitivity 为普通右删失 `1×/2×/4×`、删除全部右删失、删除左右全部
  删失；另保留包含历史 Unicode/`>>` 细节的 `paper_legacy` 精确基线。
- label 和 R²/MAE/RMSE/Spearman/Pearson 均在论文使用的
  `-log10(MIC_um / 10)` scale 上重算。主汇总是三个 group 的不加权均值和 sample s.d.；pooled
  结果也完整保留。
- strain 使用 fixed `PYTHONHASHSEED=0` reconstruction，phylum 使用可确定的 canonical taxonomy
  membership。两者均消费确定性 replay 的七成员 ensemble mean。

冻结 prediction 表没有保留 raw strain name。对于 1,822 个在 protocol/group/route/model input/
normalized strain/MIC 上完全相同的行，无法唯一恢复 raw-lineage occurrence 的配对。模型无法区分
这些行，因此分析先在该 stable key 内取 saved predictions 的均值，再附回 censor lineage；唯一
输入行完全不变。最大逐预测改变量为 `1.31e-5`，paper-legacy 指标的最大改变量为 `5.52e-9`，
详见 `duplicate_prediction_normalization_audit.csv`。

## 已验证结果（2026-08-07）

raw DBAASP selected measurements 共 105,547 条，其中普通右删失 22,158 条（20.99%）、`>>`
19 条、左删失 615 条。实际 eligible held-out measurement instances 中：

- strain：86,358 条；普通右删失 14,939 条（17.30%），`>>` 18 条，左删失 400 条；
- phylum：85,824 条；普通右删失 15,264 条（17.79%），`>>` 15 条，左删失 397 条。

下表为 reviewer/论文口径的 mean across three groups：

| Protocol | Scenario | Measurements | R² | MAE | Spearman |
| --- | --- | ---: | ---: | ---: | ---: |
| strain | `1×` | 86,340 | 0.5785 | 0.4469 | 0.7478 |
| strain | `2×` | 86,340 | 0.5813 | 0.4644 | 0.7478 |
| strain | `4×` | 86,340 | 0.5634 | 0.5088 | 0.7415 |
| strain | exclude right-censored | 71,401 | 0.5699 | 0.4594 | 0.7251 |
| strain | exclude all censored | 71,001 | 0.5700 | 0.4582 | 0.7253 |
| phylum | `1×` | 85,809 | 0.3804 | 0.5165 | 0.6282 |
| phylum | `2×` | 85,809 | 0.3879 | 0.5448 | 0.6309 |
| phylum | `4×` | 85,809 | 0.3748 | 0.5967 | 0.6269 |
| phylum | exclude right-censored | 70,545 | 0.3491 | 0.5366 | 0.5829 |
| phylum | exclude all censored | 70,148 | 0.3508 | 0.5352 | 0.5842 |

**已由代码和输出验证的事实：** `1×--4×` 内两套 protocol 的 R² 和 rank correlation 保持相近，
删除删失记录后仍保留正的 ranking signal；`2×` 并非所有 metric 上都最优，`1×` 的 MAE 更低，
而 `4×` 的误差更高。

**根据现有证据作出的推断：** held-out signal 不依赖唯一的 `2×` 取值，但绝对误差会随点值编码
发生可见变化。因此回复应承认该倍数是基于 twofold-dilution spacing 的 operational heuristic，
而不是已知的真实 MIC 或理论唯一值。

**正式采用边界（2026-08-08 作者确认）：** 本结果已加入正式 reviewer response，并以紧凑方法与
结果段落加入 DBAASP MIC preprocessing table 前后；没有新增 Supplementary table。本实验没有在每个
alternative label encoding 下重训，因此正式文稿只将其表述为 frozen-prediction label sensitivity，
不能写成训练过程对 censoring 完全稳健，也不能替代 censored-regression/interval-likelihood 模型。

## 为什么 R² 变化有限（2026-08-08 诊断）

R² 的 pooled `SSE/TSS` 分解进一步解释了表面上的稳定性：

- 只有 strain/phylum 的17.30%/17.79% eligible rows进入普通 multiplier grid；
- 论文 label 为`-log10(MIC_um/10)`，MIC每翻一倍只使受影响 label 平移
  `log10(2)=0.3010`，因此全体 label mean 每一步只移动约`0.052--0.054`；
- 从`2V`改为`V`时，strain SSE/TSS 同时下降`7.68%/8.28%`，phylum 同时下降
  `8.11%/8.35%`，所以 pooled R² 只变化`-0.0035/-0.0015`；
- 从`2V`改为`4V`时，SSE上升`14.26%/14.42%`，但TSS也上升`11.20%/11.35%`，pooled R²
  因而只下降`0.0148/0.0160`。

**已验证事实：** R²看的是`1-SSE/TSS`；修改标签同时改变分子和分母，两者同向变化构成主要抵消。
**解释边界：** 这不代表 multiplier 对所有 metric 无影响；MAE 在`4V`下明显增加，而且当前仍是
frozen-prediction分析。上述 exact 分解与解释已完整保留在本节；2026-08-09 维护审计时移除了没有
canonical 生成入口、且未被正式文稿消费的一次性 portable HTML/JSON 报告，避免把不可再生的展示
文件误当作正式分析产物；文件已移出工作区并保存在可恢复归档
`/data2/tianang/.codex-trash/20260809_synergy_mic_multiplier_diagnostics/`。

## 代码与文件系统维护审计（2026-08-09）

- 原 623 行 audit script 同时承担 CLI、重建、指标汇总和 manifest 写入，维护边界不清。现已将其
  收缩为118行参数入口，完整工作流进入上述共享 module；没有复制另一套分析实现。
- 删除了从未被任何入口或测试调用的 `censor_counts` 死代码，并将 multiplier grid、closed output
  contract、额外 MIC audit columns 与 small-molecule auxiliary columns 的兼容性纳入测试。
- 使用同一冻结输入完整执行 canonical `--overwrite` 重跑。六个 CSV 与 manifest 共七个输出的
  SHA-256 均与重构前逐字节一致，因此数值、row order、schema 和 provenance contract 未改变。
- manifest 的 `generated_on=2026-08-07` 是 canonical artifact 的冻结 provenance date，不是每次
  deterministic rerun 的 wall-clock 时间；代码已将其命名为 `CANONICAL_ANALYSIS_DATE`，避免误读为
  动态运行时间并保持已有 artifact hash 不变。
- 42 MB `row_censor_assignments.csv` 继续保留为 local-only：它是独立复算60行指标和重复记录对齐
  审计的逐行依据，位于唯一 canonical output root 且已由 `.gitignore` 精确保护，不属于重复副本。
- 退役的一次性 R² portable report 共约466 KB，已移入上述工作区外可恢复归档；其全部科学结论仍
  由本 README 的 exact 数值和 canonical CSV 支撑。

## 正式文稿落稿记录（2026-08-08）

正式落稿严格使用 `REVIEWER_RESPONSE_DRAFT.md` 中作者确认的文本与数值：

- `sn-article.tex` 在 Table `DBAASP_MIC` 前说明 `>V` 是需要 multiplier finite assignment 的最大类别
  （22,158条，相比`>>V`的19条），用于代表性 sensitivity；表后报告 strain/phylum 的受影响分母、
  `V/2V/4V` 和 exclusion R²、Spearman/MAE范围及 heuristic 结论。
- `Response to reviewers letter.docx` 中原 future-tense 占位回复被替换为三段完成时回复；新段落沿用
  原段落的 Normal、两端对齐、Arial 12 pt、红色和 SimSun East Asian 字体设置，其他段落文本和表格
  单元格未改变。
- 修改前备份位于正式文稿目录：
  `sn-article_before_mic_multiplier_sensitivity_20260808.tex` 和
  `Response to reviewers letter_before_mic_multiplier_sensitivity_20260808.docx`。
- 修改后正式 TeX/DOCX SHA-256 分别为
  `c8cc3b688fe81b73607bb30b9569b82fded70132e335fa6a84ba2f6de6d48f21` 和
  `378e2a90336f31218c307c2831fbaa3c67806c6ad9a2e2e946e332738cd7887d`。TeX 独立编译为35页，
  无 undefined citation/reference，新增内容位于第10--11页；DOCX独立渲染为32页，新回复位于
  第24--25页；两者均已目视核验。
- 正式 `sn-article.pdf` 未覆盖，SHA-256 保持
  `761b1b6cac0d17a1ade0932df1d6a4a2fa6a044c3729da5d9d1fa351b707d3c2`。

## 产物与存储边界

| 文件 | 内容 | Git 边界 |
| --- | --- | --- |
| `analysis/analysis_manifest.json` | 输入/输出 SHA-256、协议和 claim boundary | compact，可发布 |
| `analysis/source_censor_rules.csv` | raw DBAASP censor class 分母 | compact，可发布 |
| `analysis/eligible_censor_counts.csv` | protocol/group 的 censor 分母 | compact，可发布 |
| `analysis/metrics.csv` | group、pooled、mean-across-groups 全指标 | compact，可发布 |
| `analysis/metric_deltas.csv` | 相对 paper legacy 与 `2×` 的差值 | compact，可发布 |
| `analysis/duplicate_prediction_normalization_audit.csv` | 重复 stable-key 技术审计 | compact，可发布 |
| `analysis/row_censor_assignments.csv` | 172,182 条逐行 lineage/prediction | 约 42 MB，local-only |

本次正式落稿基准与实施文本见 `REVIEWER_RESPONSE_DRAFT.md`。

## 验证

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_hierarchical_mic_censor_sensitivity.py \
  tests/test_hierarchical_mic_runner.py \
  tests/test_hierarchical_mic_molecule_overlap.py
```

正式输出另已从 `row_censor_assignments.csv` 用独立 NumPy/SciPy 公式复算 60 个
`protocol × scenario × aggregation` 行：measurement/R²/MAE/RMSE/Pearson 一致到 `2.2e-16`
以内，Spearman 最大 CSV round-trip 差为 `5.03e-10`。focused tests 为28 passed；全仓回归为
196 passed（14条既有 dependency/runtime warnings）。2026-08-09 维护重构后，扩展 focused suite
为31 passed，全仓回归为205 passed（14条既有 dependency/runtime warnings）。
