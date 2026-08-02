# ReMDM remasking schedule reviewer 补实验结果

> 完成日期：2026-07-29
> 状态：36/36 generation tasks 和正式 clean-model evaluation 均已完成；尚未修改论文或
> reviewer response。
> **2026-07-29 structure-audit warning：** 本报告下列 “peptide” 数值是历史 v1
> classifier-positive operational label，不是可靠的 structure-based peptide identity。current
> window 的 213 个 full-token positives 中有 125 个没有 RDKit general amide；修正 first
> `[SEP]` 后 suffix 后仍有 105/191 没有 general amide。故 `53.9% vs 46.1%` 不能用于回答真实
> peptide/small-molecule composition，现有 reviewer-facing panel a/c/d 暂停使用；用全部
> RDKit-valid outputs 汇总的 predicted-MIC trade-off 也应在 structure-qualified subset 上复核。
> 完整证据见 `STRUCTURE_AUDIT.md`。

## 1. 结论摘要

这组实验支持一个谨慎而清楚的结论：

1. 当前 `0.55--0.45` window 不是唯一或严格最优的设置，但处于 peptide yield 与 predicted
   MIC 的合理折中区域。它相对于 earlier window 的结果非常接近；later/wider window 可以提高
   peptide yield，但 clean-model predicted MIC 变差，尤其 wider `0.55--0.25` 的 trade-off
   最明显。
2. 在 paper setting 下，395 个 RDKit-valid candidates 中，历史 v1 classifier 判定 213 个
   positive 和 182 个 negative，即 `53.9% vs 46.1%`。后续 structure audit 已确认该 operational
   label **不能直接回答** Reviewer 2 要求的真实 peptide/small-molecule composition。
3. 与 `gamma_peptide=0` 的直接对照相比，stage-2 correction 对“classifier-positive 比例”
   的提升很小：all attempts 为 `47.7% vs 47.5%`；在 valid molecules 中反而是
   `53.9% vs 55.8%`。因此不能声称 narrow window 单独大幅“overcame”预训练 domain
   imbalance。
4. correction 的可测收益主要体现在 usable peptide yield 和 clean predicted MIC：
   每 600 attempts 的 valid classifier-positive peptides 从 198 增至 213
   （`33.0% -> 35.5%`），validity 从 `59.2% -> 65.8%`，valid-molecule median predicted
   MIC 从 `56.2 -> 37.9 micromolar`。
5. 作者确认原始 interval 本来就是通过相近的 empirical window comparison 选择，只是当时没有
   完整记录过程和结果。因此，正式回复应使用本次可复核的五窗口结果展示选择过程及 trade-off
   逻辑；不声称理论唯一最优，也不把本次冻结协议逐项冒充为当年已完整归档的同一实验。

## 2. 冻结协议与完整性

- 6 conditions × 2 strains × 3 seeds × 100 raw attempts = 3,600 attempts；
- 每个 condition 恰好 600 attempts；
- 本机 12/12 tasks，node002 24/24 tasks；
- 36/36 `completed.json`、144/144 batch files；
- 3,600 个 `(task_id, batch_index, sample_index)` keys 全部唯一；
- 跨机同步后，每个 batch 的 size/SHA-256 与生成时 completion marker 完全一致；
- 3,600/3,600 attempts 得到 v1 clean-input peptide probability；
- 2,355 个 RDKit-valid molecules 全部得到 finite clean-MIC prediction；
- 每个 condition 固定 2 个 local tasks 和 4 个 node002 tasks，两个 strain 都跨 host。

五个 window 为：

| Condition | `t_on` | `t_off` | `gamma_peptide` |
| --- | ---: | ---: | ---: |
| earlier | 0.75 | 0.65 | 15 |
| current | 0.55 | 0.45 | 15 |
| later | 0.35 | 0.25 | 15 |
| narrower | 0.525 | 0.475 | 15 |
| wider | 0.55 | 0.25 | 15 |
| no peptide correction | 0.55 | 0.45 | 0 |

在代码实际使用的 `torch.linspace(1, 1e-5, 257)` 上，ReMDM 分支条件为
`t_off < t <= t_on`。对应的实际 loop updates 为：earlier 25、current 25、later 26、
narrower 13、wider 77；no-peptide-correction 与 current 同为 25。

其余生成参数固定为 256 steps、base `r_t=0.02`、`alpha_on=0.5`、
`gamma_MIC=15` 和 target MIC 1 micromolar。

## 3. Pooled 结果

所有百分比以每个 condition 的 600 raw attempts 为冻结分母，除非列名明确写为
“among valid”。“Peptide”是生成时同一个 v1 classifier 在 clean input 上
`p(peptide)>=0.5` 的 operational definition。

| Condition | Complete | RDKit valid | Classifier-positive / attempts | Valid classifier-positive yield / attempts | Classifier-positive vs negative among valid | Median predicted MIC among valid, IQR (micromolar) | Median predicted MIC among valid classifier-positive (micromolar) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| earlier | 592 (98.7%) | 389 (64.8%) | 283 (47.2%) | 210 (35.0%) | 210/179 (54.0%/46.0%) | 37.2 (16.8–81.3) | 45.3 |
| **current** | **589 (98.2%)** | **395 (65.8%)** | **286 (47.7%)** | **213 (35.5%)** | **213/182 (53.9%/46.1%)** | **37.9 (19.4–100.5)** | **49.6** |
| later | 593 (98.8%) | 388 (64.7%) | 309 (51.5%) | 224 (37.3%) | 224/164 (57.7%/42.3%) | 44.5 (21.2–109.9) | 54.2 |
| narrower | 587 (97.8%) | 381 (63.5%) | 297 (49.5%) | 207 (34.5%) | 207/174 (54.3%/45.7%) | 40.3 (18.0–89.8) | 45.3 |
| wider | 596 (99.3%) | 447 (74.5%) | 361 (60.2%) | 291 (48.5%) | 291/156 (65.1%/34.9%) | 50.0 (22.3–121.9) | 70.4 |
| no peptide correction | 590 (98.3%) | 355 (59.2%) | 285 (47.5%) | 198 (33.0%) | 198/157 (55.8%/44.2%) | 56.2 (20.3–123.0) | 53.3 |

### 3.1 Window 选择

- **Earlier vs current：** usable peptide yield 为 `35.0% vs 35.5%`，valid-molecule median
  MIC 为 `37.2 vs 37.9`。两者非常接近，不能据此声称 `0.55--0.45` 是唯一 optimum。
- **Later：** yield 增至 `37.3%`，但 median MIC 增至 `44.5`。
- **Narrower：** yield 降至 `34.5%`，median MIC 增至 `40.3`；在这两个 pooled 指标上不优于
  current。
- **Wider：** yield 明显增至 `48.5%`，但 valid-molecule/valid-peptide median MIC 分别恶化到
  `50.0/70.4`。因此 wider window 不是无代价改善。

本次结果支持把 current window 描述为兼顾 peptide yield 与 predicted activity 的合理经验折中，
而不是由理论唯一决定或对所有 molecule 自适应的时间段。

### 3.2 Stage-2 effectiveness

`current` 与 `no peptide correction` 除 `gamma_peptide=15 vs 0` 外保持相同协议：

| Metric | Current | No peptide correction | Difference |
| --- | ---: | ---: | ---: |
| Classifier-positive / all attempts | 47.7% | 47.5% | +0.2 percentage points |
| RDKit-valid / all attempts | 65.8% | 59.2% | +6.7 percentage points |
| Valid peptide yield / all attempts | 35.5% | 33.0% | +2.5 percentage points |
| Classifier-positive among valid molecules | 53.9% | 55.8% | -1.9 percentage points |
| Valid-molecule median predicted MIC | 37.9 | 56.2 | -18.4 micromolar |
| Valid-peptide median predicted MIC | 49.6 | 53.3 | -3.7 micromolar |

所以 correction 的作用不能概括成“大幅提高 peptide fraction”。更准确的说法是：在这次冻结评估
中，它增加了 valid classifier-positive candidates 的绝对产出并改变 predicted MIC，但对
classifier-positive fraction 本身的增量很小；structure audit 后不能把这一点解释为真实 peptide
或可信 activity benefit。

RDKit-valid yield 从 `59.2%` 增至 `65.8%`（`+6.7` percentage points），因此可以作为
peptide guidance 相对关闭 peptide guidance 的一个 empirical advantage：在本次其余参数相同的
direct control 中，guidance 产生了更多可解析、可继续筛选的候选。但六个 matched
`strain × seed` task 上的 two-sided exact paired sign-flip test 为 `p=0.1875`，没有达到
`p<0.05`；正式回复应同时报告 effect size 和这一不确定性，不能写成已证明的普遍显著改善。

本报告原表格中“peptide”的 operational definition 是：使用生成时同一个历史 v1 peptide classifier
checkpoint，在完整生成 token sequence 的 clean input（`t=0`）上计算 sigmoid probability，
并以 `p(peptide)>=0.5` 判为 classifier-positive。该 classifier 的正类来自
SmProt2、UniProt/UniRef 和 PeptideCLM-generated 来源，负类来自 PubChem；因此它是与实际
guidance 一致的模型判定，不是独立的结构真值或 biological peptide identity。后续
`STRUCTURE_AUDIT.md` 已进一步证明它不能用于真实 peptide/small-molecule 主分类。

## 4. 两个 strain 的结果

| Condition | Strain | Valid / attempts | Classifier-positive / attempts | Valid peptide yield / attempts | Peptides among valid | Median predicted MIC among valid |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| earlier | BAA-3170 | 68.7% | 66.0% | 47.7% | 69.4% | 31.2 |
| earlier | BAA-3197 | 61.0% | 28.3% | 22.3% | 36.6% | 45.7 |
| current | BAA-3170 | 69.7% | 64.3% | 48.3% | 69.4% | 34.3 |
| current | BAA-3197 | 62.0% | 31.0% | 22.7% | 36.6% | 45.5 |
| later | BAA-3170 | 66.3% | 69.3% | 49.3% | 74.4% | 34.6 |
| later | BAA-3197 | 63.0% | 33.7% | 25.3% | 40.2% | 51.4 |
| narrower | BAA-3170 | 68.3% | 68.0% | 47.3% | 69.3% | 36.5 |
| narrower | BAA-3197 | 58.7% | 31.0% | 21.7% | 36.9% | 43.7 |
| wider | BAA-3170 | 73.7% | 72.3% | 57.7% | 78.3% | 46.1 |
| wider | BAA-3197 | 75.3% | 48.0% | 39.3% | 52.2% | 54.5 |
| no peptide correction | BAA-3170 | 63.0% | 67.7% | 46.3% | 73.5% | 38.2 |
| no peptide correction | BAA-3197 | 55.3% | 27.3% | 19.7% | 35.5% | 65.5 |

两种 strain 的 absolute peptide proportion 差异很大，因此正式回复应同时给 pooled composition
和 strain-specific composition，不应只报告一个总体百分比。

## 5. 对 reviewer 回复和论文修改的含义

### 已由代码和实验验证的事实

- base ReMDM proposal 在 window 内对每个 eligible decoded token 独立使用
  `r_t=0.02`；实际 guided probability 还由当前部分序列和 peptide classifier reweighting
  决定。没有按 molecule 固定 remask token 数，也没有按长度设置 quota。
- 全局 window 的确不是按 molecule complexity 自适应的；但“统一允许 reversible transition
  的时间段”不等于“每个 molecule 或 token 在同一时刻必须发生 correction”。
- current paper setting 的 valid output 中，历史 v1 classifier operational labels 为
  `53.9%` positive、`46.1%` negative；structure audit 已证明不能把它们改称真实
  peptide/small molecule。
- 原 schedule sweep 显示 classifier-positive yield、RDKit parseability 和 all-valid
  predicted MIC 的变化；window trade-off 必须在 structure-qualified subset 上复核后才能作为
  peptide-generation 结论。

### 根据现有证据作出的解释

- Reviewer 所说的 hardcoded/static schedule 在“global window”层面是事实，但由此进一步推出
  所有 molecule 被强制在同一阶段、以相同方式 correction，并不符合实际逐 token stochastic
  transition。
- Reviewer 2 关于 correction 完全未评估的批评对原稿成立；本次实验补上了 raw generation 与
  direct control，但 classifier/structure audit 表明真实 composition 仍未回答。当前只支持
  RDKit parseability 和 operational classifier score 的描述，不能声称 usable-peptide 或
  activity benefit。
- 原拟回复中的 `r_t=0.02 for t_off<t<t_on` 基本正确，但必须写成
  **unguided/base remask proposal probability**，不能写成 guidance 后每个 token 的固定最终
  remask probability。

### 仍然不能声称的内容

- 不能声称原始 `0.55--0.45` 是通过本次 grid search 选出的；
- 不能声称该 window 对所有 molecular size/complexity 最优；
- 不能把 v1 classifier label 当成独立结构真值或 biological peptide identity；
- 不能把 clean-model predicted MIC 当成 wet-lab MIC；
- 不能把本次两个 strain、三个 seeds 扩大解释成所有生成任务的普遍规律；
- 不能用 reviewer 的 90%/10% 直接替代仍未恢复的 DLM 过滤后精确来源比例。

## 6. 产物与哈希

| 产物 | SHA-256 |
| --- | --- |
| `task_manifest.json` | `84463b4dd09b7c6e6e021468fba47dca850bf26eecbab7847767fe424c07b6e7` |
| `analysis/evaluated_attempts.csv` | `bd847bb5081a8851c0e87b44922a6bae1db25cabb6c6dfdec51ad3b685f05d8f` |
| `analysis/summary.json` | `4a8597e64197044b330b391c78308d9fa8a3ca6c18fb52aa56429e0c3a9096ae` |
| `analysis/provenance.json` | `7328295eeed465245fad21ebcafe84ed7a2072f864bb62cfbbcbd6373b5281fa` |

Canonical code identities：

| 文件 | SHA-256 |
| --- | --- |
| `scripts/reproduce/run_remasking_schedule_reviewer.py` | `b933af383c78456af5d88704e934e4a429f6b3fdbc2c017123eaf8130d57bea8` |
| `scripts/reproduce/prepare_remasking_schedule_reviewer_tasks.py` | `76169a854c21dfe9efb608922407bd78f03faee5d3bed721abd2786986bc303a` |
| `scripts/reproduce/orchestrate_remasking_schedule_reviewer.py` | `3bfdf2592e9ffd38f9dbe1be1e9a5e7c5aa7a90bebf718ed620a2a35753265d4` |
| `scripts/reproduce/evaluate_remasking_schedule_reviewer.py` | `e2948018f0692ea9fdeca0ad13f1f6e73e85feffaa2ccc4dc21fddf6b6b6f54d` |
| `tests/test_remasking_schedule_reviewer.py` | `e2dcb97ceba4004baa51fa37b9acb9e6236e786c6003c788d8491ff635ca9350` |

`evaluated_attempts.csv` 和 raw `runs/` 保持本地、由 `.gitignore` 排除；compact summary、
provenance、manifest、README 和本报告可进入 reviewer-facing 版本控制。

## 7. Canonical final reviewer figure（2026-07-29）

作者确认最终三面板图 stem 为：

`figures/remasking_structure_qualified_peptides_with_mic_control`

panel a 与 panel c 的 `Peptide yield` 均要求候选同时通过窄结构筛选和 first `[SEP]` 后正确
padding 的历史 v1 classifier 阈值。current guidance 与 no peptide guidance 的统一口径 yield
分别为 `20/600 = 3.3%` 和 `10/600 = 1.7%`。PDF/SVG/PNG、caption、exact plotted-data CSV
及 SHA-256 manifest 使用同一 stem。该图稿已定版；联合判据仍是 preliminary narrow
criterion，不应扩展为通用 peptide ground truth。

2026-07-31 的 canonical 修订在 panel c 同一个三行坐标区中增加 predicted-MIC 对照：
current guidance 与 no peptide guidance 的 all-RDKit-valid pooled median 分别为
`37.9` 与 `56.2` $\mu\mathrm{M}$，横向 error bars 为三个 seed-level pooled median 的 sample
s.d.（`6.34` 与 `9.71` $\mu\mathrm{M}$）。两个 yield 比较和简短图例保留；数值轴仅压缩
5--25 区间，每行标签指明各自单位。

canonical/legacy 完整索引及不复制二进制别名的 storage policy 见 `figures/README.md`。

## 8. Legacy 可视化（不再用于 reviewer）

四面板描述性图位于：

- `figures/remasking_schedule_reviewer.pdf`：历史矢量版本，当前不用于论文或 response letter；
- `figures/remasking_schedule_reviewer.svg`：可编辑矢量版本；
- `figures/remasking_schedule_reviewer.png`：快速审阅版本；
- `figures/remasking_schedule_reviewer_data.csv`：图中全部 exact plotted values；
- `figures/remasking_schedule_reviewer_manifest.json`：输入 summary、绘图脚本和输出文件的
  SHA-256 血缘。
- `figures/remasking_schedule_reviewer_violin.{pdf,svg,png}`：保留其余 panel，只将 panel b
  替换为 log-scale full predicted-MIC distribution 的备选版本；
- `figures/remasking_schedule_reviewer_violin_{data.csv,manifest.json}`：violin 版本的全部
  plotted observations 与输入/output 血缘。

四个 panel 分别展示：

1. 五个 window 的 valid classifier-positive yield；
2. 五个 window 的 RDKit-valid-molecule median clean-model predicted MIC；
3. current window 下 `gamma_peptide=15` 与 `gamma_peptide=0` 的直接对照；
4. current window 下 pooled 和两个 strain 的 classifier-positive/negative composition。

默认 bar 版本中，panel a 的 error bars 为三个 seed-level pooled rates 的 sample s.d.，每个
seed 恰好汇总两个 strain、200 attempts；白点显示三个 seed 值。内部 plotted-data
CSV/manifest 保留六个 matched `strain × seed` tasks 的 two-sided exact paired sign-flip
p-values：
classifier-positive `p=1.000`、RDKit-valid `p=0.1875`、valid peptide yield `p=0.59375`，
但 reviewer-facing figure 不显示 p-value 或 significance stars。没有为五个 window
comparisons 添加 significance stars；只有三个 seeds 时做多重 pairwise significance testing
不足以支持可靠强结论。作者决定后续 reply 和正文只报告样本量及 descriptive effect，不加入
“difference did not reach statistical significance”句子，同时也不作 significance claim。

建议英文 caption：

> **Sensitivity of generated candidates to the remasking window and peptide correction.**
> (a) Yield of RDKit-valid, peptide-classifier-positive candidates among all raw generation
> attempts across five remasking windows. (b) Median clean-model predicted MIC among RDKit-valid
> candidates; lower values indicate stronger predicted activity. (c) Direct comparison of the
> current 0.55--0.45 window with peptide correction
> ($\gamma_{\mathrm{peptide}}=15$) and with peptide correction disabled
> ($\gamma_{\mathrm{peptide}}=0$). (d) Peptide-classifier-positive and classifier-negative
> composition among RDKit-valid candidates under the current window, shown for the pooled set and
> separately for the two target strains. Each condition comprises 600 raw attempts
> (2 strains × 3 random seeds × 100 attempts). Peptide status is defined by the generation-time v1
> classifier at $p\geq0.5$; predicted MIC values are model-based and are not wet-lab measurements.
