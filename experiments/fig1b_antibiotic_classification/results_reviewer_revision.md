# Fig. 1b reviewer 修订：单成员诊断与完整 ensemble 补实验

更新时间：2026-07-21。primary comparison metric 为 pooled out-of-fold AUPRC；AUROC 同时保留。
所有 baseline 使用与 ApexOracle fine-tune 完全相同的五个 molecule-ID outer folds，模型选择只看
outer-train 内部 validation。统计使用同一样本上的分层 paired bootstrap 95% CI、双侧
prediction-swap randomization test，并在每个 model-mode × metric 的三个菌株内做 Holm 校正。

> 2026-07-20 作者确认：下列单成员 ApexOracle 与单模型 Chemprop 数值仅保留为诊断，不是最终
> Fig. 1b 结果。最终比较使用 ApexOracle 每折 10 members，三个 Chemprop baseline 也统一
> 使用 10 个模型；Liu 采用论文汇报的无 RDKit feature 版本。

## 已由运行产物验证的事实

### ApexOracle fine-tune 10-member ensemble（最终汇总）

2026-07-21 已完成 `3 strains × 5 outer folds × 10 members = 150` 个 checkpoint 的
确定性推理与组装。每个 fold 都严格包含原编号 `ensemble_0`--`ensemble_9`；组装器同时核验
样本 ID、标签和 member 数。以下共同 cohort 与 Chemprop baseline 对齐，适合最终成对比较：

| Target | n（positive） | pooled OOF AUPRC | pooled OOF AUROC | fold AUPRC mean ± s.d. | fold AUROC mean ± s.d. |
| --- | ---: | ---: | ---: | ---: | ---: |
| *E. coli* BW25113 | 2,334 (120) | 0.69045 | 0.95560 | 0.71205 ± 0.14657 | 0.95884 ± 0.03341 |
| *A. baumannii* ATCC 17978 | 7,684 (480) | 0.41636 | 0.81732 | 0.43436 ± 0.05261 | 0.82200 ± 0.02138 |
| *S. aureus* RN4220 | 39,310 (512) | 0.39442 | 0.94689 | 0.40127 ± 0.03500 | 0.95309 ± 0.00776 |

旧论文柱子使用 fold-metric mean 口径，因此与旧值比较或继续该图口径时应使用最后两列，
不能换成 pooled OOF。A. baumannii AUPRC `0.43436` 四舍五入后恰为旧图 `0.4344`；E. coli
AUROC 比旧图 `0.962` 低 `0.00316`；RN4220 AUPRC 比旧图 `0.408` 低 `0.00673`。
paired significance 和 reviewer 主比较使用同一样本上的 pooled OOF predictions；必须等待
RN4220 剩余三个 10-member Chemprop folds 完成后重算，不能沿用单模型 sensitivity 的 p 值。

canonical 汇总为
`results/fig1b_revision/apexoracle_fine_tune_10member/summary.json`，SHA-256
`234bd8da3a48d0b9d92350b50c4e84278520192873581113b858abadac290b0b`。

### 最终 10-member Chemprop baseline

三个 baseline 的 15 个 folds 已于 2026-07-22 全部完成。每个 fold 最终 prediction 都只使用
固定编号 `model_0`--`model_9`；Stokes 2020 和 Wong 2024 保留各自论文 profile 的 RDKit2D，
Liu 2023 使用作者确认的 no-RDKit ablation。

| Target | pooled OOF AUPRC | pooled OOF AUROC | fold AUPRC mean ± s.d. | fold AUROC mean ± s.d. |
| --- | ---: | ---: | ---: | ---: |
| *E. coli* BW25113 | 0.54378 | 0.87074 | 0.54535 ± 0.13277 | 0.85932 ± 0.07452 |
| *A. baumannii* ATCC 17978 | 0.29221 | 0.77288 | 0.30355 ± 0.02240 | 0.77589 ± 0.03375 |
| *S. aureus* RN4220 | 0.33459 | 0.89877 | 0.36616 ± 0.04826 | 0.94337 ± 0.01597 |

### 最终 10-member paired comparison（5,000 iterations）

以下使用共同 molecule cohort 的 pooled OOF predictions。p 值来自双侧 paired
prediction-swap test，并在每个 mode × metric 的三株内进行 Holm 校正。

| Mode | Target | Metric | ApexOracle | Baseline | Difference [95% CI] | Holm p |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| fine-tune | *E. coli* | AUPRC | 0.69045 | 0.54378 | +0.14667 [0.05452, 0.24480] | 0.02639 |
| fine-tune | *E. coli* | AUROC | 0.95560 | 0.87074 | +0.08485 [0.04443, 0.12736] | 0.01200 |
| fine-tune | *A. baumannii* | AUPRC | 0.41636 | 0.29221 | +0.12415 [0.08126, 0.16176] | 0.00060 |
| fine-tune | *A. baumannii* | AUROC | 0.81732 | 0.77288 | +0.04444 [0.02360, 0.06481] | 0.02180 |
| fine-tune | RN4220 | AUPRC | 0.39442 | 0.33459 | +0.05984 [0.02053, 0.09661] | 0.02639 |
| fine-tune | RN4220 | AUROC | 0.94689 | 0.89877 | +0.04812 [0.03581, 0.06040] | 0.01800 |
| strict zero-shot | *E. coli* | AUPRC | 0.58738 | 0.54378 | +0.04360 [-0.06451, 0.15065] | 0.54269 |
| strict zero-shot | *E. coli* | AUROC | 0.93502 | 0.87074 | +0.06427 [0.01918, 0.11180] | 0.04759 |
| strict zero-shot | *A. baumannii* | AUPRC | 0.32098 | 0.29221 | +0.02877 [-0.01551, 0.07338] | 0.54269 |
| strict zero-shot | *A. baumannii* | AUROC | 0.72408 | 0.77288 | -0.04880 [-0.08016, -0.01826] | 0.04759 |
| strict zero-shot | RN4220 | AUPRC | 0.16562 | 0.33459 | -0.16897 [-0.21391, -0.12629] | 0.00060 |
| strict zero-shot | RN4220 | AUROC | 0.76740 | 0.89877 | -0.13137 [-0.15651, -0.10657] | 0.00060 |

最终统计 JSON 为 `results/fig1b_revision/significance_10member.json`，SHA-256
`12c1d603b29679ee5f9b1bd8fcd5a60a78b5beb622fe593ca8fd6e7003577ae0`。

### Strict zero-shot 的同口径 error bar

zero-shot 每株历史上只有一个 held-target 10-member ensemble，因此原报告只有完整 target 的一个
指标。为与 fine-tune/baseline 的 error bar 口径一致，最终图可以把这组固定 prediction 按同一
五个 outer-fold test membership 分区；模型不重训，error bar 表示测试分区间 sample s.d.：

| Target | fold AUPRC mean ± s.d. | fold AUROC mean ± s.d. |
| --- | ---: | ---: |
| *E. coli* BW25113 | 0.58537 ± 0.13716 | 0.93982 ± 0.03495 |
| *A. baumannii* ATCC 17978 | 0.32515 ± 0.06589 | 0.72391 ± 0.05126 |
| *S. aureus* RN4220 | 0.16630 ± 0.05288 | 0.76676 ± 0.02141 |

这些五折 error bars 不是 10 个 ensemble members 的随机种子方差；三种方法因此保持同一种
test-fold uncertainty 解释。paired significance 仍使用上表的完整 pooled predictions。

### Common-fold Chemprop 单模型诊断（已被最终 ensemble 协议取代）

| Target | n（positive） | pooled AUPRC | pooled AUROC | fold AUPRC mean ± s.d. | fold AUROC mean ± s.d. |
| --- | ---: | ---: | ---: | ---: | ---: |
| *E. coli* BW25113 | 2,334 (120) | 0.50752 | 0.85711 | 0.54226 ± 0.12306 | 0.84907 ± 0.08587 |
| *A. baumannii* ATCC 17978 | 7,684 (480) | 0.31967 | 0.77750 | 0.35448 ± 0.04117 | 0.78754 ± 0.03376 |
| *S. aureus* RN4220 | 39,310 (512) | 0.32873 | 0.92848 | 0.33702 ± 0.06717 | 0.93004 ± 0.02649 |

E. coli 的 `ce_2244` 与 RN4220 的 `na_20640` 是同一条含异常铝配位价态的结构，Chemprop/RDKit
不能产生 prediction。固定 KFold 没有重排；配对统计仅从该 target 的双方同时排除对应记录。

### Strict zero-shot vs common-fold baseline（5,000 iterations）

| Target | Metric | ApexOracle | Baseline | Paired difference [95% CI] | raw p | Holm p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| *E. coli* BW25113 | AUPRC | 0.58738 | 0.50752 | +0.07986 [-0.03093, 0.18942] | 0.20836 | 0.41672 |
| *E. coli* BW25113 | AUROC | 0.93502 | 0.85711 | +0.07791 [0.03386, 0.12541] | 0.00680 | 0.01360 |
| *A. baumannii* ATCC 17978 | AUPRC | 0.32098 | 0.31967 | +0.00132 [-0.04345, 0.04876] | 0.96601 | 0.96601 |
| *A. baumannii* ATCC 17978 | AUROC | 0.72408 | 0.77750 | -0.05342 [-0.08612, -0.02191] | 0.03059 | 0.03059 |
| *S. aureus* RN4220 | AUPRC | 0.16562 | 0.32873 | -0.16311 [-0.20661, -0.11951] | 0.00020 | 0.00060 |
| *S. aureus* RN4220 | AUROC | 0.76740 | 0.92848 | -0.16107 [-0.18564, -0.13672] | 0.00020 | 0.00060 |

### Fine-tune sensitivity vs common-fold baseline（5,000 iterations）

为避免把不同大小的残缺 ensemble 混在一个 OOF 表中，修订 sensitivity 固定每个 outer fold
只使用原编号 `ensemble_0`。14/15 个 fold 复用历史 checkpoint；唯一缺失的 RN4220 fold 4
使用 `PYTHONHASHSEED=0`、ensemble seed 42 按旧 25-epoch 多任务协议补训，并加载 best-AUROC
checkpoint 做确定性推理。

| Target | Metric | ApexOracle | Baseline | Paired difference [95% CI] | raw p | Holm p |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| *E. coli* BW25113 | AUPRC | 0.66655 | 0.50752 | +0.15903 [0.06161, 0.26380] | 0.01180 | 0.03539 |
| *E. coli* BW25113 | AUROC | 0.95529 | 0.85711 | +0.09818 [0.05935, 0.14068] | 0.00060 | 0.00180 |
| *A. baumannii* ATCC 17978 | AUPRC | 0.35294 | 0.31967 | +0.03327 [-0.01263, 0.07580] | 0.19276 | 0.38552 |
| *A. baumannii* ATCC 17978 | AUROC | 0.77698 | 0.77750 | -0.00052 [-0.02516, 0.02468] | 0.98400 | 1.00000 |
| *S. aureus* RN4220 | AUPRC | 0.34518 | 0.32873 | +0.01645 [-0.02756, 0.06093] | 0.51650 | 0.51650 |
| *S. aureus* RN4220 | AUROC | 0.92278 | 0.92848 | -0.00570 [-0.02021, 0.00870] | 0.77345 | 1.00000 |

补训 checkpoint SHA-256 为
`68a34004a4992c0bfff3733a9e5e7135ebed79bfbf15dd38e6eca7d2199d6a87`；完整统计 JSON
SHA-256 为 `912de74c72960b99495e4a441e1dcb75b338dd7390625ac47b3468025c676e83`。

## 根据现有证据作出的解释

- strict zero-shot 并非跨三个 target 普遍优于 baseline；当前仅 E. coli AUROC 显著更高。
- A. baumannii 的 AUPRC 与 baseline 无可检测差异，但 AUROC 显著较低。
- RN4220 的 AUPRC 和 AUROC 均显著低于 baseline。因此正文和图注必须撤回笼统的优势表述。
- E. coli AUPRC 的点估计较高，但 CI 跨 0 且 Holm 校正后不显著，不能写成统计学优势。
- fine-tune sensitivity 只在 E. coli 的 AUPRC 和 AUROC 上显著优于 baseline；A. baumannii
  和 RN4220 的两个指标均无可检测差异。
- fine-tune 是每折单模型的 sensitivity analysis，不得写成已恢复旧论文的完整 ensemble 结果。

## 旧论文 fine-tuned 柱子与当前 sensitivity 的口径诊断

旧 Mac notebook 在修改前的 cell `220739609a526f79` 中直接硬编码了混合指标：RN4220
AUPRC `0.408`、E. coli AUROC `0.962`、A. baumannii AUPRC/AUROC
`0.4344/0.8262`。这些值不是当前 pooled OOF 单模型分析的输出。

| Target | 旧图值 | 已恢复的历史 ensemble 证据 | 当前 reviewer sensitivity |
| --- | ---: | ---: | ---: |
| *E. coli* BW25113 | AUROC 0.962 | 5/5 folds、每折 10 members 的 fold AUROC 均值 0.96114 | 单成员 pooled OOF AUROC 0.95529 |
| *A. baumannii* ATCC 17978 | AUPRC 0.4344 | 仅 2/5 folds 完整；两折 10-member ensemble AUPRC 均值 0.45595 | 单成员 pooled OOF AUPRC 0.35294 |
| *S. aureus* RN4220 | AUPRC 0.408 | 仅 3/5 folds 完整；三折 10-member ensemble AUPRC 均值 0.40730 | 单成员 pooled OOF AUPRC 0.34518 |

因此早先 A. baumannii 和 RN4220 的下降不能解释为重构代码错误：比较同时改变了 ensemble
大小（10 → 1）、聚合方式（fold metric mean → pooled OOF）、checkpoint 完整性和推理模式。
RN4220 fold 4 还是后补训候选。canonical runner 的 legacy 行为等价测试、strict zero-shot
逐样本 logit 对齐和历史 checkpoint 严格加载均已通过；当前证据支持“口径不同”，不支持
“重构 forward 发生行为漂移”。目前完整 `5 folds × 10 members` 及样本级预测均已补齐；最终
10-member 结果见上文。历史与补训 checkpoint、member prediction 和最终组装目录彼此隔离，
没有覆盖原始数据或历史权重。

## Chemprop 数值与来源论文的关系

| Target profile | 当前 common-fold 结果 | 来源论文主结果 | 解释 |
| --- | ---: | ---: | --- |
| Stokes 2020 / E. coli | AUROC 0.85711；AUPRC 0.50752 | AUROC 0.896；未汇报 AUPRC | 当前 AUROC 更低，AUPRC 是本次共同 folds 新增结果 |
| Liu 2023 / A. baumannii | AUROC/AUPRC 0.77750/0.31967 | 约 0.792/0.337 | 这是带 RDKit 的单模型诊断；最终改用论文汇报的 no-RDKit ablation（论文约 0.756/0.266） |
| Wong 2024 / RN4220 | AUPRC 0.32873 | AUPRC 0.364 | 当前更低 |

上述三个 baseline 都是在 ApexOracle 固定 outer folds 上重新训练的单模型 Chemprop sensitivity，
不是最终比较。最终三个 baseline 均统一为每折 10 members；Liu 不使用 RDKit2D descriptors。

## 绘图与显著性标注

- canonical 绘图实现是 `src/apexoracle/evaluation/fig1b_plot.py`，入口是
  `scripts/reproduce/plot_fig1b_revision.py`；Matplotlib 生成冻结 CSV、PDF 和 PNG。
- Mac canonical notebook 是
  `/Users/kirianozan/Documents/Study/Penn/projects/local_figs/figs.ipynb`，cell ID
  `220739609a526f79`。`/Users/kirianozan/Documents/Study/Penn/Synergy/paper_figs/figs.ipynb`
  是只有三个旧 cell 的另一份 notebook，从未包含该 Fig. 1b cell。
- Holm 校正后的 `p < 0.05` 才标为显著。完整 ensemble 完成后，新增 notebook cell 使用
  Fig. 3a bracket 样式显示 paired prediction-swap test 的 Holm-adjusted p；原始论文 cell
  不允许覆盖或改写。

## 发布同步

- 原始 Mac cell `220739609a526f79` 已从修改前备份逐字节恢复；原始硬编码论文数值和 output
  保持不变。reviewer sensitivity 已迁入新 cell `fig1b-reviewer-sensitivity-20260720`，独立输出
  `3-strain-antibiotics-reviewer-sensitivity.pdf`。两个 cell 均通过 notebook schema 与语法校验，
  恢复过程中没有执行原始 cell。
- 禁止脚本自动合并或覆盖论文 `Fig1.pdf`。总图已经从更新前备份恢复，临时 bracket 图注也已
  撤回；后续只生成独立 panel，论文图片编辑由作者完成。
- 修改前快照、最终文件和运行产物 SHA-256 见
  `reproducibility/fig1b_reviewer_revision_2026-07-20.json`。
