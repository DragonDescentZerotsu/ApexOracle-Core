# Fig. 1b reviewer 修订结果

更新时间：2026-07-20。primary comparison metric 为 pooled out-of-fold AUPRC；AUROC 同时保留。
所有 baseline 使用与 ApexOracle fine-tune 完全相同的五个 molecule-ID outer folds，模型选择只看
outer-train 内部 validation。统计使用同一样本上的分层 paired bootstrap 95% CI、双侧
prediction-swap randomization test，并在每个 model-mode × metric 的三个菌株内做 Holm 校正。

## 已由运行产物验证的事实

### Common-fold Chemprop baselines

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

因此 A. baumannii 和 RN4220 的下降不能解释为重构代码错误：比较同时改变了 ensemble
大小（10 → 1）、聚合方式（fold metric mean → pooled OOF）、checkpoint 完整性和推理模式。
RN4220 fold 4 还是后补训候选。canonical runner 的 legacy 行为等价测试、strict zero-shot
逐样本 logit 对齐和历史 checkpoint 严格加载均已通过；当前证据支持“口径不同”，不支持
“重构 forward 发生行为漂移”。若要恢复与旧柱子严格同口径的显著性结果，必须补训缺失的
46 个历史网格成员，形成每个 target 都完整的 5 folds × 10 members，并重新输出样本级预测。

## Chemprop 数值与来源论文的关系

| Target profile | 当前 common-fold 结果 | 来源论文主结果 | 解释 |
| --- | ---: | ---: | --- |
| Stokes 2020 / E. coli | AUROC 0.85711；AUPRC 0.50752 | AUROC 0.896；未汇报 AUPRC | 当前 AUROC 更低，AUPRC 是本次共同 folds 新增结果 |
| Liu 2023 / A. baumannii | AUROC/AUPRC 0.77750/0.31967 | 约 0.792/0.337 | 接近且略低；旧图 0.756/0.266 是 no-RDKit ablation，不是主模型 |
| Wong 2024 / RN4220 | AUPRC 0.32873 | AUPRC 0.364 | 当前更低 |

三个 baseline 都是在 ApexOracle 固定 outer folds 上重新训练的单模型 Chemprop sensitivity，
不是直接复制来源论文 headline。它们没有高于各自论文主结果；相对旧图变高的主要原因是
恢复了论文主 profile（尤其 Liu 的 RDKit2D descriptors），并统一了数据和 folds。

## 绘图与显著性标注

- canonical 绘图实现是 `src/apexoracle/evaluation/fig1b_plot.py`，入口是
  `scripts/reproduce/plot_fig1b_revision.py`；Matplotlib 生成冻结 CSV、PDF 和 PNG。
- Mac canonical notebook 是
  `/Users/kirianozan/Documents/Study/Penn/projects/local_figs/figs.ipynb`，cell ID
  `220739609a526f79`。`/Users/kirianozan/Documents/Study/Penn/Synergy/paper_figs/figs.ipynb`
  是只有三个旧 cell 的另一份 notebook，从未包含该 Fig. 1b cell。
- 图上 bracket 沿用 Fig. 3a 的视觉样式，但统计量使用本实验的 paired prediction-swap test；
  文本显示三菌株内 Holm 校正后的精确 p 值。legend 显式写为
  `ApexOracle, fine-tuned (1 model/fold)`，不再把 sensitivity 暗示成旧 10-member ensemble。

## 发布同步

- Mac canonical notebook cell `220739609a526f79` 已改为读取冻结 CSV、绘制三个菌株的统一
  AUPRC 及 Holm-adjusted p-value brackets；该 cell 已在 Mac `base` 环境以 `Agg` 后端执行通过。
- 新 panel 已合并到论文 `Fig1.pdf`；Results、Methods、caption 和 response letter 已同步为
  上述统计结论，论文完整编译为 28 页。
- 修改前快照、最终文件和运行产物 SHA-256 见
  `reproducibility/fig1b_reviewer_revision_2026-07-20.json`。
