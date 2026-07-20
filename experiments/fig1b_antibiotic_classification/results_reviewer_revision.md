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

### Fine-tune sensitivity

为避免把不同大小的残缺 ensemble 混在一个 OOF 表中，修订 sensitivity 固定每个 outer fold
只使用原编号 `ensemble_0`。14/15 个历史 fold 已完成确定性推理；RN4220 fold 4 没有遗留
checkpoint，正在按旧 25-epoch 多任务协议补训。该部分最终数值和统计尚未冻结。

## 根据现有证据作出的解释

- strict zero-shot 并非跨三个 target 普遍优于 baseline；当前仅 E. coli AUROC 显著更高。
- A. baumannii 的 AUPRC 与 baseline 无可检测差异，但 AUROC 显著较低。
- RN4220 的 AUPRC 和 AUROC 均显著低于 baseline。因此正文和图注必须撤回笼统的优势表述。
- E. coli AUPRC 的点估计较高，但 CI 跨 0 且 Holm 校正后不显著，不能写成统计学优势。

## 仍待完成

- 完成 RN4220 fold 4 补训、单 checkpoint 推理和三菌株 fine-tune OOF 汇总；
- 对 fine-tune vs baseline 运行同一套 5,000 次统计，并生成最终 Fig. 1b AUPRC panel；
- 同步论文 Results、Methods、caption、response letter 和 Mac 绘图 notebook。
