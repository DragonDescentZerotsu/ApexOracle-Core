# Fig. 2b 共同数据五折 benchmark 结果

完成日期：2026-07-18。

本次正式修订实验严格使用 10,886 个全部 encoder 都能处理的 molecule ID 和同一份 molecule-level 五折划分。相较原论文实验，只统一了样本集合和 fold membership；各模型的 encoder、checkpoint、prediction head、200 epochs、batch size 200、Adam、learning rate `1e-4` 以及 held-out-fold checkpoint selection 行为保持原实现。

## 汇总结果

R² 的标准差是五个 fold 最佳 mean R² 的 sample SD。

| 模型 | 共同数据 R²（mean ± SD） | 原 retained-set rerun | 变化 | 论文 Fig. 2b 近似值 | 相对论文图变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLM MTR+DLM | **0.5386 ± 0.0250** | 0.5207 | +0.0179 | 0.530 | +0.0086 |
| ChemBERTa MTR | 0.4172 ± 0.0275 | 0.4197 | -0.0025 | 0.417 | +0.0002 |
| DLM MLM（DLM-only） | 0.3765 ± 0.0239 | 0.4083 | -0.0318 | 0.408 | -0.0315 |
| ChemBERTa MLM | 0.2247 ± 0.0131 | 0.2302 | -0.0055 | 0.226 | -0.0013 |
| PeptideCLM | 0.3836 ± 0.0244 | 0.3767 | +0.0069 | 0.376 | +0.0076 |
| MolFormer | 0.3678 ± 0.0198 | 0.3726 | -0.0048 | 0.371 | -0.0032 |
| APEX | 0.4014 ± 0.0146 | 0.4050 | -0.0036 | 0.403 | -0.0016 |

共同数据上的排序为：DLM MTR+DLM、ChemBERTa MTR、APEX、PeptideCLM、DLM-only、MolFormer、ChemBERTa MLM。

## 逐 fold 结果

| 模型 | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DLM MTR+DLM | 0.526298 | 0.566994 | 0.564266 | 0.520626 | 0.514979 |
| ChemBERTa MTR | 0.400020 | 0.433026 | 0.455621 | 0.411580 | 0.385937 |
| DLM MLM（DLM-only） | 0.350616 | 0.389063 | 0.411144 | 0.361923 | 0.369984 |
| ChemBERTa MLM | 0.208353 | 0.223749 | 0.242040 | 0.216790 | 0.232368 |
| PeptideCLM | 0.352312 | 0.380115 | 0.418019 | 0.394075 | 0.373461 |
| MolFormer | 0.363893 | 0.378595 | 0.394524 | 0.359945 | 0.342193 |
| APEX | 0.391343 | 0.407716 | 0.424117 | 0.393893 | 0.389812 |

五个 test fold 的大小依次为 2,178、2,177、2,177、2,177、2,177；每个模型的五个 test fold 合计恰好覆盖全部 10,886 个 molecule，35 个模型-fold 组合均已完成且没有重复或缺失。

## 结论与解释边界

- DLM MTR+DLM 在共同数据上比原 retained-set rerun 高 0.0179，仍是性能最高的模型。
- DLM-only 降低 0.0318，是唯一出现较大变化的模型；这说明旧实验中各模型样本和 fold 不一致对该 bar 的影响不可忽略。新论文图和 reviewer response 应使用 0.3765，而不是沿用约 0.408 的旧值。
- 两个 DLM checkpoint 的容量不同：DLM-only 是 12-layer/768，MTR+DLM 是 24-layer/1024。node002 原始训练目录和源码确认 24-layer checkpoint 从一开始就采用联合 `DLM + 0.1 × MTR MSE` 目标；当前未找到同容量的纯 DLM 权重。因此二者是模型版本比较，不是容量受控的 MTR objective ablation，正文和 reviewer response 不应把 0.1621 的差异完全归因于 MTR。
- ChemBERTa-MTR、ChemBERTa-MLM、MolFormer 和 APEX 相对原 rerun 的绝对变化都小于 0.006；PeptideCLM 上升 0.0069。
- APEX 没有被重构或改变：仍使用原 23-token vocabulary、AAindex embedding、encoder、checkpoint 和 `512→256` regression head；noncanonical residue 的 `X` 按原 `onehot_encoding` 行为留在 index 0。
- 原代码没有显式固定 PyTorch 训练 seed。本次保留了这一原始行为，并在每个 fold 的 `metrics.json` 中记录 `initial_torch_seed`；因此这里是一次忠实的 stochastic rerun，而不是跨硬件逐 bit 确定的结果。
- 原代码逐 epoch 在 held-out fold 上选择最佳 checkpoint，且 APEX 评估时保留 head dropout。本次按 reviewer 要求只统一数据和 folds，没有新增 validation，也没有把划分改成 scaffold split。这一 test-set reuse 局限应在回复或补充材料中披露。

## 数据与模型溯源

- 协议版本：`fig2b-shared-native-intersection-v2`；KFold `shuffle=True, random_state=42`。
- `common_molecule_ids.csv` SHA-256：`6563ec0f19aea03c13fe9fd70c4847608adca78b669866d4d1542708c6e6578b`。
- `folds.csv` SHA-256：`500bcc58f0d28976394fa7e7000dfb8c70caa1a38f675745fdc2a6dd9cf299c4`。
- DLM-only 使用 12-layer `best_2.ckpt`；DLM MTR+DLM 使用 24-layer `best.ckpt`。
- 24-layer checkpoint 的原始训练文件位于 `node002:/data1/fangping/mdlm/outputs/openwebtext-train/2025.05.06/112126/checkpoints/`；搜索范围与解释边界详见 `MODEL_WEIGHTS.md`。
- MolFormer 使用 `ibm/MoLFormer-XL-both-10pct` revision `7b12d946c181a37f6012b9dc3b002275de070314`。这是本地已有权重对应、且与当前环境兼容的历史 revision；没有改变模型结构、权重或训练超参数。
- 完整 checkpoint、prediction 和逐任务指标位于被 Git 忽略的 `results/fig2b_shared_original_protocol/`；其中 `REPORT.md`、`comparison_summary.json`、`comparison_summary.csv` 和 `fold_metrics.csv` 可用于后续更新图表。
