# Fig. 1b 三菌株抗生素分类重构

本目录记录 Fig. 1b 三菌株分类代码家族的行为保持迁移。唯一入口为：

```bash
python scripts/reproduce/run_antibiotic_classification.py \
  --mode strict-zero-shot \
  --test-group 0 \
  --dry-run
```

`--test-group` 的固定顺序是：

0. `#004`（E. coli BW25113，genome + text）；
1. `17978`（A. baumannii ATCC 17978，genome + text）；
2. `Staphylococcus aureus RN4220`（text-only）。

## 三种模式

- `strict-zero-shot`：目标菌株全部样本只用于逐 epoch 评估和 checkpoint 选择，不进入训练；不接受 `--fold`。
- `fine-tune`：目标菌株在经过 512-token 过滤后执行 `KFold(n_splits=5, shuffle=True, random_state=42)`；必须指定 `--fold 0..4`。
- `molecule-only`：使用同一目标 KFold，但只训练 `768→384→128→1` molecular classification head；这是旧文件名中的 `wo_SAND` 对照，不是 strict zero-shot ApexOracle。

共享配置位于 `configs/antibiotic_classification/legacy_three_strain.yaml`。三种模式共享数据准备、target/fold adapter、feature loader、指标、AUROC best-tracker、checkpoint 命名和 ensemble orchestration；full-fusion 与 molecule-only 只在真实模型和训练行为不同处保留 mode adapter。

## 已由代码、日志和真实运行验证的事实

- 三个旧 MDLM driver 的前 834 行完全相同；Dataset、collate、fusion、head 和绝大部分训练逻辑属于复制代码。
- full-fusion 模式使用完整 MIC genome-text frame，同时再次把 genome-backed MIC 记录放入 text route；这会让这些 MIC 记录通过两个 route 各训练一次。统一 runner 明确保留该行为。
- batch size 为 70，默认 25 epochs、10 ensembles、Adam `1e-5`、CosineAnnealingLR、`freeze_epochs=3000`。
- strict zero-shot 的三个目标 checkpoint 网格完整，共 `3 × 10 = 30` 个文件，并有三个完整结束的日志。
- fine-tune 预期 `3 × 10 × 5 = 150` 个 checkpoint，现存 77 个；14 个日志中只有 6 个包含最终五折汇总。因此它只能标记为历史证据不完整。
- molecule-only 的 `3 × 10 × 5 = 150` checkpoint 网格和 15 个日志均完整，但它是 molecule-only 对照。
- strict zero-shot 日志最终 ensemble 指标为：`#004` 0.9360/0.5890，`17978` 0.7262/0.3243，RN4220 0.7679/0.1655（AUROC/AUPRC）。
- 真实数据 dry-run 与旧日志中的 Dataset 计数逐项一致。以 `#004` 为例，512-token 过滤后的两路 MIC 为 77,372/89,118，辅助 genome/text 分类为 7,684/39,311，held target 为 2,335。
- H100 上使用 group 0 / ensemble 0 的 capsule checkpoint、batch size 70 和全 eval checkpoint inference 时，canonical runner 与旧 capsule 的 2,335 个 molecule ID、label 和 logit 逐值完全相同；AUROC/AUPRC 均为 `0.9350169300225734 / 0.6120361666817208`。详见 `checkpoint_validation_group0_ensemble0.json`。
- 合成四路 batch 的一轮 CUDA 集成测试通过，覆盖 MIC genome、MIC text route、辅助 genome classification、辅助 text classification、scheduler 和 held-target evaluation。

checkpoint 文件网格和日志完成度来自 `checkpoint_family_audit.json`，生成命令为：

```bash
python scripts/audit/audit_antibiotic_classification_checkpoints.py \
  --output experiments/fig1b_antibiotic_classification/checkpoint_family_audit.json
```

## 必须保留的历史行为

- full-fusion 每个 epoch 只对 genome attention、text attention 和 regression head 调用 `train()`/`eval()`；classification head 在 held-target checkpoint selection 时仍保持 train mode，因此 dropout 参与 AUROC/AUPRC。
- molecule-only 在评估前显式调用 classification head 的 `eval()`。
- held target 直接用于逐 epoch best-AUROC checkpoint selection，没有独立 validation set。
- AUROC 提升时先保存 checkpoint，随后才更新 best AUPRC；因此 checkpoint 内的 `auprc` 可能落后于同 epoch 实际 AUPRC。
- strict zero-shot 的目录名仍包含历史遗留的 `10_fold`，但该模式没有 target KFold。
- RN4220 作为 target 时，两个 genome-backed auxiliary block 来自 Python set difference；其顺序仍受 `PYTHONHASHSEED` 影响。旧日志没有记录该环境变量，不能声称恢复了逐 batch 的精确历史顺序。

## 推理指标与训练日志的边界

历史训练日志中的 best prediction 来自 classification head 的 train-mode dropout。加载 checkpoint 后做确定性推理时，canonical `--evaluate-checkpoints` 会把全部消费模块设为 eval，这与 capsule 的推理契约一致，但不应被描述为逐 bit 重放训练时的随机 best predictions。

此外，bfloat16 attention 对分批方式敏感：旧 capsule 默认 batch size 64，而论文训练 driver 使用 70。group 0 / ensemble 0 在 64 与 70 下 AUROC 相同，AUPRC 相差约 `2.9e-5`；统一到 batch size 70 后逐样本 logit 完全一致。

## 仍待确认或不在本阶段声称的事项

- fine-tune 结果证据不完整，不应从现存 77 个 checkpoint 推断完整五折正式结果。
- 本阶段没有重新训练 30 个 strict zero-shot ensemble；保留并验证的是原行为和已有 checkpoint 推理。
- strict checkpoint 的历史训练时 dropout prediction 仍不可逐 bit 恢复；当前 30 个 checkpoint
  已完成的是统一 `eval()` 契约下的确定性推理。

## Reviewer 修订：三菌株统一 AUPRC 与显著性（已完成）

Reviewer 要求 Fig. 1b 对三个菌株一致报告 AUPRC，并为“优于 baseline”的表述提供统计检验。
本阶段新增两个彼此分离的入口：

- `scripts/reproduce/run_fig1b_chemprop_baselines.py`：按三个原论文的 Chemprop/RDKit
  结构，在 ApexOracle 的固定 outer folds 上训练 baseline；outer-train 内部另划 validation，
  outer test 不参与 checkpoint 选择。
- `scripts/reproduce/analyze_fig1b_significance.py`：对同一样本上的预测执行分层 paired
  bootstrap 95% CI 和双侧 prediction-swap randomization test，并在同一指标族内做 Holm 校正。

已验证事实：

- strict zero-shot 的全部 30 个 checkpoint 已在 H100 上完成确定性 ensemble inference，样本级
  预测分别覆盖 2,335、7,684、39,311 个样本；AUROC/AUPRC 为
  `0.93504/0.58738`、`0.72408/0.32098`、`0.76741/0.16562`。
- Liu 2023 主模型使用 RDKit descriptors，公开十折结果约为 AUROC/AUPRC
  `0.792/0.337`；当前 Fig. 1b 的 `0.756/0.266` 实际对应该论文的 no-RDKit ablation，
  不能继续标成主 baseline。
- E. coli 有 1 个 MDLM 可处理、但 RDKit 因异常铝配位价态拒绝的 SMILES。固定 KFold
  membership 不重排；该样本只在配对统计时从双方同时排除，并写入 exclusions。
- node002 的隔离环境固定为 Chemprop 1.5.2、RDKit 2025.03.5；PyTorch 2.7 加载受信任的
  Chemprop 1.x checkpoint 时显式设置 `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`，不改权重内容。

2026-07-20 运行状态：三个 baseline 的 15 个 fold 已全部完成。pooled OOF 的
AUROC/AUPRC 分别为 E. coli `0.85711/0.50752`、A. baumannii `0.77750/0.31967`、
RN4220 `0.92848/0.32873`。E. coli 与 RN4220 各有一条相同的异常铝配位结构不能被
Chemprop/RDKit 预测；配对分析从双方同时排除，并分别记录为 `ce_2244` 与 `na_20640`。

strict zero-shot 的 5,000 次 paired 分层 bootstrap 和 prediction-swap 检验已经完成。
以 AUPRC 为主要指标，E. coli 数值高 `0.07986` 但 Holm 校正后不显著（`p=0.4167`），
A. baumannii 几乎相同（差 `0.00132`，`p=0.9660`），RN4220 显著低
`0.16311`（`p=0.00060`）。AUROC 方面 E. coli 显著高 `0.07791`
（`p=0.01360`），A. baumannii 显著低 `0.05342`（`p=0.03059`），RN4220 显著低
`0.16107`（`p=0.00060`）。因此不能再保留“zero-shot 普遍优于 baseline”的概括。

fine-tune sensitivity 固定为每个 outer fold 恰好使用 `ensemble_0`，避免把不同大小的
残缺 ensemble 混入同一统计表。14/15 个 fold 复用历史 checkpoint；唯一缺失的 RN4220
fold 4 已在本机 H100 以 `PYTHONHASHSEED=0`、ensemble seed 42 按旧 25-epoch 协议补训，
best-AUROC checkpoint SHA-256 为
`68a34004a4992c0bfff3733a9e5e7135ebed79bfbf15dd38e6eca7d2199d6a87`。

fine-tune pooled OOF 的 AUROC/AUPRC 分别为 E. coli `0.95529/0.66655`、A. baumannii
`0.77698/0.35294`、RN4220 `0.92278/0.34518`。经 5,000 次 paired 检验和三菌株内 Holm
校正，只有 E. coli AUPRC（差 `+0.15903`，`p=0.03539`）与 AUROC（差 `+0.09818`，
`p=0.00180`）显著高于 baseline；A. baumannii 与 RN4220 的四项差异均不显著。完整结果见
`results_reviewer_revision.md`；该表明确属于单模型/折 sensitivity，不是旧完整 ensemble 结果。

Mac notebook、最终 panel、论文和回复信已同步并通过完整编译。修改前快照、运行产物及
最终文件的 SHA-256 见 `reproducibility/fig1b_reviewer_revision_2026-07-20.json`。

node002 运行环境是
`/data1/tianang/Projects/.venvs/fig1b-chemprop-v1`：Python 3.12.7、Chemprop 1.5.2、
RDKit 2025.03.5、PyTorch 2.7.1+cu126、Pandas 2.2.2、scikit-learn 1.8.0。
该 venv 使用 `--system-site-packages` 继承 node002 的 CUDA PyTorch；Chemprop 与新版 Pandas
不兼容的旧 RDKit 2023 已在 venv 内由 RDKit 2025.03.5 覆盖，未修改 conda base。

## 完整 ensemble 最终补跑（已完成）

上文已经完成的 baseline 与单成员 fine-tune 只作为 reviewer sensitivity 保留。作者随后确认
最终结果中 ApexOracle 每个 outer fold 使用 10 members。作者于 2026-07-21 决定三个 baseline
也统一使用固定编号 0--9 的 10-member ensemble，而不是照搬发布资产不同的 `20/10/20` 大小。
由于 ApexOracle 自身没有 RDKit feature augmentation，Liu 使用论文报告的 no-RDKit ablation，
不采用其 RDKit 增强主模型。新运行写入
`results/fig1b_revision/full_ensemble_reconstruction/` 和
`results/fig1b_revision/baselines_full_ensemble_no_rdkit/`，不覆盖历史 checkpoint 或原始数据。

这里的完整 fine-tune 网格是 `3 strains × 5 outer folds × 10 members = 150 checkpoints`，
不是 strict zero-shot 的 `3 × 10 = 30`。两台机器共恢复 104 个历史 checkpoint，另补完
RN4220 fold 4 member 0 后已有 105 个，因此本轮只训练剩余 45 个 member。监控中的待补完成数
只计算这 45 个新任务；例如 `0/45` 不表示已有 105 个资产不存在。

ApexOracle 完整 ensemble 已于 2026-07-21 完成。每个成功 member 固定完整训练 25 epochs，
没有 early stopping。
目标菌株四个 folds 用于训练，第五个 held fold 每个 epoch 都用于 strict highest-AUROC
checkpoint selection；没有独立 validation set。该 held-test selection 是历史协议的已知局限，
当前为保持行为而保留。随后另行加载最佳 checkpoint，以全 `eval()` 模式导出确定性 prediction。

本任务不在线调用 molecular encoder。runner 启动时一次性加载 768 维 float32 的
`Pep_emb_dict_cls_wo_pad_eval.pt`（18,029 entries，SHA-256 `c0863086…a9ff7b`）和
`SM_emb_dict_cls_wo_pad_eval.pt`（49,330 entries，SHA-256 `5d2e2f4d…e83ac`），Dataset 按
`DBAASP_id` 查表。这里的 fine-tune 只更新 strain-aware attention、regression/classification
heads 和 missing-genome parameter，不更新 DLM/MDLM molecule encoder。

最终组装严格核验 15 个 fold 均为 10 members。共同 baseline cohort 上的 fold mean ± sample
s.d.（与旧论文柱子相同口径）为：E. coli AUPRC/AUROC
`0.71205 ± 0.14657 / 0.95884 ± 0.03341`；A. baumannii
`0.43436 ± 0.05261 / 0.82200 ± 0.02138`；RN4220
`0.40127 ± 0.03500 / 0.95309 ± 0.00776`。pooled OOF AUPRC/AUROC 分别为
`0.69045/0.95560`、`0.41636/0.81732`、`0.39442/0.94689`。

15 个 baseline folds 已全部完成。baseline 的 fold-mean AUPRC/AUROC 为 E. coli
`0.54535/0.85932`、A. baumannii `0.30355/0.77589`、RN4220 `0.36616/0.94337`。
5,000 次 paired bootstrap/prediction-swap 的最终结果显示：fine-tune 在三株两个指标上均高于
baseline，且 6/6 Holm-adjusted `p < 0.05`；strict zero-shot 只有 E. coli AUROC 显著更高，
A. baumannii AUROC 显著更低，RN4220 两项均显著更低。完整数值见
`results_reviewer_revision.md`。

## 最终双指标 panel（2026-07-22）

最终绘图输入冻结在 `final_10member_dual_metric.csv`。每个柱高为五个 outer folds 的指标均值，
error bar 为五折 sample s.d.；显著性 bracket 显示同一 held-fold 样本上 prediction-swap test 的
Holm-adjusted `p`。左 panel 为 AUPRC，右 panel 为 AUROC，三种方法依次为 Chemprop baseline、
ApexOracle zero-shot 和 ApexOracle fine-tuned（每折均为固定 10-member ensemble）。

Mac canonical notebook
`/Users/kirianozan/Documents/Study/Penn/projects/local_figs/figs.ipynb` 中新增且只执行了 cell
`fig1b-final-10member-dual-metric-20260722`；说明 cell 为
`fig1b-final-10member-dual-metric-notes-20260722`。新 cell 沿用原论文 panel 的 pastel 配色、
白色柱边、虚线水平网格、italic species labels 和 despine 风格，独立输出：

- `/Users/kirianozan/Documents/Study/Penn/Synergy/paper_figs/3-strain-antibiotics-final-10member-dual-metric.pdf`
- `/Users/kirianozan/Documents/Study/Penn/Synergy/paper_figs/3-strain-antibiotics-final-10member-dual-metric.png`

原始 cell `220739609a526f79` 的 source SHA-256 仍为
`56d9eb55eb1d23ee7f30d9ad504cc49bb6f05f862517d34bc60ca9c480cc24a4`，已有 output 的语义
SHA-256 仍为 `86b36871d2ce1810dfa44e7f41c7fad475cc66923790e8f7d97006cd81fb52f2`。
本次没有执行或重写原始 cell，也没有覆盖旧 panel PDF、reviewer sensitivity panel 或论文总图。
完整 notebook、输入和输出 hash 见
`reproducibility/fig1b_reviewer_revision_2026-07-20.json`。

作者随后在原始 cell 后新增 `### Final 10-member Fig. 1b: AUPRC only version` 区域。
该区域的 placeholder 已填写为独立 cell `fig1b-final-10member-auprc-only-20260722`，只读取同一
冻结 CSV 的 AUPRC 行，沿用相同三色 palette、五折 sample s.d. 和 Holm-adjusted paired
prediction-swap `p`。它独立输出：

- `/Users/kirianozan/Documents/Study/Penn/Synergy/paper_figs/3-strain-antibiotics-final-10member-auprc-only.pdf`
- `/Users/kirianozan/Documents/Study/Penn/Synergy/paper_figs/3-strain-antibiotics-final-10member-auprc-only.png`

该图已按最终 PNG/PDF 检查，title、legend、数值标签、error bar 和六个 bracket 均无裁切或重叠。
原始 cell 和双指标 cell 均未执行或修改。给 reviewer 的最终英文回复、建议图注、Results/Methods
替换段落及相对当前论文草稿的修改清单位于 `reviewer_response_auprc_final.md`；原始 docx 和 TeX
尚未自动改写。

作者反馈初版 legend 与坐标区之间留白过大后，仅修改上述 AUPRC-only cell 的 layout：画布使用
旧论文 panel 的实际页面尺寸 `741.12 × 380.724 pt`（约 `10.293 × 5.288 in`），取消
`bbox_inches="tight"` 并显式设置 margins，同时把 y-axis 上限从 `1.12` 收紧至 `1.03`。最终
PDF 与旧 `3-strain-antibiotics.pdf` 页面尺寸逐点一致；原始和双指标 cell 仍未变化。

一次性查看本机、node001 和 node002 的合并状态：

```bash
python scripts/reproduce/monitor_fig1b_revision.py
```

每 30 秒连续刷新：

```bash
watch -n 30 python scripts/reproduce/monitor_fig1b_revision.py
```

监控同时显示 45 个待补 member 的唯一完成数与 epoch、15 个 baseline fold、动态 Apex ETA、
GPU 温度/热降频、磁盘余量和新 checkpoint 占用。总 ETA 必须等首个完整 baseline fold 完成后
再外推；Apex ETA 不应被误写成整个 reviewer 实验的完成时间。
