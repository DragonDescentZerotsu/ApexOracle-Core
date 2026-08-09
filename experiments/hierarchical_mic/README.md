# Hierarchical MIC holdouts：strain / species / phylum

本目录记录论文中三种 hierarchical MIC holdout 的统一实现。三者只在 held-out strain
集合的生成方式上不同；Dataset、molecule/genome/text fusion、regression/classification
heads、四路训练、评估、checkpoint selection 和 ensemble 现在只有一份 canonical 代码。

## Canonical 实现

- 唯一入口：`scripts/reproduce/run_hierarchical_mic.py`
- 唯一配置：`configs/hierarchical_mic/legacy_mdlm.yaml`
- 公共数据与 split adapter：
  `src/apexoracle/data/hierarchical_mic_preparation.py`
- 公共 Dataset/collate：`src/apexoracle/data/hierarchical_mic.py`
- 公共训练 primitives：`src/apexoracle/training/hierarchical_mic.py`
- 公共完整 runner：`src/apexoracle/training/hierarchical_mic_runner.py`
- 公共评估：`src/apexoracle/evaluation/hierarchical_mic.py`
- 公共 fusion/head：`src/apexoracle/models/strain_fusion.py`
- 公共 checkpoint loader：
  `src/apexoracle/models/hierarchical_mic_checkpoint.py`
- Fig. 2c online encoder profiles：
  `configs/hierarchical_mic/legacy_fig2c_comparators.yaml`

旧的 `strainwise` package 路径只保留很薄的 import facade，用于已有调用兼容；不再包含
第二套实现。

## 三个 split adapter

| `--protocol` | 论文名称 | adapter | group |
| --- | --- | --- | --- |
| `strain` | strain-wise | 每个 species 内按旧代码生成三折 strain holdout | `fold 1`–`fold 3` |
| `species` | species-wise | taxonomy tree 距离矩阵 + average-linkage 11 clusters | 11 个历史命名 group |
| `phylum` | phylum-wise | taxonomy tree 距离矩阵 + average-linkage 3 clusters | `Fungi`、`Pseudomonadati`、`Bacillati` |

`species` 使用 `All_species_gt_Taxonomy_Tree_cluster.phy`；`phylum` 使用
`All_species_gt_Taxonomy_Tree_draw.phy`。两个 adapter 均保留旧实现的 taxonomy 名称映射、
cluster 按 species 数量排序和 auxiliary strain set 迭代语义。

## 已由代码、日志和运行验证的事实

- node002 上找回的 phylum-wise MDLM 终版候选 SHA-256 为
  `36ef70bc4a20f2d94294e40b027be7b41c0c8a722c97a09bee856916622789e1`。
  它与 strain/species 终版使用相同的 768-d molecule embedding、8192-d genome
  cross-attention、4096-d text cross-attention、`12288→3072→128→1` 两个 head、
  learnable missing-genome parameter、Adam `1e-5`、batch size 80、7 seeds 和 25 epochs。
- 三个真实数据 dry-run 均成功。species group 0 的四类 regression 计数为
  `79903/1/92275/46`（长度过滤后 `77371/1/89073/44`），与保存的 MDLM 日志完全一致。
- phylum `Fungi` 的四类 regression 计数为 `75066/4838/86783/701`（长度过滤后
  `72723/4649/83790/679`），两个 auxiliary 计数为 `10019/39311`；这些数值与
  node002 终版日志逐项完全一致。
- strain `PYTHONHASHSEED=0` candidate 的 group 0 计数与冻结 manifest 完全一致；历史
  三个独立进程未记录 hash seed，因此仍不声称恢复了 2025 年精确 membership。
- 公共 runner 保留模块在 `train()` mode 下评估、dropout 参与 held-out selection、
  `zip_longest` 四路 loader、每个非空 modality batch 单独 optimizer step、每 epoch 一次
  cosine scheduler、严格 `>` best-metric 更新和原七键 checkpoint payload。
- classification 在 epoch 5000 后仍按历史行为裁剪 `reg_head` 而非 `cls_head`，text
  attention 仍不裁剪；默认 25 epochs 不会进入该分支。
- H100 一轮合成集成 smoke 已实际跑通四路训练、CUDA autocast/GradScaler、评估、
  scheduler、best-R² selection 和 checkpoint 写出。
- strain 历史 21 个 checkpoint 的结构审计和代表性严格推理证据保存在 `strain/`。

## 运行方式

三个实验只改变 `--protocol` 和 `--test-group`：

```bash
python scripts/reproduce/run_hierarchical_mic.py \
  --protocol strain \
  --test-group 0 \
  --device 0 \
  --acknowledge-dynamic-legacy-split

python scripts/reproduce/run_hierarchical_mic.py \
  --protocol species \
  --test-group 6 \
  --device 0 \
  --acknowledge-dynamic-legacy-split

python scripts/reproduce/run_hierarchical_mic.py \
  --protocol phylum \
  --test-group 0 \
  --device 0 \
  --acknowledge-dynamic-legacy-split
```

用 `--dry-run` 只核对资源、split、group 名称和数据计数，不加载大型 embedding 或训练。

### Fig. 2c molecular encoder comparator

四个 strain-wise comparator 现在复用同一个 split、fusion、四路训练、评估和 checkpoint
runner，只通过 `--molecule-encoder` 选择 online molecule backbone：

```bash
python scripts/reproduce/run_hierarchical_mic.py \
  --config configs/hierarchical_mic/legacy_fig2c_comparators.yaml \
  --protocol strain \
  --molecule-encoder chemberta_mtr \
  --test-group 0 \
  --device 0 \
  --acknowledge-dynamic-legacy-split
```

可选值为 `chemberta_mtr`、`chemberta_mlm`、`molformer` 和 `peptideclm`。profiles 保留各旧
driver 的 raw-SMILES online tokenization、first-token pooling、hidden size、backbone mode、
freeze epoch、ensemble 数、optimizer group 次序和 legacy checkpoint key；没有把四个模型强行
改成同一训练行为。Hugging Face revision 已固定用于当前复现，但旧 run 没有记录 2025 年实际
upstream commit，因此固定 revision 是经 checkpoint 兼容性验证的复现锚点，不是对旧 commit
身份的绝对证明。

PeptideCLM 需额外保留一项边界：本机较晚的 `fix` driver/checkpoint 为 eval mode、默认 25
epochs 内始终冻结、单 member；node002 还保留一个 freeze 3 epochs、7 members 的较早 driver，
但没有找到完整的 7-member checkpoint 网格。canonical profile 采用有可加载 checkpoint 支持的
本机 fixed 版本，不声称已经恢复论文绘图所用的精确 PeptideCLM ensemble。逐项证据和数据
SHA-256 见 `strain/fig2c_comparator_migration_audit.json`。

## Legacy 清理与恢复

已被统一 runner 替代的 root-level DP/in-house/SM、species/phylum、pooling/eval 脚本及
capsule 内第二份 strain driver 已删除。它们仍可从 annotated tag
`legacy-code-snapshot-2026-07-17` 恢复。
精确的逐文件删除记录见 `legacy_cleanup.json`。

以下实验不是同一模型的复制版本：

- Fig. 2c 的 ChemBERTa-MTR、ChemBERTa-MLM、MolFormer 和 PeptideCLM 已迁入显式 profiles；
- 尚未独立迁移和核验的 genome-only、text-only 与早期 modality ablation 血缘。

这一区分避免以“清理”为由丢失仍可能对应论文图表的不同实验功能。

## 证据边界

- strain-wise 保存结果 `0.4057/0.6889/0.6434`（均值 `0.5793`）仍是高置信度正式结果。
- species-wise 完整 11-group 论文运行未全部保留；现存终版日志只有 group 6–10。
- phylum-wise node002 有三个终版 MDLM 日志和 checkpoint，但现存指标与论文汇总仍有差异。
  统一代码证明代码路径和可用日志分区一致，不会把缺失的历史结果升级为“已完整复现”。

## 验证命令

```bash
pytest -q tests/test_hierarchical_mic_runner.py \
  tests/test_strainwise_legacy_equivalence.py

python scripts/reproduce/validate_hierarchical_mic_checkpoint.py \
  --checkpoint Checkpoints/genome_text_learnable_emb/strain_wise_w_SM_b_attn/MDLM_MTR_fix_7_fold_ensembles/genome_text_learnable_emb_Strain_wise_best_R2_group_0_ensemble_0.pth \
  --device cuda:0
```

## Reviewer exact-molecule overlap audit

Canonical CPU/只读入口：

```bash
PYTHONPATH=src python scripts/audit/audit_hierarchical_mic_molecule_overlap.py \
  --protocol all
```

共享逻辑位于
`src/apexoracle/evaluation/hierarchical_mic_molecule_overlap.py`，输出位于
`experiments/hierarchical_mic/molecule_overlap/`。该入口不修改训练 split 或 checkpoint；
它在与 legacy Dataset 相同的 512-token eligibility filter 后，分别按 `DBAASP_id` 和模型实际
stored-token input SHA-256 统计 train/test exact-molecule overlap。strain 总数是三个 fold 的
measurement instances，不能解释为跨 fold 去重后的唯一实验记录。

## Reviewer molecule-disjoint checkpoint replay

先生成不含 optimizer 和未使用 classification head 的推理副本：

```bash
PYTHONPATH=src python \
  scripts/reproduce/prepare_hierarchical_mic_inference_checkpoints.py \
  --protocol strain \
  --output-dir experiments/hierarchical_mic/molecule_disjoint/inference_checkpoints/strain
```

再按一个 `protocol × group × ensemble` 任务运行：

```bash
PYTHONPATH=src python \
  scripts/reproduce/evaluate_hierarchical_mic_molecule_disjoint.py \
  --protocol strain --group 0 --ensemble 0 --device cuda:0 \
  --inference-only \
  --checkpoint-dir experiments/hierarchical_mic/molecule_disjoint/inference_checkpoints/strain \
  --output-dir experiments/hierarchical_mic/molecule_disjoint/predictions
```

同一个 `--group` 可重复传入 `--ensemble`，使多个 member 复用一次大型 feature load，例如
`--ensemble 0 --ensemble 1 --ensemble 2`。

汇总 7-member sample-level ensemble、常数/peptide-mean baselines 和 exact-molecule cluster
bootstrap：

```bash
PYTHONPATH=src python \
  scripts/reproduce/summarize_hierarchical_mic_molecule_disjoint.py \
  --protocol strain --groups 3 --members 7 \
  --prediction-dir experiments/hierarchical_mic/molecule_disjoint/predictions \
  --output-dir experiments/hierarchical_mic/molecule_disjoint/analysis
```

推理副本 manifest 记录源 checkpoint path、size、SHA-256 和派生文件的 size、SHA-256。replay
不改变 pathogen holdout 或训练数据；它以精确 stored-token SHA-256 标记 full、train-seen 和
train-unseen peptide test rows，并写出 train-peptide-mean baseline。推理使用确定性 `eval()`
mode。strain 输出仍是冻结 candidate membership 上的 reconstruction，不能表述为已恢复的
2025 精确 membership。

正式 reviewer sensitivity 使用固定 membership 的 phylum `3 × 7` replay；compact 汇总位于
`molecule_disjoint/phylum_analysis/`，解释边界和英文回复草稿见
`molecule_disjoint/REVIEWER_SENSITIVITY_REPORT.md`。strain candidate 结果只作为内部
secondary robustness evidence。

## Reviewer fixed strain-wise retraining

为消除 archived strain checkpoint 与未知 2025 hash membership 不一致的问题，固定 split
重训使用冻结的 `PYTHONHASHSEED=0` candidate manifest，但不再动态重建 unordered legacy
sets。单个 `group × ensemble` 的 canonical 命令为：

```bash
PYTHONPATH=src python scripts/reproduce/run_hierarchical_mic.py \
  --protocol strain --test-group 0 --ensemble 0 --device 0 \
  --strain-manifest experiments/hierarchical_mic/strain/legacy_protocol_manifest.json \
  --output-dir experiments/hierarchical_mic/fixed_strain_retrain/checkpoints \
  --acknowledge-dynamic-legacy-split
```

`--ensemble` 只拆分原7个 seeds 以便多 GPU 并行，不改变模型、数据、optimizer、25 epochs、
held-out evaluation 或 checkpoint-selection contract。任务 ownership、GPU 和输出根目录冻结在
`fixed_strain_retrain/task_manifest.json`；本机与共享 release 的 checkpoint 不互相覆盖。
训练完成后，使用
`scripts/reproduce/evaluate_hierarchical_mic_molecule_disjoint.py` 对新 checkpoint 运行
deterministic `eval()` replay，再由
`scripts/reproduce/summarize_hierarchical_mic_molecule_disjoint.py` 汇总 full、train-seen 和
train-unseen exact-peptide 指标。

本次 `3 × 7` 训练与 replay 已全部完成。正式 pooled full/seen/unseen R2 为
`0.4638/0.5672/0.0942`；严格 unseen 有26,272条 measurements、8,259个 pooled distinct
exact peptides，Spearman/Pearson 为 `0.4070/0.4130`，其中53.12%为 MIC<=16 micromolar。
与论文口径一致的 mean-across-folds full/seen/unseen R2 为
`0.5814/0.6283/0.1089`；full `0.5814` 与历史论文值 `0.5793` 一致，不能拿 pooled
`0.4638` 直接和论文 fold mean 比较。
2,000次 exact-peptide cluster bootstrap 的 unseen R2 95% CI 为 `[0.0687, 0.1191]`。
完整结果、fold-level 边界和 reviewer 回复草稿见
`fixed_strain_retrain/REVIEWER_SENSITIVITY_REPORT.md`。

### Reviewer MIC test-distribution Supplementary Figure

使用已完成的 fixed strain-wise 七成员 ensemble replay，绘制全部86,358条 eligible held-out
measurements 的 pooled MIC histogram 和三个 test folds 的 ECDF：

```bash
python scripts/audit/plot_hierarchical_mic_test_distribution.py
```

默认输出到 `mic_distribution/`：PNG/PDF、逐 cohort summary、histogram bins 和输入 SHA-256
manifest。横轴为 MIC micromolar 的 log2 scale，虚线为16 micromolar。最终版删除总标题和
source脚注，panel标题居中且使用常规字重，并增加`a/b`标记。其PDF以相同SHA-256复制到正式
文稿目录的`Fig_SI_MIC_distribution.pdf`，由TeX作为Supplementary Fig. C3引用；脚本仍只写
`mic_distribution/`，不会自动覆盖文稿资产或论文总PDF。

若大 checkpoint 位于非默认目录，可先生成保留源 size/SHA-256 血缘的 inference-only 权重：

```bash
PYTHONPATH=src python scripts/reproduce/prepare_hierarchical_mic_inference_checkpoints.py \
  --protocol strain --group 1 \
  --strain-manifest experiments/hierarchical_mic/strain/legacy_protocol_manifest.json \
  --checkpoint-dir /path/to/checkpoints \
  --output-dir /path/to/inference_checkpoints
```

验证命令：

```bash
PYTHONPATH=src pytest -q tests/test_hierarchical_mic_runner.py \
  tests/test_strainwise_legacy_equivalence.py \
  tests/test_hierarchical_mic_molecule_overlap.py
```

## Reviewer MIC censor-multiplier sensitivity

CPU/只读 canonical 入口：

```bash
PYTHONPATH=src python \
  scripts/audit/analyze_hierarchical_mic_censor_sensitivity.py
```

该入口从 raw DBAASP concentration 重建 paper-era censor lineage，并用已有 fixed strain-wise 与
canonical phylum-wise 七成员 ensemble predictions 重算 `>V` 的 `1×/2×/4×`、删除右删失和删除
全部删失口径。输出位于 `censor_multiplier_sensitivity/analysis/`；42 MB 逐行表 local-only，
compact counts/metrics/deltas/manifest 可发布。

2026-08-09 维护后，canonical CLI 仅负责参数解析；冻结输入重建、prediction 对齐、closed output
contract 与 manifest 由
`src/apexoracle/evaluation/hierarchical_mic_censor_workflow.py` 统一实现，label/metric 计算保留在
`hierarchical_mic_censor_sensitivity.py`。同输入完整重跑的全部七个输出 SHA-256 与重构前一致。

已验证 raw DBAASP selected measurements 为105,547条，其中普通右删失22,158条；eligible
strain/phylum measurement instances 的普通右删失比例分别为17.30%/17.79%。论文式
mean-across-groups R² 在 strain `1×/2×/4×` 下为 `0.5785/0.5813/0.5634`，删除右删失为
`0.5699`；phylum 对应为 `0.3804/0.3879/0.3748`，删除右删失为 `0.3491`。这支持
held-out signal 不依赖唯一的 `2×` 选择，但 MAE 会随 point encoding 发生可见变化。

完整协议、parser 的 ASCII `>`/Unicode `≥`/`>>` 历史差异、重复 stable-key 审计和英文落稿记录
见 `censor_multiplier_sensitivity/README.md`。本实验不重训，必须表述为已报告指标的
evaluation-label sensitivity；不能升级为 training robustness 或 censor-aware regression。
