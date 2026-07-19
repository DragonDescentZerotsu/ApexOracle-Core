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

## Legacy 清理与恢复

已被统一 runner 替代的 root-level DP/in-house/SM、species/phylum、pooling/eval 脚本及
capsule 内第二份 strain driver 已删除。它们仍可从 annotated tag
`legacy-code-snapshot-2026-07-17` 恢复。
精确的逐文件删除记录见 `legacy_cleanup.json`。

以下文件不是同一模型的复制版本，因此本阶段没有删除：

- Fig. 2c 的 ChemBERTa-MTR、ChemBERTa-MLM、MolFormer 和 PeptideCLM comparator；
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
