# Modality ablation（附录图）

本目录冻结论文最终 modality-ablation 图的 12 个绘图值和绘图契约。当前支持级别为
`paper plot reproducible / legacy training rerun unavailable`：可以从版本化的小型结果表重建
论文图，但不能声称不完整的 legacy checkpoint 已经精确重算这些数值。代码收尾已完成。

## 已由最终绘图资源验证的事实

- Mac 上最终绘图 notebook 是
  `/Users/kirianozan/Documents/Study/Penn/projects/local_figs/figs.ipynb`，cell ID
  `8d84054140b51b7d`，execution count 18，执行时间为 2025-05-22。
- 该 cell 直接定义了四条曲线、三个 holdout 粒度和全部 12 个 R²；这些值已逐项冻结在
  `paper_values.csv`。
- 论文目录中 `modality-ablation.pdf` 的 SHA-256 是
  `0392846c0dd00100235da36648e41515690e2d0d275d628d6e3155cad5b6ae57`。
- 图中四条曲线不是简单的三模态比较，而是 genome-only、text-only、genome+text，以及带
  small-molecule auxiliary data 的完整 ApexOracle。
- 九个候选 genome-only/text-only/genome+text driver 在本机与 node002 的 SHA-256 完全相同；
  两台机器保留的 checkpoint group 却互相补充且均不构成可证明的完整最终运行。

## 根据现有证据作出的推断

前三条曲线最可能来自较早的 ChemBERTa-era
`MIC_with_{genome,text,text_genome}_test_on_non_seen_*` 家族。`candidate_lineage.json` 记录两台
机器上的候选 checkpoint 组别计数，便于以后继续追溯，但这些目录目前只能称为候选血缘。

## 仍待确认的事项

- 没有代码、日志、W&B output 或结果表包含全部 12 个最终数值；notebook 中的数组是当前唯一
  精确来源。
- checkpoint 的 `R2` 字段是单成员训练期间的 best held-out R²，不是图中的 ensemble R²。
  缺少精确 held-out predictions、成员选择和聚合记录时，不能从该字段反推论文图。
- 15 个旧 driver 都会在顶层原地写
  `DataPrepare/Data/DBAASP_id_bact_name_SMILES_MIC_Evo_with_genome.csv`，且没有运行时消费者。
  它们已在不执行的前提下登记 SHA-256 并从发布工作树删除；恢复 tag、分类和受保护数据哈希见
  `legacy_cleanup.json`。

## 绘图入口

安装绘图依赖后执行：

```bash
pip install -e '.[figures]'
MPLBACKEND=Agg python scripts/reproduce/plot_modality_ablation.py \
  --output-pdf /tmp/modality-ablation.pdf \
  --output-png /tmp/modality-ablation.png
```

该命令只读取 `paper_values.csv` 和 `configs/modality_ablation/paper_plot.json`，不会读取或修改
训练数据。CLI 会拒绝用输出路径覆盖上述两个输入文件。

旧 driver 的删除不代表训练血缘已被补齐。发布边界是复现论文图和披露候选家族，不提供缺失的
逐 checkpoint 训练重跑；原始数据、embedding、checkpoint 和历史结果均未删除或修改。
