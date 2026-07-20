# Strain-wise k-mer genome ablation

本目录只维护论文 Fig. 2c 中实际汇报的 **strain-wise** k-mer 消融。2026 年 3 月额外运行的
phylum-wise 和 species-wise k-mer 实验不属于论文汇报范围，不迁入发布入口。

## 已由代码、日志和最终绘图验证的事实

- Mac 最终绘图 notebook 中单模型 `DLM MTR+DLM+kmer` 为 R² `0.4507`、Spearman
  `0.6688`、Pearson `0.6793`；同图单模型 Evo-2 为 R² `0.5032`。
- `/data/fangping/kmer_embeddings/k456_combined` 包含 567 个 windowed tensor，采用
  11,000 nt window、10,000 nt step 和 k=`4,5,6`，维度为 5,376。
- `/data/fangping/kmer_embeddings/k456_global` 包含 567 个 `[1,5376]` tensor。canonical
  producer 对 E. coli ATCC 25922 的真实 FASTA 重算结果与现存 tensor 逐值相同。
- 2026 年完整 global 重建使用 25 epochs、7 个 ensemble member 和一次性未训练的冻结
  `5376→8192→8192` random projection；三组 R² 为 `0.3660`、`0.6308`、`0.5860`，均值
  `0.5276`。
- 现存 global 重建没有保存模型 checkpoint，只保留完整日志和三份最终 metrics CSV。

## 根据现有证据作出的推断

- 论文图是单模型结果，而 2026 重建脚本明确把 ensemble 数从 1 改成 7，因此 `0.5276`
  不能作为 `0.4507` 的精确复现。
- 论文所写的“11.6%”对应 `(0.5032 - 0.4507) / 0.4507 = 11.65%`；若把
  `0.5032` 当分母，下降比例是 10.43%。发布时保留论文原文，但不把两种分母混写。

## 仍待作者确认或无法恢复的事项

- 产生论文 `0.4507` 的精确 producer 模式、随机 projection 状态、训练日志和 checkpoint
  均未在本机或 node002 找到。
- 2026 reconstruction 在创建 frozen projection 之前没有固定 PyTorch seed，因此即使其他
  参数相同，也不保证重新训练得到相同数值。

## 精简入口

从 FASTA 创建一套**新的** tensor；命令拒绝覆盖非空输出目录，也拒绝写入原始 genome 目录：

```bash
python scripts/reproduce/build_kmer_embeddings.py \
  --genome-dir DataPrepare/Data/Genome/ATCC \
  --output-dir results/kmer_ablation/k456_global \
  --mode global
```

`--mode windowed` 保留 11,000/10,000 nt 的分块实现。两种模式的 alphabet 顺序按现存
producer 分别固定为 global `A,C,G,T` 和 windowed legacy `A,T,G,C`。

只读审计现存 tensor：

```bash
python scripts/audit/audit_kmer_embeddings.py \
  --embedding-dir /data/fangping/kmer_embeddings/k456_global \
  --output results/kmer_global_manifest.csv
```

运行 2026 global reconstruction candidate：

```bash
APEXORACLE_KMER_EMBEDDINGS_DIR=/data/fangping/kmer_embeddings/k456_global \
python scripts/reproduce/run_hierarchical_mic.py \
  --config configs/hierarchical_mic/legacy_kmer_reconstruction.yaml \
  --protocol strain \
  --test-group 0 \
  --acknowledge-dynamic-legacy-split
```

默认输出位于 `results/kmer_ablation/`，不会覆盖 `/data/fangping` 中的 tensor、日志或数据。

## 文件

- `paper_values.csv`：最终绘图中的论文值。
- `reconstruction_metrics.csv`：2026 global/25-epoch/7-model 完整重建结果。
- `reconstruction_audit.json`：两套血缘及差异边界。
- `global_embedding_manifest.csv`、`windowed_embedding_manifest.csv`：预计算 tensor 身份。
